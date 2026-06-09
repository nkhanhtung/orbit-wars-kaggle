from __future__ import annotations
import dataclasses
import os
import sys
from dataclasses import dataclass
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import torch
from torch import Tensor
from orbit_lite.geometry import fleet_speed
from orbit_lite.intercept_aim import intercept_angle
from orbit_lite.movement import MovementConfig, PlanetMovement
from orbit_lite.movement_step import apply_private_planned_launches, concat_launch_entries, disambiguate_duplicate_launches, ensure_planet_movement, infer_planned_launches_from_entries
from orbit_lite.obs import parse_obs
from orbit_lite.distance_cache import build_distance_cache
from orbit_lite.planner_core import _candidate_indices, _empty_entries, _greedy_select, _plan_regroup, build_target_shortlist, capture_floor, empty_action_row, entries_to_sparse_payload, largest_initial_player_count, make_launch_set, reachable_mask, reinforcement_timing_factor, safe_drain, score_candidates
from orbit_lite.adapter import single_obs_to_tensor, sparse_action_row_to_moves

@dataclass(frozen=True)
class ProducerLiteConfig:
    """Cấu hình chiến thuật cấp cao của agent heuristic.

Các tham số trong lớp này điều khiển độ sâu dự báo, số nguồn/đích được xét, số wave tối đa mỗi lượt, ngưỡng ROI, mức chia quân và cơ chế regroup. Đây không phải thuật toán học máy mà là bộ tham số hand-tuned cho planner dựa trên mô phỏng tiến về tương lai."""
    horizon: int = 18
    max_sources_per_lane: int = 14
    max_offensive_targets: int = 10
    max_defensive_targets: int = 4
    max_waves_per_turn: int = 7
    roi_threshold: float = 1.25
    min_ships_to_launch: float = 5.0
    send_fracs: tuple[float, ...] = (0.34, 0.58, 1.0)
    enable_regroup: bool = True
    max_regroup_time: float = 5.0
    regroup_pressure_delta_min: float = 0.25
    max_regroup_sources_per_lane: int = 6
    max_regroup_targets_per_source: int = 3
    regroup_pressure_norm: str = 'none'
    regroup_time_penalty_weight: float = 0.001

def _movement_config(config: ProducerLiteConfig, *, player_count: int) -> MovementConfig:
    """Tạo cấu hình mô phỏng chuyển động cho PlanetMovement.

Hàm gắn horizon của planner vào bộ dự báo, bật tracking fleet để agent có thể theo dõi đội tàu đang bay và dùng số người chơi hiện tại để mô phỏng owner/garrison đúng với trận 2 hoặc 4 người."""
    return MovementConfig(movement_horizon=int(config.horizon), drift_epsilon=0.001, track_fleets=True, player_count=int(player_count), max_tracked_fleets=128)

def cheap_enemy_pressure(obs, cache, *, horizon: float, player_id: int) -> Tensor:
    """Ước lượng nhanh áp lực địch lên từng hành tinh.

Thuật toán cộng hai nguồn áp lực: hành tinh địch có thể phóng quân tới mục tiêu trong horizon và các fleet địch đang bay gần mục tiêu. Khoảng cách càng gần và số tàu càng lớn thì điểm pressure càng cao. Đây là heuristic rẻ để hỗ trợ phòng thủ/regroup, không phải mô phỏng combat đầy đủ."""
    P = int(obs.P)
    device = obs.device
    dtype = obs.ships.dtype
    if P == 0:
        return torch.zeros(P, dtype=dtype, device=device)
    pid = int(player_id)
    H = max(float(horizon), 1e-06)
    d0 = cache.cross_dist[0].to(dtype)
    ships = obs.ships.to(dtype)
    speeds = fleet_speed(ships.clamp(min=1e-06))
    reach_dist = (speeds.view(P, 1) * H).clamp(min=1e-06)
    enemy = obs.alive & (obs.owner_abs >= 0) & (obs.owner_abs != pid)
    eye = torch.eye(P, device=device, dtype=torch.bool)
    valid = enemy.view(P, 1) & obs.alive.view(1, P) & ~eye
    decay = (1.0 - d0 / reach_dist).clamp(min=0.0)
    contrib_planets = torch.where(valid, ships.view(P, 1) * decay, torch.zeros_like(decay))
    pressure = contrib_planets.sum(dim=0)
    f_alive = obs.f_alive
    if bool(f_alive.any()):
        f_owner = obs.f_owner.to(torch.long)
        f_enemy = f_alive & (f_owner >= 0) & (f_owner != pid)
        if bool(f_enemy.any()):
            fx = obs.f_x.to(dtype)[f_enemy]
            fy = obs.f_y.to(dtype)[f_enemy]
            fs = obs.f_ships.to(dtype)[f_enemy].clamp(min=1e-06)
            f_speed = fleet_speed(fs)
            f_reach = (f_speed * H).clamp(min=1e-06)
            tx = obs.x.to(dtype).view(1, P)
            ty = obs.y.to(dtype).view(1, P)
            dxe = fx.view(-1, 1) - tx
            dye = fy.view(-1, 1) - ty
            d_ft = torch.sqrt((dxe * dxe + dye * dye).clamp(min=0.0))
            decay_f = (1.0 - d_ft / f_reach.view(-1, 1)).clamp(min=0.0)
            tgt_alive = obs.alive.view(1, P)
            decay_f = torch.where(tgt_alive, decay_f, torch.zeros_like(decay_f))
            contrib_fleets = fs.view(-1, 1) * decay_f
            pressure = pressure + contrib_fleets.sum(dim=0)
    return pressure

def plan_lite_waves(*, movement: PlanetMovement, obs, obs_tensors: dict, cache, garrison_status, prod: Tensor, alive_by_step: Tensor, config: ProducerLiteConfig, player_count: int):
    """Sinh và chọn các wave tấn công chính trong một lượt.

Luồng thuật toán: lọc nguồn có quân, dựng shortlist mục tiêu, tính safe_drain, tính capture_floor theo ETA, thử nhiều tỉ lệ gửi quân, lọc reachable_mask, tính góc bắn đón đầu bằng intercept_angle, chấm điểm bằng score_candidates, rồi dùng greedy selector để chọn những launch có lợi nhất mà không vượt ngân sách quân của từng nguồn."""
    P = obs.P
    device = obs.device
    dtype = obs.ships.dtype
    pid = int(obs.player_id)
    H_axis = int(garrison_status.ships.shape[-1])
    H = max(H_axis - 1, 0)
    K_eta = max(1, min(int(config.horizon), H))
    W = max(1, int(config.max_waves_per_turn))
    source_mask = obs.owned & obs.alive & (obs.ships >= float(config.min_ships_to_launch))
    if not bool(source_mask.any()):
        return _empty_entries(device, dtype)
    S_cap = max(1, min(int(config.max_sources_per_lane), P))
    source_idx, source_exists = _candidate_indices(obs.ships, source_mask, S_cap)
    target_idx, target_exists = build_target_shortlist(obs, obs_tensors, garrison_status, cache, config=config, K_eta=K_eta, H=H, prod=prod, source_mask=source_mask)
    if not bool(target_exists.any()):
        return _empty_entries(device, dtype)
    S = int(source_idx.shape[0])
    T = int(target_idx.shape[0])
    target_is_mine = obs.owned[target_idx.clamp(0, P - 1)]
    source_ships = obs.ships[source_idx.clamp(0, P - 1)].to(dtype)
    H_eff = torch.full((), float(H), dtype=dtype, device=device)
    drain = safe_drain(garrison_status, source_idx=source_idx, source_ships=source_ships, H_eff=H_eff, player_id=pid)
    eta_cap = torch.full((T,), float(K_eta), dtype=dtype, device=device)
    floor = capture_floor(garrison_status, target_idx=target_idx, k_max=K_eta, capture_overhead=1.0, player_id=pid)
    K = int(floor.shape[-1])
    fracs = torch.tensor(tuple(config.send_fracs), dtype=dtype, device=device).clamp(0.05, 1.0)
    G = int(fracs.numel())
    drain_st = drain.view(S, 1, 1).expand(S, T, G)
    min_send = torch.full_like(drain_st, float(config.min_ships_to_launch))
    sizes = torch.maximum((drain_st * fracs.view(1, 1, G)).floor(), min_send)
    sizes = torch.minimum(sizes, drain_st.floor()).clamp(min=0.0)
    active = reachable_mask(movement, source_idx=source_idx, target_idx=target_idx, fleet_sizes=sizes, eta_cap=eta_cap)
    aim = intercept_angle(movement, source_idx.view(S, 1, 1), target_idx.view(1, T, 1), sizes, active=active)
    angle = aim['angle']
    eta = aim['eta']
    viable = aim['viable'] & (eta <= eta_cap.view(1, T, 1))
    if K > 0:
        k_arr = (eta.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)
        floor_at_arr = floor.view(1, T, 1, K).expand(S, T, G, K).gather(-1, k_arr.unsqueeze(-1)).squeeze(-1)
    else:
        floor_at_arr = torch.ones(S, T, G, dtype=dtype, device=device)
    clears_floor = sizes >= floor_at_arr
    src_neq_tgt = source_idx.view(S, 1, 1) != target_idx.view(1, T, 1)
    valid = viable & clears_floor & (sizes >= float(config.min_ships_to_launch)) & src_neq_tgt & source_exists.view(S, 1, 1) & target_exists.view(1, T, 1)
    L = 1
    C = S * T * G
    cand_src = source_idx.view(S, 1, 1).expand(S, T, G).reshape(C, L)
    cand_tgt_slot = target_idx.view(1, T, 1).expand(S, T, G).reshape(C)
    cand_tgt_short = torch.arange(T, device=device).view(1, T, 1).expand(S, T, G).reshape(C)
    cand_send = torch.where(valid, sizes, torch.zeros_like(sizes)).reshape(C, L)
    cand_angle = angle.reshape(C, L)
    cand_eta = torch.where(valid, eta, torch.ones_like(eta)).reshape(C, L)
    cand_active = valid.reshape(C, L)
    cand_valid = valid.reshape(C)
    cand_is_def = target_is_mine[cand_tgt_short]
    launches = make_launch_set(source_slots=cand_src, target_slots=cand_tgt_slot.unsqueeze(-1).expand(C, L), ships=cand_send, eta=cand_eta, valid=cand_active & cand_valid.unsqueeze(-1), player_id=pid)
    score = score_candidates(garrison_status, prod=prod, alive_by_step=alive_by_step, player_count=int(player_count), launches=launches, player_id=pid)
    score = torch.where(cand_valid, score, torch.full_like(score, float('-inf')))
    wave_entries, leftover = _greedy_select(P=P, W=W, device=device, dtype=dtype, score=score, cand_src=cand_src, cand_send=cand_send, cand_angle=cand_angle, cand_eta=cand_eta, cand_active=cand_active, cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short, cand_is_def=cand_is_def, source_budget=obs.ships.to(dtype).clone(), target_exists=target_exists, roi_threshold=float(config.roi_threshold))
    if not bool(config.enable_regroup):
        return wave_entries
    enemy_mass = cheap_enemy_pressure(obs, cache, horizon=float(K_eta), player_id=pid)
    regroup_entries = _plan_regroup(movement=movement, obs=obs, obs_tensors=obs_tensors, garrison_status=garrison_status, leftover=leftover, original_ships=obs.ships.to(dtype), pressure=enemy_mass, config=config, H=H)
    return concat_launch_entries([wave_entries, regroup_entries])

def run_turn(obs_tensors: dict, *, config: ProducerLiteConfig, player_count: int, memory) -> dict:
    """Chạy toàn bộ pipeline ra quyết định cho một turn.

Hàm parse observation, cập nhật/khởi tạo mô hình chuyển động, dựng cache khoảng cách, lấy dự báo garrison, lập kế hoạch tấn công, lập kế hoạch regroup nếu cần, ghép các lệnh launch và chuyển thành action sparse cho Kaggle."""
    device = obs_tensors['planets'].device
    obs = parse_obs(obs_tensors)
    P = obs.P
    if P == 0:
        return empty_action_row(device)
    movement = ensure_planet_movement(obs_tensors=obs_tensors, expected_cfg=_movement_config(config, player_count=int(player_count)), cached_movement=getattr(memory, 'movement', None))
    memory.movement = movement
    cache = build_distance_cache(movement, max_k=int(config.horizon))
    H = int(config.horizon)
    status = movement.garrison_status(max_horizon=H)
    alive_by_step = movement.alive_by_step[:H + 1]
    entries = plan_lite_waves(movement=movement, obs=obs, obs_tensors=obs_tensors, cache=cache, garrison_status=status, prod=movement.planet_prod, alive_by_step=alive_by_step, config=config, player_count=int(player_count))
    entries = disambiguate_duplicate_launches(entries)
    launches = infer_planned_launches_from_entries(obs_tensors=obs_tensors, movement=movement, entries=entries, player_id=int(obs.player_id))
    apply_private_planned_launches(movement=movement, launches=launches, owner_id=int(obs.player_id), obs_tensors=obs_tensors)
    planet_ids = obs_tensors['planets'][..., 0].long()
    return entries_to_sparse_payload(entries, planet_ids=planet_ids)
CONFIG_4P = dataclasses.replace(ProducerLiteConfig(), horizon=14, max_sources_per_lane=8, max_offensive_targets=8, max_defensive_targets=3, max_waves_per_turn=7, roi_threshold=1.35, max_regroup_time=6.0, max_regroup_targets_per_source=8)

def _config_for(player_count: int) -> ProducerLiteConfig:
    """Chọn cấu hình agent tương ứng với số người chơi.

Hàm hiện dùng cùng cấu hình ProducerLiteConfig cho các chế độ, nhưng giữ điểm mở rộng để sau này có thể tách tham số riêng cho trận 2 người và 4 người."""
    return CONFIG_4P if int(player_count) >= 4 else ProducerLiteConfig()

class ProducerLiteMemory:
    """Bộ nhớ ngắn hạn giữa các lượt của agent.

Lớp giữ PlanetMovement để tránh xây lại dự báo từ đầu mỗi turn. Khi game mới bắt đầu hoặc observation không còn khớp, reset sẽ xoá trạng thái cũ để tránh dùng cache sai."""

    def __init__(self) -> None:
        """Khởi tạo bộ nhớ runtime với movement rỗng."""
        self.movement = None
        self.cached_player_count: int | None = None
        self.last_sparse_action_row: dict | None = None

    def reset(self) -> None:
        """Xoá trạng thái dự báo đã lưu, dùng khi bắt đầu ván mới hoặc cần đồng bộ lại observation."""
        self.movement = None
        self.cached_player_count = None
        self.last_sparse_action_row = None

class ProducerLiteRuntime:
    """Runtime bao quanh planner để dùng trong môi trường Kaggle.

Lớp chịu trách nhiệm giữ memory, chọn config, gọi run_turn và chuyển action tensor nội bộ thành danh sách move đúng format submission."""

    def __init__(self, memory: ProducerLiteMemory | None=None) -> None:
        """Khởi tạo runtime và bộ nhớ dùng chung cho nhiều lượt."""
        self.memory = memory if memory is not None else ProducerLiteMemory()

    def reset(self) -> None:
        """Reset memory của runtime trước một ván mới hoặc khi muốn xoá dự báo cũ."""
        self.memory.reset()

    def tensor_action(self, obs_tensors: dict):
        """Nhận observation dạng tensor và trả action sparse dạng tensor.

Đây là API nội bộ dùng cho batch/test local: observation đã được adapter chuẩn hoá, planner chỉ tập trung xử lý chiến thuật."""
        mem = self.memory
        if bool((obs_tensors['step'] == 0).all()):
            mem.cached_player_count = None
        if mem.cached_player_count is None:
            mem.cached_player_count = largest_initial_player_count(obs_tensors)
        config = _config_for(mem.cached_player_count)
        row = run_turn(obs_tensors, config=config, player_count=int(mem.cached_player_count), memory=mem)
        mem.last_sparse_action_row = row
        return row
_RUNTIME = ProducerLiteRuntime()

def agent(obs):
    """Hàm entry point bắt buộc khi nộp Kaggle Orbit Wars.

Kaggle gọi agent(obs) mỗi turn. Hàm chuyển obs thô sang tensor, gọi runtime để lập kế hoạch, rồi đổi action sparse thành list [[source_id, angle, ships], ...]. Logic phải gọn và an toàn vì lỗi ở đây sẽ làm submission fail."""
    player = obs.get('player', 0) if isinstance(obs, dict) else obs.player
    player_id = int(player)
    obs_tensors = single_obs_to_tensor(obs, player_id=player_id)
    with torch.no_grad():
        sparse_row = _RUNTIME.tensor_action(obs_tensors)
    return sparse_action_row_to_moves(sparse_row, obs, player_id=player_id)
