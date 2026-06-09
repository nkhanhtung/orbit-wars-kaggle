from __future__ import annotations
import torch
from torch import Tensor
from .garrison_launch import GarrisonFlowDiff, LaunchSet, sparse_launch_flow_delta
from .movement import PlanetGarrisonStatus, PlanetMovement
from .geometry import fleet_speed
from .intercept_aim import intercept_angle
from .movement_aiming import LAUNCH_SURFACE_OFFSET, TARGET_HIT_SURFACE_OFFSET
from .movement_step import LaunchEntries
from .distance_cache import min_distance_to_targets

def largest_initial_player_count(obs_tensors: dict) -> int:
    """Lấy số người chơi lớn nhất trong trận.

Hàm ưu tiên metadata player_count, nếu thiếu thì suy luận từ owner ban đầu của các hành tinh. Kết quả dùng cho garrison projection và scoring theo từng người chơi."""
    metadata_count = obs_tensors.get('player_count')
    if metadata_count is not None:
        count = int(metadata_count.flatten()[0].item()) if isinstance(metadata_count, Tensor) else int(metadata_count)
        if count in (2, 4):
            return count
    initial = obs_tensors['initial_planets']
    pid = initial[:, 0]
    owner = initial[:, 1]
    mask = (pid >= 0) & (owner >= 0)
    owners = owner[mask]
    n_max = 2
    if owners.numel() > 0:
        n_max = max(n_max, int(torch.unique(owners.long()).numel()))
    return n_max

def make_launch_set(*, source_slots: Tensor, target_slots: Tensor, ships: Tensor, eta: Tensor, valid: Tensor, player_id: int) -> LaunchSet:
    """Đóng gói tensor ứng viên launch thành LaunchSet.

Hàm tạo owner tensor theo player_id và ép kiểu source/target/valid để flow simulator nhận dữ liệu chuẩn."""
    owner = torch.full_like(source_slots, int(player_id), dtype=torch.long)
    return LaunchSet(source_slots=source_slots.to(torch.long), target_slots=target_slots.to(torch.long), ships=ships, eta=eta, owner=owner, valid=valid.to(torch.bool))

def competitive_score(diff: GarrisonFlowDiff, *, player_id: int) -> Tensor:
    """Tính điểm cạnh tranh của một launch.

Công thức lấy net gain của mình trừ tổng net gain của đối thủ. Launch tốt là launch vừa tăng lợi thế của mình vừa làm giảm hoặc không tăng lợi thế đối phương."""
    net = diff.net_ship_delta
    me = net[..., int(player_id)]
    opp = net.sum(dim=-1) - me
    return me - opp

def score_candidates(status: PlanetGarrisonStatus, *, prod: Tensor, alive_by_step: Tensor, player_count: int, launches: LaunchSet, player_id: int) -> Tensor:
    """Chấm điểm các ứng viên launch bằng mô phỏng flow chính xác.

Hàm gọi sparse_launch_flow_delta để so sánh projection trước/sau launch, sau đó chuyển delta thành competitive_score."""
    diff = sparse_launch_flow_delta(status, prod=prod, alive_by_step=alive_by_step, player_count=int(player_count), launches=launches, player_id=int(player_id))
    return competitive_score(diff, player_id=int(player_id))

def _stable_topk_indices(ranked: Tensor, k: int) -> Tensor:
    """Chọn top-k ổn định giữa CPU và GPU.

Thuật toán thêm tie-break theo index để kết quả không phụ thuộc cách torch.topk xử lý điểm bằng nhau trên từng thiết bị."""
    order = torch.argsort(ranked, dim=-1, descending=True, stable=True)
    return order[..., :max(1, int(k))]

def _stable_argmax(scores: Tensor) -> Tensor:
    """Lấy argmax ổn định khi có nhiều phần tử cùng điểm."""
    C = int(scores.shape[-1])
    is_max = scores == scores.max(dim=-1, keepdim=True).values
    idx = torch.arange(C, device=scores.device).expand_as(scores)
    return torch.where(is_max, idx, torch.full_like(idx, C)).argmin(dim=-1)

def _candidate_indices(values: Tensor, mask: Tensor, cap: int) -> tuple[Tensor, Tensor]:
    """Lấy chỉ số candidate tốt nhất từ một vector score và mask hợp lệ.

Hàm dùng stable top-k để tạo shortlist nguồn hoặc mục tiêu mà không làm kết quả dao động giữa môi trường chạy."""
    p_count = values.shape[0]
    k = p_count if cap <= 0 else min(int(cap), p_count)
    neg_inf = torch.full_like(values, float('-inf'))
    ranked = torch.where(mask, values, neg_inf)
    top_idx = _stable_topk_indices(ranked, max(1, k))
    top_vals = ranked[top_idx]
    return (top_idx, top_vals > float('-inf'))

def is_comet_planet(obs_tensors: dict, P: int, device: torch.device) -> Tensor | None:
    """Nhận diện hành tinh comet/sự kiện đặc biệt dựa trên metadata hình học/gameplay."""
    comet_ids = obs_tensors.get('comet_planet_ids')
    planets = obs_tensors.get('planets')
    if comet_ids is None or planets is None:
        return None
    planet_ids = planets[..., 0].long()
    comet_ids = comet_ids.to(device=device)
    mask = torch.zeros(P, dtype=torch.bool, device=device)
    for c in range(int(comet_ids.shape[-1])):
        cid = comet_ids[c]
        mask = mask | (planet_ids == cid) & (cid >= 0)
    return mask

def reinforcement_timing_factor(eta: Tensor, *, eta_free: float, eta_scale: float) -> Tensor:
    """Tính hệ số timing cho việc tăng viện.

Hệ số ưu tiên launch đến kịp thời điểm nguy hiểm; đến quá muộn hoặc quá xa horizon thì giá trị giảm."""
    scale = max(float(eta_scale), 1e-06)
    return ((eta - float(eta_free)) / scale).clamp(0.0, 1.0)

def capture_floor(garrison_status: PlanetGarrisonStatus, *, target_idx: Tensor, k_max: int, capture_overhead: float, player_id: int, reinforcement: Tensor | None=None) -> Tensor:
    """Tính số tàu tối thiểu cần gửi để chiếm hoặc giữ mục tiêu theo từng ETA.

Thuật toán đọc projection garrison tại thời điểm đến, so sánh quân mình với quân mạnh nhất của đối thủ/trung lập, rồi cộng overhead để tạo ngưỡng capture an toàn."""
    ships = garrison_status.ships
    owner = garrison_status.owner
    dtype = ships.dtype if ships.is_floating_point() else torch.float32
    T = target_idx.shape[0]
    H_axis = int(ships.shape[-1])
    P = int(ships.shape[0])
    K = max(0, min(int(k_max), H_axis - 1))
    if K == 0:
        return torch.empty(T, 0, dtype=dtype, device=ships.device)
    tgt = target_idx.clamp(min=0, max=max(P - 1, 0))
    gathered = ships[tgt].to(dtype=dtype)
    owner_g = owner[tgt]
    k_idx = torch.arange(1, K + 1, device=ships.device).view(1, K).expand(T, K)
    defenders = gathered.gather(-1, k_idx)
    mine_at_k = owner_g.gather(-1, k_idx) == int(player_id)
    if reinforcement is not None:
        assert reinforcement.shape[-1] >= K, f'reinforcement last dim {reinforcement.shape[-1]} < capture_floor K={K}'
        extra = reinforcement[..., :K].to(dtype=dtype, device=ships.device)
    else:
        extra = 0.0
    cap = (defenders + float(capture_overhead) + extra).clamp(min=1.0).ceil()
    return torch.where(mine_at_k, torch.ones_like(cap), cap)

def attack_target_mask(obs, obs_tensors: dict) -> Tensor:
    """Tạo mask mục tiêu nên xét cho tấn công.

Hàm thường chọn hành tinh địch/trung lập còn sống, loại mục tiêu không hợp lệ hoặc không đáng đánh."""
    mask = (obs.is_enemy | obs.is_neutral) & obs.alive
    comet = is_comet_planet(obs_tensors, obs.P, obs.device)
    if comet is not None:
        mask = mask & ~comet
    return mask

def friendly_flip_targets(obs, garrison_status: PlanetGarrisonStatus, *, H: int, prod: Tensor) -> tuple[Tensor, Tensor]:
    """Xác định hành tinh của mình có nguy cơ bị lật owner trong projection.

Các mục tiêu này được đưa vào nhóm phòng thủ/tăng viện thay vì tấn công thuần."""
    P = obs.P
    device = obs.device
    pid = int(obs.player_id)
    if H <= 0:
        z = torch.zeros(P, device=device)
        return (torch.zeros(P, dtype=torch.bool, device=device), z)
    owner_h = garrison_status.owner[..., 1:]
    flips = obs.owned.unsqueeze(-1) & (owner_h != pid)
    any_flip = flips.any(dim=-1)
    flip_turn = _stable_argmax(flips.to(torch.int64)) + 1
    remaining = (float(H) - flip_turn.to(prod.dtype)).clamp(min=0.0)
    urgency = prod * remaining + obs.ships
    urgency = torch.where(any_flip, urgency, torch.full_like(urgency, float('-inf')))
    return (any_flip, urgency)

def build_target_shortlist(obs, obs_tensors, garrison_status, cache, *, config, K_eta, H, prod, source_mask):
    """Dựng danh sách mục tiêu ưu tiên cho planner.

Thuật toán kết hợp mục tiêu tấn công, mục tiêu phòng thủ, production, khoảng cách, trạng thái garrison và nguy cơ bị lật để giảm không gian tìm kiếm trước khi chấm điểm sâu."""
    P = obs.P
    device = obs.device
    n_attack = max(1, min(int(config.max_offensive_targets), P))
    R = max(0, min(int(config.max_defensive_targets), P))
    attack_mask = attack_target_mask(obs, obs_tensors)
    proximity = min_distance_to_targets(cache, source_mask, attack_mask, max_k=K_eta)
    dtype = proximity.dtype
    static_outer_bonus = (~obs.is_orbiting & attack_mask).to(dtype) * 4.0
    prod_bonus = prod.to(dtype) * 8.0
    resistance_penalty = obs.ships.to(dtype).clamp(min=0.0) * 0.22
    neutral_bonus = obs.is_neutral.to(dtype) * 2.0
    capture_efficiency = prod.to(dtype) / (obs.ships.to(dtype).clamp(min=0.0) + 1.0)
    capture_efficiency_bonus = capture_efficiency * 3.5
    attack_pref_raw = -proximity + prod_bonus + static_outer_bonus + neutral_bonus + capture_efficiency_bonus - resistance_penalty
    attack_pref = torch.where(attack_mask, attack_pref_raw, torch.full_like(proximity, float('-inf')))
    atk_idx, atk_exists = _candidate_indices(attack_pref, attack_mask, n_attack)
    if R > 0:
        flip_mask, urgency = friendly_flip_targets(obs, garrison_status, H=H, prod=prod)
        def_idx, def_exists = _candidate_indices(urgency, flip_mask, R)
        target_idx = torch.cat([atk_idx, def_idx], dim=0)
        target_exists = torch.cat([atk_exists, def_exists], dim=0)
    else:
        target_idx, target_exists = (atk_idx, atk_exists)
    return (target_idx, target_exists)

def reachable_mask(movement: PlanetMovement, *, source_idx: Tensor, target_idx: Tensor, fleet_sizes: Tensor, eta_cap: Tensor, eps: float=0.0001) -> Tensor:
    """Lọc source-target-size có thể tới kịp trong horizon.

Hàm dùng khoảng cách dự báo, tốc độ fleet và ETA cap để tạo strict-superset mask. Mục tiêu là loại candidate chắc chắn vô ích trước khi gọi intercept_angle tốn hơn."""
    S, T, G = fleet_sizes.shape
    P = int(movement.P)
    dt = movement.dtype
    K = max(1, min(int(movement.movement_horizon), int(torch.ceil(eta_cap.max()).item())))
    src = source_idx.clamp(0, P - 1)
    tgt = target_idx.clamp(0, P - 1)
    sx = movement.x[0][src].view(S, 1, 1)
    sy = movement.y[0][src].view(S, 1, 1)
    tx = movement.x[:K + 1].gather(1, tgt.view(1, T).expand(K + 1, T))
    ty = movement.y[:K + 1].gather(1, tgt.view(1, T).expand(K + 1, T))
    ax = tx[:K, :].view(1, K, T)
    ay = ty[:K, :].view(1, K, T)
    bx = tx[1:, :].view(1, K, T)
    by = ty[1:, :].view(1, K, T)
    abx = bx - ax
    aby = by - ay
    apx = sx - ax
    apy = sy - ay
    denom = (abx * abx + aby * aby).clamp(min=1e-12)
    u = ((apx * abx + apy * aby) / denom).clamp(0.0, 1.0)
    cx = ax + u * abx
    cy = ay + u * aby
    seg_dist = torch.sqrt(((sx - cx) ** 2 + (sy - cy) ** 2).clamp(min=0.0))
    src_r = movement.radii[src].view(S, 1, 1)
    tgt_r = movement.radii[tgt].view(1, 1, T)
    gap = src_r + tgt_r + (LAUNCH_SURFACE_OFFSET + TARGET_HIT_SURFACE_OFFSET)
    surf = (seg_dist - gap).clamp(min=0.0)
    kv = torch.arange(1, K + 1, device=movement.device, dtype=dt).view(1, K, 1)
    ratio = surf / kv
    within = kv <= eta_cap.view(1, 1, T)
    ratio = torch.where(within, ratio, torch.full_like(ratio, float('inf')))
    min_ratio = ratio.amin(dim=1)
    speed = fleet_speed(fleet_sizes.clamp(min=1.0))
    reachable = min_ratio.unsqueeze(-1) <= speed * (1.0 + float(eps))
    distinct = (src.view(S, 1) != tgt.view(1, T)).unsqueeze(-1)
    return reachable & distinct

def _greedy_select(*, P, W, device, dtype, score, cand_src, cand_send, cand_angle, cand_eta, cand_active, cand_tgt_slot, cand_tgt_short, cand_is_def, source_budget, target_exists, roi_threshold) -> LaunchEntries:
    """Chọn tập launch cuối cùng bằng greedy có ràng buộc ngân sách nguồn.

Thuật toán sắp ứng viên theo ROI/score, duyệt lần lượt và chỉ nhận launch nếu còn đủ quân an toàn ở source, không vượt số wave tối đa và điểm vượt ngưỡng."""
    C, L = (int(cand_src.shape[0]), int(cand_src.shape[1]))
    target_taken = ~target_exists.clone()
    defended = torch.zeros(P, dtype=torch.bool, device=device)
    used_src = torch.zeros(P, dtype=torch.bool, device=device)
    w_src = torch.zeros(W, L, dtype=torch.long, device=device)
    w_send = torch.zeros(W, L, dtype=dtype, device=device)
    w_angle = torch.zeros(W, L, dtype=dtype, device=device)
    w_eta = torch.ones(W, L, dtype=dtype, device=device)
    w_tgt = torch.zeros(W, L, dtype=torch.long, device=device)
    w_active = torch.zeros(W, L, dtype=torch.bool, device=device)
    for w in range(W):
        taken_cand = target_taken[cand_tgt_short]
        budget_at = source_budget[cand_src]
        can_fund = ((cand_send <= budget_at) | ~cand_active).all(dim=-1)
        tgt_used_as_src = used_src[cand_tgt_slot]
        contrib_defended = (defended[cand_src] & cand_active).any(dim=-1)
        mask = torch.isfinite(score) & ~taken_cand & can_fund & ~tgt_used_as_src & ~contrib_defended
        masked = torch.where(mask, score, torch.full_like(score, float('-inf')))
        best_c = _stable_argmax(masked)
        best_score = masked[best_c]
        fired = bool(torch.isfinite(best_score) & (best_score > roi_threshold))
        if not fired:
            break
        sel_src = cand_src[best_c]
        sel_send = cand_send[best_c]
        sel_active = cand_active[best_c]
        w_src[w] = sel_src
        w_send[w] = torch.where(sel_active, sel_send, torch.zeros_like(sel_send))
        w_angle[w] = cand_angle[best_c]
        w_eta[w] = cand_eta[best_c]
        w_tgt[w] = cand_tgt_slot[best_c]
        w_active[w] = sel_active
        debit = torch.zeros_like(source_budget)
        debit.scatter_add_(0, sel_src, torch.where(sel_active, sel_send, torch.zeros_like(sel_send)))
        source_budget = (source_budget - debit).clamp(min=0.0)
        target_taken[cand_tgt_short[best_c]] = True
        src_mark = torch.zeros(P, dtype=torch.long, device=device)
        src_mark.scatter_add_(0, sel_src, sel_active.to(torch.long))
        used_src = used_src | (src_mark > 0)
        sel_tgt = cand_tgt_slot[best_c]
        sel_is_def = bool(cand_is_def[best_c])
        defended[sel_tgt] = defended[sel_tgt] | sel_is_def
    WL = W * L
    entries = LaunchEntries(source_slots=w_src.reshape(WL), target_slots=w_tgt.reshape(WL), ships=torch.where(w_active, w_send, torch.zeros_like(w_send)).reshape(WL), angle=torch.where(w_active, w_angle, torch.zeros_like(w_angle)).reshape(WL), eta=torch.where(w_active, w_eta, torch.ones_like(w_eta)).reshape(WL), valid=w_active.reshape(WL))
    return (entries, source_budget)

def _plan_regroup(*, movement, obs, obs_tensors, garrison_status, leftover, original_ships, pressure, config, H) -> LaunchEntries:
    """Lập kế hoạch gom quân/tăng viện giữa các hành tinh của mình.

Thuật toán tìm nguồn dư quân, tìm mục tiêu friendly chịu áp lực cao hoặc có nguy cơ mất, tính góc/ETA và gửi quân nếu timing hợp lý. Đây là lớp phòng thủ bổ sung sau offensive planner."""
    P = obs.P
    device = obs.device
    dtype = original_ships.dtype
    pid = int(obs.player_id)
    min_send = float(config.min_ships_to_launch)
    src_mask = obs.owned & obs.alive & (leftover >= min_send)
    if not bool(src_mask.any()):
        return _empty_entries(device, dtype)
    S_cap = max(1, min(int(config.max_regroup_sources_per_lane), P))
    src_idx, src_exists = _candidate_indices(leftover, src_mask, S_cap)
    S = int(src_idx.shape[0])
    leftover_s = leftover[src_idx.clamp(0, P - 1)]
    orig_s = original_ships[src_idx.clamp(0, P - 1)]
    H_eff = torch.full((), float(H), dtype=dtype, device=device)
    drain_s = safe_drain(garrison_status, source_idx=src_idx, source_ships=orig_s, H_eff=H_eff, player_id=pid)
    committed_s = (orig_s - leftover_s).clamp(min=0.0)
    regroup_cap = torch.minimum(leftover_s, (drain_s - committed_s).clamp(min=0.0)).floor()
    can_send = src_exists & (regroup_cap >= min_send)
    if not bool(can_send.any()):
        return _empty_entries(device, dtype)
    dst_mask = obs.owned & obs.alive
    comet = is_comet_planet(obs_tensors, P, device)
    if comet is not None:
        dst_mask = dst_mask & ~comet
    T_cap = max(1, min(int(config.max_regroup_targets_per_source), P))
    dst_idx, dst_exists = _candidate_indices(pressure, dst_mask, T_cap)
    T = int(dst_idx.shape[0])
    regroup_active = reachable_mask(movement, source_idx=src_idx, target_idx=dst_idx, fleet_sizes=regroup_cap.view(S, 1, 1).expand(S, T, 1), eta_cap=torch.full((T,), float(config.max_regroup_time), device=device)).squeeze(-1)
    aim = intercept_angle(movement, src_idx.unsqueeze(1), dst_idx.unsqueeze(0), regroup_cap.unsqueeze(1), active=regroup_active)
    angle = aim['angle']
    eta = aim['eta']
    viable = aim['viable']
    src_pres = pressure[src_idx.clamp(0, P - 1)].view(S, 1)
    dst_pres = pressure[dst_idx.clamp(0, P - 1)].view(1, T)
    gap = dst_pres - src_pres
    owner = garrison_status.owner
    H_axis = int(owner.shape[-1])
    dst_owner = owner[dst_idx.clamp(0, P - 1)]
    k = torch.ceil(eta).clamp(min=0, max=H_axis - 1).to(torch.long)
    owner_at_k = dst_owner.unsqueeze(0).expand(S, T, H_axis).gather(-1, k.unsqueeze(-1)).squeeze(-1)
    still_mine = owner_at_k == pid
    src_neq_dst = src_idx.view(S, 1) != dst_idx.view(1, T)
    valid = viable & still_mine & src_neq_dst & (gap > float(config.regroup_pressure_delta_min)) & (eta <= float(config.max_regroup_time)) & can_send.view(S, 1) & dst_exists.view(1, T)
    sc = torch.where(valid, gap - float(config.regroup_time_penalty_weight) * eta, torch.full_like(gap, float('-inf')))
    best_t = _stable_argmax(sc)
    best_score = sc.gather(-1, best_t.unsqueeze(-1)).squeeze(-1)
    best_valid = torch.isfinite(best_score)
    s_ar = torch.arange(S, device=device)
    best_dst = dst_idx[best_t]
    best_angle = angle[s_ar, best_t]
    best_eta = eta[s_ar, best_t]
    return LaunchEntries(source_slots=src_idx, target_slots=best_dst, ships=torch.where(best_valid, regroup_cap, torch.zeros_like(regroup_cap)), angle=torch.where(best_valid, best_angle, torch.zeros_like(best_angle)), eta=torch.where(best_valid, best_eta, torch.ones_like(best_eta)), valid=best_valid)

def _empty_entries(device: torch.device, dtype: torch.dtype) -> LaunchEntries:
    """Tạo bảng LaunchEntries rỗng đúng device/dtype.

Dùng khi không có nguồn, không có mục tiêu hoặc không có launch đạt ngưỡng."""
    z = torch.zeros(0, dtype=dtype, device=device)
    zl = torch.zeros(0, dtype=torch.long, device=device)
    return LaunchEntries(source_slots=zl, target_slots=zl, ships=z, angle=z, eta=z, valid=torch.zeros(0, dtype=torch.bool, device=device))

def entries_to_sparse_payload(entries: LaunchEntries, *, planet_ids: Tensor) -> dict[str, Tensor]:
    """Chuyển LaunchEntries thành payload sparse dạng tensor.

Hàm lọc valid entry và sắp các trường source, angle, ships để adapter đổi sang action list."""
    L = entries.source_slots.shape[0]
    device = entries.source_slots.device
    P = int(planet_ids.shape[0])
    valid_long = entries.valid.to(torch.int64)
    counts = valid_long.sum().to(torch.int32)
    max_count = int(counts.item())
    out_from = torch.full((max_count,), -1, dtype=torch.int32, device=device)
    out_angle = torch.zeros((max_count,), dtype=torch.float32, device=device)
    out_ships = torch.zeros((max_count,), dtype=torch.float32, device=device)
    if max_count == 0:
        return {'from_planet_id': out_from, 'angle': out_angle, 'num_ships': out_ships, 'counts': counts}
    safe_src = entries.source_slots.clamp(min=0, max=max(P - 1, 0))
    from_pid_full = planet_ids[safe_src].to(torch.int32)
    launch_rank = valid_long.cumsum(0) - valid_long
    l_idx = torch.where(entries.valid)[0]
    pos = launch_rank[l_idx]
    out_from[pos] = from_pid_full[l_idx]
    out_angle[pos] = entries.angle[l_idx].to(torch.float32)
    out_ships[pos] = entries.ships[l_idx].to(torch.float32)
    return {'from_planet_id': out_from, 'angle': out_angle, 'num_ships': out_ships, 'counts': counts}

def empty_action_row(device: torch.device) -> dict[str, Tensor]:
    """Tạo action sparse rỗng khi agent quyết định không phóng quân."""
    return {'from_planet_id': torch.full((0,), -1, dtype=torch.int32, device=device), 'angle': torch.zeros((0,), dtype=torch.float32, device=device), 'num_ships': torch.zeros((0,), dtype=torch.float32, device=device), 'counts': torch.zeros((), dtype=torch.int32, device=device)}

def safe_drain(garrison_status: PlanetGarrisonStatus, *, source_idx: Tensor, source_ships: Tensor, H_eff: Tensor, player_id: int=0) -> Tensor:
    """Tính số quân có thể rút khỏi source mà vẫn giữ an toàn.

Thuật toán nhìn vào projection garrison của source trong horizon, giữ lại reserve để chống áp lực hoặc nguy cơ bị chiếm, rồi trả phần quân dư có thể dùng cho attack/regroup."""
    S = source_idx.shape[0]
    ships_cache = garrison_status.ships
    dtype = ships_cache.dtype if ships_cache.is_floating_point() else torch.float32
    device = ships_cache.device
    H_axis = int(ships_cache.shape[-1])
    H = max(H_axis - 1, 0)
    P = int(ships_cache.shape[0])
    if H == 0:
        return torch.zeros(S, dtype=dtype, device=device)
    src_idx_safe = source_idx.clamp(min=0, max=max(P - 1, 0))
    src_ships_traj = ships_cache[src_idx_safe][..., 1:].to(dtype=dtype)
    src_owner_traj = garrison_status.owner[src_idx_safe][..., 1:]
    me_owned = src_owner_traj == int(player_id)
    turn_grid = torch.arange(1, H + 1, device=device, dtype=dtype).view(1, H)
    within_horizon = turn_grid <= H_eff
    held = me_owned & within_horizon & (src_ships_traj > 0.0)
    inf_fill = torch.full_like(src_ships_traj, float('inf'))
    cap_traj = torch.where(held, src_ships_traj, inf_fill)
    min_slack = cap_traj.min(dim=-1).values
    return torch.minimum(min_slack, source_ships.to(dtype)).clamp(min=0.0)
