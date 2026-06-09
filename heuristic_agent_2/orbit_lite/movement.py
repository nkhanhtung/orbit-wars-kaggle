from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor
from .aiming import orbit_phase_index_from_obs_step
from .geometry import fleet_speed
from .obs import parse_obs
from .constants import BOARD_SIZE, CENTER, SUN_RADIUS
DEFAULT_MOVEMENT_HORIZON = 20
DEFAULT_DRIFT_EPSILON = 0.0001
DEFAULT_MAX_TRACKED_FLEETS = 64

@dataclass(frozen=True)
class MovementConfig:
    """Cấu hình cho bộ dự báo chuyển động và tracking fleet.

Các tham số quyết định horizon dự báo, độ lệch chấp nhận khi khớp fleet, có tracking fleet hay không, số người chơi và dung lượng sổ cái fleet."""
    movement_horizon: int = DEFAULT_MOVEMENT_HORIZON
    drift_epsilon: float = DEFAULT_DRIFT_EPSILON
    track_fleets: bool = False
    player_count: int | None = None
    max_tracked_fleets: int = DEFAULT_MAX_TRACKED_FLEETS

@dataclass(frozen=True)
class PlanetGarrisonStatus:
    """Trajectory dự báo owner và ships của từng hành tinh.

Lớp lưu trạng thái garrison qua các bước tương lai để planner tính capture_floor, safe_drain và score launch theo thời điểm đến."""
    owner: Tensor
    ships: Tensor
    pre_combat_owner: Tensor | None = None
    pre_combat_ships: Tensor | None = None
    arrivals_by_owner: Tensor | None = None

@dataclass
class PlanetMovement:
    """Bộ mô phỏng chuyển động và trạng thái tương lai của bản đồ.

Lớp kết hợp quỹ đạo hành tinh, fleet đang bay, pending launches và projection garrison. Đây là trạng thái lõi giúp agent không chỉ nhìn hiện tại mà còn dự đoán tương lai gần."""
    x: Tensor
    y: Tensor
    alive_by_step: Tensor
    planet_ids: Tensor
    radii: Tensor
    planet_owner: Tensor
    planet_ships: Tensor
    planet_prod: Tensor
    base_step: Tensor
    comet_planet_ids: Tensor
    comet_path_index: Tensor
    movement_horizon: int = DEFAULT_MOVEMENT_HORIZON
    drift_epsilon: float = DEFAULT_DRIFT_EPSILON
    track_fleets: bool = False
    player_count: int | None = None
    max_tracked_fleets: int = DEFAULT_MAX_TRACKED_FLEETS
    fleet_buckets: Tensor | None = None
    fleet_last_step: Tensor | None = None
    tracked_fleet_ids: Tensor | None = None
    tracked_fleet_eta: Tensor | None = None
    tracked_fleet_target_slot: Tensor | None = None
    tracked_fleet_owner: Tensor | None = None
    tracked_fleet_ships: Tensor | None = None
    garrison_owner_cache: Tensor | None = None
    garrison_ships_cache: Tensor | None = None
    garrison_pre_combat_owner_cache: Tensor | None = None
    garrison_pre_combat_ships_cache: Tensor | None = None
    garrison_dirty_from: Tensor | None = None
    pending_source_planets: Tensor | None = None
    pending_ships: Tensor | None = None
    pending_angle: Tensor | None = None
    pending_target_slots: Tensor | None = None
    pending_eta: Tensor | None = None
    pending_owners: Tensor | None = None
    pending_prev_nfid: Tensor | None = None
    pending_stash_step: Tensor | None = None

    @property
    def P(self) -> int:
        """Trả số hành tinh đang được theo dõi trong mô hình chuyển động."""
        return int(self.planet_ids.shape[0])

    @property
    def device(self) -> torch.device:
        """Trả device của mô hình để tạo tensor phụ trợ đúng nơi tính toán."""
        return self.x.device

    @property
    def dtype(self) -> torch.dtype:
        """Trả dtype chính dùng trong mô hình chuyển động."""
        return self.x.dtype

    @property
    def config(self) -> MovementConfig:
        """Trả cấu hình MovementConfig hiện tại của mô hình."""
        return MovementConfig(movement_horizon=int(self.movement_horizon), drift_epsilon=float(self.drift_epsilon), track_fleets=bool(self.track_fleets), player_count=self.player_count, max_tracked_fleets=int(self.max_tracked_fleets))

    @classmethod
    def from_obs_tensors(cls, obs_tensors: dict, *, config: MovementConfig | None=None, movement_horizon: int=DEFAULT_MOVEMENT_HORIZON, drift_epsilon: float=DEFAULT_DRIFT_EPSILON, track_fleets: bool=False, player_count: int | None=None, max_tracked_fleets: int=DEFAULT_MAX_TRACKED_FLEETS) -> 'PlanetMovement':
        """Khởi tạo PlanetMovement từ observation tensor.

Thuật toán parse obs, xây trajectory vị trí tương lai, thiết lập tracking fleet và dựng projection garrison ban đầu. Đây là điểm bắt đầu khi chưa có memory từ turn trước."""
        cfg = config if config is not None else MovementConfig(movement_horizon=int(movement_horizon), drift_epsilon=float(drift_epsilon), track_fleets=bool(track_fleets), player_count=player_count, max_tracked_fleets=int(max_tracked_fleets))
        built = _build_future_from_obs(obs_tensors, int(cfg.movement_horizon))
        resolved_player_count = _resolve_player_count(obs_tensors, cfg.player_count) if cfg.track_fleets else cfg.player_count
        movement = cls(x=built['x'], y=built['y'], alive_by_step=built['alive_by_step'], planet_ids=built['planet_ids'], radii=built['radii'], planet_owner=built['owner'], planet_ships=built['ships'], planet_prod=built['prod'], base_step=built['step'], comet_planet_ids=built['comet_planet_ids'], comet_path_index=built['comet_path_index'], movement_horizon=int(cfg.movement_horizon), drift_epsilon=float(cfg.drift_epsilon), track_fleets=bool(cfg.track_fleets), player_count=resolved_player_count, max_tracked_fleets=int(cfg.max_tracked_fleets))
        if movement.track_fleets:
            movement._init_fleet_tracking(obs_tensors, reset_ledger=True)
            movement._ingest_obs_fleets(obs_tensors)
        return movement

    def update(self, obs_tensors: dict) -> 'PlanetMovement':
        """Cập nhật mô hình chuyển động bằng observation mới.

Hàm đồng bộ vị trí, owner, ships, fleet mới quan sát được, reconcile với pending launch của mình và làm bẩn cache garrison khi cần. Đây là cơ chế giữ memory nhất quán qua nhiều lượt."""
        planets = obs_tensors['planets']
        if planets.device != self.device or planets.shape[0] != self.P or int(self.x.shape[0]) != int(self.movement_horizon) + 1:
            fresh = type(self).from_obs_tensors(obs_tensors, movement_horizon=self.movement_horizon, drift_epsilon=self.drift_epsilon, track_fleets=self.track_fleets, player_count=self.player_count, max_tracked_fleets=int(self.max_tracked_fleets))
            self._copy_from(fresh)
            return self
        if self.track_fleets:
            current_player_count = _resolve_player_count(obs_tensors, self.player_count)
            if self.fleet_buckets is None or self.fleet_last_step is None or self.tracked_fleet_ids is None or (tuple(self.fleet_buckets.shape) != (self.P, int(self.movement_horizon), int(current_player_count))) or (self.fleet_buckets.device != self.device) or (int(self.tracked_fleet_ids.shape[0]) < int(self.max_tracked_fleets)):
                self.player_count = int(current_player_count)
                self._init_fleet_tracking(obs_tensors, reset_ledger=True)
        obs_for_decision = parse_obs(obs_tensors)
        H = int(self.movement_horizon)
        planet_ids_now = planets[..., 0].long()
        radii_now = planets[..., 4].to(dtype=self.dtype)
        owner_now = planets[..., 1].to(device=self.device, dtype=torch.long)
        owner_now = torch.where(obs_for_decision.alive, owner_now, torch.full_like(owner_now, -1))
        ships_now = planets[..., 5].to(device=self.device, dtype=self.dtype)
        prod_now = planets[..., 6].to(device=self.device, dtype=self.dtype)
        step_now = obs_for_decision.step.to(device=self.device, dtype=torch.long)
        comet_ids_now, comet_idx_now = _comet_metadata(obs_tensors, self.device)
        current_obs_x = planets[..., 2].to(device=self.device, dtype=self.dtype)
        current_obs_y = planets[..., 3].to(device=self.device, dtype=self.dtype)
        current_alive = obs_for_decision.alive
        ids_same = bool((planet_ids_now == self.planet_ids).all())
        same_step = bool(step_now == self.base_step)
        next_step = bool(step_now == self.base_step + 1)
        comet_same = _same_2d(comet_ids_now, self.comet_planet_ids)
        comet_idx_same = _same_2d(comet_idx_now, self.comet_path_index)
        expected_next_idx = torch.where(self.comet_path_index >= 0, self.comet_path_index + 1, self.comet_path_index)
        comet_idx_next = _same_2d(comet_idx_now, expected_next_idx)
        same_alive_ok = bool((current_alive == self.alive_by_step[0]).all())
        next_alive_ok = bool((current_alive == self.alive_by_step[1]).all())
        same_drift_ok = _position_matches(self.x[0], self.y[0], current_obs_x, current_obs_y, current_alive, float(self.drift_epsilon))
        next_drift_ok = _position_matches(self.x[1], self.y[1], current_obs_x, current_obs_y, current_alive, float(self.drift_epsilon))
        keep = ids_same and same_step and comet_same and comet_idx_same and same_alive_ok and same_drift_ok
        roll = ids_same and next_step and comet_same and comet_idx_next and next_alive_ok and next_drift_ok
        rebuild = not (keep or roll)
        if rebuild:
            built = _build_future_from_obs(obs_tensors, H)
        elif roll:
            last_offset = torch.tensor([H], dtype=torch.long, device=self.device)
            built = _build_future_from_obs(obs_tensors, H, offsets=last_offset)
        else:
            built = None
        if roll:
            assert built is not None
            self.x[:-1] = self.x[1:].clone()
            self.y[:-1] = self.y[1:].clone()
            self.alive_by_step[:-1] = self.alive_by_step[1:].clone()
            self.x[-1] = built['x'][-1]
            self.y[-1] = built['y'][-1]
            self.alive_by_step[-1] = built['alive_by_step'][-1]
            self._roll_garrison_projection()
        if rebuild:
            assert built is not None
            self.x[:] = built['x']
            self.y[:] = built['y']
            self.alive_by_step[:] = built['alive_by_step']
            self._mark_garrison_dirty_all(0)
        if roll or rebuild:
            self.planet_ids[:] = planet_ids_now
            self.radii[:] = radii_now
            self.base_step = step_now
            self.comet_planet_ids = comet_ids_now
            self.comet_path_index = comet_idx_now
        self._refresh_garrison_base({'planet_ids': planet_ids_now, 'radii': radii_now, 'owner': owner_now, 'ships': ships_now, 'prod': prod_now, 'step': step_now})
        if self.track_fleets:
            self._roll_fleet_buckets_phase1(step_now)
            if rebuild and (not ids_same):
                self._reset_fleet_tracking()
            self._reconcile_pending_own_launches(obs_tensors)
            self._ingest_obs_fleets(obs_tensors)
            self._reconcile_obs_fleets(obs_tensors)
        return self

    def all_positions(self, k: int) -> tuple[Tensor, Tensor]:
        """Trả vị trí dự báo của mọi hành tinh tại một bước tương lai."""
        idx = self._k_index(k)
        return (self.x[idx], self.y[idx])

    def alive_at(self, k: int) -> Tensor:
        """Trả mask hành tinh còn tồn tại tại bước tương lai."""
        return self.alive_by_step[self._k_index(k)]

    def position_at_slots(self, slots: Tensor, k: int) -> tuple[Tensor, Tensor]:
        """Lấy tọa độ dự báo của các hành tinh cụ thể tại một bước.

Hàm dùng index theo slot và thời gian để phục vụ aiming, distance cache và reachability."""
        slots = slots.to(device=self.device, dtype=torch.long).clamp(0, max(self.P - 1, 0))
        px, py = self.all_positions(k)
        out_x = px[slots].to(dtype=self.dtype)
        out_y = py[slots].to(dtype=self.dtype)
        return (out_x, out_y)

    def pairwise_distance(self, k: int) -> Tensor:
        """Tính khoảng cách cặp source-target tại một bước tương lai."""
        px, py = self.all_positions(k)
        dx = px.unsqueeze(1) - px.unsqueeze(0)
        dy = py.unsqueeze(1) - py.unsqueeze(0)
        return torch.sqrt((dx * dx + dy * dy).clamp(min=0.0))

    def garrison_status(self, planet_slots: Tensor | None=None, *, max_horizon: int | None=None) -> PlanetGarrisonStatus:
        """Trả projection garrison đã cache hoặc tự refresh nếu bẩn.

Planner dùng kết quả này như bản đồ tương lai của owner/ships khi chưa thêm launch mới."""
        self._require_fleet_buckets()
        slots, out_prefix = self._normalize_garrison_slots(planet_slots)
        requested_horizon = int(self.movement_horizon if max_horizon is None else max(0, min(int(max_horizon), int(self.movement_horizon))))
        self._refresh_garrison_projection(slots, requested_horizon=requested_horizon)
        assert self.garrison_owner_cache is not None
        assert self.garrison_ships_cache is not None
        assert self.garrison_dirty_from is not None
        owner = self.garrison_owner_cache[slots][:, :requested_horizon + 1].reshape(*out_prefix, requested_horizon + 1)
        ships = self.garrison_ships_cache[slots][:, :requested_horizon + 1].reshape(*out_prefix, requested_horizon + 1)
        pre_combat_owner: Tensor | None = None
        pre_combat_ships: Tensor | None = None
        if self.garrison_pre_combat_owner_cache is not None and self.garrison_pre_combat_ships_cache is not None:
            pre_combat_owner = self.garrison_pre_combat_owner_cache[slots][:, :requested_horizon + 1].reshape(*out_prefix, requested_horizon + 1)
            pre_combat_ships = self.garrison_pre_combat_ships_cache[slots][:, :requested_horizon + 1].reshape(*out_prefix, requested_horizon + 1)
        arrivals_by_owner: Tensor | None = None
        if self.fleet_buckets is not None and requested_horizon > 0:
            A = int(self.fleet_buckets.shape[-1])
            arrivals_full = self.fleet_buckets[slots].reshape(*out_prefix, int(self.movement_horizon), A)
            arrivals_trimmed = arrivals_full[..., :requested_horizon, :]
            zero_frame = torch.zeros(*out_prefix, 1, A, dtype=arrivals_trimmed.dtype, device=self.device)
            arrivals_by_owner = torch.cat([zero_frame, arrivals_trimmed], dim=-2)
        status = PlanetGarrisonStatus(owner=owner, ships=ships, pre_combat_owner=pre_combat_owner, pre_combat_ships=pre_combat_ships, arrivals_by_owner=arrivals_by_owner)
        return status

    def _clear_pending_mask(self, mask: Tensor) -> None:
        """Xoá các pending launch theo mask.

Dùng khi launch đã được khớp với fleet quan sát được hoặc hết hiệu lực."""
        if self.pending_owners is None:
            return
        self.pending_owners[mask] = -1
        assert self.pending_source_planets is not None
        self.pending_source_planets[mask] = -1
        assert self.pending_ships is not None
        self.pending_ships[mask] = 0
        assert self.pending_angle is not None
        self.pending_angle[mask] = 0.0
        assert self.pending_target_slots is not None
        self.pending_target_slots[mask] = -1
        assert self.pending_eta is not None
        self.pending_eta[mask] = 0.0
        assert self.pending_prev_nfid is not None
        self.pending_prev_nfid[mask] = 0
        assert self.pending_stash_step is not None
        self.pending_stash_step[mask] = -1

    def _ensure_pending_capacity(self, needed: int) -> None:
        """Mở rộng bộ nhớ pending launch nếu số launch vượt dung lượng hiện tại."""
        device = self.device
        if self.pending_owners is None:
            initial = max(4, int(needed))
            shape = (initial,)
            self.pending_owners = torch.full(shape, -1, dtype=torch.long, device=device)
            self.pending_source_planets = torch.full(shape, -1, dtype=torch.long, device=device)
            self.pending_ships = torch.zeros(shape, dtype=torch.long, device=device)
            self.pending_angle = torch.zeros(shape, dtype=self.dtype, device=device)
            self.pending_target_slots = torch.full(shape, -1, dtype=torch.long, device=device)
            self.pending_eta = torch.zeros(shape, dtype=self.dtype, device=device)
            self.pending_prev_nfid = torch.zeros(shape, dtype=torch.long, device=device)
            self.pending_stash_step = torch.full(shape, -1, dtype=torch.long, device=device)
            return
        assert self.pending_owners is not None
        empty_count = int((self.pending_owners == -1).sum().item())
        shortage = int(needed) - empty_count
        if shortage <= 0:
            return
        cur_L = int(self.pending_owners.shape[0])
        extra = max(shortage, cur_L)
        new_L = cur_L + extra

        def _grow(t: Tensor, fill: float | int) -> Tensor:
            """Hàm/lớp PlanetMovement._ensure_pending_capacity._grow phục vụ pipeline lập kế hoạch của agent Orbit Wars.

Ghi chú: logic gốc được giữ nguyên; phần mô tả này giải thích vị trí của thành phần trong hệ thống, các tensor đầu vào/đầu ra và vai trò thuật toán khi agent dự báo, chấm điểm hoặc đóng gói action."""
            extension = torch.full((new_L - cur_L,), fill, dtype=t.dtype, device=device)
            return torch.cat([t, extension], dim=0)
        self.pending_owners = _grow(self.pending_owners, -1)
        assert self.pending_source_planets is not None
        self.pending_source_planets = _grow(self.pending_source_planets, -1)
        assert self.pending_ships is not None
        self.pending_ships = _grow(self.pending_ships, 0)
        assert self.pending_angle is not None
        self.pending_angle = _grow(self.pending_angle, 0.0)
        assert self.pending_target_slots is not None
        self.pending_target_slots = _grow(self.pending_target_slots, -1)
        assert self.pending_eta is not None
        self.pending_eta = _grow(self.pending_eta, 0.0)
        assert self.pending_prev_nfid is not None
        self.pending_prev_nfid = _grow(self.pending_prev_nfid, 0)
        assert self.pending_stash_step is not None
        self.pending_stash_step = _grow(self.pending_stash_step, -1)

    def stash_pending_own_launches(self, *, owner_id: int | Tensor, source_slots: Tensor, ships: Tensor, angle: Tensor, target_slots: Tensor, eta: Tensor, valid: Tensor, prev_next_fleet_id: int | Tensor) -> None:
        """Lưu các launch agent vừa quyết định nhưng chưa chắc đã xuất hiện trong observation.

Cơ chế này giúp projection riêng của agent tính trước tác động của action vừa phóng, giảm độ trễ một turn so với observation công khai."""
        if not self.track_fleets:
            return
        device = self.device
        valid_mask = valid.to(device=device, dtype=torch.bool).reshape(-1)
        if not bool(valid_mask.any()):
            return
        src = source_slots.to(device=device, dtype=torch.long).reshape(-1)
        ships_t = ships.to(device=device, dtype=torch.long).reshape(-1)
        angle_t = angle.to(device=device, dtype=self.dtype).reshape(-1)
        tgt_t = target_slots.to(device=device, dtype=torch.long).reshape(-1)
        eta_t = eta.to(device=device, dtype=self.dtype).reshape(-1)
        src_safe = src.clamp(min=0, max=max(int(self.P) - 1, 0))
        source_planet_ids = self.planet_ids[src_safe]
        L_in = int(valid_mask.shape[0])
        if isinstance(prev_next_fleet_id, Tensor):
            prev_nfid_scalar = int(prev_next_fleet_id.flatten()[0].item())
        else:
            prev_nfid_scalar = int(prev_next_fleet_id)
        prev_nfid_L = torch.full((L_in,), prev_nfid_scalar, dtype=torch.long, device=device)
        owner_scalar = int(owner_id.flatten()[0].item()) if isinstance(owner_id, Tensor) else int(owner_id)
        owner_L = torch.full((L_in,), owner_scalar, dtype=torch.long, device=device)
        stash_step_scalar = int(self.base_step.item()) if isinstance(self.base_step, Tensor) else -1
        stash_step_L = torch.full((L_in,), stash_step_scalar, dtype=torch.long, device=device)
        if self.pending_owners is not None:
            same_owner = self.pending_owners == owner_scalar
            if bool(same_owner.any()):
                self._clear_pending_mask(same_owner)
        per_needed = int(valid_mask.sum().item())
        self._ensure_pending_capacity(per_needed)
        assert self.pending_owners is not None
        empty_slots = torch.nonzero(self.pending_owners == -1, as_tuple=True)[0]
        k_in = torch.nonzero(valid_mask, as_tuple=True)[0]
        slot_in_pending = empty_slots[:k_in.numel()]
        self.pending_owners[slot_in_pending] = owner_L[k_in]
        assert self.pending_source_planets is not None
        self.pending_source_planets[slot_in_pending] = source_planet_ids[k_in]
        assert self.pending_ships is not None
        self.pending_ships[slot_in_pending] = ships_t[k_in]
        assert self.pending_angle is not None
        self.pending_angle[slot_in_pending] = angle_t[k_in]
        assert self.pending_target_slots is not None
        self.pending_target_slots[slot_in_pending] = tgt_t[k_in]
        assert self.pending_eta is not None
        self.pending_eta[slot_in_pending] = eta_t[k_in]
        assert self.pending_prev_nfid is not None
        self.pending_prev_nfid[slot_in_pending] = prev_nfid_L[k_in]
        assert self.pending_stash_step is not None
        self.pending_stash_step[slot_in_pending] = stash_step_L[k_in]

    def _reconcile_pending_own_launches(self, obs_tensors: dict) -> None:
        """Khớp pending launch của mình với fleet mới quan sát được.

Thuật toán so sánh vị trí, góc, số tàu và ETA gần đúng để xác nhận launch đã thành fleet thật, từ đó tránh đếm trùng."""
        if not self.track_fleets:
            return
        if self.pending_owners is None or self.tracked_fleet_ids is None:
            return
        active_mask = self.pending_owners != -1
        if not bool(active_mask.any()):
            return
        device = self.device
        step_tensor = obs_tensors.get('step')
        if step_tensor is not None:
            assert self.pending_stash_step is not None
            step_scalar = int(step_tensor.flatten()[0].item()) if isinstance(step_tensor, Tensor) else int(step_tensor)
            advanced = step_scalar > self.pending_stash_step
            active_mask = active_mask & advanced
        if not bool(active_mask.any()):
            return
        fleets = obs_tensors['fleets'].to(device=device)
        fleet_ids = fleets[..., 0].to(dtype=torch.long)
        obs_owner = fleets[..., 1].to(dtype=torch.long)
        obs_angle = fleets[..., 4].to(dtype=self.dtype)
        obs_from = fleets[..., 5].to(dtype=torch.long)
        obs_ships = fleets[..., 6].to(dtype=torch.long)
        assert self.pending_owners is not None
        assert self.pending_source_planets is not None
        assert self.pending_ships is not None
        assert self.pending_angle is not None
        assert self.pending_target_slots is not None
        assert self.pending_eta is not None
        assert self.pending_prev_nfid is not None
        match_FL = active_mask.unsqueeze(0) & (fleet_ids.unsqueeze(1) >= 0) & (obs_owner.unsqueeze(1) == self.pending_owners.unsqueeze(0)) & (obs_from.unsqueeze(1) == self.pending_source_planets.unsqueeze(0)) & (obs_ships.unsqueeze(1) == self.pending_ships.unsqueeze(0)) & (obs_angle.unsqueeze(1) == self.pending_angle.unsqueeze(0)) & (fleet_ids.unsqueeze(1) >= self.pending_prev_nfid.unsqueeze(0))
        INF = torch.iinfo(torch.long).max
        id_for_match = torch.where(match_FL, fleet_ids.unsqueeze(1).expand_as(match_FL), torch.full_like(match_FL, INF, dtype=torch.long))
        chosen_id, _ = id_for_match.min(dim=0)
        eta_now = torch.ceil(self.pending_eta).to(dtype=torch.long) - 1
        expect_obs_match = active_mask & (eta_now > 0)
        no_match = expect_obs_match & (chosen_id == INF)
        matched = expect_obs_match & (chosen_id != INF)
        if int(active_mask.shape[0]) > 1:
            chosen_for_matched = torch.where(matched, chosen_id, torch.full_like(chosen_id, INF))
            sorted_ids, _ = chosen_for_matched.sort()
            dup = bool(((sorted_ids[1:] == sorted_ids[:-1]) & (sorted_ids[1:] != INF)).any())
            if dup:
                raise AssertionError('Pending-launch reconciliation: multiple pending entries resolved to the same engine fleet id. This usually means multi-launch from the same source with identical (ships, angle) tuples processed in an unexpected order.')
        if bool(matched.any()):
            l_idx = torch.where(matched)[0]
            real_ids = chosen_id[l_idx]
            self._ledger_bulk_insert(real_ids, eta_now[l_idx], self.pending_target_slots[l_idx], self.pending_owners[l_idx], self.pending_ships[l_idx].to(dtype=self.dtype))
        if bool(no_match.any()):
            self._decrement_unmatched_arrivals(no_match)
        self._clear_pending_mask(active_mask)

    def _decrement_unmatched_arrivals(self, no_match: Tensor) -> None:
        """Giảm ETA cho pending launch chưa khớp và loại launch đã quá hạn."""
        assert self.pending_eta is not None
        assert self.pending_owners is not None
        assert self.pending_ships is not None
        assert self.pending_target_slots is not None
        buckets = self._require_fleet_buckets()
        eta_now = torch.ceil(self.pending_eta).to(dtype=torch.long) - 1
        h_idx_now = eta_now - 1
        H = int(self.movement_horizon)
        Aowner = int(buckets.shape[2])
        valid = no_match & (h_idx_now >= 0) & (h_idx_now < H) & (self.pending_target_slots >= 0) & (self.pending_target_slots < int(self.P)) & (self.pending_owners >= 0) & (self.pending_owners < Aowner) & (self.pending_ships > 0)
        if not bool(valid.any()):
            return
        target = self.pending_target_slots[valid]
        h_idx_sel = h_idx_now[valid]
        owner_sel = self.pending_owners[valid]
        ships_sel = self.pending_ships[valid].to(dtype=self.dtype)
        buckets.index_put_((target, h_idx_sel, owner_sel), -ships_sel, accumulate=True)
        self._mark_garrison_dirty(target, h_idx_sel + 1)

    def record_fleet_arrivals(self, *, target_slots: Tensor, owner_ids: Tensor | int, ships: Tensor, eta: Tensor, valid: Tensor | None=None) -> None:
        """Ghi nhận fleet sắp/đã đến mục tiêu vào projection garrison.

Hàm đưa sự kiện arrival vào các bucket thời gian để garrison recurrence biết lúc nào cần cộng quân đến nơi."""
        buckets = self._require_fleet_buckets()
        target_slots, ships, eta = torch.broadcast_tensors(target_slots.to(device=self.device, dtype=torch.long), ships.to(device=self.device, dtype=self.dtype), eta.to(device=self.device, dtype=self.dtype))
        if isinstance(owner_ids, int):
            owner = torch.full_like(target_slots, int(owner_ids), dtype=torch.long, device=self.device)
        else:
            owner = torch.broadcast_to(owner_ids.to(device=self.device, dtype=torch.long), target_slots.shape)
        if valid is None:
            valid_mask = torch.ones_like(target_slots, dtype=torch.bool)
        else:
            valid_mask = torch.broadcast_to(valid.to(device=self.device, dtype=torch.bool), target_slots.shape)
        h_idx = torch.ceil(eta).to(dtype=torch.long) - 1
        valid_mask = valid_mask & (target_slots >= 0) & (target_slots < self.P) & (owner >= 0) & (owner < int(buckets.shape[2])) & (h_idx >= 0) & (h_idx < int(self.movement_horizon)) & (ships > 0.0)
        if not bool(valid_mask.any()):
            return
        buckets.index_put_((target_slots[valid_mask], h_idx[valid_mask], owner[valid_mask]), ships[valid_mask], accumulate=True)
        self._mark_garrison_dirty(target_slots[valid_mask], h_idx[valid_mask] + 1)

    def _normalize_garrison_slots(self, planet_slots: Tensor | None) -> tuple[Tensor, torch.Size]:
        """Chuẩn hoá danh sách slot hành tinh cần refresh garrison."""
        if planet_slots is None:
            slots = torch.arange(self.P, dtype=torch.long, device=self.device)
            return (slots, slots.shape)
        raw = planet_slots.to(device=self.device, dtype=torch.long)
        out_prefix = raw.shape
        slots = raw.reshape(-1).clamp(0, max(self.P - 1, 0))
        return (slots, out_prefix)

    def _ensure_garrison_cache(self) -> None:
        """Đảm bảo cache garrison tồn tại và còn hợp lệ trước khi truy cập."""
        self._ensure_garrison_cache_impl()

    def _ensure_garrison_cache_impl(self) -> None:
        """Triển khai nội bộ việc tạo hoặc refresh cache garrison."""
        expected_owner = (self.P, int(self.movement_horizon) + 1)
        expected_dirty = (self.P,)
        if self.garrison_owner_cache is not None and self.garrison_ships_cache is not None and (self.garrison_pre_combat_owner_cache is not None) and (self.garrison_pre_combat_ships_cache is not None) and (self.garrison_dirty_from is not None) and (tuple(self.garrison_owner_cache.shape) == expected_owner) and (tuple(self.garrison_ships_cache.shape) == expected_owner) and (tuple(self.garrison_pre_combat_owner_cache.shape) == expected_owner) and (tuple(self.garrison_pre_combat_ships_cache.shape) == expected_owner) and (tuple(self.garrison_dirty_from.shape) == expected_dirty) and (self.garrison_owner_cache.device == self.device) and (self.garrison_ships_cache.device == self.device):
            return
        horizon = int(self.movement_horizon)
        self.garrison_owner_cache = torch.full((self.P, horizon + 1), -1, dtype=torch.long, device=self.device)
        self.garrison_ships_cache = torch.zeros(self.P, horizon + 1, dtype=self.dtype, device=self.device)
        self.garrison_pre_combat_owner_cache = self.garrison_owner_cache.clone()
        self.garrison_pre_combat_ships_cache = self.garrison_ships_cache.clone()
        self.garrison_owner_cache[:, 0] = self.planet_owner
        self.garrison_ships_cache[:, 0] = self.planet_ships
        self.garrison_pre_combat_owner_cache[:, 0] = self.planet_owner
        self.garrison_pre_combat_ships_cache[:, 0] = self.planet_ships
        self.garrison_dirty_from = torch.zeros(self.P, dtype=torch.long, device=self.device)

    def _refresh_garrison_projection(self, slots: Tensor, *, requested_horizon: int | None=None) -> None:
        """Làm mới projection owner/ships của các hành tinh được đánh dấu bẩn.

Thuật toán chạy recurrence từ trạng thái hiện tại, cộng production và các arrival bucket để dự báo trạng thái qua horizon."""
        self._ensure_garrison_cache()
        assert self.fleet_buckets is not None
        assert self.garrison_owner_cache is not None
        assert self.garrison_ships_cache is not None
        assert self.garrison_dirty_from is not None
        p_idx = torch.unique(slots.reshape(-1).clamp(min=0, max=max(self.P - 1, 0)))
        if p_idx.numel() == 0:
            return
        dirty = self.garrison_dirty_from[p_idx]
        horizon = int(self.movement_horizon if requested_horizon is None else max(0, min(int(requested_horizon), int(self.movement_horizon))))
        needs_refresh = dirty <= horizon
        if not bool(needs_refresh.any()):
            return
        p_idx = p_idx[needs_refresh]
        owner = self.planet_owner[p_idx].clone()
        ships = self.planet_ships[p_idx].clone()
        self.garrison_owner_cache[p_idx, 0] = owner
        self.garrison_ships_cache[p_idx, 0] = ships
        assert self.garrison_pre_combat_owner_cache is not None
        assert self.garrison_pre_combat_ships_cache is not None
        self.garrison_pre_combat_owner_cache[p_idx, 0] = owner
        self.garrison_pre_combat_ships_cache[p_idx, 0] = ships
        prod = self.planet_prod[p_idx]
        if horizon == 0:
            self.garrison_dirty_from[p_idx] = horizon + 1
            return
        self._fill_garrison_trajectory(p_idx=p_idx, init_owner=owner, init_ships=ships, prod=prod, horizon=horizon)
        self.garrison_dirty_from[p_idx] = horizon + 1

    def _fill_garrison_trajectory(self, *, p_idx: Tensor, init_owner: Tensor, init_ships: Tensor, prod: Tensor, horizon: int) -> None:
        """Điền trajectory garrison cho một nhóm hành tinh.

Hàm mô phỏng từng bước: cộng production, cộng fleet đến, giải combat, cập nhật owner/ships. Đây là lõi dự báo phòng thủ/tấn công."""
        assert self.fleet_buckets is not None
        assert self.garrison_owner_cache is not None
        assert self.garrison_ships_cache is not None
        assert self.garrison_pre_combat_owner_cache is not None
        assert self.garrison_pre_combat_ships_cache is not None
        H = int(horizon)
        N = int(p_idx.numel())
        if N == 0 or H == 0:
            return
        alive_step = self.alive_by_step[:, p_idx].transpose(0, 1)
        alive_before = alive_step[:, :H]
        alive_now = alive_step[:, 1:]
        arrivals = self.fleet_buckets[p_idx, :H, :]
        has_any_arrival = (arrivals > 0.0).any(dim=-1).any(dim=-1)
        alive_all_true = alive_step.all(dim=1)
        simple_mask = ~has_any_arrival & alive_all_true
        alive_step_full = alive_step
        n_simple = int(simple_mask.sum().item())
        n_complex = N - n_simple
        if n_simple > 0:
            simple_p = p_idx[simple_mask]
            simple_owner = init_owner[simple_mask]
            simple_ships = init_ships[simple_mask]
            simple_prod = prod[simple_mask]
            owner_alive_factor = (simple_owner >= 0).to(dtype=simple_ships.dtype)
            k_range = torch.arange(1, H + 1, device=self.device, dtype=simple_ships.dtype)
            ships_traj = simple_ships.unsqueeze(1) + simple_prod.unsqueeze(1) * owner_alive_factor.unsqueeze(1) * k_range.unsqueeze(0)
            owner_traj = simple_owner.unsqueeze(1).expand(-1, H)
            self.garrison_owner_cache[simple_p, 1:H + 1] = owner_traj
            self.garrison_ships_cache[simple_p, 1:H + 1] = ships_traj
            self.garrison_pre_combat_owner_cache[simple_p, 1:H + 1] = owner_traj
            self.garrison_pre_combat_ships_cache[simple_p, 1:H + 1] = ships_traj
        if n_complex == 0:
            return
        complex_mask = ~simple_mask
        cp = p_idx[complex_mask]
        arrivals_c = arrivals[complex_mask]
        alive_before_c = alive_before[complex_mask]
        alive_now_c = alive_now[complex_mask]
        alive_step_c = alive_step_full[complex_mask]
        state_owner = init_owner[complex_mask].clone()
        state_ships = init_ships[complex_mask].clone()
        prod_c = prod[complex_mask]
        A = int(arrivals_c.shape[-1])
        if A >= 2:
            top2 = arrivals_c.topk(k=2, dim=-1)
            top_ships_traj = top2.values[..., 0]
            second_ships_traj = top2.values[..., 1]
            top_owner_traj = top2.indices[..., 0].to(dtype=torch.long)
        else:
            top_ships_traj, top_owner_traj = arrivals_c.max(dim=-1)
            second_ships_traj = torch.zeros_like(top_ships_traj)
            top_owner_traj = top_owner_traj.to(dtype=torch.long)
        tied = top_ships_traj == second_ships_traj
        survivor_ships_traj = torch.where(tied, torch.zeros_like(top_ships_traj), (top_ships_traj - second_ships_traj).clamp(min=0.0))
        survivor_owner_traj = top_owner_traj
        zero_ships_scalar = torch.zeros((), dtype=state_ships.dtype, device=self.device)
        neg_one_owner_scalar = torch.full((), -1, dtype=state_owner.dtype, device=self.device)
        zero_prod_scalar = torch.zeros((), dtype=prod_c.dtype, device=self.device)
        combat_event_per_step = (survivor_ships_traj > 0.0) & alive_now_c
        alive_change_per_step = alive_before_c != alive_now_c
        any_event_per_step = (combat_event_per_step | alive_change_per_step).any(dim=0)
        arange_h = torch.arange(1, H + 1, device=self.device, dtype=torch.long)
        k_last_tensor = torch.where(any_event_per_step, arange_h, torch.zeros_like(arange_h)).max()
        k_last = int(k_last_tensor.item())
        loop_iters = max(0, k_last)
        tail_steps = H - loop_iters
        if loop_iters > 0:
            for k in range(1, loop_iters + 1):
                a_before = alive_before_c[:, k - 1]
                a_now = alive_now_c[:, k - 1]
                s_owner = survivor_owner_traj[:, k - 1]
                s_ships = survivor_ships_traj[:, k - 1]
                produces = a_before & (state_owner >= 0)
                state_ships = state_ships + torch.where(produces, prod_c, zero_prod_scalar)
                pre_owner = torch.where(a_now, state_owner, neg_one_owner_scalar)
                pre_ships = torch.where(a_now, state_ships, zero_ships_scalar)
                self.garrison_pre_combat_owner_cache[cp, k] = pre_owner
                self.garrison_pre_combat_ships_cache[cp, k] = pre_ships
                has_combat = (s_ships > 0.0) & a_now
                same = state_owner == s_owner
                diff = state_ships - s_ships
                attacker_wins = ~same & (diff < 0.0)
                combat_ships = torch.where(same, state_ships + s_ships, diff.abs())
                combat_owner = torch.where(attacker_wins, s_owner, state_owner)
                state_ships = torch.where(has_combat, combat_ships, state_ships)
                state_owner = torch.where(has_combat, combat_owner, state_owner)
                state_owner = torch.where(a_now, state_owner, neg_one_owner_scalar)
                state_ships = torch.where(a_now, state_ships, zero_ships_scalar)
                self.garrison_owner_cache[cp, k] = state_owner
                self.garrison_ships_cache[cp, k] = state_ships
        if tail_steps > 0:
            alive_at_k_last = alive_step_c[:, k_last]
            state_owner = torch.where(alive_at_k_last, state_owner, neg_one_owner_scalar)
            state_ships = torch.where(alive_at_k_last, state_ships, zero_ships_scalar)
            owner_alive_factor = (state_owner >= 0).to(dtype=state_ships.dtype) * alive_at_k_last.to(dtype=state_ships.dtype)
            dk_range = torch.arange(1, tail_steps + 1, device=self.device, dtype=state_ships.dtype)
            ships_traj_tail = state_ships.unsqueeze(1) + prod_c.unsqueeze(1) * owner_alive_factor.unsqueeze(1) * dk_range.unsqueeze(0)
            owner_traj_tail = state_owner.unsqueeze(1).expand(-1, tail_steps)
            self.garrison_owner_cache[cp, k_last + 1:H + 1] = owner_traj_tail
            self.garrison_ships_cache[cp, k_last + 1:H + 1] = ships_traj_tail
            self.garrison_pre_combat_owner_cache[cp, k_last + 1:H + 1] = owner_traj_tail
            self.garrison_pre_combat_ships_cache[cp, k_last + 1:H + 1] = ships_traj_tail

    def _roll_garrison_projection(self) -> None:
        """Dịch projection garrison sang turn tiếp theo.

Khi game tiến một bước, cache được roll thay vì xây lại toàn bộ, giúp tiết kiệm thời gian."""
        if self.garrison_owner_cache is None or self.garrison_ships_cache is None or self.garrison_pre_combat_owner_cache is None or (self.garrison_pre_combat_ships_cache is None) or (self.garrison_dirty_from is None):
            return
        horizon = int(self.movement_horizon)
        if horizon > 0:
            self.garrison_owner_cache[:, :-1] = self.garrison_owner_cache[:, 1:].clone()
            self.garrison_ships_cache[:, :-1] = self.garrison_ships_cache[:, 1:].clone()
            self.garrison_pre_combat_owner_cache[:, :-1] = self.garrison_pre_combat_owner_cache[:, 1:].clone()
            self.garrison_pre_combat_ships_cache[:, :-1] = self.garrison_pre_combat_ships_cache[:, 1:].clone()
            self.garrison_dirty_from = (self.garrison_dirty_from - 1).clamp(min=0)
            self.garrison_dirty_from = torch.minimum(self.garrison_dirty_from, torch.full_like(self.garrison_dirty_from, horizon))
        else:
            self.garrison_dirty_from[:] = 0

    def _refresh_garrison_base(self, built: dict[str, Tensor]) -> None:
        """Cập nhật trạng thái gốc của garrison từ observation hiện tại."""
        owner = built['owner'].to(device=self.device, dtype=torch.long)
        ships = built['ships'].to(device=self.device, dtype=self.dtype)
        prod = built['prod'].to(device=self.device, dtype=self.dtype)
        prod_changed = tuple(self.planet_prod.shape) != tuple(prod.shape) or self.planet_prod != prod
        self.planet_owner = owner
        self.planet_ships = ships
        self.planet_prod = prod
        if self.garrison_owner_cache is None or self.garrison_ships_cache is None or self.garrison_dirty_from is None:
            return
        base_changed = (self.garrison_owner_cache[:, 0] != owner) | (self.garrison_ships_cache[:, 0] != ships)
        self.garrison_owner_cache[:, 0] = owner
        self.garrison_ships_cache[:, 0] = ships
        if self.garrison_pre_combat_owner_cache is not None:
            self.garrison_pre_combat_owner_cache[:, 0] = owner
        if self.garrison_pre_combat_ships_cache is not None:
            self.garrison_pre_combat_ships_cache[:, 0] = ships
        if bool(base_changed.any()):
            self.garrison_dirty_from[base_changed] = 0
        if isinstance(prod_changed, Tensor) and bool(prod_changed.any()):
            self.garrison_dirty_from[prod_changed] = torch.minimum(self.garrison_dirty_from[prod_changed], torch.ones_like(self.garrison_dirty_from[prod_changed]))
        elif not isinstance(prod_changed, Tensor) and prod_changed:
            self.garrison_dirty_from[:] = torch.minimum(self.garrison_dirty_from, torch.ones_like(self.garrison_dirty_from))

    def _mark_garrison_dirty(self, planet_idx: Tensor, start_step: Tensor | int) -> None:
        """Đánh dấu một nhóm hành tinh cần tính lại garrison projection."""
        if self.garrison_dirty_from is None:
            return
        p = planet_idx.to(device=self.device, dtype=torch.long)
        if isinstance(start_step, int):
            start = torch.full((), int(start_step), dtype=torch.long, device=self.device)
        else:
            start = start_step.to(device=self.device, dtype=torch.long)
        p, start = torch.broadcast_tensors(p, start)
        p = p.reshape(-1)
        start = start.reshape(-1)
        if p.numel() == 0:
            return
        start = start.clamp(min=0, max=int(self.movement_horizon))
        valid = (p >= 0) & (p < self.P)
        if not bool(valid.any()):
            return
        p = p[valid]
        start = start[valid]
        flat = self.garrison_dirty_from
        unique_idx, inverse = torch.unique(p, return_inverse=True)
        if unique_idx.numel() == p.numel():
            flat[unique_idx] = torch.minimum(flat[unique_idx], start)
            return
        sentinel = int(self.movement_horizon) + 1
        candidate = torch.full((unique_idx.shape[0],), sentinel, dtype=torch.long, device=self.device)
        candidate.scatter_reduce_(0, inverse, start, reduce='amin', include_self=True)
        flat[unique_idx] = torch.minimum(flat[unique_idx], candidate)

    def _mark_garrison_dirty_all(self, start_step: int) -> None:
        """Đánh dấu toàn bộ hành tinh cần refresh projection."""
        if self.garrison_dirty_from is None:
            return
        self.garrison_dirty_from = torch.minimum(self.garrison_dirty_from, torch.full_like(self.garrison_dirty_from, int(start_step)))

    def _init_fleet_tracking(self, obs_tensors: dict, *, reset_ledger: bool) -> None:
        """Khởi tạo cấu trúc tracking fleet và sổ cái arrival."""
        _ = reset_ledger
        player_count = _resolve_player_count(obs_tensors, self.player_count)
        self.player_count = int(player_count)
        self.fleet_buckets = torch.zeros(self.P, int(self.movement_horizon), int(player_count), dtype=self.dtype, device=self.device)
        step = obs_tensors['step'].to(device=self.device, dtype=torch.long)
        self.fleet_last_step = step.detach().clone()
        M = max(1, int(self.max_tracked_fleets))
        self.max_tracked_fleets = M
        self.tracked_fleet_ids = torch.full((M,), -1, dtype=torch.long, device=self.device)
        self.tracked_fleet_eta = torch.zeros((M,), dtype=torch.long, device=self.device)
        self.tracked_fleet_target_slot = torch.full((M,), -1, dtype=torch.long, device=self.device)
        self.tracked_fleet_owner = torch.zeros((M,), dtype=torch.long, device=self.device)
        self.tracked_fleet_ships = torch.zeros((M,), dtype=self.dtype, device=self.device)
        if self.garrison_dirty_from is not None:
            self.garrison_dirty_from[:] = torch.minimum(self.garrison_dirty_from, torch.full_like(self.garrison_dirty_from, 1))

    def _clear_tracked_rows(self) -> None:
        """Xoá các dòng fleet tracking theo mask."""
        if self.tracked_fleet_ids is None or self.tracked_fleet_eta is None or self.tracked_fleet_target_slot is None or (self.tracked_fleet_owner is None) or (self.tracked_fleet_ships is None):
            return
        self.tracked_fleet_ids[:] = -1
        self.tracked_fleet_eta[:] = 0
        self.tracked_fleet_target_slot[:] = -1
        self.tracked_fleet_owner[:] = 0
        self.tracked_fleet_ships[:] = 0.0

    def _ledger_bulk_insert(self, fleet_ids: Tensor, eta_remaining: Tensor, target_slots: Tensor, owners: Tensor, ships: Tensor) -> None:
        """Chèn hàng loạt fleet/arrival vào sổ cái tracking.

Sổ cái giúp biết fleet nào sẽ đến hành tinh nào ở thời điểm nào để cập nhật garrison projection."""
        if fleet_ids.numel() == 0:
            return
        assert self.tracked_fleet_ids is not None
        assert self.tracked_fleet_eta is not None
        assert self.tracked_fleet_target_slot is not None
        assert self.tracked_fleet_owner is not None
        assert self.tracked_fleet_ships is not None
        M = int(self.tracked_fleet_ids.shape[0])
        fleet_ids = fleet_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        eta_remaining = eta_remaining.to(device=self.device, dtype=torch.long).reshape(-1)
        target_slots = target_slots.to(device=self.device, dtype=torch.long).reshape(-1)
        owners = owners.to(device=self.device, dtype=torch.long).reshape(-1)
        ships = ships.to(device=self.device, dtype=self.dtype).reshape(-1)
        valid_rows = fleet_ids >= 0
        if not bool(valid_rows.any()):
            return
        fleet_ids = fleet_ids[valid_rows]
        eta_remaining = eta_remaining[valid_rows]
        target_slots = target_slots[valid_rows]
        owners = owners[valid_rows]
        ships = ships[valid_rows]
        n = int(fleet_ids.numel())
        empty_mask = self.tracked_fleet_ids == -1
        empty_count = int(empty_mask.sum().item())
        if n > empty_count:
            occupied_count = M - empty_count
            self._grow_ledger_capacity(occupied_count + n)
            assert self.tracked_fleet_ids is not None
            empty_mask = self.tracked_fleet_ids == -1
        empty_slots = torch.nonzero(empty_mask, as_tuple=True)[0]
        slot_idx = empty_slots[:n]
        self.tracked_fleet_ids[slot_idx] = fleet_ids
        self.tracked_fleet_eta[slot_idx] = eta_remaining
        self.tracked_fleet_target_slot[slot_idx] = target_slots
        self.tracked_fleet_owner[slot_idx] = owners
        self.tracked_fleet_ships[slot_idx] = ships

    def _grow_ledger_capacity(self, required_capacity: int) -> None:
        """Tăng dung lượng sổ cái fleet khi số bản ghi vượt giới hạn cũ."""
        if self.tracked_fleet_ids is None or self.tracked_fleet_eta is None or self.tracked_fleet_target_slot is None or (self.tracked_fleet_owner is None) or (self.tracked_fleet_ships is None):
            return
        old_capacity = int(self.tracked_fleet_ids.shape[0])
        target_capacity = max(int(required_capacity), old_capacity)
        if target_capacity <= old_capacity:
            return
        new_capacity = max(target_capacity, old_capacity * 2)
        old_ids = self.tracked_fleet_ids
        old_eta = self.tracked_fleet_eta
        old_tgt = self.tracked_fleet_target_slot
        old_owner = self.tracked_fleet_owner
        old_ships = self.tracked_fleet_ships
        self.tracked_fleet_ids = torch.full((new_capacity,), -1, dtype=torch.long, device=self.device)
        self.tracked_fleet_eta = torch.zeros((new_capacity,), dtype=torch.long, device=self.device)
        self.tracked_fleet_target_slot = torch.full((new_capacity,), -1, dtype=torch.long, device=self.device)
        self.tracked_fleet_owner = torch.zeros((new_capacity,), dtype=torch.long, device=self.device)
        self.tracked_fleet_ships = torch.zeros((new_capacity,), dtype=self.dtype, device=self.device)
        self.tracked_fleet_ids[:old_capacity] = old_ids
        self.tracked_fleet_eta[:old_capacity] = old_eta
        self.tracked_fleet_target_slot[:old_capacity] = old_tgt
        self.tracked_fleet_owner[:old_capacity] = old_owner
        self.tracked_fleet_ships[:old_capacity] = old_ships

    def _ledger_decrement_and_expire(self) -> None:
        """Giảm ETA trong sổ cái và loại các fleet đã hết hạn/đã đến."""
        if self.tracked_fleet_ids is None or self.tracked_fleet_eta is None or self.tracked_fleet_target_slot is None or (self.tracked_fleet_owner is None) or (self.tracked_fleet_ships is None):
            return
        valid = self.tracked_fleet_ids >= 0
        eta = torch.where(valid, self.tracked_fleet_eta - 1, self.tracked_fleet_eta)
        expire = valid & (eta <= 0)
        self.tracked_fleet_eta = eta
        self.tracked_fleet_ids = torch.where(expire, torch.full_like(self.tracked_fleet_ids, -1), self.tracked_fleet_ids)
        self.tracked_fleet_eta = torch.where(expire, torch.zeros_like(self.tracked_fleet_eta), self.tracked_fleet_eta)
        self.tracked_fleet_target_slot = torch.where(expire, torch.full_like(self.tracked_fleet_target_slot, -1), self.tracked_fleet_target_slot)
        self.tracked_fleet_owner = torch.where(expire, torch.zeros_like(self.tracked_fleet_owner), self.tracked_fleet_owner)
        self.tracked_fleet_ships = torch.where(expire, torch.zeros_like(self.tracked_fleet_ships), self.tracked_fleet_ships)

    def _roll_fleet_buckets_phase1(self, current_step: Tensor) -> None:
        """Roll các bucket arrival của fleet sang turn kế tiếp trước khi ingest observation mới."""
        if self.fleet_buckets is None or self.fleet_last_step is None:
            return
        step = current_step.to(device=self.device, dtype=torch.long)
        delta = step - self.fleet_last_step.to(device=self.device, dtype=torch.long)
        horizon = int(self.movement_horizon)
        reset = bool((delta < 0) | (step <= 0))
        if reset:
            self.fleet_buckets[:] = 0.0
            self._clear_tracked_rows()
            self._mark_garrison_dirty_all(1)
        rolled_once = not reset and bool(delta == 1)
        if rolled_once and horizon > 0:
            self.fleet_buckets[:, :-1, :] = self.fleet_buckets[:, 1:, :].clone()
            self.fleet_buckets[:, -1, :] = 0.0
            self._ledger_decrement_and_expire()
            self._mark_garrison_dirty_all(1)
        delta_bad = not reset and bool(delta > 1)
        if delta_bad:
            self._reset_fleet_tracking()
        self.fleet_last_step = step.detach().clone()

    def _reset_fleet_tracking(self) -> None:
        """Reset toàn bộ tracking fleet khi cần đồng bộ lại từ observation."""
        if self.fleet_buckets is None:
            return
        self.fleet_buckets[:] = 0.0
        self._clear_tracked_rows()
        self._mark_garrison_dirty_all(1)

    def _ingest_obs_fleets(self, obs_tensors: dict) -> None:
        """Đưa các fleet quan sát được vào hệ thống tracking.

Hàm ước lượng mục tiêu/ETA, ghi vào ledger và bucket arrival để projection garrison biết tác động tương lai của fleet."""
        if self.fleet_buckets is None or self.tracked_fleet_ids is None or int(self.movement_horizon) <= 0:
            return
        fleets = obs_tensors['fleets'].to(device=self.device, dtype=self.dtype)
        fleet_ids = fleets[..., 0].to(dtype=torch.long)
        alive = fleet_ids >= 0
        tracked = (fleet_ids.unsqueeze(1) == self.tracked_fleet_ids.unsqueeze(0)).any(dim=1)
        process_mask = alive & ~tracked
        n_alive = int(alive.sum().item())
        n_tracked = int((alive & tracked).sum().item())
        n_to_process = n_alive - n_tracked
        if n_to_process == 0:
            return
        fleet_slot = torch.where(process_mask)[0]
        proc_ids = fleet_ids[fleet_slot]
        estimate = _estimate_new_fleet_arrivals(movement=self, obs_fleets=fleets, fleet_slot=fleet_slot)
        valid_owner = (estimate['owner'] >= 0) & (estimate['owner'] < int(self.fleet_buckets.shape[2]))
        valid_hit = estimate['has_hit'] & valid_owner
        if not bool(valid_hit.any()):
            return
        buckets = self._require_fleet_buckets()
        buckets.index_put_((estimate['target_slot'][valid_hit], estimate['eta_index'][valid_hit], estimate['owner'][valid_hit]), estimate['ships'][valid_hit], accumulate=True)
        self._mark_garrison_dirty(estimate['target_slot'][valid_hit], estimate['eta_index'][valid_hit] + 1)
        eta_remaining = estimate['eta_index'][valid_hit].to(dtype=torch.long) + 1
        self._ledger_bulk_insert(proc_ids[valid_hit], eta_remaining, estimate['target_slot'][valid_hit], estimate['owner'][valid_hit], estimate['ships'][valid_hit])

    def _reconcile_obs_fleets(self, obs_tensors: dict) -> None:
        """Đồng bộ fleet tracking cũ với fleet trong observation mới.

Thuật toán khớp các fleet theo vị trí/owner/ships/góc để cập nhật ETA, thêm fleet mới và loại fleet biến mất."""
        if self.fleet_buckets is None or self.tracked_fleet_ids is None or self.tracked_fleet_eta is None or (self.tracked_fleet_target_slot is None) or (self.tracked_fleet_owner is None) or (self.tracked_fleet_ships is None) or (int(self.movement_horizon) <= 0):
            return
        obs_ids = obs_tensors['fleets'][..., 0].to(device=self.device, dtype=torch.long)
        in_flight = (self.tracked_fleet_ids >= 0) & (self.tracked_fleet_eta > 0)
        if not bool(in_flight.any()):
            return
        match = (self.tracked_fleet_ids.unsqueeze(1) == obs_ids.unsqueeze(0)).any(dim=1)
        phantom = in_flight & ~match
        if not bool(phantom.any()):
            return
        m_idx = torch.where(phantom)[0]
        h_idx = (self.tracked_fleet_eta[m_idx] - 1).clamp(min=0)
        P = int(self.fleet_buckets.shape[0])
        H = int(self.fleet_buckets.shape[1])
        A = int(self.fleet_buckets.shape[2])
        in_horizon = h_idx < H
        if not bool(in_horizon.any()):
            self.tracked_fleet_ids[m_idx] = -1
            self.tracked_fleet_eta[m_idx] = 0
            self.tracked_fleet_target_slot[m_idx] = -1
            self.tracked_fleet_owner[m_idx] = 0
            self.tracked_fleet_ships[m_idx] = 0.0
            return
        m_sel = m_idx[in_horizon]
        h_sel = h_idx[in_horizon]
        slots = self.tracked_fleet_target_slot[m_sel].clamp(min=0, max=max(P - 1, 0))
        owners = self.tracked_fleet_owner[m_sel].clamp(min=0, max=max(A - 1, 0))
        ships = self.tracked_fleet_ships[m_sel]
        self.fleet_buckets.index_put_((slots, h_sel, owners), -ships, accumulate=True)
        self._mark_garrison_dirty(slots, h_sel + 1)
        self.tracked_fleet_ids[m_idx] = -1
        self.tracked_fleet_eta[m_idx] = 0
        self.tracked_fleet_target_slot[m_idx] = -1
        self.tracked_fleet_owner[m_idx] = 0
        self.tracked_fleet_ships[m_idx] = 0.0

    def _require_fleet_buckets(self) -> Tensor:
        """Đảm bảo các tensor bucket arrival của fleet đã được khởi tạo."""
        if self.fleet_buckets is None:
            raise RuntimeError('PlanetMovement fleet tracking is not enabled')
        return self.fleet_buckets

    def _k_index(self, k: int) -> int:
        """Chuẩn hoá chỉ số thời gian k trong horizon projection."""
        if k < 0 or k > int(self.movement_horizon):
            raise IndexError(f'k must be in [0, {self.movement_horizon}], got {k}')
        return int(k)

    def _copy_from(self, other: 'PlanetMovement') -> None:
        """Sao chép toàn bộ trạng thái từ một PlanetMovement khác.

Dùng khi cần thay thế memory bằng mô hình mới nhưng vẫn giữ object runtime."""
        self.x = other.x
        self.y = other.y
        self.alive_by_step = other.alive_by_step
        self.planet_ids = other.planet_ids
        self.radii = other.radii
        self.planet_owner = other.planet_owner
        self.planet_ships = other.planet_ships
        self.planet_prod = other.planet_prod
        self.base_step = other.base_step
        self.comet_planet_ids = other.comet_planet_ids
        self.comet_path_index = other.comet_path_index
        self.movement_horizon = other.movement_horizon
        self.drift_epsilon = other.drift_epsilon
        self.track_fleets = other.track_fleets
        self.player_count = other.player_count
        self.max_tracked_fleets = other.max_tracked_fleets
        self.fleet_buckets = other.fleet_buckets
        self.fleet_last_step = other.fleet_last_step
        self.tracked_fleet_ids = other.tracked_fleet_ids
        self.tracked_fleet_eta = other.tracked_fleet_eta
        self.tracked_fleet_target_slot = other.tracked_fleet_target_slot
        self.tracked_fleet_owner = other.tracked_fleet_owner
        self.tracked_fleet_ships = other.tracked_fleet_ships
        self.garrison_owner_cache = other.garrison_owner_cache
        self.garrison_ships_cache = other.garrison_ships_cache
        self.garrison_dirty_from = other.garrison_dirty_from

def _resolve_player_count(obs_tensors: dict, player_count: int | None) -> int:
    """Suy luận số người chơi từ metadata hoặc owner ban đầu."""
    if player_count is not None:
        if int(player_count) not in (2, 4):
            raise ValueError('player_count must be 2 or 4')
        return int(player_count)
    metadata_count = obs_tensors.get('player_count')
    if metadata_count is not None:
        count = int(metadata_count.flatten()[0].item()) if isinstance(metadata_count, Tensor) else int(metadata_count)
        if count not in (2, 4):
            raise ValueError('player_count metadata must be 2 or 4')
        return count
    planets = obs_tensors['planets']
    fleets = obs_tensors['fleets']
    planet_alive = planets[..., 0] >= 0
    fleet_alive = fleets[..., 0] >= 0
    owner_values = []
    if bool(planet_alive.any()):
        owner_values.append(planets[..., 1][planet_alive].to(dtype=torch.long))
    if bool(fleet_alive.any()):
        owner_values.append(fleets[..., 1][fleet_alive].to(dtype=torch.long))
    if not owner_values:
        return 2
    owners = torch.cat(owner_values)
    owners = owners[owners >= 0]
    if owners.numel() == 0:
        return 2
    return 4 if int(owners.max().item()) >= 2 else 2

def _estimate_new_fleet_arrivals(*, movement: PlanetMovement, obs_fleets: Tensor, fleet_slot: Tensor) -> dict[str, Tensor]:
    """Ước lượng mục tiêu và ETA của fleet mới quan sát được.

Thuật toán dự đoán đường bay thẳng từ vị trí/góc fleet, quét các hành tinh trong tương lai và chọn va chạm hợp lệ sớm nhất."""
    N = int(fleet_slot.numel())
    device = movement.device
    dtype = movement.dtype
    H = int(movement.movement_horizon)
    P = int(movement.P)
    if N == 0:
        empty_long = torch.empty(0, dtype=torch.long, device=device)
        empty_bool = torch.empty(0, dtype=torch.bool, device=device)
        empty_float = torch.empty(0, dtype=dtype, device=device)
        return {'owner': empty_long, 'target_slot': empty_long, 'eta_index': empty_long, 'has_hit': empty_bool, 'ships': empty_float}
    rows = obs_fleets[fleet_slot]
    owner = rows[:, 1].to(dtype=torch.long)
    x = rows[:, 2].to(dtype=dtype)
    y = rows[:, 3].to(dtype=dtype)
    angle = rows[:, 4].to(dtype=dtype)
    ships = rows[:, 6].to(dtype=dtype)
    times = torch.arange(1, H + 1, dtype=dtype, device=device).view(1, H)
    speed = fleet_speed(ships).clamp(min=1e-06)
    ux = torch.cos(angle)
    uy = torch.sin(angle)
    old_x = x.view(N, 1) + ux.view(N, 1) * speed.view(N, 1) * (times - 1.0)
    old_y = y.view(N, 1) + uy.view(N, 1) * speed.view(N, 1) * (times - 1.0)
    new_x = x.view(N, 1) + ux.view(N, 1) * speed.view(N, 1) * times
    new_y = y.view(N, 1) + uy.view(N, 1) * speed.view(N, 1) * times
    in_bounds = (new_x >= 0.0) & (new_x <= BOARD_SIZE) & (new_y >= 0.0) & (new_y <= BOARD_SIZE)
    sun_dist_sq = _point_to_segment_distance_sq(torch.full_like(new_x, CENTER), torch.full_like(new_y, CENTER), old_x, old_y, new_x, new_y)
    env_kill = ~in_bounds | (sun_dist_sq < SUN_RADIUS * SUN_RADIUS)
    planet_x = movement.x.unsqueeze(0).expand(N, H + 1, P)
    planet_y = movement.y.unsqueeze(0).expand(N, H + 1, P)
    planet_alive = movement.alive_by_step.unsqueeze(0).expand(N, H + 1, P)
    radii = movement.radii.unsqueeze(0).expand(N, P).to(dtype=dtype)
    old_px = planet_x[:, :-1, :]
    old_py = planet_y[:, :-1, :]
    new_px = planet_x[:, 1:, :]
    new_py = planet_y[:, 1:, :]
    alive_old = planet_alive[:, :-1, :]
    check_collision = alive_old & (old_px >= 0.0) & (old_py >= 0.0)
    swept_collides = _swept_pair_hit_mask(old_x.unsqueeze(2), old_y.unsqueeze(2), new_x.unsqueeze(2), new_y.unsqueeze(2), old_px, old_py, new_px, new_py, radii.view(N, 1, P)) & check_collision
    step_raw_has_hit = swept_collides.any(dim=2)
    hit_rank = swept_collides.to(torch.int32).cumsum(dim=2)
    first_hit = swept_collides & (hit_rank == 1)
    step_hit_slot = first_hit.to(torch.int64).argmax(dim=2)
    step_hit_slot = step_hit_slot.where(step_raw_has_hit, torch.full_like(step_hit_slot, -1))
    kill_event = step_raw_has_hit | env_kill
    cum_kill_inclusive = kill_event.cummax(dim=1).values
    alive_before_t = torch.cat([torch.ones((N, 1), dtype=torch.bool, device=device), ~cum_kill_inclusive[:, :-1]], dim=1)
    step_has_hit = step_raw_has_hit & alive_before_t
    has_hit = step_has_hit.any(dim=1)
    eta_index = step_has_hit.to(torch.int64).argmax(dim=1)
    target_slot = step_hit_slot.gather(1, eta_index.view(N, 1)).squeeze(1).clamp(min=0, max=max(P - 1, 0))
    return {'owner': owner, 'target_slot': target_slot, 'eta_index': eta_index, 'has_hit': has_hit, 'ships': ships}

def _point_to_segment_distance_sq(px: Tensor, py: Tensor, x1: Tensor, y1: Tensor, x2: Tensor, y2: Tensor) -> Tensor:
    """Tính bình phương khoảng cách từ điểm tới đoạn thẳng.

Đây là phép hình học nền để kiểm tra một fleet có đi qua vùng va chạm của hành tinh hay không."""
    dx = x2 - x1
    dy = y2 - y1
    denom = dx * dx + dy * dy
    safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))
    t = ((px - x1) * dx + (py - y1) * dy) / safe_denom
    t = t.clamp(0.0, 1.0)
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return (px - proj_x) ** 2 + (py - proj_y) ** 2

def _swept_pair_hit_mask(ax: Tensor, ay: Tensor, bx: Tensor, by: Tensor, p0x: Tensor, p0y: Tensor, p1x: Tensor, p1y: Tensor, r: Tensor) -> Tensor:
    """Kiểm tra đoạn di chuyển của fleet có quét qua mục tiêu trong một bước hay không."""
    d0x = ax - p0x
    d0y = ay - p0y
    dvx = bx - ax - (p1x - p0x)
    dvy = by - ay - (p1y - p0y)
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - r * r
    near_static = a < 1e-12
    c_hit = c <= 0.0
    disc = b * b - 4.0 * a * c
    has_root = disc >= 0.0
    safe_a = torch.where(near_static, torch.ones_like(a), a)
    sq = torch.sqrt(torch.clamp(disc, min=0.0))
    t1 = (-b - sq) / (2.0 * safe_a)
    t2 = (-b + sq) / (2.0 * safe_a)
    quad_hit = has_root & (t2 >= 0.0) & (t1 <= 1.0)
    return torch.where(near_static, c_hit, quad_hit)

def _build_future_from_obs(obs_tensors: dict, movement_horizon: int, *, offsets: Tensor | None=None) -> dict[str, Tensor]:
    """Dựng vị trí tương lai của hành tinh từ observation.

Hàm tách hành tinh đứng yên và hành tinh quay quanh tâm, sau đó tính tọa độ từng bước trong horizon."""
    obs = parse_obs(obs_tensors)
    H = int(movement_horizon)
    planets = obs_tensors['planets']
    dtype = planets.dtype
    device = planets.device
    P, _ = planets.shape
    planet_ids = planets[..., 0].long()
    radii = planets[..., 4].to(dtype=dtype)
    owner = planets[..., 1].to(device=device, dtype=torch.long)
    owner = torch.where(obs.alive, owner, torch.full_like(owner, -1))
    ships = planets[..., 5].to(device=device, dtype=dtype)
    prod = planets[..., 6].to(device=device, dtype=dtype)
    step = obs.step.to(device=device, dtype=torch.long)
    if offsets is None:
        offsets_long = torch.arange(H + 1, dtype=torch.long, device=device)
    else:
        offsets_long = offsets.to(device=device, dtype=torch.long).reshape(-1)
    M = int(offsets_long.shape[0])
    offsets_d = offsets_long.to(dtype=dtype)
    future_phase = orbit_phase_index_from_obs_step(obs.step.to(dtype=dtype) + offsets_d).to(device=device, dtype=dtype)
    angle = obs.orb_a0.to(dtype=dtype).view(1, P) + obs.angvel.to(dtype=dtype) * future_phase.view(M, 1)
    orb_x = CENTER + obs.orb_r.to(dtype=dtype).view(1, P) * torch.cos(angle)
    orb_y = CENTER + obs.orb_r.to(dtype=dtype).view(1, P) * torch.sin(angle)
    is_orbiting = obs.is_orbiting.view(1, P)
    x = torch.where(is_orbiting, orb_x, obs.x.to(dtype=dtype).view(1, P).expand(M, P)).contiguous()
    y = torch.where(is_orbiting, orb_y, obs.y.to(dtype=dtype).view(1, P).expand(M, P)).contiguous()
    alive_by_step = obs.alive.view(1, P).expand(M, P).clone()
    comet_planet_ids, comet_path_index = _comet_metadata(obs_tensors, device)
    x, y, alive_by_step = _apply_comet_paths(x=x, y=y, alive_by_step=alive_by_step, planet_ids=planet_ids, comet_planet_ids=comet_planet_ids, comet_path_index=comet_path_index, obs_tensors=obs_tensors, offsets=offsets_long)
    zero_idx = (offsets_long == 0).nonzero(as_tuple=True)[0]
    if int(zero_idx.numel()) > 0:
        x[zero_idx, :] = obs.x.to(dtype=dtype).view(1, P)
        y[zero_idx, :] = obs.y.to(dtype=dtype).view(1, P)
        alive_by_step[zero_idx, :] = obs.alive.view(1, P)
    return {'x': x, 'y': y, 'alive_by_step': alive_by_step, 'planet_ids': planet_ids, 'radii': radii, 'owner': owner, 'ships': ships, 'prod': prod, 'step': step, 'comet_planet_ids': comet_planet_ids, 'comet_path_index': comet_path_index, '_offsets': offsets_long}

def _comet_metadata(obs_tensors: dict, device: torch.device) -> tuple[Tensor, Tensor]:
    """Tạo metadata cho các comet event nếu bản đồ có hành tinh comet."""
    comets = obs_tensors.get('comets') or {}
    comet_ids = comets.get('planet_ids')
    if comet_ids is None:
        flat_ids = obs_tensors.get('comet_planet_ids')
        if flat_ids is None:
            flat_ids = torch.full((0,), -1, dtype=torch.long, device=device)
        else:
            flat_ids = flat_ids.to(device=device, dtype=torch.long)
        path_index = torch.full((0,), -1, dtype=torch.long, device=device)
        return (flat_ids, path_index)
    comet_ids = comet_ids.to(device=device, dtype=torch.long)
    flat_ids = comet_ids.reshape(-1)
    path_index = comets.get('path_index')
    if path_index is None:
        path_index = torch.full((comet_ids.shape[0],), -1, dtype=torch.long, device=device)
    else:
        path_index = path_index.to(device=device, dtype=torch.long)
    return (flat_ids, path_index)

def _apply_comet_paths(*, x: Tensor, y: Tensor, alive_by_step: Tensor, planet_ids: Tensor, comet_planet_ids: Tensor, comet_path_index: Tensor, obs_tensors: dict, offsets: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Áp dụng đường đi comet vào trajectory tương lai.

Comet là hành tinh/event đặc biệt nên cần cập nhật alive và vị trí theo lịch spawn/đường bay riêng."""
    comets = obs_tensors.get('comets') or {}
    paths = comets.get('paths')
    ids_grid = comets.get('planet_ids')
    if paths is None or ids_grid is None or comet_planet_ids.numel() == 0:
        return (x, y, alive_by_step)
    M, P = x.shape
    paths = paths.to(device=x.device, dtype=x.dtype)
    ids_grid = ids_grid.to(device=x.device, dtype=torch.long)
    E = int(ids_grid.shape[0])
    C = int(ids_grid.shape[1])
    T = int(paths.shape[2])
    if E == 0 or C == 0 or T == 0:
        return (x, y, alive_by_step)
    flat_ids = ids_grid.reshape(E * C)
    matches = (planet_ids.unsqueeze(1) == flat_ids.unsqueeze(0)) & (flat_ids.unsqueeze(0) >= 0)
    is_comet = matches.any(dim=1)
    flat_slot = matches.to(torch.float32).argmax(dim=1).long()
    flat_paths_x = paths[..., 0].reshape(E * C, T)
    flat_paths_y = paths[..., 1].reshape(E * C, T)
    path_x_by_slot = flat_paths_x[flat_slot]
    path_y_by_slot = flat_paths_y[flat_slot]
    finite = torch.isfinite(flat_paths_x)
    path_len = finite.sum(dim=1).to(dtype=torch.long)
    len_by_slot = path_len[flat_slot]
    group_idx = (flat_slot // C).clamp(min=0, max=max(E - 1, 0))
    path_idx_by_slot = comet_path_index[group_idx]
    offsets_v = offsets.to(device=x.device, dtype=torch.long).view(M, 1)
    future_idx = path_idx_by_slot.view(1, P) + offsets_v
    valid_future = is_comet.view(1, P) & (future_idx >= 0) & (future_idx < len_by_slot.view(1, P))
    idx_clamped = future_idx.clamp(min=0, max=max(T - 1, 0))
    p_index = torch.arange(P, device=x.device).view(1, P).expand(M, P)
    comet_x = path_x_by_slot[p_index, idx_clamped]
    comet_y = path_y_by_slot[p_index, idx_clamped]
    x = torch.where(valid_future, comet_x, x)
    y = torch.where(valid_future, comet_y, y)
    alive_by_step = torch.where(is_comet.view(1, P), valid_future, alive_by_step)
    return (x, y, alive_by_step)

def _same_2d(a: Tensor, b: Tensor) -> bool:
    """So sánh hai tensor tọa độ 2D với sai số cho phép."""
    if a.shape != b.shape:
        return False
    if a.numel() == 0:
        return True
    return bool((a == b.to(device=a.device, dtype=a.dtype)).all())

def _position_matches(pred_x: Tensor, pred_y: Tensor, cur_x: Tensor, cur_y: Tensor, alive: Tensor, epsilon: float) -> bool:
    """Kiểm tra hai vị trí có khớp nhau trong ngưỡng tolerance."""
    diff = torch.maximum((pred_x - cur_x).abs(), (pred_y - cur_y).abs())
    diff = torch.where(alive, diff, torch.zeros_like(diff))
    return bool((diff <= float(epsilon)).all())
