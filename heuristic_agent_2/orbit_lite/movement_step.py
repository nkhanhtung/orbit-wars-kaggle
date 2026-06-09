from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import torch
from torch import Tensor
from .movement import MovementConfig, PlanetMovement, _estimate_new_fleet_arrivals

@dataclass(frozen=True)
class PlannedLaunches:
    """Bảng lưu các launch đã được suy luận sau khi agent phóng quân.

Thông tin này dùng để cập nhật private movement, giúp agent biết action vừa gửi sẽ tạo fleet nào và đến đâu trong các lượt sau."""
    source_slots: Tensor
    angle: Tensor
    ships: Tensor
    target_slots: Tensor
    eta_turns: Tensor
    valid: Tensor
    fleet_ids: Tensor

@dataclass(frozen=True)
class LaunchEntries:
    """Bảng action ứng viên/đã chọn ở mức planner.

Mỗi entry gồm source, target, ships, angle, ETA và valid mask. Đây là dạng trung gian trước khi đóng gói thành sparse action."""
    source_slots: Tensor
    target_slots: Tensor
    ships: Tensor
    angle: Tensor
    eta: Tensor
    valid: Tensor

    @property
    def width(self) -> int:
        """Trả số entry launch đang có trong bảng."""
        return int(self.source_slots.shape[0])

def concat_launch_entries(entries: Sequence[LaunchEntries]) -> LaunchEntries:
    """Ghép nhiều bảng LaunchEntries thành một bảng duy nhất.

Dùng khi agent có cả wave tấn công và regroup trong cùng turn."""
    if not entries:
        raise ValueError('concat_launch_entries requires at least one entry table')
    if len(entries) == 1:
        return entries[0]
    return LaunchEntries(source_slots=torch.cat([e.source_slots for e in entries], dim=0), target_slots=torch.cat([e.target_slots for e in entries], dim=0), ships=torch.cat([e.ships for e in entries], dim=0), angle=torch.cat([e.angle for e in entries], dim=0), eta=torch.cat([e.eta for e in entries], dim=0), valid=torch.cat([e.valid for e in entries], dim=0))

def disambiguate_duplicate_launches(entries: LaunchEntries, *, epsilon: float=1e-05) -> LaunchEntries:
    """Khử nhập nhằng khi nhiều launch giống nhau về source/angle/ships.

Hàm gán target/ETA rõ hơn để private tracking không nhầm các launch trùng thông số."""
    src = entries.source_slots
    ang = entries.angle
    ships = entries.ships
    valid = entries.valid
    L = src.shape[0]
    if L < 2 or not bool(valid.any()):
        return entries
    device = src.device
    src_i = src.unsqueeze(1)
    src_j = src.unsqueeze(0)
    ang_i = ang.unsqueeze(1)
    ang_j = ang.unsqueeze(0)
    ships_i = ships.unsqueeze(1)
    ships_j = ships.unsqueeze(0)
    valid_i = valid.unsqueeze(1)
    valid_j = valid.unsqueeze(0)
    j_indices = torch.arange(L, device=device).view(1, L)
    i_indices = torch.arange(L, device=device).view(L, 1)
    earlier = j_indices < i_indices
    match = valid_i & valid_j & (src_i == src_j) & (ang_i == ang_j) & (ships_i == ships_j) & earlier
    if not bool(match.any()):
        return entries
    dup_count = match.sum(dim=1).to(ang.dtype)
    new_angle = ang + dup_count * float(epsilon)
    return LaunchEntries(source_slots=entries.source_slots, target_slots=entries.target_slots, ships=entries.ships, angle=new_angle, eta=entries.eta, valid=entries.valid)

def ensure_planet_movement(*, obs_tensors: dict, expected_cfg: MovementConfig, cached_movement: PlanetMovement | None) -> PlanetMovement:
    """Đảm bảo có PlanetMovement hợp lệ cho observation hiện tại.

Nếu memory rỗng hoặc không tương thích, hàm tạo mới; nếu còn dùng được thì update theo observation mới."""
    if cached_movement is not None and cached_movement.config == expected_cfg:
        cached_movement.update(obs_tensors)
        return cached_movement
    return PlanetMovement.from_obs_tensors(obs_tensors, config=expected_cfg)

def _resolve_player_next_fleet_id(obs_tensors: dict, *, device: torch.device) -> Tensor:
    """Tìm fleet id tiếp theo của người chơi để gán cho launch mới trong private tracking."""
    next_fleet_id = obs_tensors.get('player_next_fleet_id', obs_tensors.get('next_fleet_id'))
    if next_fleet_id is None:
        return torch.zeros((), dtype=torch.long, device=device)
    return next_fleet_id.to(device=device, dtype=torch.long)

def infer_planned_launches_from_entries(*, obs_tensors: dict, movement: PlanetMovement, entries: LaunchEntries, player_id: int) -> PlannedLaunches:
    """Chuyển LaunchEntries đã chọn thành PlannedLaunches.

Hàm suy ra fleet id, target slot, ETA và valid mask để stash vào PlanetMovement sau khi action được gửi."""
    source_slots = entries.source_slots
    angle = entries.angle
    ships = entries.ships
    launch_valid = entries.valid
    L = source_slots.shape[0]
    device = source_slots.device
    P = max(int(movement.P), 1)
    next_fleet_id = _resolve_player_next_fleet_id(obs_tensors, device=device)
    launch_long = launch_valid.to(torch.long)
    launch_rank = launch_long.cumsum(0) - launch_long
    fleet_ids = next_fleet_id + launch_rank
    src_safe = source_slots.clamp(min=0, max=P - 1)
    launch_x, launch_y = movement.position_at_slots(src_safe, 0)
    source_r = movement.radii[src_safe]
    start_x = launch_x + torch.cos(angle) * (source_r + 0.1)
    start_y = launch_y + torch.sin(angle) * (source_r + 0.1)
    source_planet_ids = movement.planet_ids[src_safe]
    rows = torch.full((L, 7), -1.0, dtype=movement.dtype, device=device)
    rows[..., 0] = fleet_ids.to(dtype=movement.dtype)
    rows[..., 1] = float(player_id)
    rows[..., 2] = start_x.to(dtype=movement.dtype)
    rows[..., 3] = start_y.to(dtype=movement.dtype)
    rows[..., 4] = angle.to(dtype=movement.dtype)
    rows[..., 5] = source_planet_ids.to(dtype=movement.dtype)
    rows[..., 6] = ships.to(dtype=movement.dtype)
    rows[..., 0] = torch.where(launch_valid, rows[..., 0], torch.full_like(rows[..., 0], -1.0))
    target_slots = torch.zeros(L, dtype=torch.long, device=device)
    eta_turns = torch.zeros(L, dtype=torch.float32, device=device)
    intent_valid = torch.zeros(L, dtype=torch.bool, device=device)
    fleet_slot = torch.where(launch_valid)[0]
    if int(fleet_slot.numel()) > 0:
        estimate = _estimate_new_fleet_arrivals(movement=movement, obs_fleets=rows, fleet_slot=fleet_slot)
        valid_hit = estimate['has_hit']
        if bool(valid_hit.any()):
            src = fleet_slot[valid_hit]
            target_slots[src] = estimate['target_slot'][valid_hit]
            eta_turns[src] = estimate['eta_index'][valid_hit].to(dtype=torch.float32) + 1.0
            intent_valid[src] = True
    return PlannedLaunches(source_slots=source_slots, angle=angle, ships=ships, target_slots=target_slots, eta_turns=eta_turns, valid=intent_valid, fleet_ids=fleet_ids)

def apply_private_planned_launches(*, movement: PlanetMovement, launches: PlannedLaunches, owner_id: int, obs_tensors: dict) -> None:
    """Áp dụng các launch đã chọn vào mô hình riêng của agent.

Thuật toán trừ quân khỏi source, ghi pending launch và đánh dấu garrison dirty để projection sau đó tính cả action vừa phóng."""
    if not movement.track_fleets:
        return
    movement.record_fleet_arrivals(target_slots=launches.target_slots, owner_ids=int(owner_id), ships=launches.ships, eta=launches.eta_turns, valid=launches.valid)
    nfid = obs_tensors.get('next_fleet_id')
    if nfid is None:
        raise ValueError("obs_tensors is missing 'next_fleet_id'")
    movement.stash_pending_own_launches(owner_id=int(owner_id), source_slots=launches.source_slots, ships=launches.ships, angle=launches.angle, target_slots=launches.target_slots, eta=launches.eta_turns, valid=launches.valid, prev_next_fleet_id=nfid)
