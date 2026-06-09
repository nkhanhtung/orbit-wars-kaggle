from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor
from .constants import CENTER, ROT_RADIUS_LIMIT

@dataclass
class ParsedObs:
    """Observation đã được parse thành các trường có ý nghĩa.

Lớp tách rõ planet, fleet, owner, mask owned/enemy/neutral và tham số orbit để các module sau không phải đọc tensor thô."""
    alive: Tensor
    x: Tensor
    y: Tensor
    r: Tensor
    ships: Tensor
    prod: Tensor
    owner_abs: Tensor
    owned: Tensor
    is_enemy: Tensor
    is_neutral: Tensor
    orb_r: Tensor
    orb_a0: Tensor
    is_orbiting: Tensor
    angvel: Tensor
    step: Tensor
    f_alive: Tensor
    f_owner: Tensor
    f_x: Tensor
    f_y: Tensor
    f_angle: Tensor
    f_ships: Tensor
    player_id: int
    P: int
    F: int
    device: torch.device

def parse_obs(obs_tensors: dict, player_id: int | None=None) -> ParsedObs:
    """Parse observation tensor thành ParsedObs.

Thuật toán đọc planet/fleet arrays, xác định hành tinh sống, owner tuyệt đối, quân, production, mask bạn/địch/trung lập, bán kính và pha quỹ đạo. Đây là bước chuẩn hoá dữ liệu đầu vào cho toàn planner."""
    planets = obs_tensors['planets']
    initial = obs_tensors['initial_planets']
    fleets = obs_tensors['fleets']
    angvel = obs_tensors['angular_velocity'].float()
    step = obs_tensors['step'].float()
    if player_id is None:
        player_id = int(obs_tensors['player'].flatten()[0].item())
    P, _ = planets.shape
    F, _ = fleets.shape
    device = planets.device
    pid = planets[..., 0]
    owner_abs = planets[..., 1]
    x = planets[..., 2]
    y = planets[..., 3]
    r = planets[..., 4]
    ships = planets[..., 5]
    prod = planets[..., 6]
    alive = pid >= 0.0
    owned = alive & (owner_abs == float(player_id))
    is_enemy = alive & (owner_abs >= 0.0) & (owner_abs != float(player_id))
    is_neutral = alive & (owner_abs < 0.0)
    ix = initial[..., 2]
    iy = initial[..., 3]
    i_r = initial[..., 4]
    dx0 = ix - CENTER
    dy0 = iy - CENTER
    orb_r_raw = torch.sqrt(dx0 * dx0 + dy0 * dy0)
    orb_a0 = torch.atan2(dy0, dx0)
    is_orbiting = alive & (orb_r_raw + i_r < ROT_RADIUS_LIMIT) & (orb_r_raw > 0.5)
    orb_r = torch.where(is_orbiting, orb_r_raw, torch.zeros_like(orb_r_raw))
    f_pid = fleets[..., 0]
    f_alive = f_pid >= 0.0
    f_owner = fleets[..., 1]
    f_x = fleets[..., 2]
    f_y = fleets[..., 3]
    f_angle = fleets[..., 4]
    f_ships = fleets[..., 6]
    return ParsedObs(alive=alive, x=x, y=y, r=r, ships=ships, prod=prod, owner_abs=owner_abs, owned=owned, is_enemy=is_enemy, is_neutral=is_neutral, orb_r=orb_r, orb_a0=orb_a0, is_orbiting=is_orbiting, angvel=angvel, step=step, f_alive=f_alive, f_owner=f_owner, f_x=f_x, f_y=f_y, f_angle=f_angle, f_ships=f_ships, player_id=player_id, P=P, F=F, device=device)
