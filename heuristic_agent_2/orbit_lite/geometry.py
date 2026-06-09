from __future__ import annotations
import torch
from torch import Tensor
from .constants import MAX_SHIP_SPEED
_LOG_1000: float = float(torch.log(torch.tensor(1000.0)).item())
_FLEET_SPEED_LUT_MAX: int = 400

def _fleet_speed_formula(ships: Tensor) -> Tensor:
    """Tính tốc độ fleet theo số tàu bằng công thức gameplay.

Số tàu lớn hơn thường bay nhanh hơn nhưng tốc độ bị chặn trên bởi MAX_SHIP_SPEED. Hàm dùng log và lũy thừa để phản ánh quy luật tăng tốc phi tuyến."""
    ratio = (torch.log(ships) / _LOG_1000).clamp(max=1.0)
    return 1.0 + (MAX_SHIP_SPEED - 1.0) * ratio.pow(1.5)

def _build_fleet_speed_lut(max_ships: int) -> Tensor:
    """Tạo bảng tra cứu tốc độ fleet cho số tàu nhỏ/vừa.

LUT giúp tránh tính log lặp lại hàng nghìn lần khi planner xét nhiều ứng viên source-target."""
    idx = torch.arange(max_ships + 1, dtype=torch.float32).clamp(min=1.0)
    return _fleet_speed_formula(idx)
_FLEET_SPEED_LUT: Tensor = _build_fleet_speed_lut(_FLEET_SPEED_LUT_MAX)
_FLEET_SPEED_LUT_CACHE: dict[tuple, Tensor] = {}

def _fleet_speed_lut_on(device: torch.device, dtype: torch.dtype) -> Tensor:
    """Đưa LUT tốc độ về đúng device và dtype đang sử dụng.

Hàm có cache theo (device, dtype) để không copy tensor nhiều lần giữa CPU/GPU."""
    key = (device, dtype)
    cached = _FLEET_SPEED_LUT_CACHE.get(key)
    if cached is None:
        cached = _FLEET_SPEED_LUT.to(device=device, dtype=dtype)
        _FLEET_SPEED_LUT_CACHE[key] = cached
    return cached

def fleet_speed(ships: Tensor) -> Tensor:
    """Tính tốc độ fleet nhanh và ổn định.

Thuật toán dùng nội suy tuyến tính trên LUT cho số tàu trong giới hạn, và fallback về công thức gốc khi số tàu vượt bảng. Đây là tối ưu hiệu năng nhưng vẫn giữ độ chính xác cần thiết."""
    s = ships.clamp(min=1.0)
    s_lut = s.clamp(max=float(_FLEET_SPEED_LUT_MAX))
    lo = torch.floor(s_lut).long()
    hi = torch.ceil(s_lut).long()
    frac = s_lut - lo.to(dtype=s.dtype)
    lut = _fleet_speed_lut_on(s.device, s.dtype)
    speed = lut[lo] + (lut[hi] - lut[lo]) * frac
    over = s > float(_FLEET_SPEED_LUT_MAX)
    speed_formula = _fleet_speed_formula(s)
    return torch.where(over, speed_formula, speed)
