from __future__ import annotations
from torch import Tensor

def orbit_phase_index_from_obs_step(obs_step: Tensor) -> Tensor:
    """Tính chỉ số pha quỹ đạo từ step hiện tại.

Đây là helper nhỏ để đồng bộ chuyển động hành tinh theo thời gian rời rạc của môi trường. Khi step tăng, pha orbit thay đổi và vị trí tương lai của hành tinh cũng thay đổi."""
    s = obs_step.float()
    return (s - (s > 0).to(s.dtype)).clamp(min=0.0)
