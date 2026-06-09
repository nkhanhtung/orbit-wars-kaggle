from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor
from .movement import PlanetMovement

@dataclass
class DistanceCache:
    """Cấu trúc lưu cache khoảng cách giữa các hành tinh theo từng bước tương lai.

Cache tránh phải tính lại pairwise distance nhiều lần trong planner, đặc biệt khi xét nhiều source-target và nhiều kích thước fleet."""
    cross_dist: Tensor
    alive_by_step: Tensor
    K: int

    @property
    def P(self) -> int:
        """Trả số lượng hành tinh trong cache."""
        return int(self.cross_dist.shape[-1])

    @property
    def device(self) -> torch.device:
        """Trả device của tensor khoảng cách để các tensor mới tạo đồng bộ CPU/GPU."""
        return self.cross_dist.device

    @property
    def dtype(self) -> torch.dtype:
        """Trả dtype của tensor khoảng cách để tránh lỗi ép kiểu khi tính toán."""
        return self.cross_dist.dtype

def build_distance_cache(movement: PlanetMovement, *, max_k: int) -> DistanceCache:
    """Dựng cache khoảng cách nguồn-đích trong tương lai.

Thuật toán lấy vị trí dự báo của mọi hành tinh trong horizon rồi tính ma trận khoảng cách pairwise theo từng bước. Kết quả dùng cho shortlist mục tiêu, pressure và reachability."""
    K = max(0, min(int(max_k), int(movement.movement_horizon)))
    P = int(movement.P)
    src_x0 = movement.x[0]
    src_y0 = movement.y[0]
    tgt_x = movement.x[:K + 1]
    tgt_y = movement.y[:K + 1]
    dx = src_x0.view(1, P, 1) - tgt_x.unsqueeze(1)
    dy = src_y0.view(1, P, 1) - tgt_y.unsqueeze(1)
    cross_dist = torch.sqrt((dx * dx + dy * dy).clamp(min=0.0))
    alive_by_step = movement.alive_by_step[:K + 1]
    return DistanceCache(cross_dist=cross_dist, alive_by_step=alive_by_step, K=K)

def min_distance_to_targets(cache: DistanceCache, source_mask: Tensor, target_mask: Tensor, *, max_k: int) -> Tensor:
    """Tính khoảng cách nhỏ nhất từ mỗi hành tinh tới tập mục tiêu.

Hàm dùng reduction min trên cache khoảng cách để hỗ trợ đánh giá hành tinh nào gần mục tiêu quan trọng, phục vụ regroup hoặc ưu tiên phòng thủ."""
    if source_mask.shape[-1] != cache.P or target_mask.shape[-1] != cache.P:
        raise ValueError('source_mask and target_mask must have shape [P]')
    K = max(0, min(int(max_k), int(cache.K)))
    if K <= 0:
        return torch.zeros(cache.P, dtype=cache.dtype, device=cache.device)
    cross = cache.cross_dist[1:K + 1].clone()
    alive_steps = cache.alive_by_step[1:K + 1]
    src_mask = source_mask.to(device=cache.device, dtype=torch.bool)
    tgt_mask = target_mask.to(device=cache.device, dtype=torch.bool)
    inf_v = float('inf')
    cross.masked_fill_(~alive_steps.unsqueeze(1), inf_v)
    cross.masked_fill_(~src_mask.view(1, cache.P, 1), inf_v)
    cross.masked_fill_(~tgt_mask.view(1, 1, cache.P), inf_v)
    best_per_target = cross.amin(dim=(0, 1))
    return torch.where(torch.isfinite(best_per_target), best_per_target, torch.zeros_like(best_per_target))
