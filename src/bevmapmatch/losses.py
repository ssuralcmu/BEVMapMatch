import torch
from torch.nn import functional as F


def gaussian_targets(labels: torch.Tensor, grid_dim: int, sigma: float) -> torch.Tensor:
    center = labels.argmax(1)
    centers = torch.stack((center // grid_dim, center % grid_dim), dim=1).float()
    axis = torch.arange(grid_dim, device=labels.device)
    coordinates = torch.stack(torch.meshgrid(axis, axis, indexing="ij"), -1).reshape(-1, 2)
    distance_squared = ((coordinates[None] - centers[:, None]) ** 2).sum(2)
    targets = torch.exp(-distance_squared / (2 * sigma**2))
    return targets / targets.sum(1, keepdim=True)


def paper_loss(
    logits: torch.Tensor, labels: torch.Tensor, grid_dim: int = 10, distance_weight: float = 0.1, sigma: float = 0.8
) -> tuple[torch.Tensor, dict[str, float]]:
    binary = F.binary_cross_entropy_with_logits(logits, labels)
    soft = gaussian_targets(labels, grid_dim, sigma)
    distance = -(soft * F.log_softmax(logits, dim=1)).sum(1).mean()
    total = binary + distance_weight * distance
    return total, {"bce": binary.item(), "distance": distance.item(), "total": total.item()}

