from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel


class FrozenDINOv2(nn.Module):
    def __init__(self, name: str = "facebook/dinov2-large") -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained(name)
        self.dimension = self.model.config.hidden_size
        self.model.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.model(pixel_values=images).last_hidden_state[:, 1:]
        side = int(tokens.shape[1] ** 0.5)
        if side * side != tokens.shape[1]:
            raise ValueError("DINOv2 patch tokens do not form a square feature map")
        return tokens.transpose(1, 2).reshape(images.shape[0], tokens.shape[2], side, side)


class CoarseMatcher(nn.Module):
    """Paper coarse matcher: frozen DINOv2, 3x3 cell context and cross-attention."""

    def __init__(self, backbone: str = "facebook/dinov2-large", grid_dim: int = 10, heads: int = 8) -> None:
        super().__init__()
        self.encoder = FrozenDINOv2(backbone)
        self.grid_dim = grid_dim
        dimension = self.encoder.dimension
        self.pool = nn.AdaptiveAvgPool2d((grid_dim, grid_dim))
        self.attention = nn.MultiheadAttention(dimension, heads, batch_first=True)
        self.position = nn.Parameter(torch.zeros(1, grid_dim * grid_dim, dimension))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.temporal_fusion = nn.Sequential(
            nn.Linear(2 * dimension, dimension), nn.GELU(), nn.LayerNorm(dimension)
        )
        self.classifier = nn.Linear(dimension, grid_dim * grid_dim)

    def _descriptor(self, features: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(features, 1).flatten(1)

    def _map_tokens(self, features: torch.Tensor) -> torch.Tensor:
        grid = self.pool(features)
        patches = F.unfold(grid, kernel_size=3, padding=1)
        patches = patches.view(grid.shape[0], grid.shape[1], 9, -1).mean(2)
        return patches.transpose(1, 2) + self.position

    def forward(
        self, segmentation: torch.Tensor, basemap: torch.Tensor, neighbors: torch.Tensor | None = None
    ) -> torch.Tensor:
        query = self._descriptor(self.encoder(segmentation))
        if neighbors is not None and neighbors.numel():
            batch, count, channels, height, width = neighbors.shape
            neighbor_features = self.encoder(neighbors.reshape(batch * count, channels, height, width))
            neighbor_query = self._descriptor(neighbor_features).view(batch, count, -1).mean(1)
            query = self.temporal_fusion(torch.cat([query, neighbor_query], dim=1))
        tokens = self._map_tokens(self.encoder(basemap))
        attended, _ = self.attention(query[:, None, :], tokens, tokens, need_weights=False)
        return self.classifier(attended[:, 0])

