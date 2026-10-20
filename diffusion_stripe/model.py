from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


def _group_count(channels: int, maximum: int = 8) -> int:
    groups = min(maximum, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        if dimension < 4:
            raise ValueError("Time embedding dimension must be at least 4")
        self.dimension = dimension

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        exponent = -math.log(10_000.0) * torch.arange(
            half, device=timesteps.device, dtype=torch.float32
        ) / max(half - 1, 1)
        frequencies = exponent.exp()
        arguments = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat((arguments.sin(), arguments.cos()), dim=1)
        if embedding.shape[1] < self.dimension:
            embedding = F.pad(embedding, (0, self.dimension - embedding.shape[1]))
        return embedding


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dimension: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dimension, out_channels),
        )
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(F.silu(self.norm1(x)))
        x = x + self.time_projection(time_embedding).unsqueeze(-1).unsqueeze(-1)
        x = self.conv2(F.silu(self.norm2(x)))
        return x + residual


@dataclass(frozen=True)
class UNetConfig:
    in_channels: int = 3
    out_channels: int = 3
    base_channels: int = 32
    time_embedding_dimension: int = 128

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class TimeConditionedUNet(nn.Module):
    """Compact time-conditioned U-Net for 64x64 RGB x0 prediction."""

    def __init__(self, config: UNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or UNetConfig()
        c = self.config.base_channels
        td = self.config.time_embedding_dimension

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(c),
            nn.Linear(c, td),
            nn.SiLU(),
            nn.Linear(td, td),
        )

        self.input_conv = nn.Conv2d(self.config.in_channels, c, kernel_size=3, padding=1)

        self.encoder1 = ResidualBlock(c, c, td)
        self.down1 = nn.Conv2d(c, c * 2, kernel_size=4, stride=2, padding=1)
        self.encoder2 = ResidualBlock(c * 2, c * 2, td)
        self.down2 = nn.Conv2d(c * 2, c * 4, kernel_size=4, stride=2, padding=1)
        self.encoder3 = ResidualBlock(c * 4, c * 4, td)
        self.down3 = nn.Conv2d(c * 4, c * 8, kernel_size=4, stride=2, padding=1)

        self.middle1 = ResidualBlock(c * 8, c * 8, td)
        self.middle2 = ResidualBlock(c * 8, c * 8, td)

        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=4, stride=2, padding=1)
        self.decoder3 = ResidualBlock(c * 8, c * 4, td)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=4, stride=2, padding=1)
        self.decoder2 = ResidualBlock(c * 4, c * 2, td)
        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=4, stride=2, padding=1)
        self.decoder1 = ResidualBlock(c * 2, c, td)

        self.output_norm = nn.GroupNorm(_group_count(c), c)
        self.output_conv = nn.Conv2d(c, self.config.out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim != 1 or timesteps.shape[0] != x.shape[0]:
            raise ValueError(
                f"timesteps must have shape [batch], got {tuple(timesteps.shape)}"
            )
        time_embedding = self.time_mlp(timesteps)

        x = self.input_conv(x)
        skip1 = self.encoder1(x, time_embedding)
        skip2 = self.encoder2(self.down1(skip1), time_embedding)
        skip3 = self.encoder3(self.down2(skip2), time_embedding)

        x = self.down3(skip3)
        x = self.middle1(x, time_embedding)
        x = self.middle2(x, time_embedding)

        x = self.up3(x)
        x = self.decoder3(torch.cat((x, skip3), dim=1), time_embedding)
        x = self.up2(x)
        x = self.decoder2(torch.cat((x, skip2), dim=1), time_embedding)
        x = self.up1(x)
        x = self.decoder1(torch.cat((x, skip1), dim=1), time_embedding)
        return self.output_conv(F.silu(self.output_norm(x)))


def build_model_from_checkpoint(checkpoint: dict) -> TimeConditionedUNet:
    config_data = checkpoint.get("model_config")
    if not isinstance(config_data, dict):
        raise RuntimeError("Checkpoint is missing model_config")
    model = TimeConditionedUNet(UNetConfig(**config_data))
    model.load_state_dict(checkpoint["model_state"])
    return model
