from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return device


def make_loader(
    dataset: Dataset,
    batch_size: int,
    workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=False,
        generator=generator,
    )


def linear_beta_schedule(
    timesteps: int,
    beta_start: float,
    beta_end: float,
    device: torch.device,
) -> torch.Tensor:
    if timesteps != 100:
        raise ValueError("This experiment protocol requires exactly 100 timesteps")
    if not 0.0 < beta_start < beta_end < 1.0:
        raise ValueError("Require 0 < beta_start < beta_end < 1")
    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return torch.cat((torch.ones(1, device=device), alpha_bars), dim=0)


def ddpm_add_noise(
    x0: torch.Tensor,
    timesteps: torch.Tensor,
    alpha_bars: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if timesteps.min().item() < 1 or timesteps.max().item() >= alpha_bars.shape[0]:
        raise ValueError("Stage-1 timesteps must be integers in [1, 100]")
    noise = torch.randn_like(x0) if noise is None else noise
    selected = alpha_bars[timesteps].view(-1, 1, 1, 1)
    xt = selected.sqrt() * x0 + (1.0 - selected).sqrt() * noise
    return xt, noise


def create_grad_scaler(enabled: bool) -> torch.amp.GradScaler:
    return torch.amp.GradScaler("cuda", enabled=enabled)


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_dataclass(payload):
        payload = asdict(payload)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def ensure_fresh_or_resume(stage_dir: Path, resume: bool) -> None:
    last_checkpoint = stage_dir / "last.pt"
    if resume:
        if not last_checkpoint.is_file():
            raise FileNotFoundError(f"Cannot resume; checkpoint not found: {last_checkpoint}")
        return
    if stage_dir.exists() and any(stage_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty experiment directory: {stage_dir}. "
            "Use --resume or choose a new --output-root."
        )


class FullDataEarlyStopping:
    """Loss-based stopping for the user-requested full-data training protocol."""

    def __init__(self, min_epochs: int, patience: int, min_delta: float) -> None:
        self.min_epochs = min_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.bad_epochs = 0

    def update(self, epoch: int, loss: float) -> tuple[bool, bool]:
        improved = loss < self.best_loss - self.min_delta
        if improved:
            self.best_loss = loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        should_stop = epoch >= self.min_epochs and self.bad_epochs >= self.patience
        return improved, should_stop


def checkpoint_payload(
    *,
    stage: int,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    loss: float,
    best_loss: float,
    bad_epochs: int,
    model_config: dict[str, Any],
    train_config: dict[str, Any],
    source_checkpoint: str | None = None,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "loss": loss,
        "best_loss": best_loss,
        "bad_epochs": bad_epochs,
        "model_config": model_config,
        "train_config": train_config,
        "source_checkpoint": source_checkpoint,
        "rng_state": capture_rng_state(),
    }
