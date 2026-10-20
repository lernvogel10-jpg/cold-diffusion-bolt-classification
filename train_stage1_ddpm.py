from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from diffusion_stripe.data import StripedImageDataset
from diffusion_stripe.model import TimeConditionedUNet, UNetConfig
from diffusion_stripe.training import (
    FullDataEarlyStopping,
    append_csv,
    atomic_torch_save,
    checkpoint_payload,
    create_grad_scaler,
    ddpm_add_noise,
    ensure_fresh_or_resume,
    linear_beta_schedule,
    make_loader,
    resolve_device,
    restore_rng_state,
    save_json,
    seed_everything,
)


DEFAULT_STRIPED_ROOT = Path("data/watermarked")
DEFAULT_OUTPUT_ROOT = Path("runs/generator")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1: DDPM Gaussian-noise x0 prediction on all 6199 striped images."
    )
    parser.add_argument("--striped-root", type=Path, default=DEFAULT_STRIPED_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--expected-count", type=int, default=6199)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--min-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--time-embedding-dimension", type=int, default=128)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.image_size != 64:
        raise ValueError("This experiment protocol requires 64x64 images")
    if args.min_epochs > args.epochs:
        raise ValueError("--min-epochs cannot exceed --epochs")

    stage_dir = args.output_root.resolve() / "stage1"
    ensure_fresh_or_resume(stage_dir, resume=args.resume)
    stage_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp

    dataset = StripedImageDataset(
        striped_root=args.striped_root,
        image_size=args.image_size,
        expected_count=args.expected_count,
    )
    model_config = UNetConfig(
        base_channels=args.base_channels,
        time_embedding_dimension=args.time_embedding_dimension,
    )
    model = TimeConditionedUNet(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = create_grad_scaler(enabled=use_amp)
    alpha_bars = linear_beta_schedule(
        timesteps=args.timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device,
    )
    stopper = FullDataEarlyStopping(
        min_epochs=args.min_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
    )
    start_epoch = 1

    if args.resume:
        checkpoint = torch.load(stage_dir / "last.pt", map_location=device, weights_only=False)
        if checkpoint.get("stage") != 1:
            raise RuntimeError("The resume checkpoint is not a stage-1 checkpoint")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        stopper.best_loss = float(checkpoint["best_loss"])
        stopper.bad_epochs = int(checkpoint["bad_epochs"])
        start_epoch = int(checkpoint["epoch"]) + 1
        restore_rng_state(checkpoint.get("rng_state"))

    train_config = vars(args).copy()
    train_config["striped_root"] = str(args.striped_root.resolve())
    train_config["output_root"] = str(args.output_root.resolve())
    train_config["device_resolved"] = str(device)
    train_config["dataset_size"] = len(dataset)
    train_config["alpha_bar_100"] = float(alpha_bars[100].item())
    save_json(train_config, stage_dir / "config.json")

    print(f"Stage 1 dataset: {len(dataset)} striped images")
    print(f"Device: {device}; AMP: {use_amp}")
    print(f"alpha_bar[100]: {alpha_bars[100].item():.8f}")

    for epoch in range(start_epoch, args.epochs + 1):
        loader = make_loader(
            dataset,
            batch_size=args.batch_size,
            workers=args.workers,
            seed=args.seed + epoch,
        )
        model.train()
        running_loss = 0.0
        sample_count = 0
        started = time.perf_counter()

        for striped in loader:
            striped = striped.to(device, non_blocking=True)
            batch_size = striped.shape[0]
            timesteps = torch.randint(
                1, args.timesteps + 1, (batch_size,), device=device, dtype=torch.long
            )
            noisy, _ = ddpm_add_noise(striped, timesteps, alpha_bars)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                prediction = model(noisy, timesteps)
                loss = F.l1_loss(prediction, striped)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * batch_size
            sample_count += batch_size

        epoch_loss = running_loss / sample_count
        improved, should_stop = stopper.update(epoch, epoch_loss)
        elapsed = time.perf_counter() - started
        payload = checkpoint_payload(
            stage=1,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            loss=epoch_loss,
            best_loss=stopper.best_loss,
            bad_epochs=stopper.bad_epochs,
            model_config=model_config.to_dict(),
            train_config=train_config,
        )
        atomic_torch_save(payload, stage_dir / "last.pt")
        if improved:
            atomic_torch_save(payload, stage_dir / "best_train_loss.pt")

        append_csv(
            stage_dir / "history.csv",
            {
                "epoch": epoch,
                "train_l1": f"{epoch_loss:.10f}",
                "best_train_l1": f"{stopper.best_loss:.10f}",
                "bad_epochs": stopper.bad_epochs,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "seconds": f"{elapsed:.3f}",
            },
        )
        print(
            f"[stage1] epoch={epoch:03d} train_l1={epoch_loss:.8f} "
            f"best={stopper.best_loss:.8f} bad={stopper.bad_epochs} time={elapsed:.1f}s"
        )
        if should_stop:
            print(f"Stage 1 stopped by full-data loss plateau at epoch {epoch}.")
            break


if __name__ == "__main__":
    main()
