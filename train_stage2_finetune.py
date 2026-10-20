from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from diffusion_stripe.data import OriginalStripePairDataset, SPLITS
from diffusion_stripe.model import build_model_from_checkpoint
from diffusion_stripe.training import (
    FullDataEarlyStopping,
    append_csv,
    atomic_torch_save,
    checkpoint_payload,
    create_grad_scaler,
    ensure_fresh_or_resume,
    make_loader,
    resolve_device,
    restore_rng_state,
    save_json,
    seed_everything,
)


DEFAULT_ORIGINAL_ROOT = Path(
    r"D:\YOLO26_study\datasets\yolo26_crops64_dataset_source_grouped_split43"
)
DEFAULT_STRIPED_ROOT = Path(
    r"D:\YOLO26_study\datasets\yolo26_crops64_striped_source_grouped_split43_w1_low0p9"
)
DEFAULT_OUTPUT_ROOT = Path(r"D:\Cold_Diffusion\experiments\full6199_two_stage")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2: full-data original-to-stripe fine-tuning with fixed t=0."
    )
    parser.add_argument("--original-root", type=Path, default=DEFAULT_ORIGINAL_ROOT)
    parser.add_argument("--striped-root", type=Path, default=DEFAULT_STRIPED_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stage1-checkpoint", type=Path, default=None)
    parser.add_argument("--expected-count", type=int, default=6199)
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-early-stop", action="store_true", help="Train exactly --epochs, retaining the existing best-loss checkpoint rule.")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.image_size != 64:
        raise ValueError("This experiment protocol requires 64x64 images")
    if not 10 <= args.min_epochs <= args.epochs <= 20:
        raise ValueError("Stage 2 requires 10 <= min_epochs <= epochs <= 20")

    output_root = args.output_root.resolve()
    stage_dir = output_root / "stage2"

    source_checkpoint = (
        args.stage1_checkpoint.resolve()
        if args.stage1_checkpoint is not None
        else output_root / "stage1" / "best_train_loss.pt"
    )
    if not source_checkpoint.is_file():
        raise FileNotFoundError(f"Stage-1 checkpoint not found: {source_checkpoint}")

    dataset = OriginalStripePairDataset(
        original_root=args.original_root,
        striped_root=args.striped_root,
        image_size=args.image_size,
        expected_count=args.expected_count,
        splits=args.splits,
    )
    if args.preflight_only:
        print(f"Stage 2 preflight: {len(dataset)} pairs; splits={args.splits}; fixed t=0")
        print(f"Stage 1 checkpoint: {source_checkpoint}")
        print(f"Output (not created): {stage_dir}")
        return
    ensure_fresh_or_resume(stage_dir, resume=args.resume)
    stage_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    stage1_checkpoint = torch.load(source_checkpoint, map_location=device, weights_only=False)
    if stage1_checkpoint.get("stage") != 1:
        raise RuntimeError("--stage1-checkpoint is not a stage-1 checkpoint")
    model = build_model_from_checkpoint(stage1_checkpoint).to(device)
    model_config = stage1_checkpoint["model_config"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = create_grad_scaler(enabled=use_amp)
    stopper = FullDataEarlyStopping(
        min_epochs=args.min_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
    )
    start_epoch = 1

    if args.resume:
        checkpoint = torch.load(stage_dir / "last.pt", map_location="cpu", weights_only=False)
        if checkpoint.get("stage") != 2:
            raise RuntimeError("The resume checkpoint is not a stage-2 checkpoint")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        stopper.best_loss = float(checkpoint["best_loss"])
        stopper.bad_epochs = int(checkpoint["bad_epochs"])
        start_epoch = int(checkpoint["epoch"]) + 1
        restore_rng_state(checkpoint.get("rng_state"))

    train_config = vars(args).copy()
    train_config["original_root"] = str(args.original_root.resolve())
    train_config["striped_root"] = str(args.striped_root.resolve())
    train_config["output_root"] = str(output_root)
    train_config["stage1_checkpoint"] = str(source_checkpoint)
    train_config["device_resolved"] = str(device)
    train_config["dataset_size"] = len(dataset)
    train_config["fixed_timestep"] = 0
    save_json(train_config, stage_dir / "config.json")

    print(f"Stage 2 dataset: {len(dataset)} original/stripe pairs")
    print(f"Loaded stage-1 weights: {source_checkpoint}")
    print(f"Device: {device}; AMP: {use_amp}; fixed timestep: 0")

    for epoch in range(start_epoch, args.epochs + 1):
        loader = make_loader(
            dataset,
            batch_size=args.batch_size,
            workers=args.workers,
            seed=args.seed + 10_000 + epoch,
        )
        model.train()
        running_loss = 0.0
        sample_count = 0
        started = time.perf_counter()

        for original, striped in loader:
            original = original.to(device, non_blocking=True)
            striped = striped.to(device, non_blocking=True)
            batch_size = original.shape[0]
            timesteps = torch.zeros(batch_size, device=device, dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                prediction = model(original, timesteps)
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
            stage=2,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            loss=epoch_loss,
            best_loss=stopper.best_loss,
            bad_epochs=stopper.bad_epochs,
            model_config=model_config,
            train_config=train_config,
            source_checkpoint=str(source_checkpoint),
        )
        atomic_torch_save(payload, stage_dir / "last.pt")
        if improved:
            atomic_torch_save(payload, stage_dir / "best_train_loss.pt")
            atomic_torch_save(payload, stage_dir / "final_generator.pt")

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
            f"[stage2] epoch={epoch:02d} train_l1={epoch_loss:.8f} "
            f"best={stopper.best_loss:.8f} bad={stopper.bad_epochs} time={elapsed:.1f}s"
        )
        if should_stop and not args.no_early_stop:
            print(f"Stage 2 stopped by full-data loss plateau at epoch {epoch}.")
            break

    save_json(
        {
            "final_generator": str((stage_dir / "final_generator.pt").resolve()),
            "selection_basis": f"mean RGB L1 training loss on {len(dataset)} pairs; min_delta={args.min_delta:g}",
            "fixed_timestep": 0,
            "dataset_size": len(dataset),
            "training_splits": args.splits,
            "epochs_requested": args.epochs,
            "early_stopping_enabled": not args.no_early_stop,
        },
        stage_dir / "final_generator.json",
    )


if __name__ == "__main__":
    main()
