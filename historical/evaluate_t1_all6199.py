from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion_stripe.data import OriginalStripePairDataset
from diffusion_stripe.model import build_model_from_checkpoint
from diffusion_stripe.training import linear_beta_schedule, resolve_device


DEFAULT_ORIGINAL_ROOT = Path("data/original")
DEFAULT_STRIPED_ROOT = Path("data/watermarked")
DEFAULT_CHECKPOINT = Path("runs/aligned/stage2/final_generator.pt")
DEFAULT_NOISE_FILE = Path("runs/generator/inference_t1/noise_t1_seed43.pt")
DEFAULT_OUTPUT_DIR = Path("outputs/inference_t1")

METHOD_NAMES = (
    "original_baseline",
    "direct_prediction",
    "formula_update_x0",
    "post_formula_prediction",
)
METRIC_NAMES = (
    "mae_rgb",
    "mse_rgb",
    "rmse_rgb",
    "psnr_db",
    "ssim_rgb",
    "mae_r",
    "mae_g",
    "mae_b",
    "stripe_b_residual_cosine",
    "stripe_b_residual_norm_ratio",
    "stripe_b_residual_scale",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate all 6199 pairs at t=1 with four comparison methods."
    )
    parser.add_argument("--original-root", type=Path, default=DEFAULT_ORIGINAL_ROOT)
    parser.add_argument("--striped-root", type=Path, default=DEFAULT_STRIPED_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--noise-file", type=Path, default=DEFAULT_NOISE_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--expected-count", type=int, default=6199)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-save-images",
        action="store_true",
        help="Compute metrics without saving the three generated image sets.",
    )
    return parser.parse_args()


def gaussian_window(
    window_size: int, sigma: float, channels: int, device: torch.device
) -> torch.Tensor:
    coordinates = (
        torch.arange(window_size, dtype=torch.float32, device=device)
        - window_size // 2
    )
    kernel_1d = torch.exp(-(coordinates**2) / (2.0 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    return torch.outer(kernel_1d, kernel_1d).expand(
        channels, 1, window_size, window_size
    ).contiguous()


def ssim_per_image(
    prediction: torch.Tensor, target: torch.Tensor, window: torch.Tensor
) -> torch.Tensor:
    channels = prediction.shape[1]
    padding = window.shape[-1] // 2
    mu_x = F.conv2d(prediction, window, padding=padding, groups=channels)
    mu_y = F.conv2d(target, window, padding=padding, groups=channels)
    mu_x_sq = mu_x.square()
    mu_y_sq = mu_y.square()
    mu_xy = mu_x * mu_y
    sigma_x = (
        F.conv2d(prediction.square(), window, padding=padding, groups=channels)
        - mu_x_sq
    )
    sigma_y = (
        F.conv2d(target.square(), window, padding=padding, groups=channels)
        - mu_y_sq
    )
    sigma_xy = (
        F.conv2d(prediction * target, window, padding=padding, groups=channels)
        - mu_xy
    )
    c1 = 0.01**2
    c2 = 0.03**2
    score = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x_sq + mu_y_sq + c1) * (sigma_x + sigma_y + c2)
    )
    return score.mean(dim=(1, 2, 3))


def batch_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    original: torch.Tensor,
    window: torch.Tensor,
) -> dict[str, torch.Tensor]:
    prediction = prediction.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    difference = prediction - target
    mse = difference.square().mean(dim=(1, 2, 3))
    mae = difference.abs().mean(dim=(1, 2, 3))
    channel_mae = difference.abs().mean(dim=(2, 3))
    predicted_residual = (prediction[:, 2] - original[:, 2]).flatten(1).double()
    target_residual = (target[:, 2] - original[:, 2]).flatten(1).double()
    predicted_energy = predicted_residual.square().sum(dim=1)
    target_energy = target_residual.square().sum(dim=1)
    denominator = predicted_energy.sqrt() * target_energy.sqrt()
    dot = (predicted_residual * target_residual).sum(dim=1)
    cosine = torch.where(
        denominator > 0,
        dot / denominator.clamp_min(torch.finfo(torch.float64).eps),
        torch.zeros_like(dot),
    )
    norm_ratio = torch.where(
        target_energy > 0,
        (predicted_energy / target_energy.clamp_min(torch.finfo(torch.float64).eps)).sqrt(),
        torch.zeros_like(target_energy),
    )
    scale = torch.where(
        target_energy > 0,
        dot / target_energy.clamp_min(torch.finfo(torch.float64).eps),
        torch.zeros_like(dot),
    )
    return {
        "mae_rgb": mae,
        "mse_rgb": mse,
        "rmse_rgb": mse.sqrt(),
        "psnr_db": 10.0 * torch.log10(1.0 / mse.clamp_min(1e-20)),
        "ssim_rgb": ssim_per_image(prediction, target, window),
        "mae_r": channel_mae[:, 0],
        "mae_g": channel_mae[:, 1],
        "mae_b": channel_mae[:, 2],
        "stripe_b_residual_cosine": cosine.float(),
        "stripe_b_residual_norm_ratio": norm_ratio.float(),
        "stripe_b_residual_scale": scale.float(),
    }


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    array = (
        tensor.detach()
        .cpu()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .byte()
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def save_generated_batch(
    output_dir: Path,
    keys: list[str],
    generated: dict[str, torch.Tensor],
) -> None:
    for batch_index, key in enumerate(keys):
        relative_png = Path(key).with_suffix(".png")
        for method_name, batch in generated.items():
            path = output_dir / "images" / method_name / relative_png
            path.parent.mkdir(parents=True, exist_ok=True)
            tensor_to_image(batch[batch_index]).save(path)


def average_rows(rows: list[dict[str, object]]) -> dict[str, float]:
    return {
        metric: sum(float(row[metric]) for row in rows) / len(rows)
        for metric in METRIC_NAMES
    }


def write_dict_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {output_dir}. "
            "Use --overwrite or choose another output directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    dataset = OriginalStripePairDataset(
        original_root=args.original_root,
        striped_root=args.striped_root,
        expected_count=args.expected_count,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model_from_checkpoint(checkpoint).to(device).eval()
    stage1_config_path = args.checkpoint.parent.parent / "stage1" / "config.json"
    stage1_config = json.loads(stage1_config_path.read_text(encoding="utf-8"))
    alpha_bars = linear_beta_schedule(
        timesteps=int(stage1_config["timesteps"]),
        beta_start=float(stage1_config["beta_start"]),
        beta_end=float(stage1_config["beta_end"]),
        device=device,
    )
    alpha_bar_1 = alpha_bars[1]
    saved_noise = torch.load(args.noise_file, map_location=device, weights_only=True)
    if tuple(saved_noise.shape) != (1, 3, 64, 64):
        raise ValueError(f"Expected noise shape (1,3,64,64), got {saved_noise.shape}")
    saved_noise = saved_noise.to(device=device, dtype=torch.float32)
    window = gaussian_window(11, 1.5, 3, device)

    rows: list[dict[str, object]] = []
    offset = 0
    progress = tqdm(loader, total=len(loader), desc="Evaluating 6199 pairs", unit="batch")
    for originals, targets in progress:
        originals = originals.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = originals.shape[0]
        records = dataset.records[offset : offset + batch_size]
        keys = [record.key for record in records]
        noise = saved_noise.expand(batch_size, -1, -1, -1)
        ones = torch.ones(batch_size, device=device, dtype=torch.long)
        zeros = torch.zeros(batch_size, device=device, dtype=torch.long)

        with torch.inference_mode():
            x1 = (
                alpha_bar_1.sqrt() * originals
                + (1.0 - alpha_bar_1).sqrt() * noise
            )
            direct = model(x1, ones).clamp(0.0, 1.0)
            degraded_direct = (
                alpha_bar_1.sqrt() * direct
                + (1.0 - alpha_bar_1).sqrt() * noise
            )
            formula = (x1 - degraded_direct + direct).clamp(0.0, 1.0)
            post_formula = model(formula, zeros).clamp(0.0, 1.0)

            predictions = {
                "original_baseline": originals,
                "direct_prediction": direct,
                "formula_update_x0": formula,
                "post_formula_prediction": post_formula,
            }
            metrics_by_method = {
                method: batch_metrics(prediction, targets, originals, window)
                for method, prediction in predictions.items()
            }

        cpu_metrics = {
            method: {
                metric: values.detach().cpu().tolist()
                for metric, values in metrics.items()
            }
            for method, metrics in metrics_by_method.items()
        }
        for index, key in enumerate(keys):
            split, class_name, _filename = key.split("/", maxsplit=2)
            for method in METHOD_NAMES:
                row: dict[str, object] = {
                    "key": key,
                    "split": split,
                    "class": class_name,
                    "method": method,
                }
                for metric in METRIC_NAMES:
                    row[metric] = cpu_metrics[method][metric][index]
                rows.append(row)

        if not args.no_save_images:
            save_generated_batch(
                output_dir,
                keys,
                {
                    "direct_prediction": direct,
                    "formula_update_x0": formula,
                    "post_formula_prediction": post_formula,
                },
            )
        offset += batch_size
        progress.set_postfix(images=f"{offset}/{len(dataset)}")

    if offset != len(dataset):
        raise RuntimeError(f"Processed {offset} images, expected {len(dataset)}")

    write_dict_rows(output_dir / "per_image_metrics.csv", rows)
    grouped_overall: dict[str, list[dict[str, object]]] = defaultdict(list)
    grouped_class: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    grouped_split_class: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        method = str(row["method"])
        class_name = str(row["class"])
        split = str(row["split"])
        grouped_overall[method].append(row)
        grouped_class[(class_name, method)].append(row)
        grouped_split_class[(split, class_name, method)].append(row)

    overall_rows = [
        {"method": method, "count": len(group), **average_rows(group)}
        for method, group in sorted(grouped_overall.items())
    ]
    class_rows = [
        {
            "class": class_name,
            "method": method,
            "count": len(group),
            **average_rows(group),
        }
        for (class_name, method), group in sorted(grouped_class.items())
    ]
    split_class_rows = [
        {
            "split": split,
            "class": class_name,
            "method": method,
            "count": len(group),
            **average_rows(group),
        }
        for (split, class_name, method), group in sorted(grouped_split_class.items())
    ]
    write_dict_rows(output_dir / "aggregate_overall.csv", overall_rows)
    write_dict_rows(output_dir / "aggregate_by_class.csv", class_rows)
    write_dict_rows(output_dir / "aggregate_by_split_class.csv", split_class_rows)

    summary = {
        "count": len(dataset),
        "timestep": 1,
        "alpha_bar_1": float(alpha_bar_1.item()),
        "checkpoint": str(args.checkpoint.resolve()),
        "noise_file": str(args.noise_file.resolve()),
        "same_noise_tensor_for_all_images": True,
        "images_saved": not args.no_save_images,
        "output_image_sets": [
            "direct_prediction",
            "formula_update_x0",
            "post_formula_prediction",
        ],
        "aggregate_overall": overall_rows,
        "aggregate_by_class": class_rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Processed pairs: {len(dataset)}")
    print(f"Per-image metric rows: {len(rows)}")
    for row in overall_rows:
        print(
            f"{row['method']:>25}: MAE={row['mae_rgb']:.8f} "
            f"PSNR={row['psnr_db']:.4f}dB SSIM={row['ssim_rgb']:.6f} "
            f"B-MAE={row['mae_b']:.8f} cosine={row['stripe_b_residual_cosine']:.6f} "
            f"norm={row['stripe_b_residual_norm_ratio']:.6f} "
            f"scale={row['stripe_b_residual_scale']:.6f}"
        )
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
