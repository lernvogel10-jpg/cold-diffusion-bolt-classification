"""Frozen-generator ablation: identical t50 recursion, differing weights only.

This entry point never trains a model or edits historical experiment artifacts.
Metrics use float32 outputs BEFORE PNG quantization and the paired watermark
target. SSIM is Gaussian (11, 1.5), population covariance, valid interior.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from diffusion_stripe.data import OriginalStripePairDataset
from diffusion_stripe.model import build_model_from_checkpoint
from diffusion_stripe.training import linear_beta_schedule, seed_everything
from evaluate_t1_all6199 import gaussian_window, tensor_to_image
from run_stage2_train4951_t50_remapped_resnet18 import remap_t50
from train_stage2_train4951 import EXPECTED_COUNTS, inspect_pair, sha256


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "experiments/stage1_vs_stage2_train4951_t50_similarity_20261016_v1"
ORIGINALS = Path("data/original")
TARGETS = Path("data/watermarked")
STAGE1 = ROOT / "experiments/full6199_two_stage/stage1/best_train_loss.pt"
STAGE2 = ROOT / "experiments/stage2_train4951_from_existing_stage1_seed43/stage2/final_generator.pt"
NOISE = ROOT / "experiments/full6199_two_stage/inference_compare_t50_seed43/noise_t50_seed43.pt"
STAGE1_CONFIG = STAGE1.parent / "config.json"
LOCKED = {
    STAGE1: "10f89fc1e909d244c1976ae084016a89c283b7744dfa1a640b1bbda66c6e4dc8",
    STAGE2: "5295975111fa2330e2941c06debc20884baa611888d204b28decb9a060becec1",
    NOISE: "60876b360eeead1c35058e741dce6cd7d7b4c37c36567b7b0a7c3b7259832254",
}
METRICS = ("mae_rgb", "mse_rgb", "rmse_rgb", "psnr_db", "ssim_rgb",
           "mae_b", "psnr_b_db", "ssim_b", "watermark_b_cosine", "watermark_b_norm_ratio")
MODELS = ("stage1_only", "stage1_plus_stage2")
FINAL = ("original_baseline", "stage1_only/remapped_t0", "stage1_plus_stage2/remapped_t0")


def save_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def ssim_channels(prediction, target, window):
    channels = prediction.shape[1]
    conv = lambda x: F.conv2d(x, window, groups=channels, padding=0)
    mx, my = conv(prediction), conv(target)
    vx, vy = conv(prediction.square()) - mx.square(), conv(target.square()) - my.square()
    covariance = conv(prediction * target) - mx * my
    score = ((2 * mx * my + .01**2) * (2 * covariance + .03**2)) / (
        (mx.square() + my.square() + .01**2) * (vx + vy + .03**2))
    return score.mean(dim=(2, 3))


def image_metrics(prediction, target, original, window):
    if not torch.isfinite(prediction).all():
        raise ValueError("Non-finite image pixels")
    if prediction.min() < 0 or prediction.max() > 1:
        raise ValueError("Evaluate clamped final images only")
    difference = prediction - target
    mse = difference.square().mean(dim=(1, 2, 3))
    b_mse = difference[:, 2].square().mean(dim=(1, 2))
    if (mse == 0).any() or (b_mse == 0).any():
        raise ValueError("Exact-match infinite PSNR requires explicit handling, not an arbitrary cap")
    ssim = ssim_channels(prediction, target, window)
    pr = (prediction[:, 2] - original[:, 2]).flatten(1)
    tr = (target[:, 2] - original[:, 2]).flatten(1)
    pn, tn = pr.norm(dim=1), tr.norm(dim=1)
    if (tn == 0).any():
        raise ValueError("A watermark target has no B residual")
    cosine = torch.where(pn > 0, (pr * tr).sum(1) / (pn * tn).clamp_min(1e-20), torch.zeros_like(pn))
    return {
        "mae_rgb": difference.abs().mean(dim=(1, 2, 3)),
        "mse_rgb": mse, "rmse_rgb": mse.sqrt(), "psnr_db": -10 * torch.log10(mse),
        "ssim_rgb": ssim.mean(1), "mae_b": difference[:, 2].abs().mean(dim=(1, 2)),
        "psnr_b_db": -10 * torch.log10(b_mse), "ssim_b": ssim[:, 2],
        "watermark_b_cosine": cosine, "watermark_b_norm_ratio": pn / tn,
    }


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate_metrics():
    # The training environment need not install scikit-image. Use an independent
    # float64 NumPy reference with the same published Gaussian SSIM definition.
    def reference_ssim(a, b):
        coords = np.arange(11, dtype=np.float64) - 5
        kernel = np.exp(-(coords**2) / (2 * 1.5**2))
        kernel /= kernel.sum()
        kernel = np.outer(kernel, kernel)
        a, b = a.astype(np.float64).transpose(2, 0, 1), b.astype(np.float64).transpose(2, 0, 1)
        def conv(value):
            patches = np.lib.stride_tricks.sliding_window_view(value, (11, 11), axis=(1, 2))
            return np.einsum("cijmn,mn->cij", patches, kernel)
        ma, mb = conv(a), conv(b)
        va, vb, cov = conv(a * a) - ma * ma, conv(b * b) - mb * mb, conv(a * b) - ma * mb
        return np.mean(((2 * ma * mb + .01**2) * (2 * cov + .03**2)) /
                       ((ma * ma + mb * mb + .01**2) * (va + vb + .03**2)))
    generator = torch.Generator().manual_seed(43)
    target = torch.rand(3, 3, 64, 64, generator=generator)
    original = (target * .9).clamp(0, 1)
    prediction = (target * .96 + .01).clamp(0, 1)
    values = image_metrics(prediction, target, original, gaussian_window(11, 1.5, 3, torch.device("cpu")))
    errors = []
    for index in range(len(target)):
        a = target[index].permute(1, 2, 0).numpy()
        b = prediction[index].permute(1, 2, 0).numpy()
        expected_psnr = -10 * np.log10(np.mean((a.astype(np.float64) - b.astype(np.float64))**2))
        expected_ssim = reference_ssim(a, b)
        errors.append({"psnr_abs_error": abs(float(values["psnr_db"][index]) - expected_psnr),
                       "ssim_abs_error": abs(float(values["ssim_rgb"][index]) - float(expected_ssim))})
    assert max(x["psnr_abs_error"] for x in errors) < 1e-4
    assert max(x["ssim_abs_error"] for x in errors) < 1e-5
    return {"reference": "independent NumPy float64 PSNR/Gaussian SSIM", "errors": errors}


def preflight(args):
    for path, expected in LOCKED.items():
        assert sha256(path) == expected, f"Changed checkpoint/noise: {path}"
    stage1, stage2 = [torch.load(p, map_location="cpu", weights_only=False) for p in (STAGE1, STAGE2)]
    assert (stage1["stage"], stage1["epoch"], stage1["train_config"]["dataset_size"]) == (1, 199, 6199)
    assert (stage2["stage"], stage2["epoch"], stage2["train_config"]["dataset_size"]) == (2, 20, 4951)
    assert stage2["train_config"]["splits"] == ["train"]
    assert Path(stage2["train_config"]["stage1_checkpoint"]).resolve() == STAGE1.resolve()
    assert stage1["model_config"] == stage2["model_config"]
    config = read_json(STAGE1_CONFIG)
    assert (config["timesteps"], config["beta_start"], config["beta_end"]) == (100, .0001, .02)
    noise = torch.load(NOISE, weights_only=True, map_location="cpu")
    assert tuple(noise.shape) == (1, 3, 64, 64) and torch.isfinite(noise).all()
    dataset = OriginalStripePairDataset(ORIGINALS, TARGETS, expected_count=6199)
    counts = {s: dict(Counter(r.key.split("/")[1] for r in dataset.records if r.key.startswith(s + "/")))
              for s in EXPECTED_COUNTS}
    assert counts == EXPECTED_COUNTS
    with ThreadPoolExecutor(max_workers=8) as executor:
        manifest = list(executor.map(inspect_pair, dataset.records))
    manifest_digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    protected = {str(p): sha256(p) for p in (STAGE1, STAGE2, NOISE, STAGE1_CONFIG)}
    original_md = ROOT / "COLD_DIFFUSION_RESNET18_EXPERIMENT_SUMMARY.md"
    protected[str(original_md)] = sha256(original_md)
    protocol = {
        "experiment": "stage1_vs_stage2_train4951_t50_similarity_v1", "count": 6199,
        "training": {"stage1_only": "all6199 watermark targets; epoch199",
                     "stage1_plus_stage2": "same Stage1 initialization; train4951 pairs only; epoch20"},
        "weights_and_noise_sha256": {str(p): h for p, h in LOCKED.items()},
        "protected_files_sha256": protected, "input_manifest_sha256": manifest_digest,
        "split_class_counts": counts, "reference": "paired target watermark image, NOT original image",
        "formula": "x50=D(I,50); for t=50,...,1: x[t-1]=x[t]-D(R(x[t],t),t)+D(R(x[t],t),t-1); out=clip(R(clip(x[0]),0))",
        "network_calls_per_image_per_model": 51, "recursion_steps": 50,
        "precision": "float32, no AMP, eval mode, no gradient; identical archived noise and schedule",
        "clipping": "None inside recursion; [0,1] before final t0 remapping and after final output",
        "noise": "single archived seed43 inference epsilon shared across images, steps, and models; not a recovered training draw",
        "metrics": {"data_range": 1.0, "rgb": "all RGB channels", "B": "blue channel only",
                    "psnr_aggregation": "arithmetic mean of per-image PSNR, not PSNR of pooled MSE",
                    "ssim": "11x11 Gaussian, sigma1.5, K1=.01, K2=.03, population covariance, valid interior then channel mean",
                    "image_representation": "clamped float32 BEFORE rounding to uint8 PNG",
                    "difference_from_legacy_ssim": "legacy evaluator used zero-padded SSIM including borders; do not mix values"},
        "metric_self_test": validate_metrics(), "batch_size": args.batch_size,
        "device": args.device, "torch_version": torch.__version__,
        "script_sha256": sha256(Path(__file__)),
        "limitation": "Stage1 saw all6199 class-encoded watermark targets; split-specific summaries are not fully generator-unseen generalization tests.",
    }
    return protocol, manifest, dataset, (stage1, stage2)


def csv_write(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(output):
    protocol = read_json(output / "protocol.json")
    chunks = [read_json(p) for p in sorted((output / "chunks").glob("*.json"))]
    rows = [r for chunk in chunks for r in chunk["rows"]]
    expected_methods = {"original_baseline"} | {f"{m}/{s}" for m in MODELS for s in ("direct_t50", "recursive_x0", "remapped_t0")}
    by_method = {m: [r for r in rows if r["method"] == m] for m in expected_methods}
    assert all(len(rs) == 6199 and len({r["key"] for r in rs}) == 6199 for rs in by_method.values())
    assert len(rows) == 6199 * 7
    manifest = read_json(output / "input_manifest.json")
    assert {r["key"] for r in manifest} == {r["key"] for r in by_method[FINAL[1]]}
    aggregates = []
    for method in sorted(expected_methods):
        for split in ("all", "train", "val", "test"):
            for cls in ("all", "defect", "normal", "nut"):
                selected = [r for r in by_method[method] if (split == "all" or r["split"] == split)
                            and (cls == "all" or r["class"] == cls)]
                row = {"method": method, "split": split, "class": cls, "n": len(selected)}
                for metric in METRICS:
                    values = np.array([r[metric] for r in selected], dtype=np.float64)
                    assert np.isfinite(values).all()
                    row[metric] = float(values.mean())
                    row[metric + "_sd"] = float(values.std(ddof=1))
                aggregates.append(row)
    paired = []
    a = {r["key"]: r for r in by_method[FINAL[1]]}
    b = {r["key"]: r for r in by_method[FINAL[2]]}
    for split in ("all", "train", "val", "test"):
        keys = [k for k in a if split == "all" or a[k]["split"] == split]
        for metric in METRICS:
            deltas = np.array([b[k][metric] - a[k][metric] for k in keys])
            greater = metric.startswith(("psnr", "ssim")) or metric == "watermark_b_cosine"
            if metric == "watermark_b_norm_ratio":
                continue
            paired.append({"split": split, "metric": metric, "n": len(keys),
                           "stage2_minus_stage1": float(deltas.mean()),
                           "stage2_better_count": int(((deltas > 0) if greater else (deltas < 0)).sum())})
    csv_write(output / "per_image_metrics.csv", rows)
    csv_write(output / "aggregate_metrics.csv", aggregates)
    csv_write(output / "paired_differences.csv", paired)
    save_json({"protocol": protocol, "aggregates": aggregates, "paired_differences": paired}, output / "results.json")
    hashes = {m: {} for m in MODELS}
    for chunk in chunks:
        for model in MODELS:
            hashes[model].update(chunk["output_sha256"][model])
    assert all(len(x) == 6199 for x in hashes.values())
    for model in MODELS:
        for key, digest in hashes[model].items():
            assert sha256(output / "images" / model / key) == digest
    for path, digest in protocol["protected_files_sha256"].items():
        assert sha256(Path(path)) == digest, f"Protected source changed: {path}"
    save_json({"count_per_model": 6199, "metric_rows": len(rows), "frozen_weights_unchanged": True,
               "all_output_hashes_verified": True, "output_sha256": hashes}, output / "complete.json")
    main = [r for r in aggregates if r["method"] in FINAL and r["split"] == "all" and r["class"] == "all"]
    print(json.dumps({"main_results_all6199": main, "output": str(output)}, ensure_ascii=False), flush=True)


def run(args):
    seed_everything(43)
    torch.set_num_threads(4)
    output = args.output_dir.resolve()
    assert output.is_relative_to((ROOT / "experiments").resolve()) and output != (ROOT / "experiments").resolve()
    protocol, manifest, dataset, checkpoints = preflight(args)
    if output.exists():
        assert (output / "protocol.json").is_file(), "Do not overwrite an unrelated directory"
        assert read_json(output / "protocol.json") == protocol, "Changed protocol: choose a fresh output directory"
    else:
        output.mkdir(parents=True)
        save_json(protocol, output / "protocol.json")
        save_json(manifest, output / "input_manifest.json")
    device = torch.device(args.device)
    models = [build_model_from_checkpoint(c).to(device).eval().requires_grad_(False) for c in checkpoints]
    before = [state_hash(m) for m in models]
    chunks_dir = output / "chunks"
    chunks_dir.mkdir(exist_ok=True)
    offset = 0
    for path in sorted(chunks_dir.glob("*.json")):
        chunk = read_json(path)
        assert chunk["start"] == offset and chunk["end"] > offset
        for model in MODELS:
            for key, digest in chunk["output_sha256"][model].items():
                assert sha256(output / "images" / model / key) == digest
        offset = chunk["end"]
    noise = torch.load(NOISE, map_location=device, weights_only=True).float()
    alpha_bars = linear_beta_schedule(100, .0001, .02, device)
    window = gaussian_window(11, 1.5, 3, device)
    loader = DataLoader(Subset(dataset, range(offset, len(dataset))), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.workers, pin_memory=True,
                        persistent_workers=args.workers > 0)
    start_offset, began = offset, time.perf_counter()
    print(f"START both frozen generators: {offset}/6199; 50 updates + one t0 remapping per model", flush=True)
    with torch.inference_mode():
        for originals, targets in loader:
            originals, targets = originals.to(device), targets.to(device)
            records = dataset.records[offset:offset + len(originals)]
            predictions = {"original_baseline": originals}
            for name, model in zip(MODELS, models, strict=True):
                direct, formula, final = remap_t50(model, originals, noise, alpha_bars)
                predictions.update({f"{name}/direct_t50": direct, f"{name}/recursive_x0": formula,
                                    f"{name}/remapped_t0": final})
            rows, hashes = [], {m: {} for m in MODELS}
            for method, prediction in predictions.items():
                measures = {k: v.cpu().tolist() for k, v in image_metrics(prediction, targets, originals, window).items()}
                for i, record in enumerate(records):
                    split, cls, _ = record.key.split("/", 2)
                    row = {"key": record.key, "split": split, "class": cls, "method": method}
                    row.update({metric: values[i] for metric, values in measures.items()})
                    rows.append(row)
            for name in MODELS:
                final = predictions[f"{name}/remapped_t0"].cpu()
                for record, value in zip(records, final, strict=True):
                    path = output / "images" / name / record.key
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tensor_to_image(value).save(path)
                    hashes[name][record.key] = sha256(path)
            end = offset + len(originals)
            save_json({"start": offset, "end": end, "rows": rows, "output_sha256": hashes},
                      chunks_dir / f"{offset:06d}.json")
            offset = end
            elapsed = time.perf_counter() - began
            remaining = elapsed * (6199 - offset) / (offset - start_offset)
            print(f"PROGRESS {offset}/6199 BOTH models; elapsed={elapsed:.0f}s ETA={remaining:.0f}s", flush=True)
    assert offset == 6199 and [state_hash(m) for m in models] == before
    # Recheck every source pair, not merely the immutable model files.
    with ThreadPoolExecutor(max_workers=8) as executor:
        after_manifest = list(executor.map(inspect_pair, dataset.records))
    assert after_manifest == manifest
    save_json({"model_state_sha256_before_after": dict(zip(MODELS, before)), "parameters_unchanged": True,
               "input_images_unchanged": True}, output / "state_audit.json")
    summarize(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stage", choices=("run", "report", "self-test"), default="run")
    args = parser.parse_args()
    if args.stage == "report":
        summarize(args.output_dir)
    elif args.stage == "self-test":
        torch.set_num_threads(4)
        print(validate_metrics())
    else:
        run(args)
