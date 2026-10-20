from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent
CLASSES = ("defect", "normal", "nut")
SPLITS = ("train", "val", "test")
# Integer approximation to 8:1:1, with all three classes in every split.
TARGET_COUNTS = ((98, 52, 148), (12, 7, 18), (12, 7, 19))
DEFAULT_CLASSIFIER = Path(
    r"D:\YOLO26_study\label_texture_experiment\striped_classification_experiment.py"
)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Frozen Stage2 t=1 remapping of 373 TTBM crops and grouped 8:1:1 ResNet18 training."
    )
    result.add_argument("--crops-root", type=Path, default=WORKSPACE / "data/TTBM-Det-crops64")
    result.add_argument(
        "--checkpoint", type=Path,
        default=WORKSPACE / "experiments/full6199_two_stage/stage2/final_generator.pt",
    )
    result.add_argument(
        "--noise-file", type=Path,
        default=WORKSPACE / "experiments/full6199_two_stage/inference_compare_t1_seed43/noise_t1_seed43.pt",
    )
    result.add_argument("--classifier-script", type=Path, default=DEFAULT_CLASSIFIER)
    result.add_argument(
        "--experiment-dir", type=Path,
        default=WORKSPACE / "experiments/ttbm373_stage2_post_formula_t1_resnet18_seed43",
    )
    result.add_argument("--seed", type=int, default=43)
    result.add_argument("--epochs", type=int, default=100)
    result.add_argument("--patience", type=int, default=15)
    result.add_argument("--batch-size", type=int, default=64)
    result.add_argument("--workers", type=int, default=0)
    result.add_argument("--device", default="cuda:0")
    result.add_argument(
        "--stage", choices=("all", "preflight", "prepare", "train", "test", "report"), default="all"
    )
    return result


def preflight(args: argparse.Namespace) -> tuple[list[dict], dict]:
    from PIL import Image

    manifest = args.crops_root / "reports/crops_manifest.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 373 or Counter(row["class_name"] for row in rows) != Counter(
        {"defect": 122, "normal": 66, "nut": 185}
    ):
        raise ValueError("Expected 373 crops: defect=122, normal=66, nut=185.")
    relative_paths = [row["crop_path"] for row in rows]
    if len(set(relative_paths)) != 373:
        raise ValueError("Duplicate crop paths in manifest.")
    actual = {
        path.relative_to(args.crops_root).as_posix()
        for split in SPLITS for path in (args.crops_root / split).rglob("*.png")
    }
    if actual != set(relative_paths):
        raise ValueError("Manifest and actual input PNG files do not match.")
    photo_hashes: dict[str, str] = {}
    for row in rows:
        path = (args.crops_root / row["crop_path"]).resolve()
        if not path.is_relative_to(args.crops_root.resolve()):
            raise ValueError(f"Crop path outside input root: {path}")
        with Image.open(path) as image:
            if image.size != (64, 64) or image.mode != "RGB":
                raise ValueError(f"Expected RGB 64x64: {path}")
            image.load()
        row["crop_sha256"] = sha256(path)
        source = Path(row["source_image"]).resolve()
        if str(source) not in photo_hashes:
            photo_hashes[str(source)] = sha256(source)
        row["photo_sha256"] = photo_hashes[str(source)]
    stage1_config = args.checkpoint.parent.parent / "stage1/config.json"
    protocol = {
        "crops_root": str(args.crops_root.resolve()),
        "manifest_sha256": sha256(manifest),
        "input_images_sha256": {row["crop_path"]: row["crop_sha256"] for row in rows},
        "source_photos_sha256": photo_hashes,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "noise_file": str(args.noise_file.resolve()),
        "noise_sha256": sha256(args.noise_file),
        "stage1_config": str(stage1_config.resolve()),
        "stage1_config_sha256": sha256(stage1_config),
        "classifier_script": str(args.classifier_script.resolve()),
        "classifier_script_sha256": sha256(args.classifier_script),
        "seed": args.seed,
        "timestep": 1,
        "inference": "Same clipped-direct t=1 formula and R(formula_x0,0) as evaluate_t1_all6199.py",
        "generator_frozen": True,
        "class_labels_provided_to_generator": False,
        "split": "Source-photo grouped, exact duplicate photo/crop grouping, approximately class-stratified 8:1:1",
        "target_class_counts": {split: dict(zip(CLASSES, counts)) for split, counts in zip(SPLITS, TARGET_COUNTS)},
        "classifier": {
            "architecture": "torchvision ResNet18 + Dropout(0.30) + Linear(512,3)",
            "pretrained": False,
            "initialize_from_old_classifier": False,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "label_smoothing": 0.05,
            "class_weights": "Computed only from the new training split",
            "augmentation": "Same RandomAffine and ColorJitter as previous classifier",
            "normalization": "mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)",
            "selection_metric": "validation macro-F1",
            "selection_min_delta": 0.0001,
        },
        "interpretation": "Frozen generator transfer to new data, followed by supervised training of a new same-architecture classifier; not frozen-classifier external evaluation",
        "noise_caveat": "Reuse the previous saved seed43 inference epsilon, not an archived per-image training noise realization (training noise was not saved).",
    }
    return rows, protocol


def assign_splits(rows: list[dict], seed: int) -> dict:
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    # Union sources if a full photo or an exact crop is duplicated.
    source_ids = sorted({row["source_id"] for row in rows})
    parent = {source: source for source in source_ids}

    def find(source: str) -> str:
        while parent[source] != source:
            parent[source] = parent[parent[source]]
            source = parent[source]
        return source

    owners: dict[tuple[str, str], str] = {}
    for row in rows:
        source = row["source_id"]
        for kind in ("photo_sha256", "crop_sha256"):
            key = (kind, row[kind])
            if key in owners:
                left, right = sorted((find(source), find(owners[key])))
                parent[right] = left
            else:
                owners[key] = source
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        row["group_id"] = find(row["source_id"])
        groups[row["group_id"]].append(row)
    group_ids = sorted(groups)
    group_counts = np.array([
        [sum(row["class_name"] == name for row in groups[group]) for name in CLASSES]
        for group in group_ids
    ])
    count = len(group_ids)
    constraints = lil_matrix((count + 9, count * 3), dtype=np.float64)
    for index in range(count):
        constraints[index, index * 3:index * 3 + 3] = 1
        for split in range(3):
            for class_index in range(3):
                constraints[count + split * 3 + class_index, index * 3 + split] = group_counts[index, class_index]
    target = np.concatenate((np.ones(count), np.array(TARGET_COUNTS).flatten()))
    result = milp(
        c=np.random.default_rng(seed).uniform(0, 1, count * 3),
        integrality=np.ones(count * 3),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(constraints.tocsc(), target, target),
        options={"time_limit": 30.0, "mip_rel_gap": 0.01},
    )
    if result.x is None:
        raise RuntimeError(f"No valid grouped 8:1:1 allocation: {result.message}")
    binary = np.rint(result.x)
    if np.max(np.abs(result.x - binary)) > 1e-5 or not np.allclose(constraints @ binary, target):
        raise RuntimeError("Split solver did not return a valid integer allocation.")
    assignments = binary.reshape(count, 3).argmax(axis=1)
    group_split = {group: SPLITS[int(assignments[index])] for index, group in enumerate(group_ids)}
    for row in rows:
        row["classification_split"] = group_split[row["group_id"]]
        row["output_key"] = f"{row['classification_split']}/{row['class_name']}/{Path(row['crop_path']).name}"
    source_sets = {split: {row["source_id"] for row in rows if row["classification_split"] == split} for split in SPLITS}
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1:]:
            if source_sets[left] & source_sets[right]:
                raise RuntimeError("Source-photo leakage across splits.")
    if len({row["output_key"] for row in rows}) != 373:
        raise RuntimeError("Duplicate output keys.")
    return {
        "total_crops": len(rows),
        "total_source_photos": len(source_ids),
        "independent_groups_after_exact_duplicate_merging": len(group_ids),
        "source_group_overlap": 0,
        "splits": {
            split: {
                "crops": sum(row["classification_split"] == split for row in rows),
                "source_photos": len(source_sets[split]),
                "class_counts": dict(Counter(row["class_name"] for row in rows if row["classification_split"] == split)),
            } for split in SPLITS
        },
        "solver": "scipy.optimize.milp/HiGHS with seeded random objective; labels/counts only, no model performance used",
        "solver_status": int(result.status),
    }


def remap_batch(model, originals, alpha_bar_1, noise):
    import torch

    ones = torch.ones(len(originals), device=originals.device, dtype=torch.long)
    zeros = torch.zeros_like(ones)
    x1 = alpha_bar_1.sqrt() * originals + (1 - alpha_bar_1).sqrt() * noise
    direct = model(x1, ones).clamp(0, 1)
    degraded_direct = alpha_bar_1.sqrt() * direct + (1 - alpha_bar_1).sqrt() * noise
    formula = (x1 - degraded_direct + direct).clamp(0, 1)
    return model(formula, zeros).clamp(0, 1)


def verify_prepared(root: Path, rows: list[dict], marker: dict) -> None:
    from PIL import Image

    expected = {row["output_key"] for row in rows}
    for domain in ("original", "post_formula_prediction"):
        dataset_root = root / "datasets" / domain
        actual = {path.relative_to(dataset_root).as_posix() for path in dataset_root.rglob("*.png")}
        if actual != expected:
            raise RuntimeError(f"Expected exactly 373 matching {domain} images.")
        for row in rows:
            path = dataset_root / row["output_key"]
            expected_hash = row["crop_sha256"] if domain == "original" else marker["output_sha256"][row["output_key"]]
            if sha256(path) != expected_hash:
                raise RuntimeError(f"Prepared file changed: {path}")
            with Image.open(path) as image:
                if image.size != (64, 64) or image.mode != "RGB":
                    raise RuntimeError(f"Bad output image: {path}")
                image.load()


def prepare(args: argparse.Namespace, rows: list[dict], protocol: dict) -> None:
    root = args.experiment_dir
    marker_path = root / "generation_complete.json"
    if marker_path.is_file():
        verify_prepared(root, rows, read_json(marker_path))
        print("已有373张再映射图通过完整哈希检查，复用数据。", flush=True)
        return
    import torch
    from PIL import Image
    from diffusion_stripe.data import load_rgb_tensor
    from diffusion_stripe.model import build_model_from_checkpoint
    from diffusion_stripe.training import linear_beta_schedule, seed_everything

    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model_from_checkpoint(checkpoint).to(device).eval()
    model.requires_grad_(False)
    config = read_json(Path(protocol["stage1_config"]))
    alpha = linear_beta_schedule(config["timesteps"], config["beta_start"], config["beta_end"], device)[1]
    noise = torch.load(args.noise_file, map_location=device, weights_only=True).float()
    if tuple(noise.shape) != (1, 3, 64, 64) or not torch.isfinite(noise).all():
        raise ValueError("Expected finite saved epsilon with shape (1,3,64,64).")
    for domain in ("original", "post_formula_prediction"):
        for split in SPLITS:
            for name in CLASSES:
                (root / "datasets" / domain / split / name).mkdir(parents=True, exist_ok=True)
    output_hashes = {}
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start:start + args.batch_size]
            originals = torch.stack([load_rgb_tensor(args.crops_root / row["crop_path"]) for row in batch_rows]).to(device)
            output = remap_batch(model, originals, alpha, noise.expand(len(batch_rows), -1, -1, -1))
            if not torch.isfinite(output).all():
                raise RuntimeError("Non-finite generated images.")
            pixels = output.cpu().mul(255).round().byte().permute(0, 2, 3, 1).numpy()
            for row, array in zip(batch_rows, pixels):
                original_path = root / "datasets/original" / row["output_key"]
                if original_path.exists():
                    if sha256(original_path) != row["crop_sha256"]:
                        raise FileExistsError(f"Refusing to overwrite changed original copy: {original_path}")
                else:
                    shutil.copy2(args.crops_root / row["crop_path"], original_path)
                image_bytes = io.BytesIO()
                Image.fromarray(array).save(image_bytes, format="PNG")
                output_path = root / "datasets/post_formula_prediction" / row["output_key"]
                payload = image_bytes.getvalue()
                if output_path.exists():
                    if output_path.read_bytes() != payload:
                        raise FileExistsError(f"Refusing to overwrite differing output: {output_path}")
                else:
                    output_path.write_bytes(payload)
                output_hashes[row["output_key"]] = hashlib.sha256(payload).hexdigest()
            print(f"Stage2再映射：{min(start + args.batch_size, len(rows))}/373", flush=True)
    if sha256(args.checkpoint) != protocol["checkpoint_sha256"]:
        raise RuntimeError("Generator checkpoint changed during inference.")
    marker = {
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "count": 373,
        "alpha_bar_1": float(alpha.item()),
        "generator_checkpoint_unchanged": True,
        "same_saved_noise_for_all_images": True,
        "labels_not_used_by_generator": True,
        "output_sha256": output_hashes,
    }
    verify_prepared(root, rows, marker)
    write_json(marker_path, marker)
    print("373张RGB 64x64再映射PNG生成完成，逐张解码和哈希检查通过。", flush=True)
    # The classifier runs in a separate process; release the generator's cache.
    del model, checkpoint, originals, output, noise
    if device.type == "cuda":
        torch.cuda.empty_cache()


def run_classifier(args: argparse.Namespace, mode: str, run_dir: Path) -> None:
    script = [sys.executable, "-X", "utf8", "-u", str(args.classifier_script)]
    if mode == "train":
        command = script + [
            "train", "--clean-data", str(args.experiment_dir / "datasets/original"),
            "--striped-data", str(args.experiment_dir / "datasets/post_formula_prediction"),
            "--runs-dir", str(run_dir.parent), "--run-tag", run_dir.name,
            "--train-domain", "striped", "--selection-domain", "striped",
            "--seed", str(args.seed), "--epochs", str(args.epochs),
            "--patience", str(args.patience), "--batch-size", str(args.batch_size),
            "--workers", str(args.workers), "--device", args.device,
            "--learning-rate", "0.001", "--weight-decay", "0.0001",
            "--label-smoothing", "0.05", "--resume",
        ]
    else:
        if not (run_dir / "training_complete.json").is_file():
            raise RuntimeError("Training must complete before test evaluation.")
        if (run_dir / "test_complete.json").is_file():
            print("已有封存test结果，跳过重复评估。", flush=True)
            return
        command = script + ["test", "--run-dir", str(run_dir), "--workers", str(args.workers), "--device", args.device]
    print(subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True)


def report(args: argparse.Namespace, run_dir: Path, split_summary: dict) -> None:
    config = read_json(run_dir / "config.json")
    with (run_dir / "history.csv").open(encoding="utf-8-sig", newline="") as handle:
        history = list(csv.DictReader(handle))
    stopped_epoch = int(history[-1]["epoch"])
    comparison_path = args.experiment_dir / "CLEAN_VS_REMAPPED.md"
    baseline_marker = (
        args.experiment_dir / "runs" / f"seed{args.seed}_resnet18_scratch_clean" / "test_complete.json"
    )
    baseline_note = (
        f"373张原图基线已补充完成，采用完全相同的分组划分和训练配置；完整原图与再映射对照见[对照报告]({comparison_path.as_posix()})。本文件主体保留再映射图结果。"
        if baseline_marker.is_file() and comparison_path.is_file()
        else "本轮仅训练再映射图分类器，没有训练373张原图基线，因此不能只凭该结果认定再映射优于新数据原图。原图同分布副本已保留，可供后续公平基线对照。"
    )
    records = []
    lines = [
        "# TTBM-Det 373张：冻结Stage2再映射 + ResNet18三分类", "",
        "## 实验设置", "",
        "- 生成器：原6199张训练得到的Stage2 final_generator.pt，冻结推理，未用新373张更新。",
        "- 输入：373张RGB 64×64目标框图；类别标签仅用于分组划分和分类器监督，不输入生成器。",
        "- 沿用之前保存的seed43高斯噪声：t=1加噪、公式更新，再以t=0输入同一个Stage2网络。",
        "- 注意：该文件是之前推理保存的固定噪声，训练没有保存逐图随机噪声，不能称为训练时同一次噪声样本。",
        "- 所有中间图截断到[0,1]的规则与原6199张批量推理完全一致；最终图四舍五入为8-bit无损PNG。",
        "", r"$$x_1=\sqrt{\bar\alpha_1}I+\sqrt{1-\bar\alpha_1}\epsilon$$", "",
        r"$$p=\operatorname{clip}(R_{\theta_2}(x_1,1),0,1)$$", "",
        r"$$z=\operatorname{clip}\left(x_1-[\sqrt{\bar\alpha_1}p+\sqrt{1-\bar\alpha_1}\epsilon]+p,0,1\right)$$", "",
        r"$$G(I)=\operatorname{clip}(R_{\theta_2}(z,0),0,1)$$", "",
        "- 同一源照片的所有目标框整体分组；完全相同的源图/目标框也合并为同组。源照片交集为0。",
        "- 固定seed43，划分只依据来源与类别数，不依据模型结果。没有声称消除了所有近重复照片。",
        "", "| 集合 | defect | normal | nut | 合计 | 源照片数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        detail = split_summary["splits"][split]
        counts = detail["class_counts"]
        lines.append(f"| {split} | {counts['defect']} | {counts['normal']} | {counts['nut']} | {detail['crops']} | {detail['source_photos']} |")
    lines += [
        "", "- 分类器：沿用原程序的ResNet18 + Dropout(0.30) + Linear(512,3)，随机初始化，不加载旧分类器权重或ImageNet权重。",
        f"- AdamW，学习率0.001，weight_decay=0.0001，batch_size={args.batch_size}，label_smoothing=0.05；训练集类别权重、原增广与归一化保持不变。",
        f"- 最多{args.epochs}轮，以验证集macro-F1选模，连续{args.patience}轮未改善早停；实际结束epoch={stopped_epoch}，最佳epoch={config.get('best_epoch')}。测试集不用于训练、选模或调整参数。",
        "", "## 分类结果", "",
        "P/R/F1报告宏平均（macro，各类等权）和加权平均（weighted，按真实类样本数加权）。ACC不依赖平均方式；单标签多分类weighted-R与ACC相等。",
        "", "| 集合 | 样本数 | ACC | macro-P | macro-R | macro-F1 | weighted-P | weighted-R | weighted-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("val", "test"):
        path = run_dir / f"{split}_metrics.json"
        if not path.is_file():
            continue
        payload = read_json(path)
        if set(payload) != {"striped"}:
            raise RuntimeError("Unexpected classifier evaluation domain.")
        metrics = payload["striped"]
        matrix = metrics["confusion_matrix"]
        if sum(map(sum, matrix)) != split_summary["splits"][split]["crops"]:
            raise RuntimeError("Evaluation sample count does not match fixed split.")
        with (run_dir / f"{split}_predictions_striped.csv").open(encoding="utf-8-sig", newline="") as handle:
            predicted_rows = list(csv.DictReader(handle))
        verified_matrix = [[0] * 3 for _ in range(3)]
        if len({row['image'] for row in predicted_rows}) != metrics['samples']:
            raise RuntimeError("Duplicate/missing predictions.")
        for row in predicted_rows:
            image = Path(row["image"]).resolve()
            dataset_root = (args.experiment_dir / "datasets/post_formula_prediction").resolve()
            key = image.relative_to(dataset_root)
            if key.parts[0] != split or key.parts[1] != CLASSES[int(row["true_id"])]:
                raise RuntimeError("Wrong evaluation split or class label.")
            verified_matrix[int(row["true_id"])][int(row["predicted_id"])]+=1
        if verified_matrix != matrix:
            raise RuntimeError("Prediction CSV and confusion matrix disagree.")
        expected_support = split_summary["splits"][split]["class_counts"]
        support = [sum(row) for row in matrix]
        if support != [expected_support[name] for name in CLASSES]:
            raise RuntimeError("Per-class evaluation counts differ from the fixed split.")
        # Independent arithmetic audit of every reported P/R/F1, using the CSV.
        predicted = [sum(row[index] for row in matrix) for index in range(3)]
        precision = [matrix[index][index] / max(predicted[index], 1) for index in range(3)]
        recall = [matrix[index][index] / max(support[index], 1) for index in range(3)]
        f1 = [2 * p * r / max(p + r, 1e-12) for p, r in zip(precision, recall)]
        audited = {"accuracy": sum(matrix[index][index] for index in range(3)) / sum(support)}
        for name, values in (("precision", precision), ("recall", recall), ("f1", f1)):
            if any(abs(left - right) > 1e-10 for left, right in zip(values, metrics[name])):
                raise RuntimeError(f"Per-class {name} audit failed.")
            audited[f"macro_{name}"] = sum(values) / 3
            audited[f"weighted_{name}"] = sum(value * count for value, count in zip(values, support)) / sum(support)
        if any(abs(value - metrics[name]) > 1e-10 for name, value in audited.items()):
            raise RuntimeError("Overall classification metric arithmetic audit failed.")
        record = {"split": split, "samples": metrics["samples"]}
        names = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_precision", "weighted_recall", "weighted_f1")
        record.update({name: metrics[name] for name in names})
        for index, name in enumerate(CLASSES):
            record[f"support_{name}"] = sum(matrix[index])
            for metric in ("precision", "recall", "f1"):
                record[f"{metric}_{name}"] = metrics[metric][index]
        records.append(record)
        values = " | ".join(f"{metrics[name] * 100:.2f}%" for name in names)
        lines.append(f"| {split} | {metrics['samples']} | {values} |")
        print(f"{split}: N={metrics['samples']}, ACC={metrics['accuracy']:.4f}, macro-P/R/F1={metrics['macro_precision']:.4f}/{metrics['macro_recall']:.4f}/{metrics['macro_f1']:.4f}, weighted-P/R/F1={metrics['weighted_precision']:.4f}/{metrics['weighted_recall']:.4f}/{metrics['weighted_f1']:.4f}", flush=True)
    lines += ["", "## 逐类别结果", "", "| 集合 | 类别 | 样本数 | P | R | F1 |", "|---|---|---:|---:|---:|---:|"]
    for record in records:
        for name in CLASSES:
            values = " | ".join(f"{record[f'{metric}_{name}'] * 100:.2f}%" for metric in ("precision", "recall", "f1"))
            lines.append(f"| {record['split']} | {name} | {record[f'support_{name}']} | {values} |")
    lines += [
        "", "## 结果解释与限制", "",
        "这是**冻结旧生成器在新数据上的迁移 + 重新训练分类器**实验。不是把旧生成器和旧ResNet18都冻结后，对373张做纯外部测试。",
        "测试集normal只识别正确1/7张（R=14.29%，F1=22.22%），明显弱于defect与nut。不能只依据总体ACC判断三类均表现良好。",
        "测试集只有38张：预测对错1张就使ACC变化约2.63个百分点；normal只有7张，召回率对错1张变化约14.29个百分点。单次划分不宜过度解读。",
        baseline_note,
        "新数据没有对应条纹真值，本实验不计算图像与条纹真值的PSNR/SSIM，分类标签不被用于生成目标条纹。",
        "", "## 运行方法", "", "```powershell",
        "cd D:\\Cold_Diffusion",
        "conda run --no-capture-output -n diffusion_env python -X utf8 -u run_ttbm373_stage2_resnet18.py",
        "```", "",
        "相同命令再次运行会核验输入/输出、复用已生成图，使用分类器断点续训并跳过已封存test。参数或输入不一致时拒绝覆盖。",
        "只重建报告：加`--stage report`；只生成图片：加`--stage prepare`。",
    ]
    write_json(args.experiment_dir / "classification_metrics.json", {row["split"]: row for row in records})
    if records:
        with (args.experiment_dir / "classification_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    (args.experiment_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parser().parse_args()
    for name in ("crops_root", "checkpoint", "noise_file", "classifier_script", "experiment_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if min(args.batch_size, args.epochs, args.patience) < 1 or args.workers < 0:
        raise ValueError("Invalid training parameters.")
    rows, protocol = preflight(args)
    root = args.experiment_dir
    protocol_path = root / "experiment_protocol.json"
    if protocol_path.is_file():
        existing = read_json(protocol_path)
        if existing != protocol:
            raise ValueError("Existing experiment protocol differs; use a new experiment directory.")
    elif root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Refusing to reuse unrecognized experiment directory: {root}")
    saved_split = root / "split_manifest.json"
    if saved_split.is_file():
        split_data = read_json(saved_split)
        mapping = {row["crop_path"]: row for row in split_data["rows"]}
        if set(mapping) != {row["crop_path"] for row in rows}:
            raise RuntimeError("Saved split and input manifest differ.")
        for row in rows:
            previous = mapping[row["crop_path"]]
            for key, value in row.items():
                if previous.get(key) != value:
                    raise RuntimeError("Saved split input metadata changed.")
            row.update(previous)
        split_summary = split_data["summary"]
    else:
        split_summary = assign_splits(rows, args.seed)
    print(json.dumps(split_summary, ensure_ascii=False, indent=2), flush=True)
    if args.stage == "preflight":
        print("预检通过；未写文件、未训练。", flush=True)
        return
    root.mkdir(parents=True, exist_ok=True)
    if not protocol_path.exists():
        write_json(protocol_path, protocol)
    if not saved_split.exists():
        write_json(saved_split, {"summary": split_summary, "rows": rows})
    run_dir = root / "runs" / f"seed{args.seed}_resnet18_scratch_stage2_post_formula_t1"
    if args.stage in ("all", "prepare"):
        prepare(args, rows, protocol)
    else:
        verify_prepared(root, rows, read_json(root / "generation_complete.json"))
    if args.stage in ("all", "train"):
        run_classifier(args, "train", run_dir)
    if args.stage in ("all", "test"):
        run_classifier(args, "test", run_dir)
    if args.stage in ("all", "train", "test", "report"):
        report(args, run_dir, split_summary)
    print(f"实验目录：{root}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("已中断；重新运行相同命令可复用完成的数据和分类器断点。", file=sys.stderr)
        raise SystemExit(130)
