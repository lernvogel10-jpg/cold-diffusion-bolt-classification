"""Frozen new Stage2 t=1 remapping, then original split43 ResNet18 evaluation."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

import run_resnet18_full6199_external373 as common
from diffusion_stripe.data import StripedImageDataset, collect_images
from diffusion_stripe.model import build_model_from_checkpoint
from diffusion_stripe.training import linear_beta_schedule, save_json
from evaluate_t1_all6199 import tensor_to_image
from resnet18_full6199_loss_earlystop import seed_worker
from train_stage2_train4951 import EXPECTED_COUNTS, WORKSPACE, read_json, sha256


NEW_STAGE2 = WORKSPACE / "experiments/stage2_train4951_from_existing_stage1_seed43"
DEFAULT_ROOT = WORKSPACE / "experiments/stage2_train4951_t1_remapped_resnet18_seed43"
NOISE = WORKSPACE / "experiments/full6199_two_stage/inference_compare_t1_seed43/noise_t1_seed43.pt"
CLASSES, SPLITS = common.CLASSES, ("train", "val", "test")


@dataclass
class Selection:
    monitor: str = "val_loss"
    patience: int = 15
    min_delta: float = 1e-4
    reference: float | None = None
    selected: float | None = None
    epoch: int = 0
    waiting: int = 0

    def update(self, epoch: int, value: float) -> tuple[bool, bool]:
        if not math.isfinite(value):
            raise ValueError("Non-finite validation selection metric.")
        sign = 1 if self.monitor == "val_loss" else -1
        significant = self.reference is None or sign * value < sign * self.reference - self.min_delta
        absolute = self.selected is None or sign * value < sign * self.selected
        # Old split43 macro-F1 saves only significant improvements; loss saves its absolute minimum.
        save_best = absolute if self.monitor == "val_loss" else significant
        if save_best:
            self.selected, self.epoch = value, epoch
        if significant:
            self.reference, self.waiting = value, 0
        else:
            self.waiting += 1
        return save_best, self.waiting >= self.patience


def remap(model, original: torch.Tensor, noise: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(len(original), device=original.device, dtype=torch.long)
    zeros = torch.zeros_like(ones)
    epsilon = noise.expand(len(original), -1, -1, -1)
    x1 = alpha.sqrt() * original + (1 - alpha).sqrt() * epsilon
    direct = model(x1, ones).clamp(0, 1)
    degraded = alpha.sqrt() * direct + (1 - alpha).sqrt() * epsilon
    formula = (x1 - degraded + direct).clamp(0, 1)
    return model(formula, zeros).clamp(0, 1)


def inspect_original(record) -> dict:
    with Image.open(record.path) as image:
        if image.mode != "RGB" or image.size != (64, 64):
            raise ValueError(f"Expected RGB64x64: {record.path}")
        image.load()
    return {"key": record.key, "path": str(record.path), "sha256": sha256(record.path)}


def preflight(args) -> tuple[dict, list[dict]]:
    checkpoint_path = NEW_STAGE2 / "stage2/final_generator.pt"
    complete = read_json(NEW_STAGE2 / "training_complete.json")
    source_protocol = read_json(NEW_STAGE2 / "experiment_protocol.json")
    if sha256(checkpoint_path) != complete["final_generator_sha256"]:
        raise ValueError("The new train4951 Stage2 generator changed.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["train_config"]
    if checkpoint["stage"] != 2 or config["dataset_size"] != 4951 or config["splits"] != ["train"] or config["fixed_timestep"] != 0:
        raise ValueError("Expected the new train4951 Stage2, not the old full6199 model.")
    original_root = Path(config["original_root"]).resolve()
    records = collect_images(original_root, expected_count=6199)
    counts = {split: dict(Counter(row.key.split("/")[1] for row in records if row.key.startswith(split + "/"))) for split in SPLITS}
    if counts != EXPECTED_COUNTS:
        raise ValueError("The original 4951/619/629 class counts changed.")
    with ThreadPoolExecutor(max_workers=8) as executor:
        manifest = list(executor.map(inspect_original, records))
    training = read_json(NEW_STAGE2 / "train_pair_manifest.json")
    training_sha = {row["key"]: row["original_sha256"] for row in training["pairs"]}
    if {row["key"]: row["sha256"] for row in manifest if row["key"].startswith("train/")} != training_sha:
        raise ValueError("Training originals differ from those used by new Stage2.")
    source = Path(checkpoint["source_checkpoint"]).resolve()
    stage1_config = source.parent / "config.json"
    if sha256(source) != source_protocol["source_stage1_sha256"]:
        raise ValueError("The originating old Stage1 changed.")
    split_manifest = original_root / "reports/crops_manifest.csv"
    if sha256(split_manifest) != source_protocol["source_group_manifest_sha256"]:
        raise ValueError("The audited source-group partition changed.")
    data = read_json(stage1_config)
    if data["timesteps"] != 100 or data["beta_start"] != 0.0001 or data["beta_end"] != 0.02:
        raise ValueError("The old DDPM noise schedule changed.")
    protocol = {
        "protocol_version": "new_stage2_train4951_t1_post_formula_v1", "count": 6199,
        "original_root": str(original_root), "split_class_counts": counts,
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path),
        "generator_training_count": 4951, "generator_epoch": checkpoint["epoch"],
        "stage1_config": str(stage1_config), "stage1_config_sha256": sha256(stage1_config),
        "stage1_training_count": 6199, "noise_file": str(NOISE), "noise_sha256": sha256(NOISE),
        "timestep": 1, "beta_start": data["beta_start"], "beta_end": data["beta_end"],
        "precision": "float32, no AMP, identical to previous inference",
        "formula": "clip(R(clip(x1-D(clip(R(x1,1)),1)+clip(R(x1,1))),0))",
        "same_saved_noise_tensor_for_all_images": True, "class_labels_input_to_generator": False,
        "original_split_preserved": True, "source_group_overlap_across_splits": 0,
        "input_manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        "noise_caveat": "Fixed archived inference seed43 epsilon; not an archived historical per-image training epsilon.",
        "limitation": "Classifier and Stage2 use train4951 only, but old Stage1 saw all6199 stripes; not full-pipeline generator-unseen val/test.",
    }
    print("预检：新Stage2=train4951/20epoch；原图4951/619/629，分组及类别数保持不变。", flush=True)
    return protocol, manifest


def verify_generated(args, manifest: list[dict], marker: dict) -> None:
    root = args.output_dir / "datasets/post_formula_prediction"
    generated = collect_images(root, expected_count=6199)
    if [row.key for row in generated] != [row["key"] for row in manifest]:
        raise ValueError("Generated image split/class/filename keys differ.")
    with ThreadPoolExecutor(max_workers=8) as executor:
        hashes = list(executor.map(sha256, [row.path for row in generated]))
    if dict(zip((row.key for row in generated), hashes)) != marker["output_sha256"]:
        raise ValueError("Generated images changed after their completion seal.")


def generate(args, protocol: dict, manifest: list[dict]) -> None:
    marker_path = args.output_dir / "generation_complete.json"
    if marker_path.exists():
        marker = read_json(marker_path)
        if marker["checkpoint_sha256"] != protocol["checkpoint_sha256"]:
            raise ValueError("Existing remaps used different generator weights.")
        verify_generated(args, manifest, marker)
        print("已核验现有6199张新权重再映射图，跳过重复生成。", flush=True)
        return
    device = torch.device(args.device)
    dataset = StripedImageDataset(protocol["original_root"], expected_count=6199)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    checkpoint = torch.load(protocol["checkpoint"], map_location="cpu", weights_only=False)
    model = build_model_from_checkpoint(checkpoint).to(device).eval().requires_grad_(False)
    before = common.state_sha256(model)
    alpha = linear_beta_schedule(100, protocol["beta_start"], protocol["beta_end"], device)[1]
    noise = torch.load(protocol["noise_file"], map_location=device, weights_only=True).float()
    if tuple(noise.shape) != (1, 3, 64, 64):
        raise ValueError("Expected the saved 1x3x64x64 epsilon tensor.")
    root = args.output_dir / "datasets/post_formula_prediction"
    offset, hashes = 0, {}
    with torch.inference_mode():
        for batch in loader:
            result = remap(model, batch.to(device, non_blocking=True), noise, alpha)
            current = dataset.records[offset:offset + len(batch)]
            for record, tensor in zip(current, result, strict=True):
                path = root / record.key
                path.parent.mkdir(parents=True, exist_ok=True)
                tensor_to_image(tensor).save(path)
                hashes[record.key] = sha256(path)
            offset += len(batch)
            if offset % (args.batch_size * 10) == 0 or offset == 6199:
                print(f"新Stage2再映射：{offset}/6199", flush=True)
    if offset != 6199 or common.state_sha256(model) != before or sha256(Path(protocol["checkpoint"])) != protocol["checkpoint_sha256"]:
        raise ValueError("Generation count or frozen generator identity audit failed.")
    marker = {"count": offset, "checkpoint_sha256": protocol["checkpoint_sha256"],
              "alpha_bar_1": float(alpha), "output_sha256": hashes, "generator_state_unchanged": True}
    save_json(marker, marker_path)
    verify_generated(args, manifest, marker)
    del model, checkpoint, dataset, loader, result
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def classifier_config(args, protocol: dict, model, classifier, *, data_root: Path | None = None) -> dict:
    reference = read_json(common.REFERENCE_RUNS / common.REFERENCE_TAGS["remapped"] / "config.json")
    config = {name: reference[name] for name in ("pretrained", "seed", "learning_rate", "weight_decay",
              "label_smoothing", "class_weights_enabled", "augmentation_enabled", "amp")}
    config.update({"model": "resnet18", "parameters": sum(p.numel() for p in model.parameters()),
                   "batch_size": args.batch_size, "workers": args.workers, "device": args.device,
                   "max_epochs": args.epochs, "patience": args.patience, "min_delta": args.min_delta,
                   "selection": args.selection, "train_samples": 4951, "val_samples": 619,
                   "test_samples": 629, "initial_model_sha256": common.state_sha256(model),
                   "dataset": str(data_root if data_root is not None else args.output_dir / "datasets/post_formula_prediction"),
                   "generator_sha256": protocol["checkpoint_sha256"] if data_root is None else None,
                   "classifier_script_sha256": sha256(common.DEFAULT_CLASSIFIER_SCRIPT),
                   "initialize_from_old_classifier": False, "test_used_for_selection": False})
    if data_root is not None:
        config["input_method"] = "original_baseline"
    return config


def dataset_for(classifier, data_root: Path, split: str, training: bool, config: dict):
    dataset = classifier.image_folder(data_root, split, training, training and config["augmentation_enabled"], config["pretrained"])
    if dataset.classes != list(CLASSES) or dict(zip(CLASSES, np.bincount(dataset.targets, minlength=3).tolist())) != EXPECTED_COUNTS[split]:
        raise ValueError("Classifier dataset labels/counts do not match the original split.")
    return dataset


def train_classifier(args, protocol: dict, classifier, *, data_root: Path | None = None) -> None:
    run = args.run_dir
    if (run / "training_complete.json").exists():
        complete = read_json(run / "training_complete.json")
        if sha256(run / "best.pt") != complete["best_sha256"]:
            raise ValueError("Completed classifier weights changed.")
        print("分类器训练已完成，复用冻结最佳权重。", flush=True)
        return
    run.mkdir(parents=True, exist_ok=True)
    classifier.set_seed(43)
    device = classifier.resolve_device(args.device)
    model = classifier.build_model(False).to(device)
    config = classifier_config(args, protocol, model, classifier, data_root=data_root)
    if (run / "config.json").exists() and read_json(run / "config.json") != config:
        raise ValueError("Classifier configuration changed; choose a new output directory.")
    save_json(config, run / "config.json")
    data_root = Path(config["dataset"])
    training = dataset_for(classifier, data_root, "train", True, config)
    validation = dataset_for(classifier, data_root, "val", False, config)
    train_loader = classifier.make_loader(training, args.batch_size, args.workers, True, 43, device.type == "cuda")
    val_loader = classifier.make_loader(validation, args.batch_size, args.workers, False, 43, device.type == "cuda")
    weights, counts = classifier.class_weights(training.targets, device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=config["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler_kwargs = {"mode": "min" if args.selection == "val_loss" else "max", "factor": 0.5, "patience": 4, "min_lr": 1e-6}
    if args.selection == "val_loss":
        scheduler_kwargs.update(threshold=args.min_delta, threshold_mode="abs")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_kwargs)
    amp = config["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    selection = Selection(args.selection, args.patience, args.min_delta)
    history, start = [], 1
    if (run / "last.pt").exists():
        last = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
        if last["config"] != config:
            raise ValueError("Classifier resume checkpoint configuration differs.")
        model.load_state_dict(last["model_state_dict"])
        optimizer.load_state_dict(last["optimizer_state_dict"])
        scheduler.load_state_dict(last["scheduler_state_dict"])
        scaler.load_state_dict(last["scaler_state_dict"])
        selection = Selection(**last["selection"])
        history, start = last["history"], last["epoch"] + 1
        random.setstate(last["python_rng"])
        np.random.set_state(last["numpy_rng"])
        torch.set_rng_state(last["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(last["cuda_rng"])
        train_loader.generator.set_state(last["loader_rng"])
        del last
    print(f"ResNet18：train4951、val619；最多{args.epochs}轮，{args.selection}早停，patience={args.patience}。", flush=True)
    for epoch in range(start, args.epochs + 1):
        if selection.waiting >= args.patience:
            break
        began, lr = time.perf_counter(), optimizer.param_groups[0]["lr"]
        train_loss, train_accuracy = classifier.train_epoch(model, train_loader, criterion, optimizer, scaler, device, amp)
        val_loss, val, _, _ = classifier.evaluate(model, val_loader, torch.nn.CrossEntropyLoss(), device, amp, False)
        score = val_loss if args.selection == "val_loss" else val["macro_f1"]
        improved, stop = selection.update(epoch, score)
        scheduler.step(score)
        history.append({"epoch": epoch, "train_samples": 4951, "val_samples": 619,
                        "train_loss": train_loss, "train_accuracy": train_accuracy, "val_loss": val_loss,
                        **{f"val_{key}": val[key] for key in common.METRIC_NAMES},
                        "learning_rate": lr, "next_learning_rate": optimizer.param_groups[0]["lr"],
                        "best_epoch": selection.epoch, "best_score": selection.selected, "bad_epochs": selection.waiting,
                        "seconds": time.perf_counter() - began})
        if improved:
            common.atomic_torch_save({"model_state_dict": model.state_dict(), "config": config,
                "epoch": epoch, "score": score, "classes": list(CLASSES)}, run / "best.pt")
        common.atomic_torch_save({"model_state_dict": model.state_dict(), "config": config, "epoch": epoch,
            "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(), "selection": asdict(selection), "history": history,
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "loader_rng": train_loader.generator.get_state()}, run / "last.pt")
        common.write_csv(run / "history.csv", history)
        print(f"epoch {epoch:03d}/{args.epochs}: train-L={train_loss:.5f}, val-L={val_loss:.5f}, "
              f"val-ACC={val['accuracy']:.4f}, val-macro-F1={val['macro_f1']:.4f}, best@{selection.epoch}, "
              f"等待={selection.waiting}/{args.patience}", flush=True)
        if stop:
            break
    if not history:
        raise ValueError("No completed classifier epochs.")
    best = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    if best["epoch"] != selection.epoch or best["score"] != selection.selected:
        raise ValueError("Classifier selected checkpoint and validation history disagree.")
    save_json({"epochs_completed": history[-1]["epoch"], "selected_epoch": selection.epoch,
               "selected_score": selection.selected, "selection": asdict(selection),
               "stop_reason": "validation_patience" if selection.waiting >= args.patience else "max_epochs",
               "best_sha256": sha256(run / "best.pt"), "history_sha256": sha256(run / "history.csv"),
               "test_used_for_training_or_selection": False}, run / "training_complete.json")
    records = [classifier.EpochRecord(epoch=r["epoch"], train_loss=r["train_loss"], train_accuracy=r["train_accuracy"],
                selection_val_loss=r["val_loss"], selection_val_accuracy=r["val_accuracy"],
                selection_val_macro_precision=r["val_macro_precision"], selection_val_macro_recall=r["val_macro_recall"],
                selection_val_macro_f1=r["val_macro_f1"], learning_rate=r["learning_rate"], elapsed_seconds=r["seconds"]) for r in history]
    classifier.plot_history(records, run / "training_curves.png")
    del model, best, optimizer, scheduler, scaler, train_loader, val_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def evaluate_split(args, split: str, classifier) -> None:
    run = args.run_dir
    complete = read_json(run / "training_complete.json")
    if sha256(run / "best.pt") != complete["best_sha256"]:
        raise ValueError("Frozen classifier weights changed.")
    marker_path = run / f"{split}_complete.json"
    if marker_path.exists():
        marker = read_json(marker_path)
        if marker["best_sha256"] != complete["best_sha256"] or sha256(run / f"{split}_metrics.json") != marker["metrics_sha256"] or sha256(run / f"{split}_predictions.csv") != marker["predictions_sha256"]:
            raise ValueError("Sealed classifier evaluation changed.")
        return
    config = read_json(run / "config.json")
    classifier.set_seed(config["seed"])
    device = classifier.resolve_device(args.device)
    model = classifier.build_model(config["pretrained"]).to(device).eval()
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.requires_grad_(False)
    before = common.state_sha256(model)
    dataset = dataset_for(classifier, Path(config["dataset"]), split, False, config)
    loader = classifier.make_loader(dataset, args.batch_size, args.workers, False, config["seed"], device.type == "cuda")
    loss, metrics, probabilities, _ = classifier.evaluate(model, loader, torch.nn.CrossEntropyLoss(), device, config["amp"] and device.type == "cuda", True)
    if common.state_sha256(model) != before:
        raise ValueError("Classifier model state changed during frozen evaluation.")
    save_json({"loss": loss, **metrics}, run / f"{split}_metrics.json")
    classifier.save_predictions(run / f"{split}_predictions.csv", dataset, probabilities)
    label = config.get("input_method", "Stage2 train4951 remaps")
    classifier.plot_confusion(metrics["confusion_matrix"], run / f"{split}_confusion.png", f"{label}: {split}")
    save_json({"samples": len(dataset), "selected_epoch": complete["selected_epoch"],
               "best_sha256": complete["best_sha256"], "frozen_evaluation": True,
               "metrics_sha256": sha256(run / f"{split}_metrics.json"),
               "predictions_sha256": sha256(run / f"{split}_predictions.csv")}, marker_path)
    print(f"{split}：ACC={metrics['accuracy']:.4f}, weighted-P/R/F1={metrics['weighted_precision']:.4f}/"
          f"{metrics['weighted_recall']:.4f}/{metrics['weighted_f1']:.4f}", flush=True)
    del model, checkpoint, loader, dataset
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def audited_metrics(args, split: str, classifier) -> dict:
    marker = read_json(args.run_dir / f"{split}_complete.json")
    metrics = read_json(args.run_dir / f"{split}_metrics.json")
    if sha256(args.run_dir / f"{split}_metrics.json") != marker["metrics_sha256"] or sha256(args.run_dir / f"{split}_predictions.csv") != marker["predictions_sha256"]:
        raise ValueError("Metrics/prediction seal changed.")
    with (args.run_dir / f"{split}_predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
        predictions = list(csv.DictReader(handle))
    data_root = Path(read_json(args.run_dir / "config.json")["dataset"])
    expected = {str(p.resolve()): i for i, name in enumerate(CLASSES) for p in (data_root / split / name).glob("*.png")}
    seen, matrix = set(), torch.zeros((3, 3), dtype=torch.int64)
    for row in predictions:
        path = str(Path(row["image"]).resolve())
        target, prediction = int(row["true_id"]), int(row["predicted_id"])
        if path in seen or expected.get(path) != target or not 0 <= prediction < 3:
            raise ValueError("Duplicate, missing or mislabelled classifier prediction.")
        seen.add(path)
        matrix[target, prediction] += 1
    if seen != set(expected) or len(predictions) != sum(EXPECTED_COUNTS[split].values()):
        raise ValueError("Classifier evaluated the wrong split or count.")
    calculated = classifier.metrics_from_confusion(matrix)
    if calculated["confusion_matrix"] != metrics["confusion_matrix"] or matrix.sum(1).tolist() != [EXPECTED_COUNTS[split][name] for name in CLASSES]:
        raise ValueError("Confusion matrix/class supports differ from paired predictions.")
    if any(abs(calculated[name] - metrics[name]) > 1e-10 for name in common.METRIC_NAMES):
        raise ValueError("Classifier metric calculation audit failed.")
    return metrics


def report(args, protocol: dict, classifier) -> None:
    complete = read_json(args.run_dir / "training_complete.json")
    rows, details = [], {}
    for split in ("val", "test"):
        metrics = audited_metrics(args, split, classifier)
        details[split] = metrics
        row = {"split": split, "samples": metrics["samples"], **{name: metrics[name] for name in common.METRIC_NAMES}}
        for i, name in enumerate(CLASSES):
            row[f"support_{name}"] = EXPECTED_COUNTS[split][name]
            row.update({f"{metric}_{name}": metrics[metric][i] for metric in ("precision", "recall", "f1")})
        rows.append(row)
    common.write_csv(args.output_dir / f"classification_{args.selection}_metrics.csv", rows)
    save_json({"metrics": rows, "full_metrics": details, "training": complete}, args.output_dir / f"classification_{args.selection}_metrics.json")
    monitor = "验证交叉熵损失（最小化）" if args.selection == "val_loss" else "验证macro-F1（最大化，与旧811协议一致）"
    lines = ["# 新Stage2(train4951)再映射图：原split43 ResNet18分类", "",
             "## 数据与生成流程", "",
             "使用新Stage2（旧Stage1初始化，仅4951对train微调20轮，t=0），冻结权重后将全部6199张原图生成再映射图。保持原有同源组划分：train4951、val619、test629；不重新划分。生成器只接收原图及时间步，不接收类别编号或目标条纹图。", "",
             "流程与旧推理一致：", "",
             r"$$x_1=\sqrt{\bar\alpha_1}I+\sqrt{1-\bar\alpha_1}\epsilon,\quad d=\operatorname{clip}(R_\theta(x_1,1),0,1).$$", "",
             r"$$x_0^{\rm formula}=\operatorname{clip}\left[x_1-\left(\sqrt{\bar\alpha_1}d+\sqrt{1-\bar\alpha_1}\epsilon\right)+d,0,1\right],\quad G(I)=\operatorname{clip}\left[R_\theta(x_0^{\rm formula},0),0,1\right].$$", "",
             "同一保存的seed43噪声张量用于所有图片，beta1=0.0001，alpha_bar1≈0.9998999834，float32推理；clip位置及PNG四舍五入量化与旧程序一致。噪声是保存的固定推理epsilon，不是归档的训练阶段逐样本噪声（历史训练噪声未保存）。", "",
             "| 集合 | defect | normal | nut | 总计 |", "|---|---:|---:|---:|---:|"]
    for split in SPLITS:
        lines.append(f"| {split} | " + " | ".join(str(EXPECTED_COUNTS[split][name]) for name in CLASSES) + f" | {sum(EXPECTED_COUNTS[split].values())} |")
    lines += ["", "## ResNet18训练与选模", "",
              "ResNet18 + Dropout(0.30) + Linear(512,3)，随机初始化（不加载任何旧分类器），seed43；AdamW lr=0.001、weight_decay=0.0001、batch64、label_smoothing=0.05、AMP，按4951张训练标签计算类别权重。训练增广与mean/std=(0.5,0.5,0.5)归一化沿用旧程序。val/test不增广。", "",
              f"最多{args.epochs} epoch，按{monitor}选模/早停，patience={args.patience}、min_delta={args.min_delta:g}。学习率ReduceLROnPlateau监测同一验证指标，factor=0.5、patience=4、min_lr=1e-6。",
              "损失模式保存绝对最低验证损失权重，微小下降是否重置早停计数仍受min_delta约束；macro-F1模式沿用旧协议，仅提升超过min_delta时更新最佳权重。", "",
              f"实际训练{complete['epochs_completed']}轮，选中第{complete['selected_epoch']}轮；停止原因={complete['stop_reason']}。4951用于梯度更新、619用于选模；测试629只在训练结束、最佳权重冻结后评估，不依据测试结果调参。", "",
              "## 验证集与测试集指标", "", "P/R/F1同时提供weighted与macro；weighted-R=ACC。验证集参与选模，测试集不参与分类器选模。", "",
              "| 集合 | 数量 | ACC | weighted-P | weighted-R | weighted-F1 | macro-P | macro-R | macro-F1 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['split']} | {row['samples']} | " + " | ".join(f"{row[name]*100:.2f}%" for name in common.METRIC_NAMES) + " |")
        print(f"{row['split']} N={row['samples']}：ACC={row['accuracy']*100:.2f}%, P={row['weighted_precision']*100:.2f}%, "
              f"R={row['weighted_recall']*100:.2f}%, F1={row['weighted_f1']*100:.2f}%, macro-F1={row['macro_f1']*100:.2f}%", flush=True)
    lines += ["", "## 逐类别指标", "", "| 集合 | 类别 | 数量 | P | R | F1 |", "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        for name in CLASSES:
            lines.append(f"| {row['split']} | {name} | {row[f'support_{name}']} | " + " | ".join(f"{row[f'{metric}_{name}']*100:.2f}%" for metric in ("precision", "recall", "f1")) + " |")
    lines += ["", "## 数据隔离边界", "",
              "原始train/val/test同源组不相交，Stage2与分类器只在train4951做梯度更新。但旧Stage1曾在全6199条纹图上训练，因此不能把整个生成器流程称为从未见过验证/测试样本；本实验只隔离Stage2和分类器的训练集。", "",
              "所有旧Stage2权重、旧再映射图及旧分类器结果保持不变；本次没有重新训练原图基线。若使用val_loss，新结果与旧macro-F1选模实验不仅生成器数据范围不同，分类器选模指标也不同，不能把指标变化全部归因于Stage2样本减少。", "",
              "## 文件与复现", "",
              f"- 再映射数据：[datasets/post_formula_prediction]({(args.output_dir/'datasets/post_formula_prediction').as_posix()})",
              f"- 最佳分类器：[best.pt]({(args.run_dir/'best.pt').as_posix()})",
              f"- 总指标：[classification_{args.selection}_metrics.csv]({(args.output_dir/f'classification_{args.selection}_metrics.csv').as_posix()})",
              "- 同目录包含逐图预测、混淆矩阵、训练曲线和history.csv；生成图片及冻结分类器、结果均记录SHA256，报告指标由逐图预测重新核算。", "",
              "```powershell", "cd <repository-root>",
              f"conda run --no-capture-output -n diffusion_env python -X utf8 -u run_stage2_train4951_remapped_resnet18.py --selection {args.selection}",
              "```", "", "相同命令会核验已有数据、断点续训或复用完成结果。--stage report只重建报告；--stage prepare只生成图片。默认workers4，工作进程增广RNG不保存，跨中断续训不保证逐位复现；需要严格主进程随机流时使用--workers 0并另选目录。"]
    (args.output_dir / f"RESNET18_{args.selection.upper()}_REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--selection", choices=("val_loss", "macro_f1"), default="val_loss")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stage", choices=("all", "preflight", "prepare", "train", "test", "report"), default="all")
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.workers < 0 or args.min_delta < 0 or not math.isfinite(args.min_delta):
        raise ValueError("Invalid training or loader parameters.")
    args.output_dir = args.output_dir.resolve()
    args.run_dir = args.output_dir / "runs" / f"resnet18_scratch_{args.selection}"
    protocol, manifest = preflight(args)
    if args.stage == "preflight":
        return
    protocol_path = args.output_dir / "generation_protocol.json"
    if protocol_path.exists():
        if read_json(protocol_path) != protocol:
            raise ValueError("Existing generation protocol differs; choose a new output directory.")
    elif args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite an unrecognized nonempty output directory.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not protocol_path.exists():
        save_json(protocol, protocol_path)
        save_json(manifest, args.output_dir / "input_manifest.json")
    generate(args, protocol, manifest)
    if args.stage == "prepare":
        return
    classifier = common.load_classifier(common.DEFAULT_CLASSIFIER_SCRIPT)
    classifier.seed_worker = seed_worker
    if args.stage in ("all", "train"):
        train_classifier(args, protocol, classifier)
    if args.stage in ("all", "train", "test"):
        evaluate_split(args, "val", classifier)
    if args.stage in ("all", "test"):
        evaluate_split(args, "test", classifier)
    if args.stage in ("all", "test", "report"):
        report(args, protocol, classifier)
    if sha256(Path(protocol["checkpoint"])) != protocol["checkpoint_sha256"] or sha256(NOISE) != protocol["noise_sha256"]:
        raise ValueError("Frozen generator or saved inference noise changed.")
    print(f"实验目录：{args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
