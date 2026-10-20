from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from run_resnet18_clean_vs_post_formula import (
    CLASSES, DEFAULT_CLASSIFIER_SCRIPT, DEFAULT_CLEAN_DATA, DEFAULT_REMAPPED_DATA,
    preflight as source_pair_preflight,
)
from run_ttbm373_stage2_resnet18 import WORKSPACE, read_json, sha256, write_json


REFERENCE_RUNS = WORKSPACE / "experiments/resnet18_clean_vs_post_formula_t1_seed43/runs"
REFERENCE_TAGS = {
    "original": "seed43_split43_resnet18_scratch_clean",
    "remapped": "seed43_split43_resnet18_scratch_post_formula_t1",
}
DEFAULT_EXTERNAL = WORKSPACE / "experiments/ttbm373_stage2_post_formula_t1_resnet18_seed43"
LEGACY_OUTPUT = WORKSPACE / "experiments/resnet18_full6199_external373_t1_seed43"
DEFAULT_OUTPUT = WORKSPACE / "experiments/resnet18_full6199_external373_loss_earlystop_t1_seed43"
METHODS = ("original", "remapped")
CLASS_COUNTS_SOURCE = {"defect": 1244, "normal": 3018, "nut": 1937}
CLASS_COUNTS_EXTERNAL = {"defect": 122, "normal": 66, "nut": 185}
METRIC_NAMES = ("accuracy", "weighted_precision", "weighted_recall", "weighted_f1", "macro_precision", "macro_recall", "macro_f1")


def load_classifier(path: Path):
    spec = importlib.util.spec_from_file_location("full6199_reference_classifier", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def inspect_image(record: dict) -> dict:
    path = Path(record["path"])
    with Image.open(path) as image:
        if image.size != (64, 64) or image.mode != "RGB":
            raise ValueError(f"Expected RGB 64x64 image: {path}")
        image.load()
        pixels_hash = hashlib.sha256(image.tobytes()).hexdigest()
    return {**record, "sha256": sha256(path), "pixels_sha256": pixels_hash}


def collect(root: Path, expected: int, class_counts: dict) -> list[dict]:
    records = []
    for split in ("train", "val", "test"):
        for class_index, name in enumerate(CLASSES):
            directory = root / split / name
            if not directory.is_dir():
                raise FileNotFoundError(directory)
            for path in sorted(directory.glob("*.png")):
                records.append({"key": path.relative_to(root).as_posix(), "path": str(path.resolve()), "label": class_index, "class": name})
    records.sort(key=lambda row: row["key"])
    if len(records) != expected or len({row["key"] for row in records}) != expected:
        raise ValueError(f"Wrong sample count or duplicate keys under {root}")
    if dict(Counter(row["class"] for row in records)) != class_counts:
        raise ValueError(f"Class counts differ under {root}")
    with ThreadPoolExecutor(max_workers=8) as executor:
        return list(executor.map(inspect_image, records))


class AllImagesDataset(Dataset):
    """Merge the old directory splits without copying or modifying images."""

    def __init__(self, records: list[dict], transform):
        self.records = records
        self.transform = transform
        self.samples = [(row["path"], row["label"]) for row in records]
        self.targets = [row["label"] for row in records]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, target = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            return self.transform(image), target


def state_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def preflight(args) -> tuple[dict, dict, dict]:
    source_pair_preflight(args.original_source, args.remapped_source)
    print("核验源数据：原图与再映射图各6199张，合并全部train/val/test。", flush=True)
    source = {
        "original": collect(args.original_source, 6199, CLASS_COUNTS_SOURCE),
        "remapped": collect(args.remapped_source, 6199, CLASS_COUNTS_SOURCE),
    }
    external = {
        "original": collect(args.external_experiment / "datasets/original", 373, CLASS_COUNTS_EXTERNAL),
        "remapped": collect(args.external_experiment / "datasets/post_formula_prediction", 373, CLASS_COUNTS_EXTERNAL),
    }
    for records in (source, external):
        if [(row["key"], row["label"]) for row in records["original"]] != [(row["key"], row["label"]) for row in records["remapped"]]:
            raise ValueError("Paired domain keys/labels do not match.")
    source_pixels = {row["pixels_sha256"] for row in source["original"]}
    duplicates = [row["key"] for row in external["original"] if row["pixels_sha256"] in source_pixels]
    if duplicates:
        raise ValueError(f"Exact pixel duplicates between source training and external test: {duplicates[:10]}")
    external_protocol = read_json(args.external_experiment / "experiment_protocol.json")
    external_split = read_json(args.external_experiment / "split_manifest.json")
    external_generation = read_json(args.external_experiment / "generation_complete.json")
    originals_by_key = {row["output_key"]: row for row in external_split["rows"]}
    for row in external["original"]:
        if row["sha256"] != originals_by_key[row["key"]]["crop_sha256"]:
            raise ValueError("External original image changed.")
    for row in external["remapped"]:
        if row["sha256"] != external_generation["output_sha256"][row["key"]]:
            raise ValueError("External remapped image changed.")
    source_generation_path = args.remapped_source.parent.parent / "summary.json"
    source_generation = read_json(source_generation_path)
    if source_generation["timestep"] != 1 or str(Path(source_generation["checkpoint"]).resolve()) != str(Path(external_protocol["checkpoint"]).resolve()):
        raise ValueError("Different source/external generator or timestep.")
    if Path(source_generation["noise_file"]).resolve() != Path(external_protocol["noise_file"]).resolve():
        raise ValueError("Source/external inference noise files differ.")
    if sha256(Path(external_protocol["checkpoint"])) != external_protocol["checkpoint_sha256"] or sha256(Path(external_protocol["noise_file"])) != external_protocol["noise_sha256"]:
        raise ValueError("Frozen generator/noise file changed.")
    selections = {}
    reference_configuration = None
    for method in METHODS:
        reference = REFERENCE_RUNS / REFERENCE_TAGS[method]
        config = read_json(reference / "config.json")
        if getattr(args, "stopping", "fixed") == "fixed":
            fixed_epochs = int(config["best_epoch"])
            with (reference / "history.csv").open(encoding="utf-8-sig", newline="") as handle:
                history = list(csv.DictReader(handle))
            if any(int(row["epoch"]) != index + 1 for index, row in enumerate(history)) or len(history) < fixed_epochs:
                raise ValueError("Non-contiguous source-only learning-rate history.")
            learning_rates = [float(config["learning_rate"])] + [float(history[index - 1]["learning_rate"]) for index in range(1, fixed_epochs)]
            selections[method] = {
                "fixed_epochs": fixed_epochs, "learning_rate_per_epoch": learning_rates,
                "source_selection_run": str(reference), "source_selection_metric": "previous source-only validation macro-F1",
                "source_config_sha256": sha256(reference / "config.json"), "source_history_sha256": sha256(reference / "history.csv"),
            }
        common = {key: config[key] for key in ("pretrained", "seed", "batch_size", "learning_rate", "weight_decay", "label_smoothing", "class_weights_enabled", "augmentation_enabled", "amp")}
        if reference_configuration is not None and common != reference_configuration:
            raise ValueError("Source classifier configurations differ between domains.")
        reference_configuration = common
    if reference_configuration["seed"] != 43 or reference_configuration["pretrained"]:
        raise ValueError("Expected previous seed43, from-scratch ResNet18 protocol.")
    manifest = {"source": source, "external": external}
    protocol = {
        "purpose": "All 6199 source samples train two independent ResNet18 models; all 373 external samples test frozen models",
        "source_original": str(args.original_source), "source_remapped": str(args.remapped_source),
        "external_experiment": str(args.external_experiment), "source_count": 6199, "external_count": 373,
        "source_class_counts": CLASS_COUNTS_SOURCE, "external_class_counts": CLASS_COUNTS_EXTERNAL,
        "classifier_script": str(args.classifier_script), "classifier_script_sha256": sha256(args.classifier_script),
        "architecture": "torchvision ResNet18 + Dropout(0.30) + Linear(512,3)",
        "initialize_from_old_classifier": False, "source_only_selection": selections,
        "optimizer": "AdamW", "normalization": "mean=std=(0.5,0.5,0.5)",
        "reference_common_training_config": reference_configuration, "workers": args.workers, "seed": 43,
        "batch_size": reference_configuration["batch_size"],
        "class_weights": "Computed solely from all 6199 source training labels",
        "external_augmentation": False, "external_used_for_training_or_selection": False,
        "source_train_validation_test_directories_all_used_for_training": True,
        "external_train_validation_test_directories_all_used_for_testing": True,
        "exact_pixel_train_test_overlap": 0,
        "generator": external_protocol["checkpoint"], "generator_sha256": external_protocol["checkpoint_sha256"],
        "noise_file": external_protocol["noise_file"], "noise_sha256": external_protocol["noise_sha256"],
        "source_generation_summary_sha256": sha256(source_generation_path),
        "external_generation_marker_sha256": sha256(args.external_experiment / "generation_complete.json"),
        "input_manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        "interpretation_caveat": "373 samples were examined in earlier local experiments; this is not a pristine never-inspected final holdout, but this run does not use them for fitting, epoch/LR selection or tuning.",
    }
    print("预检通过：全量训练6199（1244/3018/1937），外部测试373（122/66/185），两域配对一致，原图像素完全重复交集为0。", flush=True)
    if getattr(args, "stopping", "fixed") == "fixed":
        print(f"源数据选定轮数：original={selections['original']['fixed_epochs']}，remapped={selections['remapped']['fixed_epochs']}；外部373不参与选模。", flush=True)
    return manifest, protocol, external_split


def train_one(args, method: str, records: list[dict], protocol: dict, classifier) -> None:
    run = args.output_dir / method
    run.mkdir(parents=True, exist_ok=True)
    if (run / "training_complete.json").is_file():
        marker = read_json(run / "training_complete.json")
        if sha256(run / "final.pt") != marker["final_sha256"]:
            raise ValueError("Final classifier checkpoint changed.")
        print(f"{method}全量训练已完成，复用冻结权重。", flush=True)
        return
    common = protocol["reference_common_training_config"]
    selection = protocol["source_only_selection"][method]
    classifier.set_seed(protocol["seed"])
    device = classifier.resolve_device(args.device)
    use_amp = common["amp"] and device.type == "cuda"
    dataset = AllImagesDataset(records, classifier.build_transform(True, common["augmentation_enabled"], common["pretrained"]))
    loader = classifier.make_loader(dataset, protocol["batch_size"], args.workers, True, protocol["seed"], device.type == "cuda")
    model = classifier.build_model(common["pretrained"]).to(device)
    initial_hash = state_sha256(model)
    weights, counts = classifier.class_weights(dataset.targets, device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights if common["class_weights_enabled"] else None, label_smoothing=common["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=common["learning_rate"], weight_decay=common["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    config = {
        "method": method, "train_samples": 6199, "train_class_counts": dict(zip(CLASSES, counts)),
        "model": "resnet18", "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "initial_model_sha256": initial_hash, "pretrained": common["pretrained"], "seed": protocol["seed"],
        "fixed_epochs": selection["fixed_epochs"], "learning_rate_per_epoch": selection["learning_rate_per_epoch"],
        "batch_size": protocol["batch_size"], "workers": args.workers, "device": str(device), "amp": use_amp,
        "weight_decay": common["weight_decay"], "label_smoothing": common["label_smoothing"],
        "class_weights": weights.cpu().tolist(), "augmentation_enabled": common["augmentation_enabled"],
        "external_test_read_during_training": False, "torch_version": torch.__version__,
    }
    if (run / "config.json").exists() and read_json(run / "config.json") != config:
        raise ValueError("Saved full-training configuration differs.")
    write_json(run / "config.json", config)
    start, history = 1, []
    last_path = run / "last.pt"
    if last_path.is_file():
        last = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(last["model_state_dict"])
        optimizer.load_state_dict(last["optimizer_state_dict"])
        scaler.load_state_dict(last["scaler_state_dict"])
        start = int(last["epoch"]) + 1
        history = last["history"]
        random.setstate(last["python_rng"])
        np.random.set_state(last["numpy_rng"])
        torch.set_rng_state(last["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(last["cuda_rng"])
        loader.generator.set_state(last["loader_rng"])
        print(f"{method}断点：从epoch {start}继续。", flush=True)
    print(f"开始全量{method}训练：N={len(dataset)}，epochs={selection['fixed_epochs']}，参数={config['parameters']:,}。", flush=True)
    for epoch in range(start, selection["fixed_epochs"] + 1):
        for group in optimizer.param_groups:
            group["lr"] = selection["learning_rate_per_epoch"][epoch - 1]
        began = time.perf_counter()
        loss, accuracy = classifier.train_epoch(model, loader, criterion, optimizer, scaler, device, use_amp)
        elapsed = time.perf_counter() - began
        history.append({"epoch": epoch, "train_samples": len(dataset), "train_loss": loss, "train_accuracy": accuracy,
                        "learning_rate": optimizer.param_groups[0]["lr"], "elapsed_seconds": elapsed})
        payload = {
            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch, "history": history, "config": config, "classes": list(CLASSES),
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [], "loader_rng": loader.generator.get_state(),
        }
        atomic_torch_save(payload, last_path)
        write_csv(run / "history.csv", history)
        print(f"{method} epoch {epoch:02d}/{selection['fixed_epochs']} | train N=6199, loss={loss:.4f}, ACC={accuracy:.4f}, lr={optimizer.param_groups[0]['lr']:.2e}, {elapsed:.1f}s | 未用外部集训练或选模", flush=True)
    final_path = run / "final.pt"
    atomic_torch_save({"model_state_dict": model.state_dict(), "config": config, "classes": list(CLASSES), "epoch": selection["fixed_epochs"]}, final_path)
    write_json(run / "training_complete.json", {"completed_at": datetime.now().isoformat(timespec="seconds"), "train_samples": 6199,
               "epochs_completed": selection["fixed_epochs"], "external_test_read": False, "final_sha256": sha256(final_path)})
    del model, optimizer, loader, dataset
    if device.type == "cuda":
        torch.cuda.empty_cache()


def freeze_training(root: Path, protocol: dict) -> dict:
    models = {}
    for method in METHODS:
        run = root / method
        marker = read_json(run / "training_complete.json")
        config = read_json(run / "config.json")
        if marker["train_samples"] != 6199 or marker["epochs_completed"] != protocol["source_only_selection"][method]["fixed_epochs"]:
            raise ValueError("Training does not meet full6199 protocol.")
        if sha256(run / "final.pt") != marker["final_sha256"]:
            raise ValueError("Frozen classifier checkpoint differs.")
        models[method] = {"final_sha256": marker["final_sha256"], "config_sha256": sha256(run / "config.json"), "initial_model_sha256": config["initial_model_sha256"]}
    if models["original"]["initial_model_sha256"] != models["remapped"]["initial_model_sha256"]:
        raise ValueError("The two classifiers did not start from identical weights.")
    frozen = {"models": models, "both_training_complete_before_external_evaluation": True, "external_samples": 373}
    path = root / "training_frozen.json"
    if path.exists():
        if read_json(path) != frozen:
            raise ValueError("Frozen model identities changed.")
    else:
        write_json(path, frozen)
    return frozen


def evaluate_one(args, method: str, records: list[dict], frozen: dict, classifier) -> None:
    run = args.output_dir / method
    if (run / "external_test_complete.json").is_file():
        saved = read_json(run / "external_test_complete.json")
        if saved["final_sha256"] != frozen["models"][method]["final_sha256"]:
            raise ValueError("Prior external test used different weights.")
        print(f"{method}已存在封存的373张外部测试，跳过重复推理。", flush=True)
        return
    config = read_json(run / "config.json")
    classifier.set_seed(config["seed"])
    device = classifier.resolve_device(args.device)
    dataset = AllImagesDataset(records, classifier.build_transform(False, False, config["pretrained"]))
    loader = classifier.make_loader(dataset, config["batch_size"], args.workers, False, config["seed"], device.type == "cuda")
    model = classifier.build_model(config["pretrained"]).to(device).eval()
    checkpoint = torch.load(run / "final.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.requires_grad_(False)
    state_before = state_sha256(model)
    loss, metrics, probabilities, _ = classifier.evaluate(model, loader, torch.nn.CrossEntropyLoss(), device,
                                                        config["amp"] and device.type == "cuda", collect_probabilities=True)
    if metrics["samples"] != 373 or state_sha256(model) != state_before:
        raise ValueError("Wrong external test count or model changed during testing.")
    predictions = []
    for record, probs in zip(records, probabilities.tolist(), strict=True):
        predicted = max(range(3), key=lambda index: probs[index])
        predictions.append({"key": record["key"], "image": record["path"], "true_id": record["label"], "true_class": record["class"],
                            "predicted_id": predicted, "predicted_class": CLASSES[predicted], "confidence": probs[predicted],
                            **{f"prob_{name}": probs[index] for index, name in enumerate(CLASSES)}})
    write_csv(run / "external_predictions.csv", predictions)
    write_json(run / "external_metrics.json", {"loss": loss, **metrics})
    classifier.plot_confusion(metrics["confusion_matrix"], run / "external_confusion.png", f"External TTBM 373: {method}, trained on all 6199")
    write_json(run / "external_test_complete.json", {"completed_at": datetime.now().isoformat(timespec="seconds"), "samples": 373,
               "final_sha256": sha256(run / "final.pt"), "frozen_during_evaluation": True,
               "metrics_sha256": sha256(run / "external_metrics.json"), "predictions_sha256": sha256(run / "external_predictions.csv")})
    print(f"{method}外部373：ACC={metrics['accuracy']:.4f}, weighted-P/R/F1={metrics['weighted_precision']:.4f}/{metrics['weighted_recall']:.4f}/{metrics['weighted_f1']:.4f}, macro-F1={metrics['macro_f1']:.4f}", flush=True)
    del model, checkpoint, loader, dataset
    if device.type == "cuda":
        torch.cuda.empty_cache()


def audited_metrics(root: Path, method: str, expected_records: list[dict], classifier) -> tuple[dict, dict]:
    run = root / method
    marker = read_json(run / "external_test_complete.json")
    if sha256(run / "external_metrics.json") != marker["metrics_sha256"] or sha256(run / "external_predictions.csv") != marker["predictions_sha256"]:
        raise ValueError("Sealed external results changed.")
    with (run / "external_predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
        predictions = list(csv.DictReader(handle))
    expected = {row["key"]: row["label"] for row in expected_records}
    matrix = torch.zeros((3, 3), dtype=torch.int64)
    correctness = {}
    for row in predictions:
        key = row["key"]
        if key in correctness or key not in expected or int(row["true_id"]) != expected[key]:
            raise ValueError("External predictions are duplicated, unpaired or mislabelled.")
        target, prediction = int(row["true_id"]), int(row["predicted_id"])
        matrix[target, prediction] += 1
        correctness[key] = target == prediction
    if set(correctness) != set(expected) or len(predictions) != 373:
        raise ValueError("External prediction count is not 373.")
    metrics = read_json(run / "external_metrics.json")
    calculated = classifier.metrics_from_confusion(matrix)
    if calculated["confusion_matrix"] != metrics["confusion_matrix"] or matrix.sum(1).tolist() != [CLASS_COUNTS_EXTERNAL[name] for name in CLASSES]:
        raise ValueError("External confusion matrix/support audit failed.")
    if any(abs(calculated[name] - metrics[name]) > 1e-10 for name in METRIC_NAMES):
        raise ValueError("External metrics do not match predictions.")
    return metrics, correctness


def report(args, manifest: dict, protocol: dict, classifier) -> None:
    root = args.output_dir
    rows, details, correctness = [], {}, {}
    for method in METHODS:
        metrics, prediction = audited_metrics(root, method, manifest["external"][method], classifier)
        details[method], correctness[method] = metrics, prediction
        row = {"method": method, "train_samples": 6199, "test_samples": 373, "training_epochs": protocol["source_only_selection"][method]["fixed_epochs"],
               **{name: metrics[name] for name in METRIC_NAMES}}
        for index, name in enumerate(CLASSES):
            row[f"support_{name}"] = CLASS_COUNTS_EXTERNAL[name]
            for metric in ("precision", "recall", "f1"):
                row[f"{metric}_{name}"] = metrics[metric][index]
        rows.append(row)
    keys = set(correctness["original"])
    if keys != set(correctness["remapped"]):
        raise ValueError("The external sets differ across methods.")
    paired = {
        "both_correct": sum(correctness["original"][key] and correctness["remapped"][key] for key in keys),
        "original_only_correct": sum(correctness["original"][key] and not correctness["remapped"][key] for key in keys),
        "remapped_only_correct": sum(not correctness["original"][key] and correctness["remapped"][key] for key in keys),
        "both_wrong": sum(not correctness["original"][key] and not correctness["remapped"][key] for key in keys),
    }
    write_csv(root / "external_comparison.csv", rows)
    write_json(root / "external_comparison.json", {"metrics": rows, "full_metrics": details, "paired_comparison": paired})
    lines = [
        "# 全6199张训练 → 全373张外部测试：ResNet18原图与Stage2再映射对照", "",
        "## 数据与流程", "",
        "- 原图组：全部6199张旧原图训练一个ResNet18，冻结后测试全部373张TTBM原图。",
        "- 再映射组：全部6199张旧Stage2 t=1公式更新后再映射图训练另一个ResNet18，冻结后测试全部373张TTBM对应再映射图。",
        "- 旧数据原有train/val/test全部合并为训练数据；新数据原有298/37/38目录全部合并为外部测试，不在373张上再划8:1:1。",
        "- 原图与再映射图的样本、标签严格配对。训练6199张的defect/normal/nut数量为1244/3018/1937，外部373张为122/66/185，来自244张新源照片。",
        "- 所有图像RGB、64×64；旧/新再映射沿用同一冻结Stage2和保存的seed43噪声，t=1→公式更新→t=0再映射；不把合成条纹参照或类别标签输入生成器。",
        "- 检查旧训练原图与新测试原图的完全相同RGB像素交集为0。这不等于已经排除了所有近重复照片或视角。",
        "", "## 训练与选模", "",
        "以前的两组分类器只训练4951张，不满足本次要求。本次两组均重新随机初始化，从头用全6199张训练，不加载旧分类器，也不加载在373张上训练过的权重。",
        "同架构ResNet18 + Dropout(0.30) + Linear(512,3)，两组初始参数SHA256完全相同，seed43；AdamW初始lr=0.001，weight_decay=0.0001，batch64，label_smoothing=0.05，训练增广和归一化沿用原程序。",
        "类别权重只用全6199张训练标签计算；训练轮数和每轮学习率来自此前旧源数据验证macro-F1选模记录，不使用373张决定。原历史CSV的学习率是当轮结束后的值，此处正确应用于下一轮训练。",
        "", "| 方法 | 全量训练轮数 | 预先固定学习率日程 |", "|---|---:|---|",
        "| 原图 | 19 | epoch1–14:0.001；epoch15–19:0.0005 |",
        "| Stage2再映射 | 23 | epoch1–20:0.001；epoch21–23:0.0005 |", "",
        "两组轮数不同，但选模依据一致且仅来自源数据。全量重训不再留源数据验证集、不按外部测试早停；使用预定最后一轮权重。两组均训练完成并保存冻结身份后，才执行外部测试。",
        "", "## 全373张外部测试结果", "",
        "P/R/F1报告weighted（按真实类样本数加权）和macro（各类等权），weighted-R与ACC相等。", "",
        "| 方法 | ACC | weighted-P | weighted-R | weighted-F1 | macro-P | macro-R | macro-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        label = "原图基线" if row["method"] == "original" else "Stage2再映射"
        lines.append(f"| {label} | " + " | ".join(f"{row[name] * 100:.2f}%" for name in METRIC_NAMES) + " |")
        print(f"{label}外部373：ACC={row['accuracy'] * 100:.2f}%, P={row['weighted_precision'] * 100:.2f}%, R={row['weighted_recall'] * 100:.2f}%, F1={row['weighted_f1'] * 100:.2f}%, macro-F1={row['macro_f1'] * 100:.2f}%", flush=True)
    lines += ["", "## 逐类别外部测试指标", "", "| 方法 | 类别 | 样本数 | P | R | F1 |", "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        label = "原图基线" if row["method"] == "original" else "Stage2再映射"
        for name in CLASSES:
            lines.append(f"| {label} | {name} | {row[f'support_{name}']} | " + " | ".join(f"{row[f'{metric}_{name}'] * 100:.2f}%" for metric in ("precision", "recall", "f1")) + " |")
    delta = {name: (details["remapped"][name] - details["original"][name]) * 100 for name in METRIC_NAMES}
    lines += [
        "", "## 配对对照与限制", "",
        f"再映射相对原图：ACC变化{delta['accuracy']:+.2f}个百分点，weighted-F1变化{delta['weighted_f1']:+.2f}个百分点，macro-F1变化{delta['macro_f1']:+.2f}个百分点。",
        f"配对预测：两组都正确{paired['both_correct']}张，仅原图正确{paired['original_only_correct']}张，仅再映射正确{paired['remapped_only_correct']}张，两组都错误{paired['both_wrong']}张。",
        "这次不是此前373张上重新训练分类器的81.58%/76.32%实验，也不是原6199内部629张test实验。本次测试数量为全部373张，训练只来自6199张旧数据。",
        "373张数据此前已经用于局部分类和图像相似度探索，因此不能称为从未查看过的最终独立留出集。但本次重训从头开始，轮数、学习率和超参数不依据373张结果选择，未对373张做梯度更新或测试时适配。",
        "单次种子结果不能证明稳定或统计显著的提升；多个目标框来自同一源照片，373个目标并不等价于373张相互独立照片。类别数量不均衡，须结合macro与逐类指标。新旧数据类别定义一致性仍依赖人工标注。",
        "", "## 冻结权重与复现", "",
        f"- 原图分类器：[original/final.pt]({(root / 'original/final.pt').as_posix()})",
        f"- 再映射分类器：[remapped/final.pt]({(root / 'remapped/final.pt').as_posix()})",
        f"- 指标：[external_comparison.csv]({(root / 'external_comparison.csv').as_posix()})",
        "", "```powershell", "cd <repository-root>",
        "conda run --no-capture-output -n diffusion_env python -X utf8 -u run_resnet18_full6199_external373.py",
        "```", "", "重复相同命令会核验输入、复用完成的冻结权重并跳过已封存外部推理；仅重建报告加`--stage report`。断点保存模型、优化器和随机状态，默认workers0以便恢复增强随机流。",
    ]
    (root / "EXTERNAL_TEST_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train on all source6199, then compare frozen ResNet18 models on all external373.")
    parser.add_argument("--original-source", type=Path, default=DEFAULT_CLEAN_DATA)
    parser.add_argument("--remapped-source", type=Path, default=DEFAULT_REMAPPED_DATA)
    parser.add_argument("--external-experiment", type=Path, default=DEFAULT_EXTERNAL)
    parser.add_argument("--classifier-script", type=Path, default=DEFAULT_CLASSIFIER_SCRIPT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--stopping", choices=("training-loss", "fixed"), default="training-loss")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum epochs for training-loss early stopping.")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stage", choices=("all", "preflight", "train", "test", "report"), default="all")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT if args.stopping == "training-loss" else LEGACY_OUTPUT
    if args.workers is None:
        args.workers = 4 if args.stopping == "training-loss" else 0
    if args.workers < 0:
        raise ValueError("Invalid workers.")
    for name in ("original_source", "remapped_source", "external_experiment", "classifier_script", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.stopping == "training-loss":
        from resnet18_full6199_loss_earlystop import run
        run(args)
        return
    manifest, protocol, _external_split = preflight(args)
    if args.stage == "preflight":
        return
    root = args.output_dir
    path = root / "experiment_protocol.json"
    if path.exists():
        if read_json(path) != protocol:
            raise ValueError("Existing full6199/external373 protocol differs; choose a new output directory.")
    elif root.exists() and any(root.iterdir()):
        raise FileExistsError("Refusing to reuse nonempty unrecognized output directory.")
    root.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        write_json(path, protocol)
        write_json(root / "input_manifest.json", manifest)
    classifier = load_classifier(args.classifier_script)
    if args.stage in ("all", "train"):
        for method in METHODS:
            train_one(args, method, manifest["source"][method], protocol, classifier)
    frozen = freeze_training(root, protocol)
    if args.stage in ("all", "test"):
        for method in METHODS:
            evaluate_one(args, method, manifest["external"][method], frozen, classifier)
    if args.stage in ("all", "test", "report"):
        report(args, manifest, protocol, classifier)
    if sha256(Path(protocol["generator"])) != protocol["generator_sha256"]:
        raise ValueError("Generator file changed.")
    print(f"实验目录：{root}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("已中断；相同命令可按保存的全量分类器断点继续。", file=sys.stderr)
        raise SystemExit(130)
