from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


DEFAULT_CLEAN_DATA = Path("data/original")
DEFAULT_STRIPED_DATA = Path("data/watermarked")
DEFAULT_RUNS = Path("runs/label_texture_classification")
CLASSES = ("defect", "normal", "nut")
IMAGE_SIZE = 64


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    train_accuracy: float
    selection_val_loss: float
    selection_val_accuracy: float
    selection_val_macro_precision: float
    selection_val_macro_recall: float
    selection_val_macro_f1: float
    learning_rate: float
    elapsed_seconds: float


def build_model(pretrained: bool) -> nn.Module:
    """构建固定的ResNet18三分类模型。"""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Sequential(
        nn.Dropout(0.30),
        nn.Linear(model.fc.in_features, len(CLASSES)),
    )
    return model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def normalization(pretrained: bool) -> transforms.Normalize:
    if pretrained:
        return transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
    return transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )


def build_transform(training: bool, augmentation: bool, pretrained: bool):
    steps: list[Any] = []
    if training and augmentation:
        # 不使用水平/垂直翻转，避免把类别编码的方向直接改变。
        steps.extend(
            [
                transforms.RandomAffine(
                    degrees=5,
                    translate=(0.03, 0.03),
                    scale=(0.95, 1.05),
                    fill=(114, 114, 114),
                ),
                transforms.ColorJitter(
                    brightness=0.10,
                    contrast=0.10,
                    saturation=0.05,
                ),
            ]
        )
    steps.extend([transforms.ToTensor(), normalization(pretrained)])
    return transforms.Compose(steps)


def validate_dataset_root(root: Path, splits: tuple[str, ...]) -> None:
    for split in splits:
        for class_name in CLASSES:
            directory = root / split / class_name
            if not directory.is_dir():
                raise FileNotFoundError(f"缺少数据目录：{directory}")


def image_folder(
    root: Path,
    split: str,
    training: bool,
    augmentation: bool,
    pretrained: bool,
) -> datasets.ImageFolder:
    dataset = datasets.ImageFolder(
        root / split,
        transform=build_transform(training, augmentation, pretrained),
    )
    expected = {name: index for index, name in enumerate(CLASSES)}
    if dataset.class_to_idx != expected:
        raise ValueError(
            f"{root / split}类别映射为{dataset.class_to_idx}，预期为{expected}"
        )
    if not dataset.samples:
        raise ValueError(f"数据集为空：{root / split}")
    return dataset


def make_loader(
    dataset: datasets.ImageFolder,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
    use_cuda: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=use_cuda,
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=generator if shuffle else None,
    )


def class_weights(targets: list[int], device: torch.device) -> tuple[torch.Tensor, list[int]]:
    counts = np.bincount(targets, minlength=len(CLASSES))
    if np.any(counts == 0):
        raise ValueError(f"存在无训练样本类别：{counts.tolist()}")
    weights = counts.sum() / (len(CLASSES) * counts.astype(np.float64))
    return torch.tensor(weights, dtype=torch.float32, device=device), counts.tolist()


def metrics_from_confusion(confusion: torch.Tensor) -> dict[str, Any]:
    matrix = confusion.to(torch.float64)
    true_positive = matrix.diag()
    predicted = matrix.sum(0)
    actual = matrix.sum(1)
    precision = true_positive / predicted.clamp_min(1)
    recall = true_positive / actual.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    total = matrix.sum().clamp_min(1)
    false_positive = predicted - true_positive
    false_negative = actual - true_positive
    true_negative = total - true_positive - false_positive - false_negative
    class_accuracy = (true_positive + true_negative) / total
    support_weights = actual / total
    weighted_precision = (precision * support_weights).sum()
    weighted_recall = (recall * support_weights).sum()
    weighted_f1 = (f1 * support_weights).sum()
    return {
        "accuracy": float(true_positive.sum() / total),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "class_accuracy": class_accuracy.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "confusion_matrix": confusion.tolist(),
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        count = targets.size(0)
        total_loss += float(loss.detach()) * count
        total_correct += int((logits.argmax(1) == targets).sum())
        total += count
    return total_loss / total, total_correct / total


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    collect_probabilities: bool,
) -> tuple[float, dict[str, Any], torch.Tensor | None, float]:
    model.eval()
    confusion = torch.zeros((len(CLASSES), len(CLASSES)), dtype=torch.int64)
    probability_batches: list[torch.Tensor] = []
    total_loss = 0.0
    total = 0
    timed_images = 0
    inference_seconds = 0.0
    warmed_up = False

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if not warmed_up:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                _ = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            warmed_up = True

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(images)
            loss = criterion(logits, targets)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds += time.perf_counter() - start
        timed_images += targets.size(0)

        probabilities = logits.softmax(1).cpu()
        predictions = probabilities.argmax(1)
        indices = targets.cpu() * len(CLASSES) + predictions
        confusion += torch.bincount(
            indices,
            minlength=len(CLASSES) ** 2,
        ).reshape(len(CLASSES), len(CLASSES))
        if collect_probabilities:
            probability_batches.append(probabilities)
        count = targets.size(0)
        total_loss += float(loss) * count
        total += count

    metrics = metrics_from_confusion(confusion)
    metrics["samples"] = total
    metrics["inference_ms_per_image"] = (
        inference_seconds * 1000 / max(timed_images, 1)
    )
    all_probabilities = (
        torch.cat(probability_batches) if collect_probabilities else None
    )
    return total_loss / total, metrics, all_probabilities, inference_seconds


def save_predictions(
    path: Path,
    dataset: datasets.ImageFolder,
    probabilities: torch.Tensor,
) -> None:
    if len(dataset.samples) != len(probabilities):
        raise ValueError("预测数量与数据集大小不一致。")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "image",
                "true_id",
                "true_class",
                "predicted_id",
                "predicted_class",
                "confidence",
                *[f"prob_{name}" for name in CLASSES],
            ]
        )
        for (image_path, true_id), probs in zip(
            dataset.samples, probabilities, strict=True
        ):
            predicted_id = int(probs.argmax())
            writer.writerow(
                [
                    image_path,
                    true_id,
                    CLASSES[true_id],
                    predicted_id,
                    CLASSES[predicted_id],
                    float(probs[predicted_id]),
                    *[float(value) for value in probs],
                ]
            )


def plot_history(history: list[EpochRecord], path: Path) -> None:
    epochs = [row.epoch for row in history]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    axes[0].plot(epochs, [row.train_loss for row in history], label="train")
    axes[0].plot(
        epochs, [row.selection_val_loss for row in history], label="selection val"
    )
    axes[0].set_title("Loss")
    axes[1].plot(
        epochs, [row.train_accuracy for row in history], label="train"
    )
    axes[1].plot(
        epochs,
        [row.selection_val_accuracy for row in history],
        label="selection val",
    )
    axes[1].set_title("Accuracy")
    axes[2].plot(
        epochs,
        [row.selection_val_macro_f1 for row in history],
        label="selection val Macro-F1",
    )
    axes[2].set_title("Macro-F1")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.3)
        axis.legend()
    axes[1].set_ylim(0, 1)
    axes[2].set_ylim(0, 1)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_confusion(confusion: list[list[int]], path: Path, title: str) -> None:
    matrix = np.asarray(confusion, dtype=np.int64)
    figure, axis = plt.subplots(figsize=(6.5, 5.7))
    image = axis.imshow(matrix, cmap="Blues", vmin=0)
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=range(len(CLASSES)),
        yticks=range(len(CLASSES)),
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        xlabel="Predicted",
        ylabel="True",
        title=title,
    )
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(len(CLASSES)):
        for column in range(len(CLASSES)):
            value = int(matrix[row, column])
            axis.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_history(path: Path, history: list[EpochRecord]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(history[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in history)


def summary_row(
    config: dict[str, Any],
    split: str,
    eval_domain: str,
    metrics: dict[str, Any],
    loss: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model": config["model"],
        "pretrained": config["pretrained"],
        "seed": config["seed"],
        "train_domain": config["train_domain"],
        "selection_domain": config.get("selection_domain", config["train_domain"]),
        "eval_domain": eval_domain,
        "split": split,
        "samples": metrics["samples"],
        "loss": loss,
        "accuracy": metrics["accuracy"],
        "precision": metrics["weighted_precision"],
        "recall": metrics["weighted_recall"],
        "f1": metrics["weighted_f1"],
        "parameters": config["parameters"],
        "inference_ms_per_image": metrics["inference_ms_per_image"],
        "best_epoch": config.get("best_epoch", ""),
    }
    for index, class_name in enumerate(CLASSES):
        row[f"accuracy_{class_name}"] = metrics["class_accuracy"][index]
        row[f"precision_{class_name}"] = metrics["precision"][index]
        row[f"recall_{class_name}"] = metrics["recall"][index]
        row[f"f1_{class_name}"] = metrics["f1"][index]
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("没有可写入的结果。")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_f1: float,
    best_epoch: int,
    no_improvement: int,
    history: list[EpochRecord],
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "best_f1": best_f1,
        "best_epoch": best_epoch,
        "epochs_without_improvement": no_improvement,
        "history": [asdict(row) for row in history],
        "config": config,
        "classes": list(CLASSES),
    }


def dataset_for_domain(clean: Path, striped: Path, domain: str) -> Path:
    if domain == "clean":
        return clean
    if domain == "striped":
        return striped
    raise ValueError(f"未知域：{domain}")


def resolve_device(value: str | None) -> torch.device:
    if value is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    normalized = value.strip().lower()
    if normalized.isdigit():
        normalized = f"cuda:{normalized}"
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了CUDA，但当前PyTorch无法使用CUDA。")
    return device


def train_command(args: argparse.Namespace) -> int:
    clean_root = args.clean_data.resolve()
    striped_root = args.striped_data.resolve()
    selection_domain = args.selection_domain or args.train_domain
    validate_dataset_root(clean_root, ("train", "val"))
    if args.train_domain == "striped" or selection_domain == "striped":
        validate_dataset_root(striped_root, ("train", "val"))
    train_root = dataset_for_domain(clean_root, striped_root, args.train_domain)
    selection_root = dataset_for_domain(clean_root, striped_root, selection_domain)
    run_dir = args.runs_dir.resolve() / args.run_tag

    if run_dir.exists() and not args.resume:
        raise FileExistsError(
            f"实验目录已存在：{run_dir}\n如需断点续训，请添加--resume。"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = resolve_device(args.device)
    use_amp = args.amp and device.type == "cuda"

    train_dataset = image_folder(
        train_root,
        "train",
        training=True,
        augmentation=not args.no_augmentation,
        pretrained=args.pretrained,
    )
    selection_val_dataset = image_folder(
        selection_root,
        "val",
        training=False,
        augmentation=False,
        pretrained=args.pretrained,
    )
    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        args.workers,
        True,
        args.seed,
        device.type == "cuda",
    )
    selection_val_loader = make_loader(
        selection_val_dataset,
        args.batch_size,
        args.workers,
        False,
        args.seed,
        device.type == "cuda",
    )

    model = build_model(args.pretrained).to(device)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    weights, counts = class_weights(train_dataset.targets, device)
    effective_weights = None if args.no_class_weights else weights
    learning_rate = args.learning_rate
    if learning_rate is None:
        learning_rate = 3e-4 if args.pretrained else 1e-3
    train_criterion = nn.CrossEntropyLoss(
        weight=effective_weights,
        label_smoothing=args.label_smoothing,
    )
    eval_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    config: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "clean_data": str(clean_root),
        "striped_data": str(striped_root),
        "train_domain": args.train_domain,
        "selection_domain": selection_domain,
        "model": "resnet18",
        "pretrained": args.pretrained,
        "seed": args.seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "learning_rate": learning_rate,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "class_weights_enabled": not args.no_class_weights,
        "class_weights": weights.detach().cpu().tolist(),
        "train_class_counts": dict(zip(CLASSES, counts, strict=True)),
        "augmentation_enabled": not args.no_augmentation,
        "device": str(device),
        "amp": use_amp,
        "parameters": total_parameters,
        "torch_version": torch.__version__,
    }
    config_path = run_dir / "config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        keys = [
            "clean_data",
            "train_domain",
            "selection_domain",
            "model",
            "pretrained",
            "seed",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "label_smoothing",
            "class_weights_enabled",
            "augmentation_enabled",
        ]
        if args.train_domain == "striped" or selection_domain == "striped":
            keys.append("striped_data")
        existing.setdefault(
            "selection_domain", existing.get("train_domain", args.train_domain)
        )
        mismatches = {
            key: (existing.get(key), config.get(key))
            for key in keys
            if existing.get(key) != config.get(key)
        }
        if mismatches:
            raise ValueError(f"断点配置与当前参数不一致：{mismatches}")
        config = existing
        config["epochs"] = args.epochs
        config["patience"] = args.patience
        config["device"] = str(device)
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    start_epoch = 1
    best_f1 = -1.0
    best_epoch = 0
    no_improvement = 0
    history: list[EpochRecord] = []
    last_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"
    if args.resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_f1 = float(checkpoint["best_f1"])
        best_epoch = int(checkpoint["best_epoch"])
        no_improvement = int(checkpoint["epochs_without_improvement"])
        history = [EpochRecord(**row) for row in checkpoint["history"]]
        print(f"断点续训：从epoch {start_epoch}继续。")

    training_already_complete = args.resume and (run_dir / "training_complete.json").is_file()
    if training_already_complete:
        print("检测到training_complete.json：复用最佳权重，仅重新汇总val。")

    print(f"实验目录：{run_dir}")
    print(
        f"模型：ResNet18，训练域：{args.train_domain}，"
        f"验证/选模域：{selection_domain}，seed={args.seed}"
    )
    print(f"数据：train={len(train_dataset)}，selection val={len(selection_val_dataset)}")
    print(f"参数量：{total_parameters:,}，设备：{device}")

    epoch_range = range(start_epoch, args.epochs + 1)
    if training_already_complete or no_improvement >= args.patience:
        epoch_range = range(0)
    for epoch in epoch_range:
        epoch_start = time.perf_counter()
        train_loss, train_accuracy = train_epoch(
            model,
            train_loader,
            train_criterion,
            optimizer,
            scaler,
            device,
            use_amp,
        )
        val_loss, val_metrics, _, _ = evaluate(
            model,
            selection_val_loader,
            eval_criterion,
            device,
            use_amp,
            collect_probabilities=False,
        )
        scheduler.step(val_metrics["macro_f1"])
        elapsed = time.perf_counter() - epoch_start
        record = EpochRecord(
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            selection_val_loss=val_loss,
            selection_val_accuracy=val_metrics["accuracy"],
            selection_val_macro_precision=val_metrics["macro_precision"],
            selection_val_macro_recall=val_metrics["macro_recall"],
            selection_val_macro_f1=val_metrics["macro_f1"],
            learning_rate=optimizer.param_groups[0]["lr"],
            elapsed_seconds=elapsed,
        )
        history.append(record)
        improved = val_metrics["macro_f1"] > best_f1 + args.min_delta
        if improved:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            no_improvement = 0
        else:
            no_improvement += 1

        payload = checkpoint_payload(
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_f1,
            best_epoch,
            no_improvement,
            history,
            config,
        )
        torch.save(payload, last_path)
        if improved:
            torch.save(payload, best_path)
        write_history(run_dir / "history.csv", history)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss={train_loss:.4f}, acc={train_accuracy:.4f} | "
            f"val loss={val_loss:.4f}, acc={val_metrics['accuracy']:.4f}, "
            f"Macro-F1={val_metrics['macro_f1']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s"
            + (" | best" if improved else "")
        )
        if no_improvement >= args.patience:
            print(f"早停：连续{args.patience}轮Macro-F1未提升。")
            break

    if not best_path.is_file():
        raise RuntimeError("没有生成best.pt。")
    plot_history(history, run_dir / "training_curves.png")
    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    best_epoch = int(best_checkpoint["best_epoch"])
    best_f1 = float(best_checkpoint["best_f1"])
    config["best_epoch"] = best_epoch
    config["best_selection_val_macro_f1"] = best_f1
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    eval_domain = selection_domain
    eval_root = dataset_for_domain(clean_root, striped_root, eval_domain)
    dataset = image_folder(
        eval_root,
        "val",
        training=False,
        augmentation=False,
        pretrained=args.pretrained,
    )
    loader = make_loader(
        dataset,
        args.batch_size,
        args.workers,
        False,
        args.seed,
        device.type == "cuda",
    )
    loss, metrics, probabilities, _ = evaluate(
        model,
        loader,
        eval_criterion,
        device,
        use_amp,
        collect_probabilities=True,
    )
    assert probabilities is not None
    val_details = {eval_domain: {"loss": loss, **metrics}}
    val_rows = [summary_row(config, "val", eval_domain, metrics, loss)]
    save_predictions(
        run_dir / f"val_predictions_{eval_domain}.csv", dataset, probabilities
    )
    plot_confusion(
        metrics["confusion_matrix"],
        run_dir / f"val_confusion_{eval_domain}.png",
        f"Val confusion: train={args.train_domain}, eval={eval_domain}",
    )
    print(
        f"val评估：train={args.train_domain}, eval={eval_domain}: "
        f"ACC={metrics['accuracy']:.4f}, "
        f"P={metrics['weighted_precision']:.4f}, "
        f"R={metrics['weighted_recall']:.4f}, "
        f"F1={metrics['weighted_f1']:.4f}"
    )

    write_csv(run_dir / "val_summary.csv", val_rows)
    (run_dir / "val_metrics.json").write_text(
        json.dumps(val_details, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "training_complete.json").write_text(
        json.dumps(
            {
                "completed_at": datetime.now().isoformat(timespec="seconds"),
                "best_epoch": best_epoch,
                "best_selection_val_macro_f1": best_f1,
                "test_read": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"训练和同域val评估完成：{run_dir}")
    print("test未读取；确定方案后再运行test子命令。")
    return 0


def test_command(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    config_path = run_dir / "config.json"
    best_path = run_dir / "best.pt"
    if not config_path.is_file() or not best_path.is_file():
        raise FileNotFoundError(f"实验目录缺少config.json或best.pt：{run_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    clean_root = Path(config["clean_data"])
    striped_root = Path(config["striped_data"])
    validate_dataset_root(clean_root, ("test",))
    validate_dataset_root(striped_root, ("test",))
    set_seed(int(config["seed"]))
    device = resolve_device(args.device)
    use_amp = bool(config["amp"]) and device.type == "cuda"
    if config.get("model") != "resnet18":
        raise ValueError(f"该程序只支持ResNet18，权重模型为：{config.get('model')}")
    model = build_model(bool(config["pretrained"])).to(device)
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.CrossEntropyLoss()

    eval_domain = config["train_domain"]
    eval_root = dataset_for_domain(clean_root, striped_root, eval_domain)
    dataset = image_folder(
        eval_root,
        "test",
        training=False,
        augmentation=False,
        pretrained=bool(config["pretrained"]),
    )
    loader = make_loader(
        dataset,
        int(config["batch_size"]),
        args.workers,
        False,
        int(config["seed"]),
        device.type == "cuda",
    )
    loss, metrics, probabilities, _ = evaluate(
        model,
        loader,
        criterion,
        device,
        use_amp,
        collect_probabilities=True,
    )
    assert probabilities is not None
    details = {eval_domain: {"loss": loss, **metrics}}
    test_rows = [summary_row(config, "test", eval_domain, metrics, loss)]
    save_predictions(
        run_dir / f"test_predictions_{eval_domain}.csv", dataset, probabilities
    )
    plot_confusion(
        metrics["confusion_matrix"],
        run_dir / f"test_confusion_{eval_domain}.png",
        f"Test confusion: train={config['train_domain']}, eval={eval_domain}",
    )
    print(
        f"同域test评估={eval_domain}: "
        f"ACC={metrics['accuracy']:.4f}, "
        f"P={metrics['weighted_precision']:.4f}, "
        f"R={metrics['weighted_recall']:.4f}, "
        f"F1={metrics['weighted_f1']:.4f}"
    )

    write_csv(run_dir / "test_summary.csv", test_rows)
    (run_dir / "test_metrics.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "test_complete.json").write_text(
        json.dumps(
            {
                "completed_at": datetime.now().isoformat(timespec="seconds"),
                "test_read": True,
                "evaluated_domains": [eval_domain],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"封存test评估完成：{run_dir / 'test_summary.csv'}")
    return 0


def compare_command(args: argparse.Namespace) -> int:
    rows: list[dict[str, str]] = []
    filename = f"{args.split}_summary.csv"
    for run_dir in args.run_dirs:
        path = run_dir.resolve() / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                row = {"run_dir": str(run_dir.resolve()), **row}
                rows.append(row)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output, rows)
    print(f"合并{args.split}结果：{output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="0/1类别条纹数据的ResNet18严格同域对照实验。"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="训练并仅评估val。")
    train_parser.add_argument("--clean-data", type=Path, default=DEFAULT_CLEAN_DATA)
    train_parser.add_argument("--striped-data", type=Path, default=DEFAULT_STRIPED_DATA)
    train_parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    train_parser.add_argument("--run-tag", required=True)
    train_parser.add_argument("--train-domain", choices=("clean", "striped"), required=True)
    train_parser.add_argument(
        "--selection-domain",
        choices=("clean", "striped"),
        default=None,
        help=(
            "用于早停、最佳权重选择和最终val评估的数据域；"
            "默认与train-domain相同。"
        ),
    )
    train_parser.add_argument("--pretrained", action="store_true")
    train_parser.add_argument("--seed", type=int, default=43)
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--patience", type=int, default=15)
    train_parser.add_argument("--min-delta", type=float, default=1e-4)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--workers", type=int, default=4)
    train_parser.add_argument("--learning-rate", type=float, default=None)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--label-smoothing", type=float, default=0.05)
    train_parser.add_argument("--device", default=None)
    train_parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    train_parser.add_argument("--no-class-weights", action="store_true")
    train_parser.add_argument("--no-augmentation", action="store_true")
    train_parser.add_argument("--resume", action="store_true")
    train_parser.set_defaults(func=train_command)

    test_parser = subparsers.add_parser(
        "test", help="显式读取封存test，并且只在与训练相同的域评估。"
    )
    test_parser.add_argument("--run-dir", type=Path, required=True)
    test_parser.add_argument("--workers", type=int, default=4)
    test_parser.add_argument("--device", default=None)
    test_parser.set_defaults(func=test_command)

    compare_parser = subparsers.add_parser("compare", help="合并多个实验结果。")
    compare_parser.add_argument("--split", choices=("val", "test"), default="val")
    compare_parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    compare_parser.set_defaults(func=compare_command)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
