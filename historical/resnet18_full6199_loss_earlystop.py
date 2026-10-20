"""Full-source loss-based stopping; external 373 images never select weights."""
from __future__ import annotations

import csv
import gc
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import run_resnet18_full6199_external373 as base


def seed_worker(worker_id: int) -> None:
    """Same reference seeding, but importable by Windows spawned workers."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


@dataclass
class LossStopping:
    patience: int = 15
    min_delta: float = 1e-4
    reference_loss: float = float("inf")
    minimum_loss: float = float("inf")
    best_epoch: int = 0
    bad_epochs: int = 0

    def update(self, epoch: int, loss: float) -> tuple[bool, bool]:
        if not math.isfinite(loss):
            raise ValueError(f"Non-finite training loss at epoch {epoch}: {loss}")
        # Keep the absolute minimum even if its decrease is smaller than min_delta.
        new_minimum = loss < self.minimum_loss
        if new_minimum:
            self.minimum_loss, self.best_epoch = loss, epoch
        if loss < self.reference_loss - self.min_delta:
            self.reference_loss, self.bad_epochs = loss, 0
        else:
            self.bad_epochs += 1
        return new_minimum, self.bad_epochs >= self.patience


def protocol_for(args, old_protocol: dict) -> dict:
    protocol = dict(old_protocol)
    # Historical 19/23 epochs and validation-derived LR sequences are NOT used.
    protocol.pop("source_only_selection")
    protocol.update({
        "protocol_version": "full6199_training_loss_earlystop_v1",
        "selection": {
            "monitor": "sample-count-weighted mean online training cross-entropy over all 6199 augmented samples",
            "mode": "min", "max_epochs": args.epochs, "patience": args.patience,
            "min_delta": args.min_delta, "checkpoint": "absolute minimum training loss",
            "external_used": False, "source_validation_used": False,
        },
        "scheduler": {"name": "ReduceLROnPlateau", "mode": "min", "factor": 0.5,
                      "patience": 4, "threshold": args.min_delta, "threshold_mode": "abs", "min_lr": 1e-6},
        "resume_worker_rng_exact": args.workers == 0,
    })
    return protocol


def train_one(args, method: str, records: list[dict], protocol: dict, classifier) -> None:
    run_dir = args.output_dir / method
    run_dir.mkdir(parents=True, exist_ok=True)
    marker_path = run_dir / "training_complete.json"
    if marker_path.is_file():
        marker = base.read_json(marker_path)
        if base.sha256(run_dir / "final.pt") != marker["final_sha256"]:
            raise ValueError("Completed classifier changed.")
        print(f"{method}损失早停训练已完成，复用最低训练损失权重。", flush=True)
        return
    common = protocol["reference_common_training_config"]
    selection = protocol["selection"]
    classifier.set_seed(protocol["seed"])
    device = classifier.resolve_device(args.device)
    use_amp = common["amp"] and device.type == "cuda"
    dataset = base.AllImagesDataset(records, classifier.build_transform(True, common["augmentation_enabled"], common["pretrained"]))
    loader = classifier.make_loader(dataset, protocol["batch_size"], args.workers, True, protocol["seed"], device.type == "cuda")
    model = classifier.build_model(common["pretrained"]).to(device)
    initial_hash = base.state_sha256(model)
    weights, counts = classifier.class_weights(dataset.targets, device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights if common["class_weights_enabled"] else None, label_smoothing=common["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=common["learning_rate"], weight_decay=common["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **{
        name: value for name, value in protocol["scheduler"].items() if name != "name"
    })
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    stopping = LossStopping(args.patience, args.min_delta)
    config = {
        "method": method, "train_samples": len(dataset), "train_class_counts": dict(zip(base.CLASSES, counts)),
        "model": "resnet18", "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "initial_model_sha256": initial_hash, "pretrained": common["pretrained"], "seed": protocol["seed"],
        "selection": selection, "scheduler": protocol["scheduler"],
        "batch_size": protocol["batch_size"], "workers": args.workers, "device": str(device), "amp": use_amp,
        "learning_rate": common["learning_rate"], "weight_decay": common["weight_decay"],
        "label_smoothing": common["label_smoothing"], "class_weights": weights.cpu().tolist(),
        "augmentation_enabled": common["augmentation_enabled"], "external_used_for_training_or_selection": False,
        "torch_version": torch.__version__,
    }
    config_path = run_dir / "config.json"
    if config_path.exists() and base.read_json(config_path) != config:
        raise ValueError("Saved training configuration differs; choose a new output directory.")
    base.write_json(config_path, config)
    start, history = 1, []
    last_path, best_path = run_dir / "last.pt", run_dir / "best_train_loss.pt"
    if last_path.is_file():
        last = torch.load(last_path, map_location="cpu", weights_only=False)
        if last["config"] != config:
            raise ValueError("Resume checkpoint configuration differs.")
        model.load_state_dict(last["model_state_dict"])
        optimizer.load_state_dict(last["optimizer_state_dict"])
        scheduler.load_state_dict(last["scheduler_state_dict"])
        scaler.load_state_dict(last["scaler_state_dict"])
        stopping = LossStopping(**last["stopping"])
        start, history = int(last["epoch"]) + 1, last["history"]
        random.setstate(last["python_rng"])
        np.random.set_state(last["numpy_rng"])
        torch.set_rng_state(last["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(last["cuda_rng"])
        loader.generator.set_state(last["loader_rng"])
        del last
        print(f"{method}断点：从epoch {start}继续；最佳epoch={stopping.best_epoch}，等待={stopping.bad_epochs}/{args.patience}。", flush=True)
    print(f"开始{method}：全6199张，最多{args.epochs}轮，按训练损失早停；373张不参与。", flush=True)
    payload = None
    for epoch in range(start, args.epochs + 1):
        if stopping.bad_epochs >= args.patience:
            break
        lr = optimizer.param_groups[0]["lr"]
        began = time.perf_counter()
        loss, accuracy = classifier.train_epoch(model, loader, criterion, optimizer, scaler, device, use_amp)
        elapsed = time.perf_counter() - began
        improved, should_stop = stopping.update(epoch, loss)
        scheduler.step(loss)
        history.append({
            "epoch": epoch, "train_samples": len(dataset), "train_loss": loss, "train_accuracy": accuracy,
            "learning_rate": lr, "next_learning_rate": optimizer.param_groups[0]["lr"],
            "best_train_loss": stopping.minimum_loss, "best_epoch": stopping.best_epoch,
            "bad_epochs": stopping.bad_epochs, "elapsed_seconds": elapsed,
        })
        if improved:
            base.atomic_torch_save({"model_state_dict": model.state_dict(), "config": config,
                                    "classes": list(base.CLASSES), "epoch": epoch,
                                    "train_loss": loss}, best_path)
        payload = {
            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(), "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch, "history": history, "config": config, "classes": list(base.CLASSES),
            "stopping": dict(vars(stopping)), "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "loader_rng": loader.generator.get_state(),
        }
        base.atomic_torch_save(payload, last_path)
        base.write_csv(run_dir / "history.csv", history)
        print(f"{method} epoch {epoch:03d}/{args.epochs} | loss={loss:.6f}, ACC={accuracy:.4f}, lr={lr:.2e}, "
              f"best={stopping.minimum_loss:.6f}@{stopping.best_epoch}, 等待={stopping.bad_epochs}/{args.patience}, {elapsed:.1f}s", flush=True)
        if should_stop:
            print(f"{method}触发训练损失早停：epoch {epoch}。", flush=True)
            break
    if not history or not best_path.is_file():
        raise ValueError("No trained checkpoint available.")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    if best["epoch"] != stopping.best_epoch or best["train_loss"] != stopping.minimum_loss:
        raise ValueError("Best checkpoint and loss history disagree.")
    final_path = run_dir / "final.pt"
    base.atomic_torch_save(best, final_path)
    epochs_completed = history[-1]["epoch"]
    stop_reason = "training_loss_patience" if stopping.bad_epochs >= args.patience else "max_epochs"
    base.write_json(marker_path, {
        "completed_at": datetime.now().isoformat(timespec="seconds"), "train_samples": len(dataset),
        "epochs_completed": epochs_completed, "best_epoch": stopping.best_epoch,
        "best_train_loss": stopping.minimum_loss, "stop_reason": stop_reason,
        "bad_epochs": stopping.bad_epochs, "max_epochs": args.epochs,
        "external_used_for_training_or_selection": False, "final_sha256": base.sha256(final_path),
        "history_sha256": base.sha256(run_dir / "history.csv"),
    })
    print(f"{method}完成：训练{epochs_completed}轮，保留epoch {stopping.best_epoch}，loss={stopping.minimum_loss:.6f}。", flush=True)
    del model, optimizer, scheduler, scaler, loader, dataset, payload, best
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def freeze_training(root: Path, protocol: dict) -> dict:
    models = {}
    for method in base.METHODS:
        run_dir = root / method
        marker = base.read_json(run_dir / "training_complete.json")
        config = base.read_json(run_dir / "config.json")
        if marker["train_samples"] != 6199 or not 1 <= marker["best_epoch"] <= marker["epochs_completed"] <= protocol["selection"]["max_epochs"]:
            raise ValueError("Invalid loss-early-stop training completion.")
        if marker["stop_reason"] == "max_epochs":
            if marker["epochs_completed"] != protocol["selection"]["max_epochs"]:
                raise ValueError("Training terminated before maximum epochs without early stopping.")
        elif marker["stop_reason"] == "training_loss_patience":
            if marker["bad_epochs"] < protocol["selection"]["patience"]:
                raise ValueError("Training did not meet patience threshold.")
        else:
            raise ValueError("Unknown stopping reason.")
        if base.sha256(run_dir / "final.pt") != marker["final_sha256"] or base.sha256(run_dir / "history.csv") != marker["history_sha256"]:
            raise ValueError("Frozen weights or training history changed.")
        with (run_dir / "history.csv").open(encoding="utf-8-sig", newline="") as handle:
            history = list(csv.DictReader(handle))
        audit = LossStopping(protocol["selection"]["patience"], protocol["selection"]["min_delta"])
        if len(history) != marker["epochs_completed"]:
            raise ValueError("Training history length differs from completion marker.")
        for epoch, row in enumerate(history, 1):
            if int(row["epoch"]) != epoch or int(row["train_samples"]) != 6199:
                raise ValueError("Training history is not contiguous or did not use all 6199.")
            _, should_stop = audit.update(epoch, float(row["train_loss"]))
            if should_stop and epoch != len(history):
                raise ValueError("Training continued after the declared early-stop threshold.")
            if int(row["bad_epochs"]) != audit.bad_epochs or int(row["best_epoch"]) != audit.best_epoch:
                raise ValueError("Training history early-stop counters are inconsistent.")
        if audit.best_epoch != marker["best_epoch"] or audit.minimum_loss != marker["best_train_loss"] or audit.bad_epochs != marker["bad_epochs"]:
            raise ValueError("Selected weights do not correspond to the loss-based stopping history.")
        models[method] = {"final_sha256": marker["final_sha256"], "config_sha256": base.sha256(run_dir / "config.json"),
                          "initial_model_sha256": config["initial_model_sha256"], "best_epoch": marker["best_epoch"]}
    if models["original"]["initial_model_sha256"] != models["remapped"]["initial_model_sha256"]:
        raise ValueError("The two classifiers did not start from identical weights.")
    frozen = {"models": models, "both_training_complete_before_external_evaluation": True, "external_samples": 373,
              "selection": protocol["selection"]}
    path = root / "training_frozen.json"
    if path.exists():
        if base.read_json(path) != frozen:
            raise ValueError("Frozen model identities changed.")
    else:
        base.write_json(path, frozen)
    return frozen


def report(args, manifest: dict, protocol: dict, classifier) -> None:
    root = args.output_dir
    rows, details, correctness, training = [], {}, {}, {}
    for method in base.METHODS:
        metrics, prediction = base.audited_metrics(root, method, manifest["external"][method], classifier)
        marker = base.read_json(root / method / "training_complete.json")
        training[method] = marker
        details[method], correctness[method] = metrics, prediction
        row = {"method": method, "train_samples": 6199, "test_samples": 373,
               "training_epochs": marker["epochs_completed"], "selected_epoch": marker["best_epoch"],
               "best_train_loss": marker["best_train_loss"], "stop_reason": marker["stop_reason"],
               **{name: metrics[name] for name in base.METRIC_NAMES}}
        for index, name in enumerate(base.CLASSES):
            row[f"support_{name}"] = base.CLASS_COUNTS_EXTERNAL[name]
            for metric in ("precision", "recall", "f1"):
                row[f"{metric}_{name}"] = metrics[metric][index]
        rows.append(row)
    keys = set(correctness["original"])
    if keys != set(correctness["remapped"]):
        raise ValueError("The two external sets differ.")
    paired = {
        "both_correct": sum(correctness["original"][key] and correctness["remapped"][key] for key in keys),
        "original_only_correct": sum(correctness["original"][key] and not correctness["remapped"][key] for key in keys),
        "remapped_only_correct": sum(not correctness["original"][key] and correctness["remapped"][key] for key in keys),
        "both_wrong": sum(not correctness["original"][key] and not correctness["remapped"][key] for key in keys),
    }
    base.write_csv(root / "external_comparison.csv", rows)
    base.write_json(root / "external_comparison.json", {"metrics": rows, "full_metrics": details,
                    "training": training, "selection": protocol["selection"], "paired_comparison": paired})
    lines = [
        "# 全6199训练、训练损失早停 → 全373外部测试", "",
        "## 数据与模型", "",
        "原图组：全6199张旧原图训练ResNet18，测试全373张TTBM原图。再映射组：全6199张旧Stage2再映射图训练另一个同架构ResNet18，测试全373张TTBM对应再映射图。旧、新数据目录中的train/val/test均合并，分别只用于训练、测试；本次不再划8:1:1。",
        "训练类别数defect/normal/nut=1244/3018/1937，外部测试=122/66/185（244张源照片）。所有图RGB、64×64，两组键和标签严格配对，源训练/外部测试原图完全相同RGB像素交集为0，但不保证排除近重复照片。",
        "再映射沿用同一冻结Stage2、保存的seed43推理噪声：t=1加噪→公式更新→t=0再映射。保存的是固定推理噪声，不是已归档的训练阶段逐样本噪声；历史训练噪声未保存。生成器没有在373张上重新训练，类别标签不输入生成器。",
        "", "## 训练、早停与选模", "",
        "两组从相同随机初始参数重新训练，不加载旧19/23轮分类器或曾在373张上训练的分类器。ResNet18 + Dropout(0.30) + Linear(512,3)，seed43，batch64；AdamW初始lr=0.001、weight_decay=0.0001，训练增广和归一化沿用原程序。按全6199标签计算类别权重，交叉熵label_smoothing=0.05。",
        f"最多{args.epochs}轮；监测每轮全6199样本训练时的平均分类交叉熵（有增广、类别权重、label smoothing）。连续{args.patience}轮未比早停参考最优损失降低超过{args.min_delta:g}则早停；微小下降累积超过阈值后可以重置计数。",
        "每次出现绝对更低的训练损失都会保存最佳权重，最终final.pt为最低训练损失那一轮，而非最后一轮。学习率ReduceLROnPlateau同样监测训练损失，mode=min、factor=0.5、patience=4、min_lr=1e-6。两组规则完全相同，不使用历史验证集选定的19/23轮或学习率日程。",
        "373张只用于两组训练完成、权重冻结后的外部测试，不参与梯度、早停、学习率调整或选模。为了全6199训练，本次没有验证集；训练损失早停不能保证泛化最优，也不能替代验证损失早停。",
        "", "| 方法 | 实际训练轮数 | 最低损失轮数 | 最低训练损失 | 停止原因 |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        label = "原图基线" if row["method"] == "original" else "Stage2再映射"
        reason = "训练损失早停" if row["stop_reason"] == "training_loss_patience" else "达到最大轮数"
        lines.append(f"| {label} | {row['training_epochs']} | {row['selected_epoch']} | {row['best_train_loss']:.6f} | {reason} |")
    lines += ["", "## 全373张外部测试指标", "", "P/R/F1同时报告weighted（真实类样本数加权）和macro（各类等权）；weighted-R=ACC。", "",
              "| 方法 | ACC | weighted-P | weighted-R | weighted-F1 | macro-P | macro-R | macro-F1 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        label = "原图基线" if row["method"] == "original" else "Stage2再映射"
        lines.append(f"| {label} | " + " | ".join(f"{row[name] * 100:.2f}%" for name in base.METRIC_NAMES) + " |")
        print(f"{label}外部373：ACC={row['accuracy'] * 100:.2f}%, P/R/F1={row['weighted_precision'] * 100:.2f}%/"
              f"{row['weighted_recall'] * 100:.2f}%/{row['weighted_f1'] * 100:.2f}%, macro-F1={row['macro_f1'] * 100:.2f}%", flush=True)
    lines += ["", "## 逐类别外部测试", "", "| 方法 | 类别 | 数量 | P | R | F1 |", "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        label = "原图基线" if row["method"] == "original" else "Stage2再映射"
        for name in base.CLASSES:
            lines.append(f"| {label} | {name} | {row[f'support_{name}']} | " + " | ".join(f"{row[f'{metric}_{name}'] * 100:.2f}%" for metric in ("precision", "recall", "f1")) + " |")
    delta = {name: (details["remapped"][name] - details["original"][name]) * 100 for name in base.METRIC_NAMES}
    lines += ["", "## 对照与限制", "",
              f"再映射相对原图：ACC变化{delta['accuracy']:+.2f}个百分点，weighted-F1变化{delta['weighted_f1']:+.2f}个百分点，macro-F1变化{delta['macro_f1']:+.2f}个百分点。",
              f"两组都正确{paired['both_correct']}张，仅原图正确{paired['original_only_correct']}张，仅再映射正确{paired['remapped_only_correct']}张，两组都错误{paired['both_wrong']}张。",
              "旧固定19/23轮结果单独保留，不能混为本次早停结果。373张此前已经用于局部训练及图像相似度探索，不能称为从未查看过的最终独立留出集；本次两组分类器从头训练且不据外部结果调参。单次种子无法说明统计显著性，多个目标框来自同一照片，不能视为373张独立照片。类别不均衡需结合macro与逐类指标。",
              "", "## 结果文件与运行", "",
              f"- 原图权重：[original/final.pt]({(root / 'original/final.pt').as_posix()})",
              f"- 再映射权重：[remapped/final.pt]({(root / 'remapped/final.pt').as_posix()})",
              f"- 总指标：[external_comparison.csv]({(root / 'external_comparison.csv').as_posix()})",
              "- 两组history.csv记录每轮训练损失、ACC、实际/下一轮学习率、最佳轮数和早停计数；last.pt是恢复断点，best_train_loss.pt/final.pt是最低损失权重。",
              "", "```powershell", "cd D:\\Cold_Diffusion",
              "conda run --no-capture-output -n diffusion_env python -X utf8 -u run_resnet18_full6199_external373.py",
              "```", "", "默认训练损失早停，--epochs 100 --patience 15 --min-delta 0.0001；默认workers4加速数据读取。相同命令自动续训或复用已完成结果，仅重建报告加--stage report。断点保留模型、优化器、学习率调度、早停计数及主进程随机状态；多进程增广的工作进程随机状态不保存，跨中断续训不保证逐位复现，需严格连续随机流时使用--workers 0并另选目录。--stopping fixed用于复现旧固定19/23轮协议，写入旧目录。"]
    (root / "EXTERNAL_TEST_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args) -> None:
    if args.epochs < 1 or args.patience < 1 or args.min_delta < 0 or not math.isfinite(args.min_delta):
        raise ValueError("epochs/patience must be positive; min_delta must be finite and nonnegative.")
    manifest, old_protocol, _ = base.preflight(args)
    protocol = protocol_for(args, old_protocol)
    print(f"本次两组统一：最多{args.epochs}轮；训练损失早停patience={args.patience}, min_delta={args.min_delta:g}；不采用旧19/23轮。", flush=True)
    if args.stage == "preflight":
        return
    root = args.output_dir
    path = root / "experiment_protocol.json"
    if path.exists():
        if base.read_json(path) != protocol:
            raise ValueError("Existing protocol differs; select a new output directory.")
    elif root.exists() and any(root.iterdir()):
        raise FileExistsError("Refusing to reuse unrecognized nonempty output directory.")
    root.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        base.write_json(path, protocol)
        base.write_json(root / "input_manifest.json", manifest)
    classifier = base.load_classifier(args.classifier_script)
    classifier.seed_worker = seed_worker
    if args.stage in ("all", "train"):
        for method in base.METHODS:
            train_one(args, method, manifest["source"][method], protocol, classifier)
    frozen = freeze_training(root, protocol)
    if args.stage in ("all", "test"):
        for method in base.METHODS:
            base.evaluate_one(args, method, manifest["external"][method], frozen, classifier)
    if args.stage in ("all", "test", "report"):
        report(args, manifest, protocol, classifier)
    if base.sha256(Path(protocol["generator"])) != protocol["generator_sha256"]:
        raise ValueError("Generator file changed.")
    print(f"实验目录：{root}", flush=True)
