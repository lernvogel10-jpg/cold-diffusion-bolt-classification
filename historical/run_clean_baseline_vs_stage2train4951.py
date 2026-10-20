"""Re-train only the raw-image baseline; merge the frozen validation-loss comparison."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

import run_stage2_train4951_remapped_resnet18 as pipeline
from diffusion_stripe.training import save_json
from resnet18_full6199_loss_earlystop import seed_worker
from train_stage2_train4951 import WORKSPACE, read_json, sha256


REMAPPED_ROOT = pipeline.DEFAULT_ROOT
REMAPPED_RUN = REMAPPED_ROOT / "runs/resnet18_scratch_val_loss"
DEFAULT_ROOT = WORKSPACE / "experiments/resnet18_original_vs_stage2train4951_t1_val_loss_seed43"
SHARED_KEYS = (
    "pretrained", "seed", "learning_rate", "weight_decay", "label_smoothing",
    "class_weights_enabled", "augmentation_enabled", "amp", "model", "parameters",
    "batch_size", "workers", "device", "max_epochs", "patience", "min_delta", "selection",
    "train_samples", "val_samples", "test_samples", "initial_model_sha256",
    "classifier_script_sha256", "initialize_from_old_classifier", "test_used_for_selection",
)


def require_same_config(original: dict, remapped: dict) -> None:
    mismatches = {name: (original.get(name), remapped.get(name)) for name in SHARED_KEYS if original.get(name) != remapped.get(name)}
    if mismatches:
        raise ValueError(f"Unmatched classifier controls: {mismatches}")


def audit_history(run: Path, config: dict) -> dict:
    complete = read_json(run / "training_complete.json")
    if sha256(run / "history.csv") != complete["history_sha256"] or sha256(run / "best.pt") != complete["best_sha256"]:
        raise ValueError("Classifier training history or selected weights changed.")
    with (run / "history.csv").open(encoding="utf-8-sig", newline="") as handle:
        history = list(csv.DictReader(handle))
    state = pipeline.Selection(config["selection"], config["patience"], config["min_delta"])
    if len(history) != complete["epochs_completed"] or len(history) > config["max_epochs"]:
        raise ValueError("Invalid classifier training duration.")
    for epoch, row in enumerate(history, 1):
        if int(row["epoch"]) != epoch or int(row["train_samples"]) != 4951 or int(row["val_samples"]) != 619:
            raise ValueError("Noncontiguous classifier history or incorrect sample count.")
        _, stop = state.update(epoch, float(row["val_loss"]))
        if stop and epoch != len(history):
            raise ValueError("Classifier continued training after patience threshold.")
        if int(row["best_epoch"]) != state.epoch or int(row["bad_epochs"]) != state.waiting:
            raise ValueError("Validation-loss selection counters do not match history.")
    if state.epoch != complete["selected_epoch"] or state.selected != complete["selected_score"]:
        raise ValueError("Selected classifier checkpoint does not match the minimum validation loss.")
    if state.waiting < config["patience"] and len(history) != config["max_epochs"]:
        raise ValueError("Classifier ended before either early stopping or the epoch cap.")
    return complete


def prediction_correctness(run: Path, split: str) -> dict[str, bool]:
    root = Path(read_json(run / "config.json")["dataset"])
    with (run / f"{split}_predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {Path(row["image"]).resolve().relative_to(root).as_posix(): int(row["true_id"]) == int(row["predicted_id"]) for row in rows}


def preserve_check(protocol: dict) -> None:
    for path, expected in protocol["preserved_reference_files_sha256"].items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Preserved old or remapped result changed: {path}")


def preflight(args, classifier) -> tuple[dict, dict]:
    reference = read_json(REMAPPED_RUN / "config.json")
    if reference["selection"] != "val_loss" or reference["max_epochs"] != 100 or reference["patience"] != 15:
        raise ValueError("Expected the completed new-remap validation-loss ResNet18 experiment.")
    for attribute, field in (("epochs", "max_epochs"), ("patience", "patience"), ("min_delta", "min_delta"),
                             ("batch_size", "batch_size"), ("workers", "workers"), ("device", "device"), ("selection", "selection")):
        setattr(args, attribute, reference[field])
    args.run_dir = args.output_dir / "runs/resnet18_scratch_original_val_loss"
    generation_protocol, manifest = pipeline.preflight(args)
    if generation_protocol != read_json(REMAPPED_ROOT / "generation_protocol.json"):
        raise ValueError("The original data or remapped generation protocol changed.")
    pipeline.verify_generated(argparse.Namespace(output_dir=REMAPPED_ROOT), manifest,
                              read_json(REMAPPED_ROOT / "generation_complete.json"))
    original_root = Path(generation_protocol["original_root"])
    classifier.set_seed(reference["seed"])
    model = classifier.build_model(reference["pretrained"])
    raw_config = pipeline.classifier_config(args, generation_protocol, model, classifier, data_root=original_root)
    require_same_config(raw_config, reference)
    del model
    old_raw = pipeline.common.REFERENCE_RUNS / pipeline.common.REFERENCE_TAGS["original"]
    protected = {str(path.resolve()): sha256(path) for run in (old_raw, REMAPPED_RUN) for path in sorted(run.iterdir()) if path.is_file()}
    stage2 = read_json(pipeline.NEW_STAGE2 / "experiment_protocol.json")
    protected.update(stage2["preserved_old_files_sha256"])
    protected[generation_protocol["checkpoint"]] = generation_protocol["checkpoint_sha256"]
    protocol = {
        "protocol_version": "original_vs_stage2train4951_same_validation_loss_v1",
        "original_dataset": str(original_root), "remapped_dataset": reference["dataset"],
        "original_run": str(args.run_dir), "remapped_run": str(REMAPPED_RUN),
        "shared_classifier_config": {key: reference[key] for key in SHARED_KEYS},
        "original_raw_baseline_retrained_from_scratch": True, "remapped_results_reused_without_training_or_retesting": True,
        "original_input_manifest_sha256": generation_protocol["input_manifest_sha256"],
        "remapped_generation_complete_sha256": sha256(REMAPPED_ROOT / "generation_complete.json"),
        "preserved_reference_files_sha256": protected,
        "split_class_counts": pipeline.EXPECTED_COUNTS,
        "scheduler": {"name": "ReduceLROnPlateau", "mode": "min", "factor": 0.5, "patience": 4,
                      "threshold": args.min_delta, "threshold_mode": "abs", "min_lr": 1e-6},
        "generator_limitation": generation_protocol["limitation"],
    }
    audit_history(REMAPPED_RUN, reference)
    for split in ("val", "test"):
        pipeline.audited_metrics(argparse.Namespace(run_dir=REMAPPED_RUN), split, classifier)
    print("预检通过：两组相同初始参数、超参数、划分及验证损失选模；仅重训原图基线。", flush=True)
    return protocol, generation_protocol


def comparison(args, protocol: dict, classifier) -> None:
    preserve_check(protocol)
    raw_config = read_json(args.run_dir / "config.json")
    remapped_config = read_json(REMAPPED_RUN / "config.json")
    require_same_config(raw_config, remapped_config)
    training = {"original": audit_history(args.run_dir, raw_config), "remapped": audit_history(REMAPPED_RUN, remapped_config)}
    rows, details, paired = [], {}, {}
    for split in ("val", "test"):
        details[split] = {}
        correct = {}
        for method, run in (("original", args.run_dir), ("remapped", REMAPPED_RUN)):
            metrics = pipeline.audited_metrics(argparse.Namespace(run_dir=run), split, classifier)
            details[split][method] = metrics
            row = {"method": method, "split": split, "samples": metrics["samples"],
                   "training_epochs": training[method]["epochs_completed"], "selected_epoch": training[method]["selected_epoch"],
                   "loss": metrics["loss"], **{name: metrics[name] for name in pipeline.common.METRIC_NAMES}}
            for i, name in enumerate(pipeline.CLASSES):
                row[f"support_{name}"] = pipeline.EXPECTED_COUNTS[split][name]
                row.update({f"{metric}_{name}": metrics[metric][i] for metric in ("precision", "recall", "f1")})
            rows.append(row)
            correct[method] = prediction_correctness(run, split)
        keys = set(correct["original"])
        if keys != set(correct["remapped"]):
            raise ValueError("Original and remapped predictions do not refer to the same samples.")
        paired[split] = {
            "both_correct": sum(correct["original"][key] and correct["remapped"][key] for key in keys),
            "original_only_correct": sum(correct["original"][key] and not correct["remapped"][key] for key in keys),
            "remapped_only_correct": sum(not correct["original"][key] and correct["remapped"][key] for key in keys),
            "both_wrong": sum(not correct["original"][key] and not correct["remapped"][key] for key in keys),
        }
    pipeline.common.write_csv(args.output_dir / "original_vs_remapped_metrics.csv", rows)
    save_json({"metrics": rows, "full_metrics": details, "training": training, "paired_comparison": paired,
               "classifier_controls_matched": True}, args.output_dir / "original_vs_remapped_metrics.json")
    lines = ["# 原图与新Stage2(train4951)再映射：相同验证损失选模的ResNet18对照", "",
             "## 对照设置", "",
             "原图基线重新随机初始化训练，不加载旧macro-F1选模基线；新Stage2再映射分类器复用已冻结的权重和结果，没有重新训练或重复测试。两组初始参数SHA256相同，输入图像及处理方式不同，其余分类器设置一致。", "",
             "原split43不变，train4951、val619、test629，类序defect/normal/nut；train类数990/2417/1544、val125/295/199、test129/306/194，同源组不跨集合。原图与再映射图样本键及标签严格配对。", "",
             "同架构ResNet18 + Dropout(0.30) + Linear(512,3)，随机初始化、seed43，batch64、workers4、AMP；AdamW lr=0.001、weight_decay=0.0001，训练交叉熵label_smoothing=0.05，类别权重仅由train4951计算；增广、归一化沿用旧程序，val/test不增广。", "",
             "两组都最多100 epoch，以未加权验证交叉熵损失最小化选模/早停，patience15、min_delta=0.0001，保留绝对最低验证损失的权重。学习率ReduceLROnPlateau监测验证损失，mode=min、factor0.5、patience4、threshold=0.0001(abs)、min_lr=1e-6。测试集不参与梯度、早停、学习率调整或选模。", "",
             "| 方法 | 实际训练轮数 | 选中轮数 | 最低验证损失 | 停止原因 |", "|---|---:|---:|---:|---|"]
    for method in ("original", "remapped"):
        record = training[method]
        label = "原图（重训基线）" if method == "original" else "新Stage2再映射"
        reason = "验证损失早停" if record["stop_reason"] == "validation_patience" else "最大轮数"
        lines.append(f"| {label} | {record['epochs_completed']} | {record['selected_epoch']} | {record['selected_score']:.6f} | {reason} |")
    lines += ["", "## 验证集与测试集总体指标", "", "P/R/F1报告weighted及macro；weighted-R=ACC。", "",
              "| 集合 | 方法 | 数量 | ACC | weighted-P | weighted-R | weighted-F1 | macro-P | macro-R | macro-F1 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        label = "原图（重训）" if row["method"] == "original" else "新Stage2再映射"
        lines.append(f"| {row['split']} | {label} | {row['samples']} | " + " | ".join(f"{row[name]*100:.2f}%" for name in pipeline.common.METRIC_NAMES) + " |")
        print(f"{row['split']} {label}：ACC={row['accuracy']*100:.2f}%, P/R/F1={row['weighted_precision']*100:.2f}%/"
              f"{row['weighted_recall']*100:.2f}%/{row['weighted_f1']*100:.2f}%, macro-F1={row['macro_f1']*100:.2f}%", flush=True)
    lines += ["", "## 再映射减原图：百分点差值", "", "| 集合 | ΔACC | Δweighted-P | Δweighted-R | Δweighted-F1 | Δmacro-F1 |", "|---|---:|---:|---:|---:|---:|"]
    for split in ("val", "test"):
        delta = {name: (details[split]["remapped"][name]-details[split]["original"][name])*100 for name in pipeline.common.METRIC_NAMES}
        lines.append(f"| {split} | " + " | ".join(f"{delta[name]:+.2f}" for name in ("accuracy", "weighted_precision", "weighted_recall", "weighted_f1", "macro_f1")) + " |")
    lines += ["", "## 逐类别指标", "", "| 集合 | 方法 | 类别 | 数量 | P | R | F1 |", "|---|---|---|---:|---:|---:|---:|"]
    for row in rows:
        label = "原图（重训）" if row["method"] == "original" else "新Stage2再映射"
        for name in pipeline.CLASSES:
            lines.append(f"| {row['split']} | {label} | {name} | {row[f'support_{name}']} | " + " | ".join(f"{row[f'{metric}_{name}']*100:.2f}%" for metric in ("precision", "recall", "f1")) + " |")
    test_raw, test_remapped = details["test"]["original"], details["test"]["remapped"]
    defect_index = pipeline.CLASSES.index("defect")
    lines += ["", "## 结果解读", "",
              f"在此次相同验证损失选模协议下，再映射测试ACC提高{(test_remapped['accuracy']-test_raw['accuracy'])*100:.2f}个百分点，"
              f"weighted-F1提高{(test_remapped['weighted_f1']-test_raw['weighted_f1'])*100:.2f}个百分点，"
              f"macro-F1提高{(test_remapped['macro_f1']-test_raw['macro_f1'])*100:.2f}个百分点。", "",
              f"但并非所有类别均改善：测试defect召回率从{test_raw['recall'][defect_index]*100:.2f}%降至"
              f"{test_remapped['recall'][defect_index]*100:.2f}%，F1从{test_raw['f1'][defect_index]*100:.2f}%降至"
              f"{test_remapped['f1'][defect_index]*100:.2f}%。整体指标的改善不能解读为缺陷检出能力也提高。", "",
              "本次最低验证损失权重不是按最高ACC或macro-F1挑选，因此原图基线数值不必与旧macro-F1选模实验相同。"]
    lines += ["", "## 配对预测与解释边界", ""]
    for split in ("val", "test"):
        record = paired[split]
        lines.append(f"{split}：两组均正确{record['both_correct']}张，仅原图正确{record['original_only_correct']}张，仅再映射正确{record['remapped_only_correct']}张，两组均错误{record['both_wrong']}张。")
        lines.append("")
    lines += ["本次已统一分类器的训练/选模条件，但两组实际早停轮数可以不同。单次seed43结果不能证明稳定或统计显著提升；多个裁剪来自同一源照片，并非完全独立样本。", "",
              "新Stage2只用train4951微调20轮；再映射沿用固定保存seed43噪声、t=1→公式更新→R(x0,0)。然而旧Stage1曾在全6199条纹图上训练，所以统一分类器规则不等于解决整个生成器的验证/测试数据暴露。不能称为全流程从未见过测试数据的泛化对照。", "",
              "旧macro-F1选模原图基线（val68.34%、test72.66%）保留，仅作历史记录，不与本次val_loss基线混用。旧Stage2、新Stage2和已完成的新再映射分类器及结果均校验SHA256未改变。", "",
              "## 文件与复现", "",
              f"- 重训原图权重：[best.pt]({(args.run_dir/'best.pt').as_posix()})",
              f"- 新再映射权重：[best.pt]({(REMAPPED_RUN/'best.pt').as_posix()})",
              f"- 合并指标：[original_vs_remapped_metrics.csv]({(args.output_dir/'original_vs_remapped_metrics.csv').as_posix()})",
              "- 各组逐图预测、混淆矩阵与训练曲线保留在各自run目录；合并指标由逐图预测重新计算并核验。", "",
              "```powershell", "cd <repository-root>",
              "conda run --no-capture-output -n diffusion_env python -X utf8 -u run_clean_baseline_vs_stage2train4951.py",
              "```", "", "相同命令核验并复用完成结果或断点续训；--stage report仅合并已有结果。多进程增广工作进程RNG不保存，跨中断续训不保证逐位复现。"]
    (args.output_dir / "ORIGINAL_VS_NEW_STAGE2_REMAPPED_REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--stage", choices=("all", "preflight", "train", "test", "report"), default="all")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    classifier = pipeline.common.load_classifier(pipeline.common.DEFAULT_CLASSIFIER_SCRIPT)
    classifier.seed_worker = seed_worker
    protocol, generation = preflight(args, classifier)
    if args.stage == "preflight":
        return
    path = args.output_dir / "comparison_protocol.json"
    if path.exists():
        if read_json(path) != protocol:
            raise ValueError("Existing comparison protocol differs; choose a new output directory.")
    elif args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite an unrecognized nonempty comparison directory.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        save_json(protocol, path)
    if args.stage in ("all", "train"):
        pipeline.train_classifier(args, generation, classifier, data_root=Path(protocol["original_dataset"]))
    if args.stage in ("all", "train", "test"):
        pipeline.evaluate_split(args, "val", classifier)
    if args.stage in ("all", "test"):
        pipeline.evaluate_split(args, "test", classifier)
    if args.stage in ("all", "test", "report"):
        comparison(args, protocol, classifier)
    preserve_check(protocol)
    print(f"统一选模的对照目录：{args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
