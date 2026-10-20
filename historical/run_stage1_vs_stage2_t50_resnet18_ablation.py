"""Matched ResNet18 ablation on the two verified t50 image sets.

Train only the missing Stage1-only classifier. The Stage1+Stage2 images are
byte-identical to an existing completed experiment, whose sealed results are
read-only reused. Never retrain a generator or edit the manuscript/master MD.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

import run_stage2_train4951_remapped_resnet18 as pipeline
import run_clean_baseline_vs_stage2train4951 as controls
from diffusion_stripe.data import collect_images
from diffusion_stripe.training import save_json
from resnet18_full6199_loss_earlystop import seed_worker
from train_stage2_train4951 import read_json, sha256


WORKSPACE = Path(__file__).resolve().parent
IMAGES = WORKSPACE / "experiments/stage1_vs_stage2_train4951_t50_similarity_20261016_v1"
STAGE1_DATA = IMAGES / "images/stage1_only"
STAGE2_DATA = IMAGES / "images/stage1_plus_stage2"
REFERENCE_ROOT = WORKSPACE / "experiments/stage2_train4951_t50_remapped_resnet18_seed43"
REFERENCE_RUN = REFERENCE_ROOT / "runs/resnet18_scratch_val_loss"
DEFAULT_OUTPUT = WORKSPACE / "experiments/resnet18_stage1_vs_stage2_t50_val_loss_seed43_20261016_v1"
STAGE1_CHECKPOINT = WORKSPACE / "experiments/full6199_two_stage/stage1/best_train_loss.pt"
BASE_CONFIG = pipeline.classifier_config


def stage1_classifier_config(args, protocol, model, classifier, *, data_root=None):
    config = BASE_CONFIG(args, protocol, model, classifier, data_root=STAGE1_DATA)
    config.update(input_method="Stage 1 only, t50 recursion + t0 remapping",
                  generator_sha256=protocol["checkpoint_sha256"])
    return config


def verify_images(root, expected):
    records = collect_images(root, expected_count=6199)
    assert {r.key for r in records} == set(expected)
    with ThreadPoolExecutor(max_workers=8) as executor:
        hashes = list(executor.map(sha256, [r.path for r in records]))
    assert dict(zip((r.key for r in records), hashes)) == expected, f"Changed images: {root}"
    for split, counts in pipeline.EXPECTED_COUNTS.items():
        actual = {name: sum(r.key.startswith(f"{split}/{name}/") for r in records) for name in pipeline.CLASSES}
        assert actual == counts


def preflight(args, classifier):
    reference = read_json(REFERENCE_RUN / "config.json")
    assert reference["selection"] == "val_loss" and reference["max_epochs"] == 100
    for attribute, field in (("epochs", "max_epochs"), ("patience", "patience"), ("min_delta", "min_delta"),
                             ("batch_size", "batch_size"), ("workers", "workers"), ("device", "device"),
                             ("selection", "selection")):
        setattr(args, attribute, reference[field])
    args.run_dir = args.output_dir / "runs/resnet18_stage1_only_t50_val_loss"
    complete = read_json(IMAGES / "complete.json")
    assert complete["count_per_model"] == 6199 and complete["frozen_weights_unchanged"]
    verify_images(STAGE1_DATA, complete["output_sha256"]["stage1_only"])
    verify_images(STAGE2_DATA, complete["output_sha256"]["stage1_plus_stage2"])
    historical = read_json(REFERENCE_ROOT / "generation_complete.json")
    assert complete["output_sha256"]["stage1_plus_stage2"] == historical["output_sha256"]
    verify_images(Path(reference["dataset"]), historical["output_sha256"])
    source = read_json(IMAGES / "protocol.json")
    source_hashes = source["weights_and_noise_sha256"]
    assert sha256(STAGE1_CHECKPOINT) == source_hashes[str(STAGE1_CHECKPOINT)]
    assert source["count"] == 6199 and source["recursion_steps"] == 50
    split_path = Path(r"D:\YOLO26_study\datasets\yolo26_crops64_dataset_source_grouped_split43\reports\crops_manifest.csv")
    with split_path.open(encoding="utf-8-sig", newline="") as handle:
        split_manifest = list(csv.DictReader(handle))
    groups = {s: {r["canonical_source_group"] for r in split_manifest if r["split"] == s} for s in pipeline.SPLITS}
    assert all(not groups[a] & groups[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test")))
    assert {r["crop_path"] for r in split_manifest} == set(complete["output_sha256"]["stage1_only"])
    classifier.set_seed(43)
    model = classifier.build_model(False)
    generation = {"checkpoint_sha256": sha256(STAGE1_CHECKPOINT)}
    config = stage1_classifier_config(args, generation, model, classifier)
    controls.require_same_config(config, reference)
    del model
    controls.audit_history(REFERENCE_RUN, reference)
    for split in ("val", "test"):
        pipeline.audited_metrics(argparse.Namespace(run_dir=REFERENCE_RUN), split, classifier)
    protected = {str(p): sha256(p) for p in sorted(REFERENCE_RUN.iterdir()) if p.is_file()}
    protected.update(source_hashes)
    for p in (IMAGES / "complete.json", IMAGES / "protocol.json", split_path,
              WORKSPACE / "COLD_DIFFUSION_RESNET18_EXPERIMENT_SUMMARY.md",
              Path(pipeline.__file__), pipeline.common.DEFAULT_CLASSIFIER_SCRIPT):
        protected[str(p)] = sha256(p)
    protocol = {
        "version": "stage1_vs_stage2_t50_resnet18_val_loss_seed43_v1",
        "stage1_images": str(STAGE1_DATA), "stage2_images": str(STAGE2_DATA),
        "stage1_run": str(args.run_dir), "stage2_read_only_run": str(REFERENCE_RUN),
        "stage2_all6199_png_sha256_identical_to_current_main_experiment": True,
        "shared_classifier_config": {k: reference[k] for k in controls.SHARED_KEYS},
        "split_class_counts": pipeline.EXPECTED_COUNTS,
        "stage1_classifier_trained_from_scratch": True,
        "stage2_completed_classifier_and_evaluation_reused_without_retraining_or_retesting": True,
        "preserved_reference_files_sha256": protected,
        "selection": "minimum val_loss, patience15, min_delta=.0001, max100 epochs",
        "test_role": "test629 evaluated only after classifier training and checkpoint selection finish",
        "gpu_pacing": "idle-only inter-epoch cooling; no change to RNG, tensor operations, optimizer or data order",
        "limitation": "Both generators inherit Stage1 all6199 class-encoded watermark exposure; not full-pipeline unseen-test generalization; single seed43.",
        "implementation_sha256": sha256(Path(__file__)),
    }
    print("PRECHECK PASS: all image hashes and keys verified; identical classifier initialization/config; Stage2 sealed result reused.", flush=True)
    return protocol, generation


def install_cooling(classifier):
    original_train = classifier.train_epoch
    def paced_train(*args, **kwargs):
        # This runs outside the epoch's random/data/gradient operations.
        reply = subprocess.run([r"C:\Windows\System32\nvidia-smi.exe", "--query-gpu=temperature.gpu",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
        temperature = int(reply.stdout.strip().splitlines()[0])
        while temperature >= 83:
            print(f"THERMAL_IDLE {temperature}C, resting 5s", flush=True)
            time.sleep(5)
            reply = subprocess.run([r"C:\Windows\System32\nvidia-smi.exe", "--query-gpu=temperature.gpu",
                                    "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
            temperature = int(reply.stdout.strip().splitlines()[0])
        result = original_train(*args, **kwargs)
        torch.cuda.synchronize()
        time.sleep(2)
        return result
    classifier.train_epoch = paced_train


def report(args, protocol, classifier):
    controls.preserve_check(protocol)
    current = read_json(args.run_dir / "config.json")
    controls.require_same_config(current, read_json(REFERENCE_RUN / "config.json"))
    rows, full, training, paired = [], {}, {}, {}
    for label, run in (("stage1_only", args.run_dir), ("stage1_plus_stage2", REFERENCE_RUN)):
        training[label] = controls.audit_history(run, read_json(run / "config.json"))
    for split in ("val", "test"):
        full[split], correctness = {}, {}
        for label, run in (("stage1_only", args.run_dir), ("stage1_plus_stage2", REFERENCE_RUN)):
            metrics = pipeline.audited_metrics(argparse.Namespace(run_dir=run), split, classifier)
            full[split][label] = metrics
            row = {"method": label, "split": split, "samples": metrics["samples"],
                   "training_epochs": training[label]["epochs_completed"], "selected_epoch": training[label]["selected_epoch"],
                   **{k: metrics[k] for k in pipeline.common.METRIC_NAMES}}
            for index, name in enumerate(pipeline.CLASSES):
                row[f"support_{name}"] = pipeline.EXPECTED_COUNTS[split][name]
                row.update({f"{metric}_{name}": metrics[metric][index] for metric in ("precision", "recall", "f1")})
            rows.append(row)
            correctness[label] = controls.prediction_correctness(run, split)
        a, b = correctness["stage1_only"], correctness["stage1_plus_stage2"]
        assert a.keys() == b.keys()
        paired[split] = {"both_correct": sum(a[k] and b[k] for k in a),
                         "stage1_only_correct": sum(a[k] and not b[k] for k in a),
                         "stage2_only_correct": sum(not a[k] and b[k] for k in a),
                         "both_wrong": sum(not a[k] and not b[k] for k in a)}
    pipeline.common.write_csv(args.output_dir / "classification_ablation_metrics.csv", rows)
    save_json({"metrics": rows, "full_metrics": full, "training": training, "paired_comparison": paired,
               "classifier_controls_matched": True}, args.output_dir / "classification_ablation_metrics.json")
    lines = ["# Stage 1 / Stage 1 + Stage 2：t50再映射图的ResNet18分类消融", "",
             "## 对照与数据范围", "",
             "使用已核验的两组t50全量再映射图，不重新训练生成器。两组各6199张，保持同源分组split43：4951训练、619验证、629测试。不是将全部6199张用于训练分类器。Stage2组PNG与当前主实验全部6199张逐文件SHA256相同，因此只读复用已完成分类器及已冻结预测；Stage1组从相同随机初始参数开始训练。", "",
             "相同ResNet18 + Dropout(0.30) + Linear(512,3)、seed43、不使用预训练；相同训练增广、归一化、类别权重、标签平滑、AdamW及调度器。最多100轮，按验证损失选模并早停（patience15，min_delta=0.0001），保存绝对最低验证损失权重。测试集不参与梯度、学习率调整、早停或权重选择。", "",
             "## 训练记录", "", "| Configuration | Epochs completed | Selected epoch | Minimum val loss |",
             "|---|---:|---:|---:|"]
    labels = {"stage1_only": "Stage 1 only", "stage1_plus_stage2": "Stage 1 + Stage 2"}
    for key, record in training.items():
        lines.append(f"| {labels[key]} | {record['epochs_completed']} | {record['selected_epoch']} | {record['selected_score']:.6f} |")
    lines += ["", "## 总体分类指标", "", "P、R、F1分别提供weighted及macro，不混淆口径。单标签三分类中weighted-R=ACC属于定义上的正常现象。", "",
              "| Split | Configuration | ACC | Weighted P | Weighted R | Weighted F1 | Macro P | Macro R | Macro F1 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['split']} | {labels[row['method']]} | " + " | ".join(f"{row[k]*100:.2f}%" for k in pipeline.common.METRIC_NAMES) + " |")
    lines += ["", "## 逐类别指标", "", "| Split | Configuration | Class | Support | P | R | F1 |",
              "|---|---|---|---:|---:|---:|---:|"]
    for row in rows:
        for name in pipeline.CLASSES:
            lines.append(f"| {row['split']} | {labels[row['method']]} | {name} | {row[f'support_{name}']} | " +
                         " | ".join(f"{row[f'{metric}_{name}']*100:.2f}%" for metric in ("precision", "recall", "f1")) + " |")
    a, b = full["test"]["stage1_only"], full["test"]["stage1_plus_stage2"]
    deltas = {k: (b[k] - a[k]) * 100 for k in pipeline.common.METRIC_NAMES}
    lines += ["", "## 测试集差值（Stage2减Stage1，单位：百分点）", "",
              " | ".join(f"{k}: {v:+.6f}" for k, v in deltas.items()), "",
              f"配对正确情况：{json.dumps(paired['test'], ensure_ascii=False)}。", "",
              "## 解释边界", "",
              "该消融比较在相同分类器协议下加入Stage2前后的最终流程。Stage1预训练已见过全部6199张带类别方向编码的目标水印图，包含当前val/test对应图像，因此不是完整流程从未见过测试数据的严格独立泛化评估。仅seed43一次运行，不据此声称多随机种子稳定性或统计显著性。类别数据不均衡，应同时关注Macro-F1及defect的P/R/F1，不能只凭ACC说明所有类别都改善。旧t1 Stage1消融采用不同选模协议，未混入本次对照。", "",
              "## 文件", "",
              "- classification_ablation_metrics.csv/json：两组val/test总体及逐类别指标。",
              "- protocol.json：相同初始权重、控制条件、图像身份、只读参考文件哈希。",
              "- runs/resnet18_stage1_only_t50_val_loss：新训练历史、最佳权重、逐图预测、混淆矩阵。",
              f"- Stage2只读参考分类器：{REFERENCE_RUN.as_posix()}", "",
              "原Word和主实验汇总MD未修改。所有旧分类器、生成器和指标的受保护文件均通过SHA256检查。"]
    (args.output_dir / "CLASSIFICATION_ABLATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"test": full["test"], "test_deltas_pp": deltas,
                      "training": training, "output": str(args.output_dir)}, ensure_ascii=False), flush=True)
    controls.preserve_check(protocol)
    # Seal all image identities again, independently of training checkpoints.
    complete = read_json(IMAGES / "complete.json")
    for model, root in (("stage1_only", STAGE1_DATA), ("stage1_plus_stage2", STAGE2_DATA)):
        verify_images(root, complete["output_sha256"][model])
    save_json({"complete": True, "classifier_controls_matched": True,
               "stage2_results_read_only_reused": True, "all_generated_images_unchanged": True,
               "test_predictions_audited_from_sample_keys_and_confusion": True,
               "results_sha256": sha256(args.output_dir / "classification_ablation_metrics.json")},
              args.output_dir / "ablation_complete.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=("all", "preflight", "report"), default="all")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    assert args.output_dir.is_relative_to((WORKSPACE / "experiments").resolve())
    assert args.output_dir not in (IMAGES, REFERENCE_ROOT, WORKSPACE / "experiments")
    classifier = pipeline.common.load_classifier(pipeline.common.DEFAULT_CLASSIFIER_SCRIPT)
    classifier.seed_worker = seed_worker
    pipeline.classifier_config = stage1_classifier_config
    protocol, generation = preflight(args, classifier)
    if args.stage == "preflight":
        return
    protocol_path = args.output_dir / "protocol.json"
    if protocol_path.exists():
        assert read_json(protocol_path) == protocol, "Changed protocol: choose a fresh output root"
    else:
        assert not args.output_dir.exists(), "Refuse unrelated existing output directory"
        args.output_dir.mkdir(parents=True)
        save_json(protocol, protocol_path)
    if args.stage == "all":
        install_cooling(classifier)
        pipeline.train_classifier(args, generation, classifier, data_root=STAGE1_DATA)
        controls.require_same_config(read_json(args.run_dir / "config.json"), read_json(REFERENCE_RUN / "config.json"))
        controls.audit_history(args.run_dir, read_json(args.run_dir / "config.json"))
        for split in ("val", "test"):
            pipeline.evaluate_split(args, split, classifier)
    report(args, protocol, classifier)


if __name__ == "__main__":
    main()
