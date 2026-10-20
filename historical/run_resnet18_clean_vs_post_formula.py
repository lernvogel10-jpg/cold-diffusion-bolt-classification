from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DEFAULT_CLEAN_DATA = Path(
    r"D:\YOLO26_study\datasets\yolo26_crops64_dataset_source_grouped_split43"
)
DEFAULT_REMAPPED_DATA = Path(
    r"D:\Cold_Diffusion\experiments\full6199_two_stage"
    r"\inference_compare_t1_all6199\images\post_formula_prediction"
)
DEFAULT_CLASSIFIER_SCRIPT = Path(
    r"D:\YOLO26_study\label_texture_experiment"
    r"\striped_classification_experiment.py"
)
DEFAULT_EXPERIMENT_DIR = Path(
    r"D:\Cold_Diffusion\experiments\resnet18_clean_vs_post_formula_t1_seed43"
)
CLASSES = ("defect", "normal", "nut")
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
EXPECTED_COUNTS = {
    ("train", "defect"): 990,
    ("train", "normal"): 2417,
    ("train", "nut"): 1544,
    ("val", "defect"): 125,
    ("val", "normal"): 295,
    ("val", "nut"): 199,
    ("test", "defect"): 129,
    ("test", "normal"): 306,
    ("test", "nut"): 194,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "使用完全相同的split43和ResNet18配置，对比6199张原图与"
            "6199张公式结果再映射图的三分类性能。"
        )
    )
    parser.add_argument("--clean-data", type=Path, default=DEFAULT_CLEAN_DATA)
    parser.add_argument("--remapped-data", type=Path, default=DEFAULT_REMAPPED_DATA)
    parser.add_argument(
        "--classifier-script", type=Path, default=DEFAULT_CLASSIFIER_SCRIPT
    )
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument(
        "--stage",
        choices=("all", "train", "test", "report"),
        default="all",
        help=(
            "all=训练两组并在权重冻结后各测试一次；train=只训练和val；"
            "test=只测试已有权重；report=只重新合并已有结果。"
        ),
    )
    parser.add_argument("--rerun-test", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def display_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def run_step(label: str, command: list[str], dry_run: bool) -> None:
    print("\n" + "=" * 88, flush=True)
    print(label, flush=True)
    print(display_command(command), flush=True)
    print("=" * 88, flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def collect_relative_images(root: Path) -> set[str]:
    if not root.is_dir():
        raise FileNotFoundError(f"数据目录不存在：{root}")
    relative_paths: set[str] = set()
    for split in SPLITS:
        for class_name in CLASSES:
            directory = root / split / class_name
            if not directory.is_dir():
                raise FileNotFoundError(f"缺少目录：{directory}")
            files = sorted(
                path
                for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            expected = EXPECTED_COUNTS[(split, class_name)]
            if len(files) != expected:
                raise RuntimeError(
                    f"{root} 的 {split}/{class_name} 应有{expected}张，"
                    f"实际{len(files)}张。"
                )
            for path in files:
                key = path.relative_to(root).with_suffix("").as_posix()
                if key in relative_paths:
                    raise RuntimeError(f"发现重复样本键：{key}")
                relative_paths.add(key)
    return relative_paths


def preflight(clean_root: Path, remapped_root: Path) -> dict[str, object]:
    clean_keys = collect_relative_images(clean_root)
    remapped_keys = collect_relative_images(remapped_root)
    if len(clean_keys) != 6199 or len(remapped_keys) != 6199:
        raise RuntimeError(
            f"数据总数错误：clean={len(clean_keys)}, remapped={len(remapped_keys)}"
        )
    missing_remapped = sorted(clean_keys - remapped_keys)
    extra_remapped = sorted(remapped_keys - clean_keys)
    if missing_remapped or extra_remapped:
        raise RuntimeError(
            "原图与再映射图样本键不一致。"
            f"缺少再映射图（前5）：{missing_remapped[:5]}；"
            f"多余再映射图（前5）：{extra_remapped[:5]}"
        )
    return {
        "clean_count": len(clean_keys),
        "remapped_count": len(remapped_keys),
        "paired_keys_equal": True,
        "split_class_counts": {
            f"{split}/{class_name}": count
            for (split, class_name), count in EXPECTED_COUNTS.items()
        },
    }


def write_protocol(
    path: Path,
    args: argparse.Namespace,
    preflight_result: dict[str, object],
    clean_run: Path,
    remapped_run: Path,
) -> None:
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "ResNet18 clean vs t=1 post-formula-remapped classification",
        "clean_data": str(args.clean_data.resolve()),
        "remapped_data": str(args.remapped_data.resolve()),
        "classifier_script": str(args.classifier_script.resolve()),
        "split_protocol": "existing source-grouped split43 (4951/619/629)",
        "seed": args.seed,
        "model": "resnet18",
        "pretrained": args.pretrained,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "class_weights": not args.no_class_weights,
        "augmentation": not args.no_augmentation,
        "clean_run": str(clean_run),
        "remapped_run": str(remapped_run),
        "metrics": {
            "overall": [
                "accuracy",
                "weighted_precision",
                "weighted_recall",
                "weighted_f1",
                "macro_precision",
                "macro_recall",
                "macro_f1",
            ],
            "per_class": ["precision", "recall", "f1"],
        },
        "scientific_limitations": [
            (
                "The t=1 post-formula images were generated by a cold-diffusion "
                "network trained/fine-tuned with all 6199 original/stripe pairs."
            ),
            (
                "Therefore its val/test images are not generator-unseen. Results are "
                "closed-set/in-sample generator comparisons, not leakage-free "
                "generalization estimates."
            ),
            (
                "The classifier split itself preserves the source-grouped split43 "
                "with no source-group overlap."
            ),
        ],
        "preflight": preflight_result,
    }
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        stable_keys = (
            "clean_data",
            "remapped_data",
            "classifier_script",
            "split_protocol",
            "seed",
            "model",
            "pretrained",
            "batch_size",
            "class_weights",
            "augmentation",
        )
        mismatches = {
            key: (existing.get(key), payload.get(key))
            for key in stable_keys
            if existing.get(key) != payload.get(key)
        }
        if mismatches:
            raise ValueError(f"已有实验协议与当前参数不一致：{mismatches}")
        return
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_common_train_args(command: list[str], args: argparse.Namespace) -> None:
    if args.pretrained:
        command.append("--pretrained")
    if args.no_augmentation:
        command.append("--no-augmentation")
    if args.no_class_weights:
        command.append("--no-class-weights")


def read_summary(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"预期{path}只有1行结果，实际{len(rows)}行。")
    return rows[0]


def write_full_metrics_report(
    output: Path,
    split: str,
    runs: tuple[tuple[str, Path], ...],
) -> None:
    rows: list[dict[str, object]] = []
    for method_label, run_dir in runs:
        metrics_path = run_dir / f"{split}_metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(metrics_path)
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if len(payload) != 1:
            raise RuntimeError(f"{metrics_path}应只有一个评估域。")
        eval_domain, metrics = next(iter(payload.items()))
        row: dict[str, object] = {
            "method": method_label,
            "split": split,
            "eval_domain": eval_domain,
            "samples": metrics["samples"],
            "accuracy": metrics["accuracy"],
            "weighted_precision": metrics["weighted_precision"],
            "weighted_recall": metrics["weighted_recall"],
            "weighted_f1": metrics["weighted_f1"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
        }
        for index, class_name in enumerate(CLASSES):
            row[f"precision_{class_name}"] = metrics["precision"][index]
            row[f"recall_{class_name}"] = metrics["recall"][index]
            row[f"f1_{class_name}"] = metrics["f1"][index]
        rows.append(row)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_final_table(clean_run: Path, remapped_run: Path) -> None:
    print("\n最终test结果（总体P/R/F1为weighted）：", flush=True)
    print("method\tACC\tP\tR\tF1", flush=True)
    for label, run_dir in (("clean", clean_run), ("post_formula", remapped_run)):
        row = read_summary(run_dir / "test_summary.csv")
        print(
            f"{label}\t{float(row['accuracy']):.4f}\t"
            f"{float(row['precision']):.4f}\t{float(row['recall']):.4f}\t"
            f"{float(row['f1']):.4f}",
            flush=True,
        )


def main() -> int:
    args = build_parser().parse_args()
    clean_root = args.clean_data.resolve()
    remapped_root = args.remapped_data.resolve()
    classifier_script = args.classifier_script.resolve()
    experiment_dir = args.experiment_dir.resolve()
    if not classifier_script.is_file():
        raise FileNotFoundError(classifier_script)

    preflight_result = preflight(clean_root, remapped_root)
    print("数据预检通过：原图与再映射图各6199张，样本键和split/class完全一致。")
    if args.preflight_only:
        print(json.dumps(preflight_result, ensure_ascii=False, indent=2))
        return 0

    weight_tag = "pretrained" if args.pretrained else "scratch"
    runs_dir = experiment_dir / "runs"
    clean_tag = f"seed{args.seed}_split43_resnet18_{weight_tag}_clean"
    remapped_tag = (
        f"seed{args.seed}_split43_resnet18_{weight_tag}_post_formula_t1"
    )
    clean_run = runs_dir / clean_tag
    remapped_run = runs_dir / remapped_tag
    if not args.dry_run:
        experiment_dir.mkdir(parents=True, exist_ok=True)
        write_protocol(
            experiment_dir / "experiment_protocol.json",
            args,
            preflight_result,
            clean_run,
            remapped_run,
        )

    python = str(Path(sys.executable).resolve())
    base_train = [
        python,
        "-u",
        str(classifier_script),
        "train",
        "--clean-data",
        str(clean_root),
        "--striped-data",
        str(remapped_root),
        "--runs-dir",
        str(runs_dir),
        "--seed",
        str(args.seed),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(args.workers),
        "--device",
        str(args.device),
        "--resume",
    ]
    append_common_train_args(base_train, args)

    if args.stage in ("all", "train"):
        run_step(
            "[1/6] 训练原图ResNet18：clean train -> clean val",
            base_train
            + [
                "--train-domain",
                "clean",
                "--selection-domain",
                "clean",
                "--run-tag",
                clean_tag,
            ],
            args.dry_run,
        )
        run_step(
            "[2/6] 训练再映射图ResNet18：post-formula train -> post-formula val",
            base_train
            + [
                "--train-domain",
                "striped",
                "--selection-domain",
                "striped",
                "--run-tag",
                remapped_tag,
            ],
            args.dry_run,
        )

    val_comparison = experiment_dir / "val_comparison.csv"
    if args.stage in ("all", "train", "report"):
        run_step(
            "[3/6] 合并两组val指标",
            [
                python,
                "-u",
                str(classifier_script),
                "compare",
                "--split",
                "val",
                "--run-dirs",
                str(clean_run),
                str(remapped_run),
                "--output",
                str(val_comparison),
            ],
            args.dry_run,
        )
        if not args.dry_run:
            write_full_metrics_report(
                experiment_dir / "val_metrics_full.csv",
                "val",
                (("clean", clean_run), ("post_formula", remapped_run)),
            )

    if args.stage in ("all", "test"):
        if not args.dry_run:
            for run_dir in (clean_run, remapped_run):
                if not (run_dir / "training_complete.json").is_file():
                    raise RuntimeError(f"尚未完成训练，禁止读取test：{run_dir}")
        for step, label, run_dir in (
            (4, "原图ResNet18 -> 原图test", clean_run),
            (5, "再映射ResNet18 -> 再映射test", remapped_run),
        ):
            marker = run_dir / "test_complete.json"
            if marker.is_file() and not args.rerun_test and not args.dry_run:
                print(f"[{step}/6] 已存在{marker}，跳过重复test。", flush=True)
                continue
            run_step(
                f"[{step}/6] {label}",
                [
                    python,
                    "-u",
                    str(classifier_script),
                    "test",
                    "--run-dir",
                    str(run_dir),
                    "--workers",
                    str(args.workers),
                    "--device",
                    str(args.device),
                ],
                args.dry_run,
            )

    test_comparison = experiment_dir / "test_comparison.csv"
    if args.stage in ("all", "test", "report"):
        run_step(
            "[6/6] 合并两组test的ACC、P、R、F1及逐类别指标",
            [
                python,
                "-u",
                str(classifier_script),
                "compare",
                "--split",
                "test",
                "--run-dirs",
                str(clean_run),
                str(remapped_run),
                "--output",
                str(test_comparison),
            ],
            args.dry_run,
        )
        if not args.dry_run:
            write_full_metrics_report(
                experiment_dir / "test_metrics_full.csv",
                "test",
                (("clean", clean_run), ("post_formula", remapped_run)),
            )

    if args.dry_run:
        print("\nDry-run完成：没有启动训练、没有读取test。")
        return 0

    if args.stage in ("all", "test", "report"):
        print_final_table(clean_run, remapped_run)
    print(f"\n实验目录：{experiment_dir}")
    print(f"val对比：{val_comparison}")
    print(f"test对比：{test_comparison}")
    print(f"完整test指标：{experiment_dir / 'test_metrics_full.csv'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n用户中断。重新运行相同命令即可断点续训。", file=sys.stderr)
        raise SystemExit(130)
