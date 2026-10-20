"""Retain full6199 Stage2; fine-tune the same old Stage1 on train4951 only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from PIL import Image

import train_stage2_finetune as stage2
from diffusion_stripe.data import collect_pairs
from diffusion_stripe.training import ensure_fresh_or_resume, save_json


WORKSPACE = Path(__file__).resolve().parent
OLD_ROOT = WORKSPACE / "experiments/full6199_two_stage"
DEFAULT_OUTPUT = WORKSPACE / "experiments/stage2_train4951_from_existing_stage1_seed43"
EXPECTED_COUNTS = {"train": {"defect": 990, "normal": 2417, "nut": 1544},
                   "val": {"defect": 125, "normal": 295, "nut": 199},
                   "test": {"defect": 129, "normal": 306, "nut": 194}}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_pair(record) -> dict:
    row = {"key": record.key}
    for name, path in (("original", record.original_path), ("striped", record.striped_path)):
        with Image.open(path) as image:
            if image.size != (64, 64) or image.mode != "RGB":
                raise ValueError(f"Expected RGB 64x64: {path}")
            image.load()
        row[f"{name}_path"] = str(path)
        row[f"{name}_sha256"] = sha256(path)
    return row


def preflight(output_root: Path) -> tuple[dict, dict, list[dict]]:
    if output_root == OLD_ROOT or output_root.is_relative_to(OLD_ROOT):
        raise ValueError("New experiment must be outside the preserved full6199 experiment.")
    config = read_json(OLD_ROOT / "stage2/config.json")
    if config["dataset_size"] != 6199 or config["epochs"] != 20 or config["fixed_timestep"] != 0:
        raise ValueError("Expected the existing full6199, 20-epoch, t=0 Stage2 reference.")
    original_root, striped_root = Path(config["original_root"]), Path(config["striped_root"])
    if output_root.is_relative_to(original_root.resolve()) or output_root.is_relative_to(striped_root.resolve()):
        raise ValueError("Cannot write an experiment into the source dataset.")
    source = Path(config["stage1_checkpoint"]).resolve()
    if source != (OLD_ROOT / "stage1/best_train_loss.pt").resolve():
        raise ValueError("The original Stage2 did not use the expected old Stage1 best checkpoint.")
    preserved = {str(path.resolve()): sha256(path) for path in sorted((OLD_ROOT / "stage2").iterdir()) if path.is_file()}
    preserved[str(source)] = sha256(source)
    pairs = collect_pairs(original_root, striped_root, expected_count=6199)
    counts = {split: dict(Counter(pair.key.split("/")[1] for pair in pairs if pair.key.startswith(split + "/")))
              for split in EXPECTED_COUNTS}
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Existing split/class counts differ: {counts}")
    train_pairs = collect_pairs(original_root, striped_root, expected_count=4951, splits=("train",))
    if any(not pair.key.startswith("train/") for pair in train_pairs):
        raise ValueError("Validation or test sample entered Stage2 training.")
    manifest_path = original_root / "reports/crops_manifest.csv"
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        split_rows = list(csv.DictReader(handle))
    if len(split_rows) != 6199 or {row["crop_path"] for row in split_rows} != {pair.key for pair in pairs}:
        raise ValueError("Existing source-group manifest and current pair keys differ.")
    groups = {split: set() for split in EXPECTED_COUNTS}
    for row in split_rows:
        split, name, *_ = Path(row["crop_path"]).parts
        if split != row["split"] or name != row["class_name"]:
            raise ValueError("Source-group manifest split/class does not match the image path.")
        groups[split].add(row["canonical_source_group"])
    if any(groups[a] & groups[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Source-group overlaps detected in the original partition.")
    with ThreadPoolExecutor(max_workers=8) as executor:
        manifest = list(executor.map(inspect_pair, train_pairs))
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if checkpoint["stage"] != 1 or checkpoint["train_config"]["dataset_size"] != 6199:
        raise ValueError("Expected an existing Stage1 trained on all 6199 stripes.")
    protocol = {
        "protocol": "old_stage1_full6199_to_new_stage2_train4951_20epochs_v1",
        "output_root": str(output_root), "source_stage1_checkpoint": str(source),
        "source_stage1_sha256": preserved[str(source)], "source_stage1_epoch": checkpoint["epoch"],
        "source_stage1_training_samples": 6199, "preserved_old_files_sha256": preserved,
        "old_stage2_training_samples": 6199, "old_stage2_epochs": 20,
        "new_stage2_training_samples": 4951, "training_splits": ["train"],
        "excluded_stage2_validation_samples": 619, "excluded_stage2_test_samples": 629,
        "split_class_counts": counts, "source_group_overlap_across_splits": 0,
        "source_group_manifest_sha256": sha256(manifest_path), "model_config": checkpoint["model_config"],
        "fixed_timestep": 0, "epochs": 20, "early_stopping_enabled": False,
        "loss": "mean RGB three-channel L1", "input_noise_in_stage2": False,
        "class_labels_input_to_network": False, "optimizer": "AdamW",
        "reference_hyperparameters": {name: config[name] for name in (
            "image_size", "batch_size", "workers", "learning_rate", "weight_decay", "grad_clip",
            "seed", "device", "no_amp", "patience", "min_delta")},
        "train_pair_manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        "code_sha256": {name: sha256(WORKSPACE / name) for name in (
            "train_stage2_train4951.py", "train_stage2_finetune.py", "diffusion_stripe/data.py",
            "diffusion_stripe/model.py", "diffusion_stripe/training.py")},
        "limitation": "Stage2 excludes val/test, but reused Stage1 saw all6199 stripe images; this is not full-pipeline unseen validation/test.",
    }
    print("预检通过：只训练4951对（defect990、normal2417、nut1544）；619/629不参与Stage2。", flush=True)
    print(f"旧Stage1 epoch={checkpoint['epoch']}，新Stage2固定t=0，训练满20轮；旧Stage2保持原位。", flush=True)
    del checkpoint
    return config, protocol, manifest


def stage2_arguments(config: dict, output_root: Path, source: str, resume: bool) -> list[str]:
    arguments = ["--original-root", config["original_root"], "--striped-root", config["striped_root"],
                 "--output-root", str(output_root), "--stage1-checkpoint", source,
                 "--splits", "train", "--expected-count", "4951", "--min-epochs", "20",
                 "--epochs", "20", "--no-early-stop"]
    for name in ("image_size", "batch_size", "workers", "learning_rate", "weight_decay", "grad_clip",
                 "seed", "device", "patience", "min_delta"):
        arguments += ["--" + name.replace("_", "-"), str(config[name])]
    if config["no_amp"]:
        arguments.append("--no-amp")
    if resume:
        arguments.append("--resume")
    return arguments


def finish(output_root: Path, protocol: dict) -> None:
    for path, expected in protocol["preserved_old_files_sha256"].items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"A preserved old file changed: {path}")
    run_dir = output_root / "stage2"
    last = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
    best = torch.load(run_dir / "final_generator.pt", map_location="cpu", weights_only=False)
    config = last["train_config"]
    if last["stage"] != 2 or last["epoch"] != 20 or config["dataset_size"] != 4951 or config["splits"] != ["train"] or config["fixed_timestep"] != 0 or not config["no_early_stop"]:
        raise ValueError("New Stage2 checkpoint does not meet the requested train4951, 20-epoch, t=0 protocol.")
    if last["model_config"] != protocol["model_config"] or Path(last["source_checkpoint"]).resolve() != Path(protocol["source_stage1_checkpoint"]):
        raise ValueError("New model architecture or initialization source changed.")
    with (run_dir / "history.csv").open(encoding="utf-8-sig", newline="") as handle:
        history = list(csv.DictReader(handle))
    if [int(row["epoch"]) for row in history] != list(range(1, 21)):
        raise ValueError("Training history must contain exactly epochs 1 through 20.")
    result = {"epochs_completed": 20, "train_samples": 4951, "training_splits": ["train"],
              "fixed_timestep": 0, "selected_epoch": best["epoch"], "selected_train_l1": best["loss"],
              "last_epoch_train_l1": last["loss"], "final_generator_sha256": sha256(run_dir / "final_generator.pt"),
              "old_weights_unchanged": True, "validation_or_test_used_in_stage2": False}
    save_json(result, output_root / "training_complete.json")
    h = protocol["reference_hyperparameters"]
    lines = ["# 新Stage2：旧Stage1初始化，仅4951对训练，固定t=0、20 epoch", "",
             "旧Stage2（全6199对微调20轮）保留，原目录所有文件的SHA256在新训练前后核验一致。新实验不加载旧Stage2继续训练，而是重新加载相同的旧Stage1最佳权重。", "",
             "## 训练配置", "",
             "- 仅使用原split43的train：4951对原图/条纹图，defect990、normal2417、nut1544。",
             "- val619与test629不参与本次Stage2梯度更新、早停或选模；原有分组与文件不改变。",
             "- 与旧Stage2同一个三通道时间条件U-Net；原图输入、对应RGB条纹图监督，类别编号不输入网络。",
             "- 固定t=0，不对Stage2输入加噪；训练满20 epoch，不提前早停。",
             f"- AdamW，lr={h['learning_rate']:g}、weight_decay={h['weight_decay']:g}、batch={h['batch_size']}、grad_clip={h['grad_clip']:g}、seed={h['seed']}、workers={h['workers']}；AMP沿用旧配置。",
             "- 保留旧Stage2的训练L1选模规则（min_delta=1e-5）；final_generator.pt为最佳训练损失检查点，last.pt为第20轮断点。", "",
             "## 公式", "",
             r"$$\hat S_i=R_{\theta}(I_i,0),\qquad \mathcal L_{\mathrm{stage2}}=\frac1{3HW|\mathcal B|}\sum_{i\in\mathcal B}\sum_{c,h,w}|R_{\theta}(I_i,0)_{c,h,w}-S_{i,c,h,w}|.$$ ",
             "", r"$$\theta_{\mathrm{init}}=\theta_{\mathrm{old\ stage1}},\quad \mathcal D_{\mathrm{stage2}}=\mathcal D_{\mathrm{train4951}},\quad t=0,\quad E=20.$$ ", "",
             "## 结果与权重", "",
             f"完成20轮；最终选中epoch {best['epoch']}，对应训练平均RGB L1={best['loss']:.8f}；第20轮L1={last['loss']:.8f}。",
             f"- 新权重：[stage2/final_generator.pt]({(run_dir / 'final_generator.pt').as_posix()})",
             f"- 第20轮断点：[stage2/last.pt]({(run_dir / 'last.pt').as_posix()})",
             f"- 旧权重：[旧Stage2/final_generator.pt]({(OLD_ROOT / 'stage2/final_generator.pt').as_posix()})",
             f"- 来源：[旧Stage1/best_train_loss.pt]({Path(protocol['source_stage1_checkpoint']).as_posix()})", "",
             "本次只训练新Stage2，没有改动已有再映射图或ResNet18结果，也没有自动用新生成器替换原实验的权重路径。新旧训练损失基于不同样本集合，不能单凭它们判断泛化是否改善。", "",
             "## 数据隔离的边界", "",
             "旧划分的train/val/test同源组无交集；本次Stage2只使用train。但复用的旧Stage1曾用全6199条纹图训练，因此验证和测试样本并非对整个生成器训练流程完全未见。这个实验只比较Stage2微调数据范围变化；不能据此声称全流程独立留出。如要全流程隔离，还需另行只用train4951训练Stage1（不属于本次改动）。", "",
             "## 终端运行", "", "```powershell", "cd D:\\Cold_Diffusion",
             "conda run --no-capture-output -n diffusion_env python -X utf8 -u train_stage2_train4951.py",
             "```", "", "仅核验加--preflight-only；中断续训加--resume；非空目录默认拒绝覆盖。"]
    (output_root / "STAGE2_TRAIN4951_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"新Stage2完成20轮；选中epoch {best['epoch']}，L1={best['loss']:.8f}；旧权重SHA256全部不变。", flush=True)
    print(f"新生成器：{run_dir / 'final_generator.pt'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    config, protocol, manifest = preflight(output_root)
    if args.preflight_only:
        return
    protocol_path = output_root / "experiment_protocol.json"
    if protocol_path.is_file():
        if read_json(protocol_path) != protocol:
            raise ValueError("Saved train4951 protocol differs; use a new output directory.")
    elif output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("Refusing to write into an unrecognized nonempty output directory.")
    ensure_fresh_or_resume(output_root / "stage2", resume=args.resume)
    output_root.mkdir(parents=True, exist_ok=True)
    if not protocol_path.exists():
        save_json(protocol, protocol_path)
        save_json({"count": len(manifest), "pairs": manifest}, output_root / "train_pair_manifest.json")
    stage2.main(stage2_arguments(config, output_root, protocol["source_stage1_checkpoint"], args.resume))
    finish(output_root, protocol)


if __name__ == "__main__":
    main()
