"""Frozen train4951 Stage2: full t=50->0 recursion, remapping, and ResNet18."""
from __future__ import annotations

import argparse
import gc
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

import run_clean_baseline_vs_stage2train4951 as controls
import run_stage2_train4951_remapped_resnet18 as pipeline
from diffusion_stripe.data import StripedImageDataset
from diffusion_stripe.model import build_model_from_checkpoint
from diffusion_stripe.training import linear_beta_schedule, save_json
from evaluate_t1_all6199 import tensor_to_image
from resnet18_full6199_loss_earlystop import seed_worker
from train_stage2_train4951 import read_json, sha256


DEFAULT_ROOT = pipeline.WORKSPACE / "experiments/stage2_train4951_t50_remapped_resnet18_seed43"
NOISE_T50 = pipeline.WORKSPACE / "experiments/full6199_two_stage/inference_compare_t50_seed43/noise_t50_seed43.pt"
RAW_RUN = controls.DEFAULT_ROOT / "runs/resnet18_scratch_original_val_loss"
T1_RUN = controls.REMAPPED_RUN
TIMESTEP = 50


def remap_t50(model, original: torch.Tensor, noise: torch.Tensor, alpha_bars: torch.Tensor):
    """Match evaluate_t50_all6199: no clipping within the 50-step recursion."""
    if alpha_bars.ndim != 1 or len(alpha_bars) <= TIMESTEP:
        raise ValueError("Expected a DDPM schedule containing t=0..50.")
    epsilon = noise.expand(len(original), -1, -1, -1)
    alpha = alpha_bars[TIMESTEP]
    state = alpha.sqrt() * original + (1 - alpha).sqrt() * epsilon
    direct_raw = None
    for timestep in range(TIMESTEP, 0, -1):
        times = torch.full((len(original),), timestep, device=original.device, dtype=torch.long)
        predicted_x0 = model(state, times)
        if timestep == TIMESTEP:
            direct_raw = predicted_x0
        current, before = alpha_bars[timestep], alpha_bars[timestep - 1]
        degraded_current = current.sqrt() * predicted_x0 + (1 - current).sqrt() * epsilon
        degraded_before = before.sqrt() * predicted_x0 + (1 - before).sqrt() * epsilon
        state = state - degraded_current + degraded_before
    formula = state.clamp(0, 1)
    zeros = torch.zeros(len(original), device=original.device, dtype=torch.long)
    post_formula = model(formula, zeros).clamp(0, 1)
    return direct_raw.clamp(0, 1), formula, post_formula


def preflight(args, classifier):
    # Reuse the existing checks for the exact generator, originals, split groups, and Stage1.
    protocol, manifest = pipeline.preflight(args)
    reference = read_json(T1_RUN / "config.json")
    for attribute, field in (("epochs", "max_epochs"), ("patience", "patience"), ("min_delta", "min_delta"),
                             ("batch_size", "batch_size"), ("workers", "workers"), ("device", "device"),
                             ("selection", "selection")):
        setattr(args, attribute, reference[field])
    if args.selection != "val_loss" or args.epochs != 100:
        raise ValueError("Expected the completed validation-loss reference configuration.")
    args.run_dir = args.output_dir / "runs/resnet18_scratch_val_loss"
    noise = torch.load(NOISE_T50, map_location="cpu", weights_only=True)
    old_t1_noise = torch.load(pipeline.NOISE, map_location="cpu", weights_only=True)
    if tuple(noise.shape) != (1, 3, 64, 64) or not torch.equal(noise, old_t1_noise) or not torch.isfinite(noise).all():
        raise ValueError("Expected the archived fixed t50 noise, identical in value to the t1 epsilon.")
    protocol.update({
        "protocol_version": "new_stage2_train4951_t50_full_recursion_post_formula_v1",
        "noise_file": str(NOISE_T50), "noise_sha256": sha256(NOISE_T50), "timestep": TIMESTEP,
        "recursion_timesteps": list(range(TIMESTEP, 0, -1)), "network_calls_per_image": TIMESTEP + 1,
        "formula": "x50=D(I,50,epsilon); for s=50..1: xs-1=xs-D(R(xs,s),s,epsilon)+D(R(xs,s),s-1,epsilon); output=clip(R(clip(x0),0))",
        "intermediate_clipping": "No prediction or state clipping inside recursion; clip x0 before R(x0,0), then clip final output.",
        "precision": "float32, no AMP, identical to the historical t50 recursion implementation",
        "t50_noise_numerically_equal_to_saved_t1_noise": True,
    })
    classifier.set_seed(reference["seed"])
    model = classifier.build_model(reference["pretrained"])
    candidate = pipeline.classifier_config(args, protocol, model, classifier)
    controls.require_same_config(candidate, reference)
    del model
    original_comparison = read_json(controls.DEFAULT_ROOT / "comparison_protocol.json")
    protected = dict(original_comparison["preserved_reference_files_sha256"])
    for run in (RAW_RUN, T1_RUN):
        for path in sorted(run.iterdir()):
            if path.is_file():
                protected[str(path.resolve())] = sha256(path)
        config = read_json(run / "config.json")
        controls.require_same_config(candidate, config)
        controls.audit_history(run, config)
        for split in ("val", "test"):
            pipeline.audited_metrics(argparse.Namespace(run_dir=run), split, classifier)
    protected[str(NOISE_T50)] = protocol["noise_sha256"]
    protected[str(pipeline.NOISE)] = sha256(pipeline.NOISE)
    protocol["preserved_reference_files_sha256"] = protected
    protocol["classifier_shared_config"] = {key: candidate[key] for key in controls.SHARED_KEYS}
    protocol["reference_results_read_only"] = {"original": str(RAW_RUN), "t1": str(T1_RUN)}
    protocol["implementation_sha256"] = sha256(Path(__file__))
    protocol["shared_trainer_sha256"] = sha256(Path(pipeline.__file__))
    controls.preserve_check(protocol)
    print("预检通过：t50完整50步递推后再映射；仅训练本次ResNet18，原图/t1结果只读。", flush=True)
    return protocol, manifest


def verify_partial(args, protocol: dict, manifest: list[dict], progress: dict) -> None:
    offset = progress["count"]
    if progress["checkpoint_sha256"] != protocol["checkpoint_sha256"] or not 0 <= offset <= len(manifest):
        raise ValueError("Invalid partial generation provenance or count.")
    expected = {row["key"] for row in manifest[:offset]}
    hashes = progress["output_sha256"]
    if set(hashes) != expected:
        raise ValueError("Partial generation is not a contiguous original manifest prefix.")
    root = args.output_dir / "datasets/post_formula_prediction"
    paths = [root / row["key"] for row in manifest[:offset]]
    with ThreadPoolExecutor(max_workers=8) as executor:
        actual = list(executor.map(sha256, paths))
    if any(actual_hash != hashes[row["key"]] for row, actual_hash in zip(manifest[:offset], actual, strict=True)):
        raise ValueError("Partially generated images changed.")


def generate(args, protocol: dict, manifest: list[dict]) -> None:
    marker_path = args.output_dir / "generation_complete.json"
    progress_path = args.output_dir / "generation_progress.json"
    if marker_path.exists():
        marker = read_json(marker_path)
        if marker["checkpoint_sha256"] != protocol["checkpoint_sha256"] or marker["timestep"] != TIMESTEP:
            raise ValueError("Existing images used a different generator or timestep.")
        pipeline.verify_generated(args, manifest, marker)
        print("已核验6199张t50再映射图，跳过重复生成。", flush=True)
        return
    progress = read_json(progress_path) if progress_path.exists() else {
        "count": 0, "output_sha256": {}, "checkpoint_sha256": protocol["checkpoint_sha256"]}
    verify_partial(args, protocol, manifest, progress)
    offset, hashes = progress["count"], progress["output_sha256"]
    dataset = StripedImageDataset(protocol["original_root"], expected_count=6199)
    device = torch.device(args.device)
    # Keep generator initialization and inference flags the same as the existing t1 experiment.
    checkpoint = torch.load(protocol["checkpoint"], map_location="cpu", weights_only=False)
    model = build_model_from_checkpoint(checkpoint).to(device).eval().requires_grad_(False)
    before = pipeline.common.state_sha256(model)
    alpha_bars = linear_beta_schedule(100, protocol["beta_start"], protocol["beta_end"], device)
    noise = torch.load(protocol["noise_file"], map_location=device, weights_only=True).float()
    loader = DataLoader(Subset(dataset, range(offset, len(dataset))), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    root = args.output_dir / "datasets/post_formula_prediction"
    began = time.perf_counter()
    start_offset = offset
    print(f"开始t50全量再映射，已完成{offset}/6199；每图执行50次递推预测及1次t0再映射。", flush=True)
    with torch.inference_mode():
        for batch in loader:
            _, _, result = remap_t50(model, batch.to(device, non_blocking=True), noise, alpha_bars)
            if not torch.isfinite(result).all():
                raise ValueError("Non-finite t50 remapped pixels.")
            result = result.cpu()
            current = dataset.records[offset:offset + len(batch)]
            for record, tensor in zip(current, result, strict=True):
                path = root / record.key
                path.parent.mkdir(parents=True, exist_ok=True)
                tensor_to_image(tensor).save(path)
                hashes[record.key] = sha256(path)
            offset += len(batch)
            save_json({"count": offset, "output_sha256": hashes,
                       "checkpoint_sha256": protocol["checkpoint_sha256"]}, progress_path)
            elapsed = time.perf_counter() - began
            remaining = elapsed * (6199 - offset) / (offset - start_offset)
            print(f"t50再映射：{offset}/6199，耗时{elapsed:.0f}s，按当前速度剩余约{remaining:.0f}s", flush=True)
    if offset != 6199 or pipeline.common.state_sha256(model) != before:
        raise ValueError("Incorrect image count or generator state mutation.")
    controls.preserve_check(protocol)
    marker = {"count": offset, "checkpoint_sha256": protocol["checkpoint_sha256"], "timestep": TIMESTEP,
              "alpha_bar_50": float(alpha_bars[TIMESTEP]), "recursion_steps": TIMESTEP,
              "output_sha256": hashes, "generator_state_unchanged": True,
              "generator_class_labels_used": False}
    save_json(marker, marker_path)
    pipeline.verify_generated(args, manifest, marker)
    del model, checkpoint, dataset, loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def report(args, protocol: dict, classifier) -> None:
    controls.preserve_check(protocol)
    config = read_json(args.run_dir / "config.json")
    controls.require_same_config(config, read_json(T1_RUN / "config.json"))
    complete = controls.audit_history(args.run_dir, config)
    rows, details = [], {}
    for split in ("val", "test"):
        metrics = pipeline.audited_metrics(args, split, classifier)
        details[split] = metrics
        row = {"split": split, "samples": metrics["samples"],
               **{name: metrics[name] for name in pipeline.common.METRIC_NAMES}}
        for index, name in enumerate(pipeline.CLASSES):
            row[f"support_{name}"] = pipeline.EXPECTED_COUNTS[split][name]
            row.update({f"{metric}_{name}": metrics[metric][index] for metric in ("precision", "recall", "f1")})
        rows.append(row)
        print(f"t50 {split}：ACC={row['accuracy']*100:.2f}%，P/R/F1="
              f"{row['weighted_precision']*100:.2f}%/{row['weighted_recall']*100:.2f}%/{row['weighted_f1']*100:.2f}%"
              f"，macro-F1={row['macro_f1']*100:.2f}%", flush=True)
    pipeline.common.write_csv(args.output_dir / "classification_val_loss_metrics.csv", rows)
    save_json({"metrics": rows, "full_metrics": details, "training": complete},
              args.output_dir / "classification_val_loss_metrics.json")
    comparison_rows = []
    for split in ("val", "test"):
        for label, run in (("original", RAW_RUN), ("t1_remapped", T1_RUN), ("t50_remapped", args.run_dir)):
            controls.require_same_config(config, read_json(run / "config.json"))
            metrics = pipeline.audited_metrics(argparse.Namespace(run_dir=run), split, classifier)
            comparison_rows.append({"method": label, "split": split, "samples": metrics["samples"],
                                    **{key: metrics[key] for key in pipeline.common.METRIC_NAMES}})
    pipeline.common.write_csv(args.output_dir / "original_t1_t50_comparison.csv", comparison_rows)
    save_json(comparison_rows, args.output_dir / "original_t1_t50_comparison.json")
    lines = ["# 新Stage2(train4951)：t=50完整公式递推、t=0再映射及ResNet18分类", "",
             "## 生成器与流程", "",
             "复用旧Stage1初始化、仅train4951对原图/条纹图在t=0微调20轮的已完成新Stage2权重，未重训生成器。推理保持train4951、val619、test629的原split43分组、类别和文件键。类别标签及目标条纹图不输入生成器。", "",
             r"$$x_{50}=\sqrt{\bar\alpha_{50}}I+\sqrt{1-\bar\alpha_{50}}\epsilon,\qquad \bar\alpha_{50}\approx0.7771798968.$$", "",
             r"$$\hat x_0^{(s)}=R_\theta(x_s,s),\quad x_{s-1}=x_s-D(\hat x_0^{(s)},s;\epsilon)+D(\hat x_0^{(s)},s-1;\epsilon),\quad s=50,49,\ldots,1.$$", "",
             r"$$D(z,s;\epsilon)=\sqrt{\bar\alpha_s}z+\sqrt{1-\bar\alpha_s}\epsilon,\qquad G_{50}(I)=\operatorname{clip}\!\left[R_\theta(\operatorname{clip}(x_0,0,1),0),0,1\right].$$", "",
             "与历史t50程序一致：完整50步递推，递推内部不裁剪网络预测或状态；得到x0后裁剪至[0,1]再输入R(x0,0)，最后再裁剪并量化为RGB64×64 PNG。推理float32、无AMP。固定复用已保存seed43 epsilon，与旧t1 epsilon数值相同；它不是归档的历史逐样本训练噪声。", "",
             "| 集合 | defect | normal | nut | 合计 |", "|---|---:|---:|---:|---:|"]
    for split in pipeline.SPLITS:
        counts = pipeline.EXPECTED_COUNTS[split]
        lines.append(f"| {split} | " + " | ".join(str(counts[name]) for name in pipeline.CLASSES) + f" | {sum(counts.values())} |")
    lines += ["", "## 分类器配置与训练", "",
              "完全复用已完成t1组的训练函数和超参数：ResNet18 + Dropout(0.30) + Linear(512,3)，相同随机初始参数、seed43、无预训练；AdamW lr=0.001、weight_decay=0.0001、batch64、workers4、AMP、训练标签平滑0.05、仅train4951计算类别权重，RandomAffine+ColorJitter，mean/std=(0.5,0.5,0.5)。val/test不增广，评价用无权重、无标签平滑交叉熵。", "",
              "最多100轮，验证损失选模/早停，patience15、min_delta=0.0001，保存绝对最低验证损失权重。ReduceLROnPlateau同样监测val_loss，mode=min、factor0.5、patience4、threshold0.0001(abs)、min_lr=1e-6。测试629不参与梯度、早停、学习率或选模。", "",
              f"实际训练{complete['epochs_completed']}轮，选中第{complete['selected_epoch']}轮，最低验证损失{complete['selected_score']:.6f}，停止原因：{complete['stop_reason']}。", "",
              "## 本次t50验证集与测试集指标", "", "P/R/F1为weighted，并同时列出macro指标；单标签多分类weighted-R=ACC。", "",
              "| 集合 | 数量 | ACC | weighted-P | weighted-R | weighted-F1 | macro-P | macro-R | macro-F1 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['split']} | {row['samples']} | " + " | ".join(f"{row[name]*100:.2f}%" for name in pipeline.common.METRIC_NAMES) + " |")
    lines += ["", "## 逐类别P/R/F1", "", "| 集合 | 类别 | 数量 | P | R | F1 |", "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        for name in pipeline.CLASSES:
            lines.append(f"| {row['split']} | {name} | {row[f'support_{name}']} | " + " | ".join(f"{row[f'{metric}_{name}']*100:.2f}%" for metric in ("precision", "recall", "f1")) + " |")
    lines += ["", "## 原图、t1与t50对照（相同val_loss协议）", "",
              "原图与t1组直接复用已冻结预测，未重新训练或重复测试。所有分类器控制条件与初始权重核验相同，实际早停轮数可能不同。", "",
              "| 集合 | 输入 | ACC | weighted-P | weighted-R | weighted-F1 | macro-F1 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    labels = {"original": "原图基线", "t1_remapped": "新Stage2 t1再映射", "t50_remapped": "新Stage2 t50再映射"}
    for row in comparison_rows:
        lines.append(f"| {row['split']} | {labels[row['method']]} | " + " | ".join(f"{row[key]*100:.2f}%" for key in ("accuracy", "weighted_precision", "weighted_recall", "weighted_f1", "macro_f1")) + " |")
    lines += ["", "## 解释边界", "",
              "Stage2和分类器均仅用train4951更新权重，但初始化来源的旧Stage1见过全部6199目标条纹图，不能称为全流程未见测试数据的独立泛化结果。固定单一seed43和epsilon，未验证多随机种子稳定性。整体ACC不能替代defect逐类召回率；没有仅凭单次结果声称统计显著性。旧Stage1、旧/新Stage2、原图基线和t1分类器文件的SHA256均核验未改变。", "",
              "## 文件与终端复现", "",
              f"- 再映射图：[datasets/post_formula_prediction]({(args.output_dir/'datasets/post_formula_prediction').as_posix()})",
              f"- 最佳ResNet18：[best.pt]({(args.run_dir/'best.pt').as_posix()})",
              f"- 指标：[classification_val_loss_metrics.csv]({(args.output_dir/'classification_val_loss_metrics.csv').as_posix()})",
              f"- 原图/t1/t50对照：[original_t1_t50_comparison.csv]({(args.output_dir/'original_t1_t50_comparison.csv').as_posix()})",
              "- run目录保留val/test逐图预测、混淆矩阵、训练曲线和训练历史；指标从逐图预测重新核算。", "",
              "```powershell", "cd <repository-root>",
              "conda run --no-capture-output -n diffusion_env python -X utf8 -u run_stage2_train4951_t50_remapped_resnet18.py",
              "```", "", "--stage preflight仅检查；--stage prepare仅生成图片；--stage report仅汇总。相同命令核验后复用完成结果，支持按完整生成批次及分类器last.pt断点继续；多进程增广RNG不保存，分类器续训不保证逐位复现。"]
    (args.output_dir / "RESNET18_T50_VAL_LOSS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--stage", choices=("all", "preflight", "prepare", "train", "test", "report"), default="all")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir in (pipeline.DEFAULT_ROOT.resolve(), controls.DEFAULT_ROOT.resolve(), pipeline.NEW_STAGE2.resolve()):
        raise ValueError("Refusing to use an existing generator, baseline or t1 result root.")
    classifier = pipeline.common.load_classifier(pipeline.common.DEFAULT_CLASSIFIER_SCRIPT)
    classifier.seed_worker = seed_worker
    protocol, manifest = preflight(args, classifier)
    if args.stage == "preflight":
        return
    protocol_path = args.output_dir / "generation_protocol.json"
    if protocol_path.exists():
        if read_json(protocol_path) != protocol:
            raise ValueError("Generation/training implementation or protocol changed; use a new directory.")
    elif args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite an unrecognized nonempty experiment directory.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not protocol_path.exists():
        save_json(protocol, protocol_path)
        save_json(manifest, args.output_dir / "input_manifest.json")
    generate(args, protocol, manifest)
    if args.stage == "prepare":
        return
    if args.stage in ("all", "train"):
        pipeline.train_classifier(args, protocol, classifier)
    if args.stage in ("all", "train", "test"):
        pipeline.evaluate_split(args, "val", classifier)
    if args.stage in ("all", "test"):
        pipeline.evaluate_split(args, "test", classifier)
    if args.stage in ("all", "test", "report"):
        report(args, protocol, classifier)
    controls.preserve_check(protocol)
    print(f"t50实验目录：{args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
