"""Paired CNN backbones: original versus frozen full6199-Stage1/train4951-Stage2 t50 PNGs."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import torch
import torchvision
from torch import nn
from torchvision import models

import run_clean_baseline_vs_stage2train4951 as controls
import run_stage2_train4951_remapped_resnet18 as pipeline
from diffusion_stripe.data import collect_images
from diffusion_stripe.training import save_json
from resnet18_full6199_loss_earlystop import seed_worker
from train_stage2_train4951 import read_json, sha256

WORKSPACE = Path(__file__).resolve().parent
SOURCE = WORKSPACE / 'experiments/stage2_train4951_t50_remapped_resnet18_seed43'
MAP_ROOT = SOURCE / 'datasets/post_formula_prediction'
SOURCE_RUN = SOURCE / 'runs/resnet18_scratch_val_loss'
RAW_REFERENCE = controls.DEFAULT_ROOT / 'runs/resnet18_scratch_original_val_loss'
GENERATOR_ROOT = WORKSPACE / 'experiments/stage2_train4951_from_existing_stage1_seed43'
STAGE1_ROOT = WORKSPACE / 'experiments/full6199_two_stage/stage1'
DEFAULT_ROOT = WORKSPACE / 'experiments/multibackbone_original_vs_stage2train4951_t50_seed43_20261017_v1'
ARCHITECTURES = ('resnet50', 'densenet121', 'mobilenet_v3_large', 'efficientnet_b0')
ARMS = ('original', 't50_remapped')
VERSION = 'paired_backbones_full6199stage1_train4951stage2_t50_v1'
REPORT_NAME = 'MULTIBACKBONE_T50_COMPARISON_REPORT.md'
CODE_FILES = (Path(__file__).name, 'run_stage2_train4951_remapped_resnet18.py',
              'run_clean_baseline_vs_stage2train4951.py', 'run_resnet18_full6199_external373.py',
              'resnet18_full6199_loss_earlystop.py', 'diffusion_stripe/data.py',
              'diffusion_stripe/model.py', 'diffusion_stripe/training.py')


def build_model(architecture, pretrained=False):
    if pretrained or architecture not in (*ARCHITECTURES, 'resnet18'):
        raise ValueError('Use a supported randomly initialized classifier; no ImageNet weights')
    model = getattr(models, architecture)(weights=None)
    if architecture.startswith('resnet'):
        model.fc = nn.Sequential(nn.Dropout(.30), nn.Linear(model.fc.in_features, 3))
    elif architecture == 'densenet121':
        model.classifier = nn.Sequential(nn.Dropout(.30), nn.Linear(model.classifier.in_features, 3))
    else:
        # Keep each model's native projection/nonlinearity, unify head dropout and output classes.
        for layer in model.classifier:
            if isinstance(layer, nn.Dropout):
                layer.p = .30
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 3)
    return model


def guard_output(output):
    output = output.resolve()
    if output.parent != (WORKSPACE / 'experiments').resolve():
        raise ValueError('Use a dedicated direct child of workspace experiments')
    if output.exists():
        marker = output / 'experiment_protocol.json'
        if any(output.iterdir()) and (not marker.exists() or read_json(marker).get('protocol_version') != VERSION):
            raise ValueError('Cannot overwrite another or unrecognized experiment')
    return output


def preserve_check(protocol):
    controls.preserve_check(protocol)
    for name, digest in protocol['code_sha256'].items():
        if sha256(WORKSPACE / name) != digest:
            raise ValueError(f'Shared/entry code changed: {name}')


def verify_maps():
    marker = read_json(SOURCE / 'generation_complete.json')
    records = collect_images(MAP_ROOT, expected_count=6199)
    with ThreadPoolExecutor(max_workers=8) as executor:
        hashes = list(executor.map(sha256, [r.path for r in records]))
    if dict(zip((r.key for r in records), hashes)) != marker['output_sha256']:
        raise ValueError('Frozen t50 images differ from their completion seal')
    counts = {s: dict(Counter(r.key.split('/')[1] for r in records if r.key.startswith(s + '/')))
              for s in pipeline.SPLITS}
    if counts != pipeline.EXPECTED_COUNTS:
        raise ValueError('Frozen t50 split/class counts changed')
    return marker, records


def preflight(args, classifier):
    source = read_json(SOURCE / 'generation_protocol.json')
    if (source['stage1_training_count'], source['generator_training_count'], source['generator_epoch'], source['timestep']) != (6199, 4951, 20, 50):
        raise ValueError('Wrong generator version: require full6199 Stage1 -> train4951 Stage2 t0, t50 inference')
    checkpoint = torch.load(source['checkpoint'], map_location='cpu', weights_only=False)
    cfg = checkpoint['train_config']
    if cfg['dataset_size'] != 4951 or cfg['splits'] != ['train'] or cfg['fixed_timestep'] != 0:
        raise ValueError('Stage2 training scope/timestep differs')
    if Path(checkpoint['source_checkpoint']).resolve() != (STAGE1_ROOT / 'best_train_loss.pt').resolve():
        raise ValueError('Stage2 does not initialize from full6199 Stage1')
    if sha256(Path(source['checkpoint'])) != source['checkpoint_sha256'] or read_json(STAGE1_ROOT / 'config.json')['dataset_size'] != 6199:
        raise ValueError('Generator weights or Stage1 training count changed')
    del checkpoint
    marker, mapped = verify_maps()
    if marker['checkpoint_sha256'] != source['checkpoint_sha256'] or marker['timestep'] != 50 or marker['recursion_steps'] != 50:
        raise ValueError('Images came from another generator/timestep/process')
    original_root = Path(source['original_root']).resolve()
    records = collect_images(original_root, expected_count=6199)
    if [r.key for r in records] != [r.key for r in mapped]:
        raise ValueError('Original and remapped sample keys do not match')
    with ThreadPoolExecutor(max_workers=8) as executor:
        manifest = list(executor.map(pipeline.inspect_original, records))
    input_sha = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if input_sha != source['input_manifest_sha256']:
        raise ValueError('Original image files differ from t50 generation inputs')
    split_path = original_root / 'reports/crops_manifest.csv'
    if sha256(split_path) != read_json(GENERATOR_ROOT / 'experiment_protocol.json')['source_group_manifest_sha256']:
        raise ValueError('Original source-group split manifest changed since Stage2 training')
    with split_path.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    groups = {s: {r['canonical_source_group'] for r in rows if r['split'] == s} for s in pipeline.SPLITS}
    if {r['crop_path'] for r in rows} != {r.key for r in records} or any(
        groups[a] & groups[b] for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))
    ):
        raise ValueError('Split manifest mismatch or source-group overlap')
    reference = read_json(SOURCE_RUN / 'config.json')
    if sha256(pipeline.common.DEFAULT_CLASSIFIER_SCRIPT) != reference['classifier_script_sha256']:
        raise ValueError('The original shared classifier implementation changed')
    for attribute, field in (('epochs', 'max_epochs'), ('patience', 'patience'), ('min_delta', 'min_delta'),
                             ('batch_size', 'batch_size'), ('workers', 'workers'), ('device', 'device'), ('selection', 'selection')):
        setattr(args, attribute, reference[field])
    if args.selection != 'val_loss' or args.epochs != 100 or reference['pretrained']:
        raise ValueError('Keep the existing scratch/max100/val_loss classifier protocol')
    protected = {str(Path(source['checkpoint']).resolve()): source['checkpoint_sha256'],
                 str(pipeline.common.DEFAULT_CLASSIFIER_SCRIPT): reference['classifier_script_sha256'],
                 str(STAGE1_ROOT / 'best_train_loss.pt'): sha256(STAGE1_ROOT / 'best_train_loss.pt'),
                 str(Path(source['noise_file']).resolve()): source['noise_sha256'],
                 str(SOURCE / 'generation_protocol.json'): sha256(SOURCE / 'generation_protocol.json'),
                 str(SOURCE / 'generation_complete.json'): sha256(SOURCE / 'generation_complete.json')}
    for root in (RAW_REFERENCE, SOURCE_RUN, GENERATOR_ROOT / 'stage2'):
        protected.update({str(p.resolve()): sha256(p) for p in root.iterdir() if p.is_file()})
    specifications = {}
    for architecture in ('resnet18', *ARCHITECTURES):
        classifier.set_seed(43)
        model = build_model(architecture)
        specifications[architecture] = {'parameters': sum(p.numel() for p in model.parameters()),
                                       'initial_model_sha256': pipeline.common.state_sha256(model)}
        del model
    if specifications['resnet18']['initial_model_sha256'] != reference['initial_model_sha256']:
        raise ValueError('ResNet18 constructor does not match historical baseline initialization')
    controls.require_same_config(reference, read_json(RAW_REFERENCE / 'config.json'))
    for run in (RAW_REFERENCE, SOURCE_RUN):
        controls.audit_history(run, read_json(run / 'config.json'))
        for split in ('val', 'test'):
            pipeline.audited_metrics(argparse.Namespace(run_dir=run), split, classifier)
    protocol = {'protocol_version': VERSION, 'original_root': str(original_root), 'remapped_root': str(MAP_ROOT),
        'checkpoint': source['checkpoint'], 'checkpoint_sha256': source['checkpoint_sha256'],
        'stage1_training_count': 6199, 'stage2_training_count': 4951, 'stage2_timestep': 0,
        'timestep': 50, 'recursion_steps': 50, 'final_t0_remapping': True,
        'count': 6199, 'split_class_counts': pipeline.EXPECTED_COUNTS, 'input_manifest_sha256': input_sha,
        'source_group_manifest_sha256': sha256(split_path), 'source_group_overlap': 0,
        'noise_file': source['noise_file'], 'noise_sha256': source['noise_sha256'],
        'noise_note': 'Existing fixed shared inference epsilon, not an archived actual Stage1 training noise.',
        'all_models_reuse_same_frozen6199_pngs_without_rerendering': True,
        'architecture_specs': specifications, 'pretrained': False, 'head_dropout': .30, 'seed': 43,
        'max_epochs': args.epochs, 'patience': args.patience, 'min_delta': args.min_delta, 'selection': 'val_loss',
        'shared_classifier_reference': {k: v for k, v in reference.items() if k in controls.SHARED_KEYS and k not in ('model', 'parameters', 'initial_model_sha256')},
        'resnet18_reference_runs': {'original': str(RAW_REFERENCE), 't50_remapped': str(SOURCE_RUN)},
        'preserved_reference_files_sha256': protected, 'code_sha256': {name: sha256(WORKSPACE / name) for name in CODE_FILES},
        'torch_version': torch.__version__, 'torchvision_version': torchvision.__version__,
        'limitation': 'Stage1 saw all6199 target stripes, including held-out splits. Not a full-pipeline unseen-test generalization experiment.'}
    preserve_check(protocol)
    gc.collect()
    print('预检通过：冻结full6199 Stage1 -> train4951 Stage2的6199张t50再映射图；原split43不变。', flush=True)
    return protocol


@contextmanager
def architecture_adapter(classifier, architecture, protocol):
    original_builder, original_config = classifier.build_model, pipeline.classifier_config

    def config(args, source, model, loaded_classifier, *, data_root=None):
        if data_root is None:
            raise ValueError('Explicitly select original or existing frozen t50 dataset')
        root = Path(data_root).resolve()
        arm = 'original' if root == Path(protocol['original_root']).resolve() else 't50_remapped' if root == MAP_ROOT.resolve() else None
        if arm is None:
            raise ValueError('Unapproved classifier dataset')
        result = original_config(args, source, model, loaded_classifier, data_root=root)
        result.update(model=architecture, input_method=arm, generator_sha256=None if arm == 'original' else protocol['checkpoint_sha256'],
                      backbone_adapter_sha256=protocol['code_sha256'][Path(__file__).name], head_dropout=.30)
        expected = protocol['architecture_specs'][architecture]
        if result['initial_model_sha256'] != expected['initial_model_sha256'] or result['parameters'] != expected['parameters']:
            raise ValueError('Classifier initialization differs across paired arms')
        return result

    classifier.build_model = lambda pretrained: build_model(architecture, pretrained)
    pipeline.classifier_config = config
    try:
        yield
    finally:
        classifier.build_model, pipeline.classifier_config = original_builder, original_config


def run_args(args, architecture, arm):
    return argparse.Namespace(**{**vars(args), 'run_dir': args.output_dir / 'runs' / architecture / arm})


def report(args, protocol, classifier):
    preserve_check(protocol)
    verify_maps()
    runs = {'resnet18': {'original': RAW_REFERENCE, 't50_remapped': SOURCE_RUN}}
    pending = []
    for architecture in ARCHITECTURES:
        pair = {arm: run_args(args, architecture, arm).run_dir for arm in ARMS}
        if all((run / 'test_complete.json').exists() and (run / 'val_complete.json').exists() for run in pair.values()):
            runs[architecture] = pair
        else:
            pending.append(architecture)
    rows, details, durations, per_class, deltas, paired = [], {}, {}, [], [], {}
    for architecture, arms in runs.items():
        configs = {arm: read_json(run / 'config.json') for arm, run in arms.items()}
        controls.require_same_config(*configs.values())
        if any(cfg['model'] != architecture for cfg in configs.values()):
            raise ValueError('Wrong architecture in classifier checkpoint config')
        if Path(configs['original']['dataset']).resolve() != Path(protocol['original_root']).resolve() or Path(configs['t50_remapped']['dataset']).resolve() != MAP_ROOT.resolve():
            raise ValueError('Paired classifier trained the wrong input images')
        durations[architecture] = {arm: controls.audit_history(run, configs[arm]) for arm, run in arms.items()}
        details[architecture], paired[architecture] = {}, {}
        for split in ('val', 'test'):
            details[architecture][split] = {}
            for arm, run in arms.items():
                metric = pipeline.audited_metrics(argparse.Namespace(run_dir=run), split, classifier)
                details[architecture][split][arm] = metric
                done = durations[architecture][arm]
                rows.append({'architecture': architecture, 'input': arm, 'split': split, 'samples': metric['samples'],
                    'parameters': configs[arm]['parameters'], 'epochs_completed': done['epochs_completed'],
                    'selected_epoch': done['selected_epoch'], 'source': 'existing_frozen' if architecture == 'resnet18' else 'newly_trained',
                    **{name: metric[name] for name in pipeline.common.METRIC_NAMES}})
                for i, name in enumerate(pipeline.CLASSES):
                    per_class.append({'architecture': architecture, 'input': arm, 'split': split, 'class': name,
                        'support': pipeline.EXPECTED_COUNTS[split][name],
                        **{key: metric[key][i] for key in ('precision', 'recall', 'f1')}})
            raw = controls.prediction_correctness(arms['original'], split)
            mapped = controls.prediction_correctness(arms['t50_remapped'], split)
            if raw.keys() != mapped.keys():
                raise ValueError('Original/remapped predictions are not paired on the same files')
            paired[architecture][split] = {'both_correct': sum(raw[k] and mapped[k] for k in raw),
                'original_only_correct': sum(raw[k] and not mapped[k] for k in raw),
                'remapped_only_correct': sum(not raw[k] and mapped[k] for k in raw),
                'both_wrong': sum(not raw[k] and not mapped[k] for k in raw)}
            deltas.append({'architecture': architecture, 'split': split,
                **{name: 100*(details[architecture][split]['t50_remapped'][name]-details[architecture][split]['original'][name])
                   for name in pipeline.common.METRIC_NAMES}})
    pipeline.common.write_csv(args.output_dir / 'multibackbone_metrics.csv', rows)
    pipeline.common.write_csv(args.output_dir / 'multibackbone_per_class.csv', per_class)
    pipeline.common.write_csv(args.output_dir / 'remapped_minus_original_deltas_pp.csv', deltas)
    save_json({'metrics': rows, 'full_metrics': details, 'training': durations, 'paired_correctness': paired,
        'deltas_percentage_points': deltas, 'completed_architectures': list(runs), 'pending_architectures': pending,
        'generator_sha256': protocol['checkpoint_sha256'], 'matched_initialization_within_each_architecture': True},
        args.output_dir / 'multibackbone_metrics.json')
    names = {'original': '原图', 't50_remapped': 'Stage2 t50再映射'}
    lines = ['# 多分类器对照：原图 vs Stage2 t=50再映射', '',
        '## 实验版本与比较对象', '',
        '生成器为Stage1在全部6199张条纹图上预训练，再以train4951对原图/条纹图在t=0微调Stage2满20轮的版本；不是Stage1也仅用4951张训练的另一版。', '',
        f"权重：`{protocol['checkpoint']}`；SHA256 `{protocol['checkpoint_sha256']}`。", '',
        '比较单位是同一分类器下的原图输入与生成图输入：C_k(I) 对比 C_k(G50(I))，其中G50为冻结生成器的完整50步公式更新后R(x0,0)再映射。Stage2是图像生成器，不是与ResNet直接比较类别预测的分类器。', '',
        '所有模型复用同一批已冻结6199张RGB64×64 PNG，原图与生成图文件键和split/class目录一致；本次不重训生成器、不重新加噪或生成图片。噪声仍是该MD版本保存的单份固定推理epsilon，全部6199图及递推步共享，并非原Stage1训练噪声的恢复记录。', '',
        '## 数据与训练控制', '',
        '| 集合 | defect | normal | nut | 总数 |', '|---|---:|---:|---:|---:|']
    for split in pipeline.SPLITS:
        lines.append(f"| {split} | " + ' | '.join(str(pipeline.EXPECTED_COUNTS[split][c]) for c in pipeline.CLASSES) + f" | {sum(pipeline.EXPECTED_COUNTS[split].values())} |")
    lines += ['', '原图和再映射每组均仅4951图更新分类器参数，619验证图用于验证损失早停/选模，629测试图不参与梯度、学习率或epoch选择。沿用原同源组隔离划分，不重新切分。', '',
        '新增ResNet50、DenseNet121、MobileNetV3-Large、EfficientNet-B0，均随机初始化，不加载ImageNet权重；同一架构的两组具有相同初始权重SHA256。保留各模型原生主体/投影，末端三分类，分类头Dropout统一0.30。输入仍64×64，不为某组单独放大。', '',
        '所有新增组继承原ResNet18训练函数和公共超参数：seed43，AdamW lr0.001、weight_decay0.0001，batch64、workers4、AMP，训练集类别权重和标签平滑0.05；RandomAffine+ColorJitter、mean/std0.5。验证测试不增广，使用无权重且无标签平滑交叉熵。最多100轮，按绝对最低验证损失保存权重，patience15/min_delta0.0001；ReduceLROnPlateau同样监测val_loss。', '',
        'ResNet18两组只读复用同版本已经冻结的结果；其余四种各两组本次新训练。统一协议用于控制变量，不代表已经分别充分调参得到各架构的最高能力。不同架构参数量/计算预算不同，不能把更大分类器的优势直接当作生成器的效果。', '',
        '## 训练轮次', '', '| 分类器 | 输入 | 参数量 | 完成轮次 | 选中轮次 | 最低验证损失 | 来源 |', '|---|---|---:|---:|---:|---:|---|']
    for architecture, arms in runs.items():
        for arm, run in arms.items():
            done = durations[architecture][arm]
            lines.append(f"| {architecture} | {names[arm]} | {protocol['architecture_specs'][architecture]['parameters']} | {done['epochs_completed']} | {done['selected_epoch']} | {done['selected_score']:.6f} | " + ('历史冻结' if architecture == 'resnet18' else '本次新训练') + ' |')
    for split, title in (('val', '验证集（619张）'), ('test', '测试集（629张）')):
        lines += ['', f'## {title}', '', 'P/R/F1列为weighted，另外提供macro-P/R/F1；单标签多分类weighted-R等于ACC。', '',
            '| 分类器 | 输入 | ACC | weighted-P | weighted-R | weighted-F1 | macro-P | macro-R | macro-F1 |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for row in rows:
            if row['split'] == split:
                lines.append(f"| {row['architecture']} | {names[row['input']]} | " + ' | '.join(f"{100*row[k]:.2f}%" for k in pipeline.common.METRIC_NAMES) + ' |')
    lines += ['', '## 相同分类器下再映射减原图（百分点）', '', '| 分类器 | 集合 | ΔACC | Δweighted-F1 | Δmacro-F1 |', '|---|---|---:|---:|---:|']
    for row in deltas:
        lines.append(f"| {row['architecture']} | {row['split']} | {row['accuracy']:+.2f} | {row['weighted_f1']:+.2f} | {row['macro_f1']:+.2f} |")
    lines += ['', '## 逐类别P/R/F1', '', '| 分类器 | 集合 | 输入 | 类别 | 数量 | P | R | F1 |', '|---|---|---|---|---:|---:|---:|---:|']
    for row in per_class:
        lines.append(f"| {row['architecture']} | {row['split']} | {names[row['input']]} | {row['class']} | {row['support']} | " + ' | '.join(f"{100*row[k]:.2f}%" for k in ('precision', 'recall', 'f1')) + ' |')
    lines += ['', '## 解释边界', '',
        'Stage2与分类器只训练train4951，但Stage1曾训练全部6199目标条纹图，包括val/test对应目标；目标条纹方向还由类别定义。因此属于旧生成器数据暴露设定下的多分类器对照，不能作为全流程未见测试样本的严格泛化结果。换分类器无法消除这种暴露。', '',
        '仅seed43和单个固定epsilon，不声称统计显著性或跨种子稳定性。同一测试集已经用于多次方法比较，不能称为未参与方法开发的最终独立测试。分类指标不能代替MAE/PSNR/SSIM，本次不重算图像相似度。', '',
        f"完成分类器：{', '.join(runs)}。待完成：{', '.join(pending) or '无'}。", '', '## 文件与运行', '',
        f'- 程序：[{Path(__file__).name}]({Path(__file__).as_posix()})',
        f'- 冻结再映射图：[post_formula_prediction]({MAP_ROOT.as_posix()})',
        f'- 总指标：[multibackbone_metrics.json]({(args.output_dir / "multibackbone_metrics.json").as_posix()})',
        f'- 指标表：[multibackbone_metrics.csv]({(args.output_dir / "multibackbone_metrics.csv").as_posix()})',
        f'- 逐类表：[multibackbone_per_class.csv]({(args.output_dir / "multibackbone_per_class.csv").as_posix()})',
        f'- 差值：[remapped_minus_original_deltas_pp.csv]({(args.output_dir / "remapped_minus_original_deltas_pp.csv").as_posix()})',
        '- 新训练的每组runs/<模型>/<original或t50_remapped>/均含best.pt、last.pt、config.json、history.csv、val/test预测和混淆矩阵。', '',
        '模型构造接口参考：[TorchVision官方模型文档](https://docs.pytorch.org/vision/stable/models.html)。', '',
        '```powershell', 'cd D:\\Cold_Diffusion', f'conda run --no-capture-output -n diffusion_env python -X utf8 -u {Path(__file__).name}', '```', '',
        '--stage preflight仅核验，train训练指定模型两组并评估val，test仅冻结评估，report只读既有预测汇总；默认all完整运行。--models可选本次训练模型子集。非空未知目录拒绝覆盖；相同命令核验后复用完成组或按last.pt继续，多进程增广续训不保证逐位复现。旧权重/结果、总实验MD和论文均不改动。']
    (args.output_dir / REPORT_NAME).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    for row in rows:
        if row['split'] == 'test':
            print(f"{row['architecture']} {row['input']}: test ACC={100*row['accuracy']:.2f}%, weightedF1={100*row['weighted_f1']:.2f}%, macroF1={100*row['macro_f1']:.2f}%", flush=True)
    preserve_check(protocol)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--models', nargs='+', choices=ARCHITECTURES, default=list(ARCHITECTURES))
    parser.add_argument('--stage', choices=('all', 'preflight', 'train', 'test', 'report'), default='all')
    args = parser.parse_args()
    if len(set(args.models)) != len(args.models):
        raise ValueError('Do not repeat models')
    args.output_dir = guard_output(args.output_dir)
    classifier = pipeline.common.load_classifier(pipeline.common.DEFAULT_CLASSIFIER_SCRIPT)
    classifier.seed_worker = seed_worker
    protocol = preflight(args, classifier)
    if args.stage == 'preflight':
        return
    marker = args.output_dir / 'experiment_protocol.json'
    if marker.exists() and read_json(marker) != protocol:
        raise ValueError('Existing generator/data/noise/model/config/source protocol differs')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not marker.exists():
        save_json(protocol, marker)
    if args.stage in ('all', 'train'):
        for architecture in args.models:
            with architecture_adapter(classifier, architecture, protocol):
                for arm in ARMS:
                    print(f'开始 {architecture} / {arm}，train4951/val619，最大100轮、val_loss早停。', flush=True)
                    selected_args = run_args(args, architecture, arm)
                    data_root = Path(protocol['original_root']) if arm == 'original' else MAP_ROOT
                    pipeline.train_classifier(selected_args, protocol, classifier, data_root=data_root)
                    pipeline.evaluate_split(selected_args, 'val', classifier)
                    preserve_check(protocol)
                    save_json({'phase': 'training', 'architecture': architecture, 'input': arm}, args.output_dir / 'progress.json')
    if args.stage in ('all', 'test'):
        # Freeze all requested classifiers before inspecting any new test result.
        for architecture in args.models:
            with architecture_adapter(classifier, architecture, protocol):
                for arm in ARMS:
                    selected_args = run_args(args, architecture, arm)
                    pipeline.evaluate_split(selected_args, 'val', classifier)
                    pipeline.evaluate_split(selected_args, 'test', classifier)
    if args.stage in ('all', 'test', 'report'):
        report(args, protocol, classifier)
        results = read_json(args.output_dir / 'multibackbone_metrics.json')
        save_json({'phase': 'complete' if not results['pending_architectures'] else 'partial',
                   'completed_architectures': results['completed_architectures'],
                   'pending_architectures': results['pending_architectures']}, args.output_dir / 'progress.json')
    preserve_check(protocol)
    print(f'实验目录：{args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
