"""Add four fixed CNN backbones to the frozen original/t50 paired experiment."""
from __future__ import annotations

import argparse
import gc
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn
from torchvision import models

import run_multibackbone_stage2_t50_comparison as base
from diffusion_stripe.training import save_json
from resnet18_full6199_loss_earlystop import seed_worker
from train_stage2_train4951 import read_json, sha256

pipeline, controls = base.pipeline, base.controls
WORKSPACE, PARENT = base.WORKSPACE, base.DEFAULT_ROOT
DEFAULT_ROOT = WORKSPACE / 'experiments/multibackbone_plus4_original_vs_stage2train4951_t50_seed43_20261017_v1'
ARCHITECTURES = ('resnet34', 'shufflenet_v2_x1_0', 'squeezenet1_1', 'regnet_y_400mf')
ARMS = base.ARMS
VERSION = 'four_more_fixed_backbones_full6199stage1_train4951stage2_t50_v1'
REPORT_NAME = 'NINE_BACKBONE_T50_COMPARISON_REPORT.md'
NAMES = {'resnet18': 'ResNet18', 'resnet50': 'ResNet50', 'densenet121': 'DenseNet121',
         'mobilenet_v3_large': 'MobileNetV3-Large', 'efficientnet_b0': 'EfficientNet-B0',
         'resnet34': 'ResNet34', 'shufflenet_v2_x1_0': 'ShuffleNetV2 x1.0',
         'squeezenet1_1': 'SqueezeNet1.1', 'regnet_y_400mf': 'RegNetY-400MF'}
INPUT_NAMES = {'original': '原图', 't50_remapped': 't50再映射'}


def build_model(architecture, pretrained=False):
    if pretrained or architecture not in ARCHITECTURES:
        raise ValueError('Require one of the four predeclared backbones with weights=None')
    if architecture == 'squeezenet1_1':
        # Preserve the native Conv1x1 -> ReLU -> global-pooling head and its special initialization.
        return models.squeezenet1_1(weights=None, num_classes=3, dropout=.30)
    model = getattr(models, architecture)(weights=None)
    model.fc = nn.Sequential(nn.Dropout(.30), nn.Linear(model.fc.in_features, 3))
    return model


def guard_output(output):
    output = output.resolve()
    if output.parent != (WORKSPACE / 'experiments').resolve():
        raise ValueError('Use a dedicated direct child of workspace experiments')
    if output.exists() and any(output.iterdir()):
        marker = output / 'experiment_protocol.json'
        if not marker.exists() or read_json(marker).get('protocol_version') != VERSION:
            raise ValueError('Cannot overwrite another experiment')
    return output


def run_args(args, architecture, arm):
    if architecture not in ARCHITECTURES or arm not in ARMS:
        raise ValueError('Unplanned backbone or input arm')
    return argparse.Namespace(**{**vars(args), 'run_dir': args.output_dir / 'runs' / architecture / arm})


def old_runs():
    runs = {'resnet18': {'original': base.RAW_REFERENCE, 't50_remapped': base.SOURCE_RUN}}
    runs.update({arch: {arm: PARENT / 'runs' / arch / arm for arm in ARMS} for arch in base.ARCHITECTURES})
    return runs


def preserve_check(protocol):
    controls.preserve_check(protocol)
    for path, expected in protocol['code_files_sha256'].items():
        if sha256(Path(path)) != expected:
            raise ValueError(f'Entry/shared code changed: {path}')


def preflight(args, classifier):
    parent = base.preflight(args, classifier)
    if parent != read_json(PARENT / 'experiment_protocol.json'):
        raise ValueError('Reference generator/data/code/protocol differs from the completed five-backbone experiment')
    protected = dict(parent['preserved_reference_files_sha256'])
    for path in PARENT.iterdir():
        if path.is_file():
            protected[str(path)] = sha256(path)
    old_details = old_runs()
    for architecture, arms in old_details.items():
        configs = [read_json(run / 'config.json') for run in arms.values()]
        controls.require_same_config(*configs)
        for arm, run in arms.items():
            cfg = read_json(run / 'config.json')
            if cfg['model'] != architecture:
                raise ValueError('Wrong historical classifier architecture')
            controls.audit_history(run, cfg)
            for split in ('val', 'test'):
                pipeline.audited_metrics(argparse.Namespace(run_dir=run), split, classifier)
            protected.update({str(p): sha256(p) for p in run.iterdir() if p.is_file()})
    summary = WORKSPACE / 'COLD_DIFFUSION_RESNET18_EXPERIMENT_SUMMARY.md'
    protected[str(summary)] = sha256(summary)
    specs = dict(parent['architecture_specs'])
    for architecture in ARCHITECTURES:
        classifier.set_seed(43)
        model = build_model(architecture)
        specs[architecture] = {
            'parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'initial_model_sha256': pipeline.common.state_sha256(model),
            'head': 'Dropout0.30 -> Conv1x1(512,3) -> ReLU -> AdaptiveAvgPool2d' if architecture == 'squeezenet1_1'
                    else f'Dropout0.30 -> Linear({model.fc[-1].in_features},3)',
        }
        del model
    dependencies = {str(WORKSPACE / name): sha256(WORKSPACE / name) for name in base.CODE_FILES}
    dependencies[str(Path(__file__).resolve())] = sha256(Path(__file__))
    dependencies[str(pipeline.common.DEFAULT_CLASSIFIER_SCRIPT)] = parent['shared_classifier_reference']['classifier_script_sha256']
    protocol = {
        'protocol_version': VERSION, 'predeclared_new_backbones': list(ARCHITECTURES),
        'parent_protocol_path': str(PARENT / 'experiment_protocol.json'),
        'parent_protocol_sha256': sha256(PARENT / 'experiment_protocol.json'),
        'original_root': parent['original_root'], 'remapped_root': str(base.MAP_ROOT),
        'checkpoint': parent['checkpoint'], 'checkpoint_sha256': parent['checkpoint_sha256'],
        'stage1_training_count': 6199, 'stage2_training_count': 4951, 'stage2_timestep': 0,
        'timestep': 50, 'recursion_steps': 50, 'final_t0_remapping': True,
        'noise_file': parent['noise_file'], 'noise_sha256': parent['noise_sha256'], 'noise_note': parent['noise_note'],
        'split_class_counts': parent['split_class_counts'], 'source_group_overlap': 0,
        'input_manifest_sha256': parent['input_manifest_sha256'],
        'source_group_manifest_sha256': parent['source_group_manifest_sha256'],
        'architecture_specs': specs, 'shared_classifier_reference': parent['shared_classifier_reference'],
        'pretrained': False, 'seed': 43, 'head_dropout': .30,
        'max_epochs': 100, 'selection': 'val_loss', 'patience': 15, 'min_delta': .0001,
        'all_new_weights_frozen_before_any_new_test': True,
        'all_models_reuse_same_frozen6199_pngs_without_rerendering': True,
        'preserved_reference_files_sha256': protected, 'code_files_sha256': dependencies,
        'limitation': parent['limitation'],
    }
    preserve_check(protocol)
    gc.collect()
    print('新增4种分类器预检通过：原图/既有t50 PNG配对，原split43与生成器不变。', flush=True)
    for architecture in ARCHITECTURES:
        print(f"  {NAMES[architecture]}：{specs[architecture]['parameters']:,}参数；{specs[architecture]['head']}", flush=True)
    return protocol


@contextmanager
def architecture_adapter(classifier, architecture, protocol):
    if architecture not in ARCHITECTURES:
        raise ValueError('Only adapt the four predeclared new classifiers')
    before_builder, before_config = classifier.build_model, pipeline.classifier_config

    def config(args, source, model, loaded_classifier, *, data_root=None):
        if data_root is None:
            raise ValueError('Explicit original or frozen-remapped dataset required')
        root = Path(data_root).resolve()
        arm = 'original' if root == Path(protocol['original_root']).resolve() else 't50_remapped' if root == base.MAP_ROOT.resolve() else None
        if arm is None:
            raise ValueError('Unexpected classifier dataset')
        result = before_config(args, source, model, loaded_classifier, data_root=root)
        result.update(model=architecture, input_method=arm,
                      generator_sha256=None if arm == 'original' else protocol['checkpoint_sha256'],
                      backbone_adapter_sha256=protocol['code_files_sha256'][str(Path(__file__).resolve())],
                      head_dropout=.30, head=protocol['architecture_specs'][architecture]['head'])
        spec = protocol['architecture_specs'][architecture]
        if result['initial_model_sha256'] != spec['initial_model_sha256'] or result['parameters'] != spec['parameters']:
            raise ValueError('Classifier initialization differs from its predeclared spec')
        for key, expected in protocol['shared_classifier_reference'].items():
            if result.get(key) != expected:
                raise ValueError(f'Common classifier control changed: {key}')
        return result

    classifier.build_model = lambda pretrained: build_model(architecture, pretrained)
    pipeline.classifier_config = config
    try:
        yield
    finally:
        classifier.build_model, pipeline.classifier_config = before_builder, before_config


def require_all_new_frozen(args, protocol):
    sealed = {}
    for architecture in ARCHITECTURES:
        for arm in ARMS:
            run = run_args(args, architecture, arm).run_dir
            cfg = read_json(run / 'config.json')
            done = controls.audit_history(run, cfg)
            if cfg['model'] != architecture or cfg['initial_model_sha256'] != protocol['architecture_specs'][architecture]['initial_model_sha256']:
                raise ValueError('Wrong architecture or initialization in freeze barrier')
            sealed[f'{architecture}/{arm}'] = {'selected_epoch': done['selected_epoch'], 'best_sha256': done['best_sha256']}
    return sealed


def report(args, protocol, classifier):
    preserve_check(protocol)
    base.verify_maps()
    runs, pending = old_runs(), []
    for architecture in ARCHITECTURES:
        pair = {arm: run_args(args, architecture, arm).run_dir for arm in ARMS}
        if all((run / f'{split}_complete.json').exists() for run in pair.values() for split in ('val', 'test')):
            runs[architecture] = pair
        else:
            pending.append(architecture)
    rows, per_class, details, training, deltas, paired = [], [], {}, {}, [], {}
    for architecture, arms in runs.items():
        configs = {arm: read_json(run / 'config.json') for arm, run in arms.items()}
        controls.require_same_config(*configs.values())
        for arm, cfg in configs.items():
            expected_root = Path(protocol['original_root']) if arm == 'original' else base.MAP_ROOT
            if cfg['model'] != architecture or Path(cfg['dataset']).resolve() != expected_root.resolve():
                raise ValueError('Wrong model or paired dataset in report')
        training[architecture] = {arm: controls.audit_history(run, configs[arm]) for arm, run in arms.items()}
        details[architecture], paired[architecture] = {}, {}
        for split in ('val', 'test'):
            details[architecture][split] = {}
            for arm, run in arms.items():
                metric = pipeline.audited_metrics(argparse.Namespace(run_dir=run), split, classifier)
                done = training[architecture][arm]
                details[architecture][split][arm] = metric
                rows.append({'architecture': architecture, 'input': arm, 'split': split, 'samples': metric['samples'],
                             'parameters': configs[arm]['parameters'], 'epochs_completed': done['epochs_completed'],
                             'selected_epoch': done['selected_epoch'], 'loss': metric['loss'],
                             'source': 'newly_trained' if architecture in ARCHITECTURES else 'existing_frozen',
                             'run_dir': str(run), **{k: metric[k] for k in pipeline.common.METRIC_NAMES}})
                for i, name in enumerate(pipeline.CLASSES):
                    per_class.append({'architecture': architecture, 'input': arm, 'split': split, 'class': name,
                                      'support': pipeline.EXPECTED_COUNTS[split][name],
                                      **{k: metric[k][i] for k in ('precision', 'recall', 'f1')}})
            raw = controls.prediction_correctness(arms['original'], split)
            mapped = controls.prediction_correctness(arms['t50_remapped'], split)
            if raw.keys() != mapped.keys():
                raise ValueError('Original/remapped evaluation images are not paired')
            paired[architecture][split] = {
                'both_correct': sum(raw[k] and mapped[k] for k in raw),
                'original_only_correct': sum(raw[k] and not mapped[k] for k in raw),
                'remapped_only_correct': sum(not raw[k] and mapped[k] for k in raw),
                'both_wrong': sum(not raw[k] and not mapped[k] for k in raw)}
            deltas.append({'architecture': architecture, 'split': split,
                           **{k: 100*(details[architecture][split]['t50_remapped'][k]-details[architecture][split]['original'][k])
                              for k in pipeline.common.METRIC_NAMES}})
    pipeline.common.write_csv(args.output_dir / 'nine_backbone_metrics.csv', rows)
    pipeline.common.write_csv(args.output_dir / 'nine_backbone_per_class.csv', per_class)
    pipeline.common.write_csv(args.output_dir / 'remapped_minus_original_deltas_pp.csv', deltas)
    save_json({'metrics': rows, 'per_class': per_class, 'full_metrics': details, 'training': training,
               'paired_correctness': paired, 'deltas_percentage_points': deltas,
               'completed_architectures': list(runs), 'pending_architectures': pending,
               'generator_sha256': protocol['checkpoint_sha256'],
               'matched_initialization_within_each_architecture': True}, args.output_dir / 'nine_backbone_metrics.json')
    lines = ['# 九种分类器对照：原图 vs Stage2 t50再映射', '', '## 实验范围与版本', '',
             '新增名单在训练前固定：ResNet34、ShuffleNetV2 x1.0、SqueezeNet1.1、RegNetY-400MF，各原图/再映射两组，共8次新训练。不因分数高低替换模型或删除负结果。已有ResNet18、ResNet50、DenseNet121、MobileNetV3-Large、EfficientNet-B0两组只读复用，不重训或重复测试。', '',
             '生成器仍为Stage1全6199条纹图预训练→Stage2仅4951对原图/条纹图在t=0微调20轮的旧版本；不是两阶段都只train4951的另一版。', '',
             f"生成器：`{protocol['checkpoint']}`；SHA256：`{protocol['checkpoint_sha256']}`。", '',
             '所有分类器共用已有6199张RGB64×64 t50 PNG，完整50步递推至x0后R(x0,0)。未重新生成图像，未改噪声；沿用单份共享固定推理epsilon，不是恢复的实际Stage1训练噪声。', '',
             '## 数据与训练控制', '',
             '沿用原split43：4951训练、619验证、629测试，同源组跨集合重叠为0。原图/再映射的split/class/filename逐图对应；训练DataLoader同步打乱图像与标签，不改变集合归属。', '',
             '全部随机初始化weights=None、seed43；同架构两组初始权重SHA256一致，不同架构不要求初始权重相同。仅适配三分类头并统一头Dropout0.30，保留原生主体。ResNet34/ShuffleNetV2/RegNetY为Dropout+Linear；SqueezeNet保留原生Dropout+Conv1x1+ReLU+全局池化及原生初始化，不强行替换为线性分类头。', '',
             '公共超参数保持旧实验：AdamW lr0.001/weight_decay0.0001/betas(0.9,0.999)/eps1e-8，batch64、workers4、AMP；训练类别权重仅由4951标签计算、label_smoothing0.05，RandomAffine±5度/translate0.03/scale0.95～1.05和ColorJitter，mean/std0.5，不使用翻转。val/test不增广，使用无权重无平滑交叉熵。', '',
             '最多100轮，按val_loss保存绝对最低损失权重，早停patience15/min_delta0.0001；ReduceLROnPlateau factor0.5/patience4/threshold0.0001(abs)/min_lr1e-6。测试不用于梯度、学习率、早停或选模；全部8个新分类器冻结后，才统一评价新测试结果。实际早停轮数可以不同。', '',
             '公共训练函数的终端标题仍写ResNet18，这是旧固定文本；实际结构由本入口构造并通过参数量、初始化哈希、checkpoint config和输出维度核验。旧公共程序没有改写。', '',
             '## 参数量与训练轮次', '', '| 分类器 | 参数量 | 输入 | 完成/所选轮次 | 最低验证损失 | 来源 |', '|---|---:|---|---:|---:|---|']
    for architecture, arms in runs.items():
        for arm in ARMS:
            done = training[architecture][arm]
            lines.append(f"| {NAMES[architecture]} | {protocol['architecture_specs'][architecture]['parameters']} | {INPUT_NAMES[arm]} | {done['epochs_completed']}/{done['selected_epoch']} | {done['selected_score']:.6f} | " + ('本次新训练' if architecture in ARCHITECTURES else '历史冻结复用') + ' |')
    for split, title in (('val', '验证集（619张）'), ('test', '测试集（629张）')):
        lines += ['', f'## {title}', '', '单位%；P/R/F1同时报告weighted与macro，weighted-R等于ACC。', '',
                  '| 分类器 | 输入 | ACC | Weighted P | Weighted R | Weighted F1 | Macro P | Macro R | Macro F1 |',
                  '|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for row in rows:
            if row['split'] == split:
                lines.append(f"| {NAMES[row['architecture']]} | {INPUT_NAMES[row['input']]} | " +
                             ' | '.join(f"{100*row[k]:.2f}" for k in pipeline.common.METRIC_NAMES) + ' |')
    lines += ['', '## 测试ACC配对简表', '', '| 分类器 | 参数量 | 原图ACC | t50再映射ACC | ΔACC（百分点） |', '|---|---:|---:|---:|---:|']
    for architecture in runs:
        raw, mapped = (details[architecture]['test'][arm] for arm in ARMS)
        lines.append(f"| {NAMES[architecture]} | {protocol['architecture_specs'][architecture]['parameters']} | {100*raw['accuracy']:.2f} | {100*mapped['accuracy']:.2f} | {100*(mapped['accuracy']-raw['accuracy']):+.2f} |")
    lines += ['', '## 再映射减原图（百分点）', '', '| 分类器 | 集合 | ΔACC | ΔWeighted F1 | ΔMacro F1 |', '|---|---|---:|---:|---:|']
    for row in deltas:
        lines.append(f"| {NAMES[row['architecture']]} | {row['split']} | " + ' | '.join(f"{row[k]:+.2f}" for k in ('accuracy', 'weighted_f1', 'macro_f1')) + ' |')
    lines += ['', '## 验证与测试逐类别P/R/F1', '', '| 分类器 | 集合 | 输入 | 类别 | 数量 | P | R | F1 |', '|---|---|---|---|---:|---:|---:|---:|']
    for row in per_class:
        lines.append(f"| {NAMES[row['architecture']]} | {row['split']} | {INPUT_NAMES[row['input']]} | {row['class']} | {row['support']} | " +
                     ' | '.join(f"{100*row[k]:.2f}" for k in ('precision', 'recall', 'f1')) + ' |')
    lines += ['', '## 路径与复现', '', f'- 程序：[{Path(__file__).name}]({Path(__file__).as_posix()})',
              f'- 已有五分类器报告：[{base.REPORT_NAME}]({(PARENT / base.REPORT_NAME).as_posix()})',
              f'- 共用t50图：[post_formula_prediction]({base.MAP_ROOT.as_posix()})',
              '- `nine_backbone_metrics.json/csv`保存总体指标、原始混淆矩阵、轮次、参数量；`nine_backbone_per_class.csv`保存逐类指标；`remapped_minus_original_deltas_pp.csv`保存配对差值。', '',
              '| 分类器 | 原图组目录 | 再映射组目录 |', '|---|---|---|']
    for architecture, arms in runs.items():
        lines.append(f"| {NAMES[architecture]} | [原图组]({arms['original'].as_posix()}) | [再映射组]({arms['t50_remapped'].as_posix()}) |")
    lines += ['', '每组目录含best.pt、last.pt、config.json、history.csv、val/test指标及逐图预测、混淆矩阵和训练曲线。', '',
              '```powershell', 'cd <repository-root>',
              'conda run --no-capture-output -n diffusion_env python -X utf8 -u run_four_more_backbones_stage2_t50.py', '```', '',
              '`--stage preflight`只核验；`train`只训练并评价val；`test`要求8组全部冻结后评价；`report`只汇总已封存结果；默认all完整运行。--models可指定4个新增模型中的子集用于分段训练，但测试屏障仍要求8组全部完成。复用完成组或按last.pt续训；多进程增广跨中断续训不保证逐位复现。', '',
              '模型参考：[ResNet34](https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.resnet34.html)、[ShuffleNetV2](https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.shufflenet_v2_x1_0.html)、[SqueezeNet1.1](https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.squeezenet1_1.html)、[RegNetY-400MF](https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.regnet_y_400mf.html)。', '',
              '## 解释边界', '',
              '这是同一生成器、同一数据及公共训练协议下的模型适配比较，不是各架构分别充分调参后的最高能力，也不能单独把参数量与架构影响分离。单次seed43不能证明稳定性或统计显著性；新增模型也不保证提升。', '',
              'Stage1见过全6199目标条纹图，含val/test对应目标；目标方向含类别编码，旧测试集也已用于方法开发。因此换更多分类器不消除生成器数据暴露，不能称为全流程严格独立泛化。分类精度不是图像相似度，本次不新增PSNR/SSIM。', '',
              f"完成：{', '.join(runs)}。待完成：{', '.join(pending) or '无'}。旧结果、生成器、噪声、图片、共享源程序和总实验MD保持不变。"]
    (args.output_dir / REPORT_NAME).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    for row in rows:
        if row['split'] == 'test' and row['architecture'] in ARCHITECTURES:
            print(f"{NAMES[row['architecture']]} {INPUT_NAMES[row['input']]}：test ACC={100*row['accuracy']:.2f}%, P/R/F1=" +
                  '/'.join(f"{100*row[k]:.2f}%" for k in ('weighted_precision', 'weighted_recall', 'weighted_f1')), flush=True)
    preserve_check(protocol)
    save_json({'phase': 'complete' if not pending else 'partial', 'completed_architectures': list(runs),
               'pending_architectures': pending}, args.output_dir / 'progress.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--models', nargs='+', choices=ARCHITECTURES, default=list(ARCHITECTURES))
    parser.add_argument('--stage', choices=('all', 'preflight', 'train', 'test', 'report'), default='all')
    args = parser.parse_args()
    if len(set(args.models)) != len(args.models):
        raise ValueError('Do not repeat model names')
    args.output_dir = guard_output(args.output_dir)
    classifier = pipeline.common.load_classifier(pipeline.common.DEFAULT_CLASSIFIER_SCRIPT)
    classifier.seed_worker = seed_worker
    protocol = preflight(args, classifier)
    if args.stage == 'preflight':
        return
    marker = args.output_dir / 'experiment_protocol.json'
    if marker.exists() and read_json(marker) != protocol:
        raise ValueError('Existing protocol differs; do not overwrite it')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not marker.exists():
        save_json(protocol, marker)
    if args.stage in ('all', 'train'):
        for architecture in args.models:
            with architecture_adapter(classifier, architecture, protocol):
                for arm in ARMS:
                    selected = run_args(args, architecture, arm)
                    root = Path(protocol['original_root']) if arm == 'original' else base.MAP_ROOT
                    save_json({'phase': 'training', 'architecture': architecture, 'input': arm}, args.output_dir / 'progress.json')
                    print(f'开始 {NAMES[architecture]} / {INPUT_NAMES[arm]}，train4951/val619，max100、val_loss早停。', flush=True)
                    pipeline.train_classifier(selected, protocol, classifier, data_root=root)
                    pipeline.evaluate_split(selected, 'val', classifier)
                    preserve_check(protocol)
    if args.stage in ('all', 'test'):
        frozen = require_all_new_frozen(args, protocol)
        barrier = args.output_dir / 'all_new_models_frozen_before_test.json'
        freeze_record = {'models': frozen, 'selection': 'val_loss', 'new_models': list(ARCHITECTURES),
                         'all_eight_training_runs_complete_before_any_new_test': True}
        if barrier.exists() and read_json(barrier) != freeze_record:
            raise ValueError('Previously frozen selected weights changed')
        if not barrier.exists():
            save_json(freeze_record, barrier)
        for architecture in ARCHITECTURES:
            with architecture_adapter(classifier, architecture, protocol):
                for arm in ARMS:
                    selected = run_args(args, architecture, arm)
                    save_json({'phase': 'frozen_test', 'architecture': architecture, 'input': arm}, args.output_dir / 'progress.json')
                    pipeline.evaluate_split(selected, 'val', classifier)
                    pipeline.evaluate_split(selected, 'test', classifier)
        if require_all_new_frozen(args, protocol) != frozen:
            raise ValueError('Selected models changed during frozen testing')
    if args.stage in ('all', 'test', 'report'):
        report(args, protocol, classifier)
    preserve_check(protocol)
    print(f'实验目录：{args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
