"""Portable path adapters around the original paper implementations.

The U-Net, diffusion update, classifier architectures, augmentation, training
loop, checkpoint selection, and evaluation are imported from the archived code.
This file replaces machine-specific paths/configuration lookup only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'historical'))
sys.path.insert(0, str(ROOT / 'reference'))
ARCHITECTURES = ('resnet18', 'densenet121', 'mobilenet_v3_large',
                 'efficientnet_b0', 'shufflenet_v2_x1_0', 'regnet_y_400mf')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol():
    return json.loads((ROOT / 'configs/paper_protocol.json').read_text(encoding='utf-8'))


def validate_dataset(root):
    from collections import Counter
    from diffusion_stripe.data import collect_images
    records = collect_images(root, expected_count=6199)
    counts = {s: dict(Counter(r.key.split('/')[1] for r in records
                             if r.key.startswith(s + '/'))) for s in ('train','val','test')}
    if counts != protocol()['split_counts']:
        raise ValueError(f'Dataset split/class counts differ from the paper: {counts}')
    return records


def safe_output(output, input_root=None):
    output = output.resolve()
    if input_root and (output == input_root.resolve() or output.is_relative_to(input_root.resolve())):
        raise ValueError('Write outputs outside the input dataset.')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output must be a new/empty directory: {output}')
    output.mkdir(parents=True, exist_ok=True)
    return output


def infer(args):
    import torch
    from torch.utils.data import DataLoader
    from diffusion_stripe.data import StripedImageDataset
    from diffusion_stripe.model import build_model_from_checkpoint
    from diffusion_stripe.training import linear_beta_schedule, resolve_device, seed_everything, save_json
    from run_stage2_train4951_t50_remapped_resnet18 import remap_t50
    from evaluate_t1_all6199 import tensor_to_image

    records = validate_dataset(args.original_root)
    output = safe_output(args.output, args.original_root)
    checkpoint_hash, noise_hash = digest(args.checkpoint), digest(args.noise_file)
    seed_everything(43)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = build_model_from_checkpoint(checkpoint).float().to(device).eval().requires_grad_(False)
    epsilon = torch.load(args.noise_file, map_location=device, weights_only=True).float()
    if tuple(epsilon.shape) != (1,3,64,64) or not torch.isfinite(epsilon).all():
        raise ValueError('Expected a finite archived Gaussian noise tensor of shape (1,3,64,64).')
    schedule = linear_beta_schedule(100, 1e-4, 0.02, device)
    dataset = StripedImageDataset(args.original_root, expected_count=6199)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    offset = 0
    with torch.inference_mode():
        for images in loader:
            _, _, remapped = remap_t50(model, images.float().to(device), epsilon, schedule)
            if not torch.isfinite(remapped).all():
                raise ValueError('Non-finite generator outputs.')
            for record, image in zip(records[offset:offset+len(images)], remapped.cpu(), strict=True):
                path = output / record.key
                path.parent.mkdir(parents=True, exist_ok=True)
                tensor_to_image(image).save(path)
            offset += len(images)
            print(f'Remapped {offset}/6199', flush=True)
    assert offset == 6199 and digest(args.checkpoint) == checkpoint_hash and digest(args.noise_file) == noise_hash
    save_json({'count': offset, 'checkpoint_sha256': checkpoint_hash, 'noise_sha256':noise_hash,
               'checkpoint_stage':checkpoint['stage'], 'inference':protocol()['inference'],
               'split_class_counts':protocol()['split_counts']}, output/'generation_complete.json')


def configure_classifier(args):
    import run_stage2_train4951_remapped_resnet18 as pipeline
    import run_multibackbone_stage2_t50_comparison as base
    import run_four_more_backbones_stage2_t50 as extra
    from resnet18_full6199_loss_earlystop import seed_worker
    source = ROOT / 'reference/striped_classification_experiment.py'
    pipeline.common.DEFAULT_CLASSIFIER_SCRIPT = source
    classifier = pipeline.common.load_classifier(source)
    classifier.seed_worker = seed_worker
    original_builder = classifier.build_model
    def builder(pretrained=False):
        if args.architecture == 'resnet18':
            return original_builder(pretrained)
        if args.architecture in extra.ARCHITECTURES:
            return extra.build_model(args.architecture, pretrained)
        return base.build_model(args.architecture, pretrained)
    classifier.build_model = builder

    def config_adapter(runtime_args, generation, model, module, *, data_root=None):
        config = dict(protocol()['classifier'])
        config.update(model=args.architecture, parameters=sum(p.numel() for p in model.parameters()),
                      batch_size=runtime_args.batch_size, workers=runtime_args.workers,
                      device=runtime_args.device, max_epochs=runtime_args.epochs,
                      patience=runtime_args.patience, min_delta=runtime_args.min_delta,
                      selection=runtime_args.selection,
                      initial_model_sha256=pipeline.common.state_sha256(model),
                      dataset=str(args.dataset.resolve()), generator_sha256=args.generator_sha256,
                      classifier_script_sha256=digest(source), initialize_from_old_classifier=False,
                      test_used_for_selection=False, input_method=args.input_label)
        return config
    pipeline.classifier_config = config_adapter
    return pipeline, classifier


def classify(args):
    from diffusion_stripe.training import resolve_device
    validate_dataset(args.dataset)
    args.device = str(resolve_device(args.device))
    args.selection, args.epochs, args.patience, args.min_delta = 'val_loss',100,15,1e-4
    args.output_dir = args.run_dir
    pipeline, classifier = configure_classifier(args)
    if args.phase in ('train','all'):
        # This calls the verbatim training loop used for the paper.
        pipeline.train_classifier(args, {'checkpoint_sha256':args.generator_sha256}, classifier,
                                  data_root=args.dataset.resolve())
    if args.phase in ('evaluate','all'):
        for split in ('val','test'):
            pipeline.evaluate_split(args, split, classifier)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    check=sub.add_parser('check-data',help='Verify the 6199 images and original split/class counts.')
    check.add_argument('--dataset',type=Path,required=True)
    p=sub.add_parser('infer',help='Run the original t=50 recursion and terminal t=0 remapping.')
    p.add_argument('--original-root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--noise-file',type=Path,default=ROOT/'assets/noise_t50_seed43.pt')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='auto')
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--workers',type=int,default=4)
    p=sub.add_parser('classify',help='Run the original validation-loss classifier trainer/evaluator.')
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--architecture',choices=ARCHITECTURES,required=True)
    p.add_argument('--input-label',required=True,help='e.g. original, stage1_only, or remapped')
    p.add_argument('--generator-sha256',default=None)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--phase',choices=['train','evaluate','all'],default='train')
    p.add_argument('--device',default='auto')
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.command=='check-data':
        print(json.dumps({'samples':len(validate_dataset(args.dataset)), 'split_counts':protocol()['split_counts']},indent=2))
    elif args.command=='infer':infer(args)
    else:classify(args)


if __name__=='__main__':main()
