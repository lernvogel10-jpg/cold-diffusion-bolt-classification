# Cold Diffusion Bolt Classification

Research code accompanying **Improving Transmission Line Bolt Classification and Detection Using a Cold Diffusion Model**.

The method learns a time-conditioned U-Net from directional watermark targets, fine-tunes it on paired original/watermarked training images, and performs 50 recursive denoising updates followed by a final mapping at `t = 0`. The resulting images are classified using six CNN architectures.

This repository contains the actual experiment source files, their local dependencies, a portable path adapter, the numerical results used in the manuscript, and source SHA-256 hashes. It does not include the manuscript, inspection images, generator/classifier checkpoints, Word/Visio/Origin files, or unrelated experiments. The archived fixed inference noise tensor is included because its exact values affect the experiment.

## Experimental scope

| Stage | Data | Objective or protocol |
|---|---|---|
| Stage 1 | All 6199 watermarked images | Direct clean-signal prediction, Gaussian degradation, mean RGB L1 loss, 200 epochs in the recorded run |
| Stage 2 | 4951 original/watermark training pairs | Original input without added noise, fixed `t = 0`, mean RGB L1, 20 epochs |
| Inference | 6199 original images | Shared archived Gaussian noise; `t = 50` to `0`; then terminal `R(x0, 0)` |
| Classification | Train 4951 / validation 619 / test 629 | Random initialization; select minimum validation loss; fixed seed 43 |

**Evaluation boundary:** Stage 1 was pretrained on all 6199 class-encoded watermark targets, including the counterparts of validation/test images. The reported results are not a fully unseen-test evaluation of the entire pipeline. They are single-seed results. The six models are the subset reported in the paper, not every architecture explored during development.

| Class | Training | Validation | Test | Total |
|---|---:|---:|---:|---:|
| defect | 990 | 125 | 129 | 1244 |
| normal | 2417 | 295 | 306 | 3018 |
| nut | 1544 | 199 | 194 | 1937 |
| Total | 4951 | 619 | 629 | 6199 |

## Installation

Use Python 3.11 and a compatible PyTorch/torchvision installation. The verified local environment used PyTorch 2.6.0 with CUDA 12.4 and torchvision 0.21.0. Install a CUDA build appropriate for your machine when GPU execution is needed.

```bash
python -m venv .venv
# Activate the environment using the command appropriate to your shell.
python -m pip install -r requirements.txt
```

The original experiment ran on Windows. The path adapter avoids the historical `D:` drive paths; execution on other operating systems has not been verified. See [environment details](docs/ENVIRONMENT.json).

## Data preparation

Provide RGB 64 × 64 images under this existing source-grouped split. Do not randomly repartition individual augmented images.

```text
data/original/
  train/{defect,normal,nut}/*.png
  val/{defect,normal,nut}/*.png
  test/{defect,normal,nut}/*.png
  reports/crops_manifest.csv
```

`crops_manifest.csv` is required by the original watermark preparation script. Its columns include `split`, `class_name`, `crop_path`, and `source_id`; the historical audit also checks `canonical_source_group`. The raw data and manifest are not distributed here.

```bash
python paper_cli.py check-data --dataset data/original
python reference/generate_striped_dataset.py --source data/original --output data/watermarked --stripe-width 1 --mask-low 0.9 --expected-total 6199
```

Horizontal, vertical, and upper-left-to-lower-right diagonal masks correspond to `defect`, `normal`, and `nut`. Only B is modulated; R and G remain unchanged. Watermark directions encode training target classes; labels or target images are not supplied to the generator during inference.

## Train the two generator stages

These are the original training implementations with explicit portable paths. The historical Stage 2 audit runner is also retained under `historical/`, but a fresh run need not recreate an older, superseded full-data Stage 2 experiment merely to use its configuration.

```bash
python train_stage1_ddpm.py --striped-root data/watermarked --output-root runs/generator --expected-count 6199 --epochs 200 --device auto
python train_stage2_finetune.py --original-root data/original --striped-root data/watermarked --stage1-checkpoint runs/generator/stage1/best_train_loss.pt --output-root runs/aligned --splits train --expected-count 4951 --min-epochs 20 --epochs 20 --no-early-stop --device auto
```

Stage 1 uses AdamW, learning rate `2e-4`, weight decay `1e-4`, batch size 64, a linear 100-step beta schedule from `1e-4` to `0.02`, and seed 43. Stage 2 uses learning rate `2e-5`. Both generator stages implement L1 loss in the recorded source. Existing nonempty outputs require the original scripts' explicit `--resume` option.

## Generate remapped images and the Stage 1 ablation

```bash
python paper_cli.py infer --original-root data/original --checkpoint runs/aligned/stage2/final_generator.pt --output data/remapped --device auto
python paper_cli.py infer --original-root data/original --checkpoint runs/generator/stage1/best_train_loss.pt --output data/stage1_only --device auto
```

The adapter calls the **unchanged** `remap_t50` function used in the experiments. It does not clip predictions or states inside recursion. It clips the final recursive state before the terminal mapping and clips the mapped output to `[0, 1]`, saving RGB PNGs using the original conversion function. All images and steps share the archived inference epsilon, not a recovered historical training-noise sample.

The generator checkpoints used for the reported results have SHA-256 hashes:

- Stage 1: `10f89fc1e909d244c1976ae084016a89c283b7744dfa1a640b1bbda66c6e4dc8`
- Stage 2: `5295975111fa2330e2941c06debc20884baa611888d204b28decb9a060becec1`
- Archived noise file: `60876b360eeead1c35058e741dce6cd7d7b4c37c36567b7b0a7c3b7259832254`

The weights are not included in this source-code release. New training may produce different weights; identical seeds do not guarantee bitwise agreement across hardware, package versions, worker scheduling, or interrupted runs.

## Classifier training and evaluation

Supported architecture keys are `resnet18`, `densenet121`, `mobilenet_v3_large`, `efficientnet_b0`, `shufflenet_v2_x1_0`, and `regnet_y_400mf`.

```bash
python paper_cli.py classify --dataset data/original --architecture resnet18 --input-label original --run-dir runs/resnet18/original --phase train
python paper_cli.py classify --dataset data/remapped --architecture resnet18 --input-label remapped --run-dir runs/resnet18/remapped --phase train
python paper_cli.py classify --dataset data/stage1_only --architecture resnet18 --input-label stage1_only --run-dir runs/resnet18/stage1_only --phase train
```

Use the same commands with the other five architecture keys and corresponding run directories. Freeze all planned classifier runs before testing. Then replace `--phase train` with `--phase evaluate` for each run. The adapter reuses the original training and evaluation functions and substitutes configuration/path lookup only. It never launches historical experiment scripts' `main()` functions.

Classifiers use no ImageNet weights, dropout 0.30, AdamW at `0.001`, weight decay `0.0001`, class weights from the training labels, label smoothing 0.05, batch 64, and a maximum of 100 epochs. Selection and early stopping use validation loss, patience 15, and minimum improvement `0.0001`. Data augmentation, normalization, and ReduceLROnPlateau settings are preserved in the source. Test data do not select epochs or learning rates. See [the complete configuration](configs/paper_protocol.json).

## ROC and AUC

Create a CSV with columns `classifier,input,run_dir`, for example:

```csv
classifier,input,run_dir
ResNet18,Original,runs/resnet18/original
ResNet18,Remapped,runs/resnet18/remapped
```

```bash
python compute_roc_auc.py --runs configs/roc_runs.example.csv --output-dir outputs/roc
```

This reads frozen `test_predictions.csv` files with `true_id`, `true_class`, and `prob_defect/prob_normal/prob_nut`. The ROC functions are extracted verbatim from the original analysis script. Scores are not renormalized or smoothed. Macro-AUC is the equal-weight mean of the three class-specific one-vs-rest AUCs; P, R, and F1 in the main comparison table are support-weighted. Weighted R equals ACC for this single-label task.

## Reported results and source provenance

- [Table 2 classification ablation](results/table2_ablation.csv)
- [Table 3 comparison](results/table3_comparison.csv)
- [Class-specific and macro AUC](results/auc_summary.csv)
- [Full ablation metrics](results/ablation_full_metrics.csv)
- [Paper section to program mapping](docs/CODE_MAP.md)
- [Verbatim source file hashes](SOURCE_MANIFEST.json)
- [Release checks](docs/VALIDATION.json)

The tables were extracted from the supplied English manuscript and checked against the frozen experiment results. Packaging does not retrain any model or replace the reported metrics. The historical files retain machine-specific paths and references to sealed experiment artifacts; they document the original runs and should not be treated as standalone fresh-run commands. Use the explicit-path commands above for a new run.
