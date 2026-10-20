# Paper sections and source programs

| Paper component | Original implementation in this repository |
|---|---|
| Time-conditioned U-Net | `diffusion_stripe/model.py` |
| RGB image loading and paired datasets | `diffusion_stripe/data.py` |
| Gaussian schedule, RNG, generator training utilities | `diffusion_stripe/training.py` |
| Section 3.1 / Algorithm 1 | `train_stage1_ddpm.py` |
| Section 3.2 / Algorithm 2 | `train_stage2_finetune.py`; historical train-only audit in `historical/train_stage2_train4951.py` |
| Directional B-channel watermark construction | `reference/generate_striped_dataset.py` |
| Original source-group partition | `reference/regroup_crops64_dataset.py` |
| Section 3.3 / Algorithm 3 | `historical/run_stage2_train4951_t50_remapped_resnet18.py`, function `remap_t50` |
| Section 4.2 / Table 2 | `historical/run_stage1_vs_stage2_t50_resnet18_ablation.py` |
| Stage 1 versus Stage 2 matched inference | `historical/evaluate_stage1_vs_stage2_t50_similarity.py` |
| Section 4.3 / Table 3 | `historical/run_multibackbone_stage2_t50_comparison.py` and `historical/run_four_more_backbones_stage2_t50.py` |
| Shared classifier training, early stopping, evaluation | `historical/run_stage2_train4951_remapped_resnet18.py` |
| Classifier transforms, class weights, metrics | `reference/striped_classification_experiment.py` |
| ROC / AUC analysis | `historical/compute_roc_auc_original.py`; extracted functions in `reference/roc_math.py` |
| Portable new-run commands | `paper_cli.py` and `compute_roc_auc.py` |

`SOURCE_MANIFEST.json` distinguishes byte-for-byte originals from release copies whose machine-specific paths were sanitized. For sanitized copies it records both the original and released file hashes; learning and evaluation algorithms are unchanged. Original local source files and experiment directories are not edited. Other historical modules are retained because the primary scripts import them; their presence does not mean their older external-data protocols are the reported paper protocol.

The historical ROC script also validates an old Word draft and editing snapshots. Those private working documents are intentionally absent. The portable ROC entry point reads probability CSVs directly and reuses its numerical functions without those document dependencies.

The portable classifier adapter substitutes configuration values recorded in the completed paper run for historical configuration-file lookups. Its model constructors, augmentation, optimizer/training loop, selection logic, and evaluation are the original functions. Fresh runs are separate from the sealed, published numerical results and must not overwrite them.
