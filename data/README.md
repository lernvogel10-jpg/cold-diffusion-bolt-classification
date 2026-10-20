# Dataset placement

The study dataset is currently not publicly available. This source-code release does not redistribute original inspection images, watermarked images, remapped images, image-level source/split manifests, or trained weights. No dataset download link or access commitment is provided. The instructions below apply only to users who already have authorization to use compatible data.

Put the original, source-grouped RGB 64 × 64 crops in `data/original/{train,val,test}/{defect,normal,nut}`. Keep every augmented/rotated image and crop sharing the same source group in the same split. Preserve the original 4951/619/629 split and its class counts.

The original preparation script expects `data/original/reports/crops_manifest.csv`. This manifest is not public and is not provided here. For authorized compatible data, the required fields include `split`, `class_name`, `crop_path`, and `source_id`. The historical grouping audit also uses `canonical_source_group`.

`paper_cli.py check-data` validates the counts, but counts alone do not prove source-group disjointness. The original audited manifest and dataset identity are needed to reproduce that split exactly. Do not generate a new random single-image split and describe it as the paper split.

Directions of watermark targets encode the training class. They are used during target construction and supervised training only. Inference reads originals, a fixed noise tensor, and a frozen generator, without giving it the ground-truth class or target image.
