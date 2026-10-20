from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


DEFAULT_SOURCE = Path("data/original")
DEFAULT_OUTPUT = Path("data/watermarked")
CLASSES = ("defect", "normal", "nut")
SPLITS = ("train", "val", "test")
PATTERN_BY_CLASS = {
    "defect": "horizontal",
    "normal": "vertical",
    "nut": "diagonal_top_left_to_bottom_right",
}
IMAGE_SIZE = 64
EXPECTED_TOTAL = 6199


def build_binary_masks(size: int, stripe_width: int) -> dict[str, np.ndarray]:
    if size <= 0:
        raise ValueError("size必须大于0。")
    if stripe_width <= 0:
        raise ValueError("stripe_width必须大于0。")

    y, x = np.indices((size, size), dtype=np.int32)
    period = stripe_width * 2
    masks = {
        "defect": ((y // stripe_width) % 2 == 0).astype(np.uint8),
        "normal": ((x // stripe_width) % 2 == 0).astype(np.uint8),
        # x-y为常数的直线方向是左上到右下（反斜杠方向）。
        "nut": (((x - y) % period) < stripe_width).astype(np.uint8),
    }
    for class_name, mask in masks.items():
        if mask.shape != (size, size):
            raise AssertionError(f"{class_name}掩码形状错误：{mask.shape}")
        if mask.dtype != np.uint8:
            raise AssertionError(f"{class_name}掩码类型不是uint8。")
        if np.unique(mask).tolist() != [0, 1]:
            raise AssertionError(
                f"{class_name}掩码不只包含0和1：{np.unique(mask).tolist()}"
            )
    return masks


def build_class_masks(
    size: int,
    stripe_width: int,
    mask_low: float,
) -> dict[str, np.ndarray]:
    if not 0.0 <= mask_low < 1.0:
        raise ValueError("mask_low必须满足0 <= mask_low < 1。")
    binary_masks = build_binary_masks(size, stripe_width)
    if mask_low == 0.0:
        return binary_masks
    return {
        class_name: np.where(binary == 1, 1.0, mask_low).astype(np.float32)
        for class_name, binary in binary_masks.items()
    }


def expected_blue_channel(blue: np.ndarray, mask: np.ndarray) -> np.ndarray:
    product = blue.astype(np.float32) * mask.astype(np.float32)
    return np.rint(product).clip(0, 255).astype(np.uint8)


def multiply_blue_channel(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if rgb.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(f"输入图像应为64×64×3，实际为{rgb.shape}")
    if mask.shape != rgb.shape[:2]:
        raise ValueError(f"掩码{mask.shape}与图像{rgb.shape[:2]}不匹配。")

    striped = rgb.copy()
    striped[:, :, 2] = expected_blue_channel(striped[:, :, 2], mask)
    return striped


def verify_pixel_relation(
    original: np.ndarray,
    mask: np.ndarray,
    striped: np.ndarray,
) -> None:
    if original.shape != striped.shape:
        raise ValueError(f"原图{original.shape}与条纹图{striped.shape}形状不同。")
    if not np.array_equal(original[:, :, 0], striped[:, :, 0]):
        raise ValueError("R通道发生了非预期改变。")
    if not np.array_equal(original[:, :, 1], striped[:, :, 1]):
        raise ValueError("G通道发生了非预期改变。")
    if not np.array_equal(
        expected_blue_channel(original[:, :, 2], mask),
        striped[:, :, 2],
    ):
        raise ValueError("B通道不满足B_new=B_original×mask。")


def load_manifest(source: Path) -> tuple[list[dict[str, str]], list[str]]:
    manifest_path = source / "reports" / "crops_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"找不到原始清单：{manifest_path}")
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    required = {"split", "class_name", "crop_path", "source_id"}
    if not required.issubset(fieldnames):
        raise ValueError(f"原始清单缺少字段：{sorted(required - set(fieldnames))}")
    return rows, fieldnames


def validate_manifest(rows: list[dict[str, str]], expected_total: int) -> dict:
    if expected_total > 0 and len(rows) != expected_total:
        raise ValueError(f"清单应有{expected_total}行，实际为{len(rows)}行。")

    counts: Counter[tuple[str, str]] = Counter()
    source_splits: defaultdict[str, set[str]] = defaultdict(set)
    paths: set[str] = set()
    for row in rows:
        split = row["split"]
        class_name = row["class_name"]
        relative_path = row["crop_path"].replace("\\", "/")
        if split not in SPLITS:
            raise ValueError(f"未知划分：{split}")
        if class_name not in CLASSES:
            raise ValueError(f"未知类别：{class_name}")
        if relative_path in paths:
            raise ValueError(f"清单存在重复crop_path：{relative_path}")
        paths.add(relative_path)
        counts[(split, class_name)] += 1
        source_splits[row["source_id"]].add(split)

    leaked = {
        source_id: sorted(splits)
        for source_id, splits in source_splits.items()
        if len(splits) > 1
    }
    if leaked:
        preview = list(leaked.items())[:10]
        raise ValueError(f"检测到{len(leaked)}个同源组跨集合：{preview}")

    return {
        "total": len(rows),
        "source_groups": len(source_splits),
        "cross_split_source_groups": 0,
        "counts": {
            split: {
                class_name: counts[(split, class_name)]
                for class_name in CLASSES
            }
            for split in SPLITS
        },
    }


def save_masks(
    output: Path,
    masks: dict[str, np.ndarray],
    stripe_width: int,
    mask_low: float,
) -> dict[str, str]:
    mask_dir = output / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    relative_paths: dict[str, str] = {}
    for class_name, mask in masks.items():
        if mask_low == 0.0:
            value_tag = "01"
        else:
            low_tag = f"{mask_low:g}".replace(".", "p")
            value_tag = f"low{low_tag}_1"
        stem = f"{class_name}_{PATTERN_BY_CLASS[class_name]}_w{stripe_width}_{value_tag}"
        npy_path = mask_dir / f"{stem}.npy"
        intensity_png_path = mask_dir / f"{stem}.png"
        visual_path = mask_dir / f"{stem}_visual.png"
        np.save(npy_path, mask)
        intensity = np.rint(mask.astype(np.float32) * 255).astype(np.uint8)
        stored_png = mask.astype(np.uint8) if mask_low == 0.0 else intensity
        Image.fromarray(stored_png, mode="L").save(intensity_png_path)
        Image.fromarray(intensity, mode="L").save(visual_path)
        relative_paths[class_name] = npy_path.relative_to(output).as_posix()
    return relative_paths


def make_preview(
    original: np.ndarray,
    mask: np.ndarray,
    striped: np.ndarray,
    class_name: str,
    output_path: Path,
) -> None:
    scale = 6
    margin = 18
    header = 42
    panels = [
        Image.fromarray(original, mode="RGB"),
        Image.fromarray(
            np.rint(mask.astype(np.float32) * 255).astype(np.uint8),
            mode="L",
        ).convert("RGB"),
        Image.fromarray(striped, mode="RGB"),
    ]
    labels = ["Original RGB", "Mask intensity", "B channel x mask"]
    panel_size = IMAGE_SIZE * scale
    canvas = Image.new(
        "RGB",
        (margin * 4 + panel_size * 3, header + panel_size + margin),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, (panel, label) in enumerate(zip(panels, labels, strict=True)):
        x = margin + index * (panel_size + margin)
        draw.text((x, 7), f"{class_name}: {label}", fill="black")
        canvas.paste(
            panel.resize((panel_size, panel_size), Image.Resampling.NEAREST),
            (x, header),
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def write_derived_manifest(
    output: Path,
    source_fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    extra_fields = [
        "clean_crop_path",
        "striped_crop_path",
        "pattern",
        "stripe_width",
        "mask_path",
        "mask_dtype",
        "mask_unique_values",
        "operation",
    ]
    fieldnames = source_fieldnames + [
        name for name in extra_fields if name not in source_fieldnames
    ]
    path = output / "reports" / "striped_manifest.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    standard_path = output / "reports" / "crops_manifest.csv"
    with standard_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=source_fieldnames)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in source_fieldnames}
            for row in rows
        )


def generate(args: argparse.Namespace) -> int:
    source = args.source.resolve()
    output = args.output.resolve()
    rows, source_fieldnames = load_manifest(source)
    manifest_summary = validate_manifest(rows, args.expected_total)
    selected_splits = tuple(dict.fromkeys(args.splits))
    selected_rows = [row for row in rows if row["split"] in selected_splits]
    if not selected_rows:
        raise ValueError(f"所选split没有样本：{selected_splits}")
    masks = build_class_masks(IMAGE_SIZE, args.stripe_width, args.mask_low)

    if output.exists() and not args.resume:
        raise FileExistsError(
            f"输出目录已存在：{output}\n"
            "为防止覆盖，请确认它属于本实验后添加 --resume。"
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "reports" / "previews").mkdir(parents=True, exist_ok=True)
    mask_paths = save_masks(output, masks, args.stripe_width, args.mask_low)

    classes_path = source / "classes.txt"
    if classes_path.is_file() and not (output / "classes.txt").exists():
        shutil.copy2(classes_path, output / "classes.txt")

    generated = 0
    reused = 0
    derived_rows: list[dict[str, str]] = []
    preview_done: set[str] = set()
    for index, row in enumerate(selected_rows, start=1):
        split = row["split"]
        class_name = row["class_name"]
        relative_path = Path(row["crop_path"])
        source_path = source / relative_path
        target_path = output / relative_path
        if source_path.suffix.lower() != ".png":
            raise ValueError(
                f"为保证逐像素无损校验，只接受PNG；发现：{source_path}"
            )
        if not source_path.is_file():
            raise FileNotFoundError(source_path)

        with Image.open(source_path) as image:
            original = np.asarray(image.convert("RGB"), dtype=np.uint8)
        mask = masks[class_name]
        expected = multiply_blue_channel(original, mask)

        if target_path.exists():
            with Image.open(target_path) as image:
                actual = np.asarray(image.convert("RGB"), dtype=np.uint8)
            verify_pixel_relation(original, mask, actual)
            if not np.array_equal(expected, actual):
                raise ValueError(f"已有输出与确定性生成结果不一致：{target_path}")
            reused += 1
        else:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(expected, mode="RGB").save(target_path, format="PNG")
            with Image.open(target_path) as image:
                saved = np.asarray(image.convert("RGB"), dtype=np.uint8)
            verify_pixel_relation(original, mask, saved)
            if not np.array_equal(expected, saved):
                raise ValueError(f"PNG保存后像素不一致：{target_path}")
            generated += 1

        if split == "train" and class_name not in preview_done:
            make_preview(
                original,
                mask,
                expected,
                class_name,
                output / "reports" / "previews" / f"{class_name}_preview.png",
            )
            preview_done.add(class_name)

        derived_row = dict(row)
        derived_row.update(
            {
                "clean_crop_path": str(source_path),
                "striped_crop_path": str(target_path),
                "pattern": PATTERN_BY_CLASS[class_name],
                "stripe_width": str(args.stripe_width),
                "mask_path": mask_paths[class_name],
                "mask_dtype": str(mask.dtype),
                "mask_unique_values": f"{args.mask_low:g}|1",
                "operation": "RGB[:,:,2]=round(RGB[:,:,2]*mask)",
            }
        )
        derived_rows.append(derived_row)

        if index % args.progress_every == 0 or index == len(selected_rows):
            print(
                f"处理进度：{index}/{len(selected_rows)}，新增={generated}，复用={reused}"
            )

    if "train" in selected_splits and preview_done != set(CLASSES):
        raise ValueError(f"未能为所有类别生成预览：{sorted(preview_done)}")

    write_derived_manifest(output, source_fieldnames, derived_rows)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_dataset": str(source),
        "output_dataset": str(output),
        "image_size": [IMAGE_SIZE, IMAGE_SIZE],
        "classes": list(CLASSES),
        "patterns": PATTERN_BY_CLASS,
        "stripe_width": args.stripe_width,
        "mask_low": args.mask_low,
        "mask_dtype": str(next(iter(masks.values())).dtype),
        "mask_unique_values": [args.mask_low, 1.0],
        "operation": "RGB[:,:,2] = round(RGB[:,:,2] * mask)",
        "selected_splits": list(selected_splits),
        "selected_total": len(selected_rows),
        "selected_counts": {
            split: {
                class_name: sum(
                    row["split"] == split and row["class_name"] == class_name
                    for row in selected_rows
                )
                for class_name in CLASSES
            }
            for split in selected_splits
        },
        "generated_this_run": generated,
        "reused_after_verification": reused,
        **manifest_summary,
    }
    (output / "reports" / "generation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n条纹数据集生成并逐像素验证完成。")
    print(json.dumps(summary["selected_counts"], ensure_ascii=False, indent=2))
    print(f"本次所选总数：{len(selected_rows)} / 来源总数：{manifest_summary['total']}")
    print(f"同源跨集合：{manifest_summary['cross_split_source_groups']}")
    print(f"输出目录：{output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "在64×64框图的B通道中加入类别对应的指定低位/1条纹，"
            "并保持原train/val/test划分。"
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stripe-width", type=int, default=4)
    parser.add_argument(
        "--mask-low",
        type=float,
        default=0.0,
        help="条纹低位强度，必须满足0<=值<1；高位固定为1。",
    )
    parser.add_argument("--expected-total", type=int, default=EXPECTED_TOTAL)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
        help="仅生成指定split；默认生成train、val、test。",
    )
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="续生成：已有图片先逐像素校验，正确则跳过。",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(generate(build_parser().parse_args()))
