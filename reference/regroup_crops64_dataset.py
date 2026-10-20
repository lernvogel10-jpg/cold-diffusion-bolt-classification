from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


CLASSES = ("defect", "normal", "nut")
SPLITS = ("train", "val", "test")
DEFAULT_SOURCE = Path("data/source_crops")
DEFAULT_OUTPUT = Path("data/original")
EXPECTED_TOTAL = 6199


def canonical_name(source_id: str) -> str:
    """Return the pre-augmentation/pre-copy identity encoded in a file name."""
    value = source_id.strip().casefold()
    value = re.sub(r"_aug\d+$", "", value, flags=re.IGNORECASE)
    value = re.sub(
        r"_(?:jpg|jpeg|png)\.rf\.[0-9a-f]+$",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"\s*\(\d+\)$", "", value)
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        a = self.find(left)
        b = self.find(right)
        if a == b:
            return
        # Stable representative makes a resumed run deterministic.
        if a < b:
            self.parent[b] = a
        else:
            self.parent[a] = b


@dataclass(frozen=True)
class GroupStats:
    group_id: str
    crop_count: int
    source_image_count: int
    class_counts: tuple[int, int, int]

    @property
    def signature(self) -> tuple[int, ...]:
        result = tuple(index for index, count in enumerate(self.class_counts) if count)
        return result or (-1,)


def read_manifest(source: Path) -> tuple[list[dict[str, str]], list[str]]:
    path = source / "reports" / "crops_manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(f"找不到裁剪清单：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    required = {
        "split",
        "class_name",
        "crop_path",
        "source_id",
        "source_image",
    }
    missing = required - set(fieldnames)
    if missing:
        raise ValueError(f"清单缺少字段：{sorted(missing)}")
    return rows, fieldnames


def validate_input(rows: list[dict[str, str]], source: Path, expected: int) -> None:
    if expected > 0 and len(rows) != expected:
        raise ValueError(f"预期{expected}个框图，实际{len(rows)}个。")
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        split = row["split"]
        class_name = row["class_name"]
        relative = row["crop_path"].replace("\\", "/")
        if split not in SPLITS:
            raise ValueError(f"第{index}行split非法：{split}")
        if class_name not in CLASSES:
            raise ValueError(f"第{index}行类别非法：{class_name}")
        if relative in seen:
            raise ValueError(f"重复crop_path：{relative}")
        seen.add(relative)
        path = source / Path(relative)
        if not path.is_file():
            raise FileNotFoundError(path)


def build_leakage_safe_groups(
    rows: list[dict[str, str]],
    source: Path,
) -> tuple[dict[int, str], dict[str, object]]:
    initial = [canonical_name(row["source_id"]) for row in rows]
    uf = UnionFind(sorted(set(initial)))

    source_hash_owner: dict[str, str] = {}
    source_paths: dict[str, tuple[Path, str]] = {}
    for row, group_id in zip(rows, initial, strict=True):
        path = Path(row["source_image"])
        source_paths.setdefault(str(path.resolve()).casefold(), (path, group_id))
    for path, group_id in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256(path)
        owner = source_hash_owner.setdefault(digest, group_id)
        uf.union(owner, group_id)

    crop_hash_owner: dict[str, str] = {}
    for row, group_id in zip(rows, initial, strict=True):
        path = source / Path(row["crop_path"])
        digest = sha256(path)
        owner = crop_hash_owner.setdefault(digest, group_id)
        uf.union(owner, group_id)

    row_groups = {index: uf.find(group_id) for index, group_id in enumerate(initial)}
    group_members: defaultdict[str, set[str]] = defaultdict(set)
    for original, group_id in zip(initial, row_groups.values(), strict=True):
        group_members[group_id].add(original)
    audit = {
        "initial_source_ids": len({row["source_id"] for row in rows}),
        "filename_canonical_groups": len(set(initial)),
        "final_groups_after_exact_hash_union": len(set(row_groups.values())),
        "unique_source_images_hashed": len(source_paths),
        "unique_source_image_hashes": len(source_hash_owner),
        "unique_crop_hashes": len(crop_hash_owner),
        "merged_group_components": sum(len(value) > 1 for value in group_members.values()),
    }
    return row_groups, audit


def largest_remainder(total: int, ratios: tuple[float, float, float]) -> list[int]:
    normalized = [value / sum(ratios) for value in ratios]
    exact = [total * value for value in normalized]
    counts = [int(value) for value in exact]
    remainder = total - sum(counts)
    order = sorted(
        range(len(exact)),
        key=lambda index: exact[index] - counts[index],
        reverse=True,
    )
    for index in order[:remainder]:
        counts[index] += 1
    return counts


def make_group_stats(
    rows: list[dict[str, str]],
    row_groups: dict[int, str],
) -> list[GroupStats]:
    class_to_id = {name: index for index, name in enumerate(CLASSES)}
    crop_counts: Counter[str] = Counter()
    class_counts: defaultdict[str, Counter[int]] = defaultdict(Counter)
    source_images: defaultdict[str, set[str]] = defaultdict(set)
    for index, row in enumerate(rows):
        group_id = row_groups[index]
        crop_counts[group_id] += 1
        class_counts[group_id][class_to_id[row["class_name"]]] += 1
        source_images[group_id].add(str(Path(row["source_image"]).resolve()).casefold())
    return [
        GroupStats(
            group_id=group_id,
            crop_count=crop_counts[group_id],
            source_image_count=len(source_images[group_id]),
            class_counts=tuple(class_counts[group_id][index] for index in range(3)),
        )
        for group_id in sorted(crop_counts)
    ]


def split_score(
    assignments: dict[str, list[GroupStats]],
    totals: tuple[int, int, int],
    total_crops: int,
    total_images: int,
    ratios: tuple[float, float, float],
) -> float:
    normalized = [value / sum(ratios) for value in ratios]
    score = 0.0
    for split_index, split in enumerate(SPLITS):
        groups = assignments[split]
        crop_count = sum(group.crop_count for group in groups)
        image_count = sum(group.source_image_count for group in groups)
        target_crops = total_crops * normalized[split_index]
        target_images = total_images * normalized[split_index]
        score += 4.0 * ((crop_count - target_crops) / max(target_crops, 1.0)) ** 2
        score += 2.0 * ((image_count - target_images) / max(target_images, 1.0)) ** 2
        for class_id in range(3):
            actual = sum(group.class_counts[class_id] for group in groups)
            target = totals[class_id] * normalized[split_index]
            score += ((actual - target) / max(target, 1.0)) ** 2
    return score


def grouped_stratified_split(
    groups: list[GroupStats],
    ratios: tuple[float, float, float],
    seed: int,
    trials: int,
) -> dict[str, str]:
    buckets: defaultdict[tuple[int, ...], list[GroupStats]] = defaultdict(list)
    for group in groups:
        buckets[group.signature].append(group)
    totals = tuple(sum(group.class_counts[index] for group in groups) for index in range(3))
    total_crops = sum(group.crop_count for group in groups)
    total_images = sum(group.source_image_count for group in groups)
    rng = random.Random(seed)
    best_score = float("inf")
    best: dict[str, list[GroupStats]] | None = None

    for _ in range(max(1, trials)):
        candidate = {split: [] for split in SPLITS}
        for signature in sorted(buckets):
            bucket = buckets[signature].copy()
            rng.shuffle(bucket)
            counts = largest_remainder(len(bucket), ratios)
            cursor = 0
            for split, count in zip(SPLITS, counts, strict=True):
                candidate[split].extend(bucket[cursor : cursor + count])
                cursor += count
        current = split_score(candidate, totals, total_crops, total_images, ratios)
        if current < best_score:
            best_score = current
            best = {split: values.copy() for split, values in candidate.items()}
    if best is None:
        raise RuntimeError("同源分组划分失败。")
    return {
        group.group_id: split
        for split, values in best.items()
        for group in values
    }


def verify_cross_split(
    rows: list[dict[str, str]],
    row_groups: dict[int, str],
    assignment: dict[str, str],
) -> dict[str, object]:
    group_splits: defaultdict[str, set[str]] = defaultdict(set)
    source_id_splits: defaultdict[str, set[str]] = defaultdict(set)
    for index, row in enumerate(rows):
        group_id = row_groups[index]
        split = assignment[group_id]
        group_splits[group_id].add(split)
        source_id_splits[row["source_id"]].add(split)
    crossed = {key: sorted(value) for key, value in group_splits.items() if len(value) > 1}
    if crossed:
        raise RuntimeError(f"修正同源组仍跨集合：{list(crossed.items())[:10]}")
    return {
        "cross_split_final_groups": 0,
        "cross_split_recorded_source_ids": sum(
            len(value) > 1 for value in source_id_splits.values()
        ),
    }


def validate_resume_configuration(
    source: Path,
    output: Path,
    split_seed: int,
    ratios: tuple[float, float, float],
    resume: bool,
) -> None:
    if not output.exists():
        return
    if not resume:
        raise FileExistsError(
            f"输出目录已存在：{output}\n确认属于本实验后使用--resume。"
        )
    summary_path = output / "reports" / "dataset_summary.json"
    if not summary_path.is_file():
        # An interrupted first copy is safe to resume because target files are
        # verified byte-for-byte. Once a summary exists, the split settings are
        # frozen and must not be changed in place.
        return
    existing = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_ratios = dict(zip(SPLITS, ratios, strict=True))
    mismatches: dict[str, tuple[object, object]] = {}
    checks = {
        "source_dataset": str(source),
        "split_seed": split_seed,
        "split_ratios": expected_ratios,
    }
    for key, expected in checks.items():
        if existing.get(key) != expected:
            mismatches[key] = (existing.get(key), expected)
    if mismatches:
        raise ValueError(
            "续跑配置与已有数据集不一致，不能在同一目录混合不同划分："
            f"{mismatches}。请使用新的--output目录。"
        )


def copy_and_write(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    row_groups: dict[int, str],
    assignment: dict[str, str],
    source: Path,
    output: Path,
    resume: bool,
) -> tuple[list[dict[str, str]], int, int]:
    output.mkdir(parents=True, exist_ok=True)
    (output / "reports").mkdir(parents=True, exist_ok=True)
    copied = 0
    reused = 0
    derived: list[dict[str, str]] = []
    target_paths: set[str] = set()

    for index, row in enumerate(rows, start=1):
        group_id = row_groups[index - 1]
        new_split = assignment[group_id]
        old_relative = Path(row["crop_path"])
        new_relative = Path(new_split) / row["class_name"] / old_relative.name
        relative_key = new_relative.as_posix().casefold()
        if relative_key in target_paths:
            raise ValueError(f"重新划分后文件名冲突：{new_relative}")
        target_paths.add(relative_key)

        source_path = source / old_relative
        target_path = output / new_relative
        if target_path.exists():
            if sha256(source_path) != sha256(target_path):
                raise ValueError(f"已有输出与来源不同：{target_path}")
            reused += 1
        else:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
            copied += 1

        item = dict(row)
        item["original_split"] = row["split"]
        item["original_crop_path"] = row["crop_path"]
        item["original_source_id"] = row["source_id"]
        item["split"] = new_split
        item["crop_path"] = new_relative.as_posix()
        item["source_id"] = group_id
        item["canonical_source_group"] = group_id
        derived.append(item)

        if index % 500 == 0 or index == len(rows):
            print(f"复制进度：{index}/{len(rows)}，新增={copied}，复用={reused}")

    extras = [
        "original_split",
        "original_crop_path",
        "original_source_id",
        "canonical_source_group",
    ]
    output_fields = fieldnames + [name for name in extras if name not in fieldnames]
    manifest_path = output / "reports" / "crops_manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(derived)

    classes_path = source / "classes.txt"
    if classes_path.is_file():
        shutil.copy2(classes_path, output / "classes.txt")
    else:
        (output / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
    return derived, copied, reused


def output_summary(
    rows: list[dict[str, str]],
    source: Path,
    output: Path,
    split_seed: int,
    ratios: tuple[float, float, float],
    grouping_audit: dict[str, object],
    copied: int,
    reused: int,
) -> dict[str, object]:
    counts = {
        split: {
            "crops": sum(row["split"] == split for row in rows),
            "class_crops": {
                class_name: sum(
                    row["split"] == split and row["class_name"] == class_name
                    for row in rows
                )
                for class_name in CLASSES
            },
            "source_groups": len(
                {row["source_id"] for row in rows if row["split"] == split}
            ),
            "source_images": len(
                {row["source_image"] for row in rows if row["split"] == split}
            ),
        }
        for split in SPLITS
    }
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_dataset": str(source),
        "output_dataset": str(output),
        "split_seed": split_seed,
        "split_ratios": dict(zip(SPLITS, ratios, strict=True)),
        "classes": list(CLASSES),
        "total_crops": len(rows),
        "copied_this_run": copied,
        "reused_after_hash_verification": reused,
        "grouping": grouping_audit,
        "cross_split_final_groups": 0,
        "splits": counts,
    }
    (output / "reports" / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="按增强前、复制前来源归组后，重新划分64×64三分类框图数据。"
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1))
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--expected-total", type=int, default=EXPECTED_TOTAL)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    ratios = tuple(args.ratios)
    if any(value <= 0 for value in ratios):
        raise ValueError("三个划分比例都必须大于0。")

    source = args.source.resolve()
    output = args.output.resolve()
    validate_resume_configuration(source, output, args.split_seed, ratios, args.resume)
    rows, fieldnames = read_manifest(source)
    validate_input(rows, source, args.expected_total)
    print(f"输入框图：{len(rows)}，开始计算修正同源组和精确哈希。")
    row_groups, grouping_audit = build_leakage_safe_groups(rows, source)
    groups = make_group_stats(rows, row_groups)
    assignment = grouped_stratified_split(
        groups,
        ratios,
        args.split_seed,
        args.trials,
    )
    grouping_audit.update(verify_cross_split(rows, row_groups, assignment))
    derived, copied, reused = copy_and_write(
        rows,
        fieldnames,
        row_groups,
        assignment,
        source,
        output,
        args.resume,
    )
    summary = output_summary(
        derived,
        source,
        output,
        args.split_seed,
        ratios,
        grouping_audit,
        copied,
        reused,
    )
    print("\n重新划分完成，同源组跨集合=0。")
    for split in SPLITS:
        item = summary["splits"][split]
        print(
            f"{split}: crops={item['crops']}, groups={item['source_groups']}, "
            f"classes={item['class_crops']}"
        )
    print(f"输出：{output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n用户中断；使用相同命令和--resume可继续。")
        raise SystemExit(130)
