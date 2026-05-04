import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CLASSES = ("benign", "malignant")


def iter_split_images(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in ("train", "val", "test"):
        split_dir = root / split
        if not split_dir.exists():
            continue
        for class_dir in sorted(item for item in split_dir.iterdir() if item.is_dir()):
            for image_path in sorted(class_dir.iterdir()):
                if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                rows.append(
                    {
                        "split": split,
                        "class": class_dir.name,
                        "path": image_path,
                        "filename": image_path.name,
                    }
                )
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_image(path: Path) -> tuple[bool, str | None]:
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:
        return False, str(exc)
    return True, None


def summarize_counts(rows: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for split in ("train", "val", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        class_counts = defaultdict(int)
        for row in split_rows:
            class_counts[str(row["class"])] += 1
        total = sum(class_counts.values())
        summary[split] = {
            "total": int(total),
            "class_counts": {name: int(class_counts[name]) for name in sorted(class_counts)},
            "malignant_ratio": (
                float(class_counts["malignant"] / total)
                if total and "malignant" in class_counts
                else None
            ),
        }
    return summary


def find_cross_split_overlaps(rows: list[dict[str, object]], key: str) -> list[dict[str, object]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    examples: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        value = str(row[key])
        grouped[value].add(str(row["split"]))
        if len(examples[value]) < 5:
            examples[value].append(str(row["path"]))

    overlaps = []
    for value, splits in grouped.items():
        if len(splits) > 1:
            overlaps.append(
                {
                    key: value,
                    "splits": sorted(splits),
                    "examples": examples[value],
                }
            )
    return overlaps


def audit_manifest(root: Path) -> dict[str, object]:
    manifest_path = root / "split_manifest.csv"
    if not manifest_path.exists():
        return {
            "available": False,
            "note": "未找到 split_manifest.csv，无法进行 lesion_id 分组泄漏审计。",
        }

    manifest = pd.read_csv(manifest_path)
    group_column = "lesion_id" if "lesion_id" in manifest.columns else None
    if group_column is None:
        candidates = [column for column in manifest.columns if column.endswith("_id")]
        group_column = candidates[0] if candidates else None
    if group_column is None or "split" not in manifest.columns:
        return {
            "available": True,
            "group_column": group_column,
            "leakage_count": None,
            "note": "manifest 中缺少 split 或可用分组字段，无法检查分组泄漏。",
        }

    split_counts = manifest.groupby(group_column)["split"].nunique()
    leaked_groups = split_counts[split_counts > 1].index.tolist()
    return {
        "available": True,
        "group_column": group_column,
        "leakage_count": int(len(leaked_groups)),
        "leaked_group_examples": [str(item) for item in leaked_groups[:10]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="审计 ImageFolder 数据集的数据质量和划分泄漏风险。")
    parser.add_argument("--root", default="data/ham10000_binary", help="包含 train/val/test 的数据集根目录。")
    parser.add_argument(
        "--output",
        default=None,
        help="报告输出路径，默认写入数据集根目录 data_quality_report.json。",
    )
    parser.add_argument(
        "--skip-hash",
        action="store_true",
        help="跳过 SHA256 哈希检查；数据量很大且只想快速统计时可启用。",
    )
    args = parser.parse_args()

    root = Path(args.root)
    output_path = Path(args.output) if args.output else root / "data_quality_report.json"
    rows = iter_split_images(root)

    unreadable = []
    for row in rows:
        ok, error = verify_image(Path(row["path"]))
        if not ok:
            unreadable.append({"path": str(row["path"]), "error": error})

    filename_overlaps = find_cross_split_overlaps(rows, key="filename")
    hash_overlaps = []
    if not args.skip_hash:
        hashed_rows = []
        for row in rows:
            hashed = dict(row)
            hashed["sha256"] = file_sha256(Path(row["path"]))
            hashed_rows.append(hashed)
        hash_overlaps = find_cross_split_overlaps(hashed_rows, key="sha256")

    report = {
        "dataset_root": str(root),
        "total_images": int(len(rows)),
        "counts": summarize_counts(rows),
        "unreadable_images": unreadable,
        "cross_split_filename_overlap_count": int(len(filename_overlaps)),
        "cross_split_filename_overlap_examples": filename_overlaps[:10],
        "cross_split_hash_overlap_count": int(len(hash_overlaps)),
        "cross_split_hash_overlap_examples": hash_overlaps[:10],
        "manifest_audit": audit_manifest(root),
        "note": (
            "该报告用于训练前质量控制和论文可复现性说明。正式测试指标必须在完整测试集上报告，"
            "不得根据测试结果事后剔除样本。"
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print(f"数据质量审计完成：{output_path}")
    print(f"图像总数：{len(rows)}")
    print(f"坏图数量：{len(unreadable)}")
    print(f"跨 split 文件名重叠：{len(filename_overlaps)}")
    print(f"跨 split 哈希重叠：{len(hash_overlaps)}")


if __name__ == "__main__":
    main()
