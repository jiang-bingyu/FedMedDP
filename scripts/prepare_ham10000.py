import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


MALIGNANT = {"mel", "bcc", "akiec"}
BENIGN = {"nv", "bkl", "df", "vasc"}


def resolve_image_path(src: Path, image_id: str) -> Path:
    candidates = [
        src / f"{image_id}.jpg",
        src / "HAM10000_images_part_1" / f"{image_id}.jpg",
        src / "HAM10000_images_part_2" / f"{image_id}.jpg",
        src / "images" / f"{image_id}.jpg",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Image not found for image_id={image_id}")


def simplify_label(dx: str) -> str | None:
    if dx in MALIGNANT:
        return "malignant"
    if dx in BENIGN:
        return "benign"
    return None


def copy_split(frame: pd.DataFrame, split: str, src: Path, dst: Path) -> None:
    for _, row in frame.iterrows():
        image_path = resolve_image_path(src, row["image_id"])
        class_dir = dst / split / row["binary_label"]
        class_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, class_dir / image_path.name)


def split_by_group(
    metadata: pd.DataFrame,
    group_field: str,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if group_field not in metadata.columns:
        raise KeyError(f"元数据中缺少分组字段：{group_field}")

    group_table = (
        metadata.groupby(group_field, as_index=False)
        .agg(binary_label=("binary_label", "first"), image_count=("image_id", "count"))
        .copy()
    )

    train_groups, temp_groups = train_test_split(
        group_table,
        test_size=val_ratio + test_ratio,
        random_state=seed,
        stratify=group_table["binary_label"],
    )
    relative_val_ratio = val_ratio / (val_ratio + test_ratio)
    val_groups, test_groups = train_test_split(
        temp_groups,
        test_size=1.0 - relative_val_ratio,
        random_state=seed,
        stratify=temp_groups["binary_label"],
    )

    train_ids = set(train_groups[group_field])
    val_ids = set(val_groups[group_field])
    test_ids = set(test_groups[group_field])
    overlaps = (train_ids & val_ids) | (train_ids & test_ids) | (val_ids & test_ids)
    if overlaps:
        raise RuntimeError(f"分组划分失败，检测到交叉泄露：{sorted(list(overlaps))[:5]}")

    train_df = metadata[metadata[group_field].isin(train_ids)].copy()
    val_df = metadata[metadata[group_field].isin(val_ids)].copy()
    test_df = metadata[metadata[group_field].isin(test_ids)].copy()
    return train_df, val_df, test_df


def build_split_summary(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_field: str,
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, object]:
    def summarize(frame: pd.DataFrame) -> dict[str, object]:
        class_counts = frame["binary_label"].value_counts().to_dict()
        return {
            "images": int(len(frame)),
            "groups": int(frame[group_field].nunique()),
            "class_counts": {key: int(value) for key, value in class_counts.items()},
        }

    return {
        "group_field": group_field,
        "seed": int(seed),
        "val_ratio": float(val_ratio),
        "test_ratio": float(test_ratio),
        "note": (
            "HAM10000 原始元数据未提供显式 patient_id；本脚本使用 lesion_id 进行病灶级隔离划分，"
            "确保同一 lesion 的所有图像只出现在 train/val/test 之一中，防止数据泄露。"
        ),
        "splits": {
            "train": summarize(train_df),
            "val": summarize(val_df),
            "test": summarize(test_df),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="原始 HAM10000 数据目录")
    parser.add_argument("--dst", required=True, help="输出的二分类数据目录")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument(
        "--group-field",
        type=str,
        default="lesion_id",
        help="用于隔离划分的分组字段。HAM10000 建议使用 lesion_id。",
    )
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    metadata_path = src / "HAM10000_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"未找到元数据文件：{metadata_path}")

    metadata = pd.read_csv(metadata_path)
    metadata["binary_label"] = metadata["dx"].map(simplify_label)
    metadata = metadata.dropna(subset=["binary_label"]).copy()
    train_df, val_df, test_df = split_by_group(
        metadata=metadata,
        group_field=args.group_field,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    copy_split(train_df, "train", src, dst)
    copy_split(val_df, "val", src, dst)
    copy_split(test_df, "test", src, dst)

    manifest = pd.concat(
        [
            train_df.assign(split="train"),
            val_df.assign(split="val"),
            test_df.assign(split="test"),
        ],
        ignore_index=True,
    )[
        ["image_id", args.group_field, "dx", "binary_label", "split"]
    ]
    manifest.to_csv(dst / "split_manifest.csv", index=False)

    summary = build_split_summary(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        group_field=args.group_field,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    with (dst / "split_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print("HAM10000 二分类数据集整理完成。")
    print(f"源目录：{src}")
    print(f"目标目录：{dst}")
    print(f"隔离划分字段：{args.group_field}")
    print("类别数量统计：")
    print(metadata["binary_label"].value_counts().to_string())
    print("各数据集划分统计：")
    for split_name, frame in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(
            f"- {split_name}: 图像 {len(frame)} 张，"
            f"{args.group_field} 数量 {frame[args.group_field].nunique()}，"
            f"类别分布 {frame['binary_label'].value_counts().to_dict()}"
        )


if __name__ == "__main__":
    main()
