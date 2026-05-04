import argparse
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fedmeddp.simulate import save_curves


def regenerate_one(experiment_dir: Path) -> bool:
    history_path = experiment_dir / "history.csv"
    summary_path = experiment_dir / "summary.json"
    if not history_path.exists() or not summary_path.exists():
        return False

    history = pd.read_csv(history_path)
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    save_curves(
        history=history,
        output_dir=experiment_dir,
        privacy_enabled=bool(summary.get("privacy_enabled", False)),
        mechanism=str(summary.get("mechanism", "none")),
        privacy_accountant=str(summary.get("privacy_accountant", "not_applicable")),
        privacy_accountant_note=str(summary.get("privacy_accountant_note", "")),
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="根据 history.csv 重新生成训练曲线图。")
    parser.add_argument(
        "--outputs-dir",
        default=str(ROOT / "outputs"),
        help="实验输出根目录。",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="只重画指定实验；不指定则按 prefix 批量重画。",
    )
    parser.add_argument(
        "--prefix",
        default="ham10000_",
        help="批量重画时匹配的实验目录前缀，留空表示全部。",
    )
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    if args.experiment_name:
        targets = [outputs_dir / args.experiment_name]
    else:
        targets = [
            item
            for item in sorted(outputs_dir.iterdir())
            if item.is_dir() and (not args.prefix or item.name.startswith(args.prefix))
        ]

    count = 0
    for experiment_dir in targets:
        if regenerate_one(experiment_dir):
            count += 1
            print(f"已重画：{experiment_dir / 'curves.png'}")

    if count == 0:
        raise FileNotFoundError("未找到可重画的实验目录，请检查 outputs-dir / experiment-name / prefix。")
    print(f"完成，重画曲线数量：{count}")


if __name__ == "__main__":
    main()
