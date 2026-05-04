import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fedmeddp.simulate import run_simulation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "demo.yaml"),
        help="YAML 配置文件路径。",
    )
    args = parser.parse_args()
    output_dir = run_simulation(args.config)
    print(f"实验已完成，输出结果保存在：{output_dir}")


if __name__ == "__main__":
    main()
