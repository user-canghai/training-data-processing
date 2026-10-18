import argparse
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = os.environ.get("TRAINING_DATA_URL")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载原始数据")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/raw/raw.jsonl"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.output.exists():
        print(f"数据已存在: {args.output}")
        return

    if not args.url:
        raise SystemExit(
            "未找到数据文件，请通过 --url 或 TRAINING_DATA_URL 指定原始数据地址"
        )

    urllib.request.urlretrieve(args.url, args.output)
    print(f"已下载: {args.output}")


if __name__ == "__main__":
    main()
