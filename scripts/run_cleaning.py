import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training_data_processing.cleaning import DataCleaningPipeline
from training_data_processing.io import load_jsonl, save_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="数据清洗流水线")
    parser.add_argument(
        "--input", type=Path, default=ROOT / "data/raw/raw.jsonl"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/processed/cleaned.jsonl",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "outputs/reports/cleaning_report.json",
    )
    parser.add_argument("--min-length", type=int, default=20)
    parser.add_argument("--max-dup-char", type=float, default=0.4)
    parser.add_argument(
        "--mask-mode", choices=["replace", "mask"], default="mask"
    )
    parser.add_argument("--language", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.input)

    pipeline = DataCleaningPipeline(
        sensitive_mask_mode=args.mask_mode,
        quality_min_length=args.min_length,
        quality_max_dup_char=args.max_dup_char,
        quality_language=args.language,
    )
    cleaned_records, stats = pipeline.clean_records(records, verbose=True)

    DataCleaningPipeline.print_report(stats)

    print("\n── 逐条清洗结果 ──")
    cleaned_iterator = iter(cleaned_records)
    for record, decision in zip(records, pipeline.decisions):
        if decision["kept"]:
            cleaned = next(cleaned_iterator)
            preview = cleaned["text"][:60].replace("\n", " ")
        else:
            preview = record.get("text", "")[:60].replace("\n", " ")
        status = "保留" if decision["kept"] else "移除"
        print(
            f"    [{decision['index']:2d}] {record.get('id', 'N/A')} "
            f"{status} ({decision['reason']}) {preview}"
        )

    save_jsonl(cleaned_records, args.output)

    removed = stats.total - stats.passed
    report = {
        "total": stats.total,
        "passed": stats.passed,
        "removed": removed,
        "removed_ratio": removed / max(stats.total, 1),
        "removed_garbled": stats.removed_garbled,
        "removed_short": stats.removed_short,
        "removed_quality": stats.removed_quality,
        "sensitive_counts": stats.sensitive_counts,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"\n最终保留: {stats.passed}/{stats.total} 篇 "
        f"(移除 {removed} 篇, "
        f"{removed / max(stats.total, 1) * 100:.1f}%)"
    )
    print("清洗完成!")
    print(f"清洗结果: {args.output}")
    print(f"清洗报告: {args.report}")


if __name__ == "__main__":
    main()
