import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training_data_processing.dedup import PipelineDeduplicator
from training_data_processing.io import load_jsonl, save_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="数据去重流水线")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/processed/cleaned.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/processed/deduplicated.jsonl",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "outputs/reports/dedup_report.json",
    )
    parser.add_argument("--text-threshold", type=float, default=0.5)
    parser.add_argument("--semantic-threshold", type=float, default=0.5)
    parser.add_argument("--no-semantic", action="store_true")
    return parser.parse_args()


def build_exact_labels(texts):
    labels = {}
    first_seen = {}
    for index, text in enumerate(texts):
        digest = hashlib.md5(text.strip().encode()).hexdigest()
        if digest in first_seen:
            labels[index] = f"←精确重复(#{first_seen[digest]})"
        else:
            first_seen[digest] = index
    return labels


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.input)
    texts = [record["text"] for record in records]
    labels = build_exact_labels(texts)

    print("=" * 60)
    print(f"    测试数据集  (共 {len(texts)} 篇文档)")
    print("=" * 60)
    for index, text in enumerate(texts):
        print(f"    [{index:2d}] {text[:65]}...  {labels.get(index, '')}")

    print("\n\n" + "▓" * 60)
    print("    运行完整去重流水线")
    print("▓" * 60)

    final_docs = PipelineDeduplicator.run(
        texts,
        text_threshold=args.text_threshold,
        semantic_threshold=args.semantic_threshold,
        do_semantic=not args.no_semantic,
    )

    print("\n最终保留的文档:")
    for index, document in enumerate(final_docs):
        print(f"    [{index:2d}] {document[:75]}...")

    records_by_text = {}
    for record in records:
        records_by_text.setdefault(record["text"], record)

    final_records = [
        dict(records_by_text[document])
        for document in final_docs
        if document in records_by_text
    ]
    save_jsonl(final_records, args.output)

    report = dict(PipelineDeduplicator.last_report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\n完成: {len(texts)} → {len(final_docs)} 篇")
    print(f"去重结果: {args.output}")
    print(f"去重报告: {args.report}")


if __name__ == "__main__":
    main()
