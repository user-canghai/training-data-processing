import hashlib
import time
from typing import Any, Dict, List, Set

from tqdm import tqdm

from .semantic_deduplicator import SemanticDeduplicator
from .text_deduplicator import TextDeduplicator


class PipelineDeduplicator:
    last_report: Dict[str, Any] = {}

    @staticmethod
    def exact_dedup(documents: List[str]) -> List[str]:
        print("=" * 60)
        print("[Step 1] 精确去重 (MD5)")
        print("=" * 60)

        seen: Set[str] = set()
        result = []

        for document in tqdm(documents, desc="精确去重"):
            digest = hashlib.md5(document.strip().encode()).hexdigest()
            if digest not in seen:
                seen.add(digest)
                result.append(document)

        removed = len(documents) - len(result)
        print(f"    {len(documents)} → {len(result)} (移除 {removed} 篇)")
        return result

    @staticmethod
    def run(
        documents: List[str],
        text_threshold: float = 0.8,
        semantic_threshold: float = 0.85,
        do_semantic: bool = True,
    ) -> List[str]:
        original = len(documents)
        total_start_time = time.time()

        print(f"\n{'=' * 60}")
        print("    大模型预训练数据集去重流水线")
        print(f"    输入文档数: {original}")
        print(f"{'=' * 60}\n")

        if not documents:
            PipelineDeduplicator.last_report = {
                "original": 0,
                "after_exact": 0,
                "after_minhash": 0,
                "after_semantic": 0 if do_semantic else None,
                "final": 0,
                "removed_ratio": 0.0,
                "text_threshold": text_threshold,
                "semantic_threshold": semantic_threshold if do_semantic else None,
                "elapsed_seconds": 0.0,
            }
            print("没有可处理的文档")
            return []

        documents = PipelineDeduplicator.exact_dedup(documents)
        after_exact = len(documents)

        print("\n" + "=" * 60)
        print("[Step 2] MinHash 文本去重")
        print("=" * 60)

        text_deduplicator = TextDeduplicator(
            num_hashes=128,
            num_bands=16,
            ngram_size=5,
            similarity_threshold=text_threshold,
        )
        documents = text_deduplicator.deduplicate(documents)
        after_minhash = len(documents)
        after_semantic = after_minhash

        if do_semantic:
            print("\n" + "=" * 60)
            print("[Step 3] 语义去重")
            print("=" * 60)

            semantic_deduplicator = SemanticDeduplicator(
                backend="tfidf",
                similarity_threshold=semantic_threshold,
            )
            documents = semantic_deduplicator.deduplicate(documents)
            after_semantic = len(documents)

        total_elapsed = time.time() - total_start_time

        report = {
            "original": original,
            "after_exact": after_exact,
            "after_minhash": after_minhash,
            "after_semantic": after_semantic if do_semantic else None,
            "final": len(documents),
            "removed_ratio": 1 - len(documents) / original,
            "text_threshold": text_threshold,
            "semantic_threshold": semantic_threshold if do_semantic else None,
            "elapsed_seconds": total_elapsed,
        }
        PipelineDeduplicator.last_report = report

        print(f"\n{'=' * 60}")
        print("    ✓ 去重完成!")
        print(f"    原始文档: {original}")
        print(f"    最终文档: {len(documents)}")
        print(f"    移除比例: {(1 - len(documents) / original) * 100:.1f}%")
        print(f"    总耗时:   {total_elapsed:.2f}s")
        print(f"{'=' * 60}")

        return documents
