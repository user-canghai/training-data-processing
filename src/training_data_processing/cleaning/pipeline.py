from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from .garbled_cleaner import GarbledTextCleaner
from .html_cleaner import HTMLCleaner
from .normalizer import TextNormalizer
from .quality_filter import QualityFilter
from .sensitive_cleaner import SensitiveDataCleaner
from .stats import CleaningStats


class DataCleaningPipeline:
    def __init__(
        self,
        sensitive_mask_mode: str = "replace",
        sensitive_extra_rules: Optional[list] = None,
        quality_min_length: int = 100,
        quality_max_length: int = 100000,
        quality_max_dup_char: float = 0.3,
        quality_language: Optional[str] = None,
    ):
        self.html = HTMLCleaner()
        self.garbled = GarbledTextCleaner()
        self.sensitive = SensitiveDataCleaner(
            extra_rules=sensitive_extra_rules,
            mask_mode=sensitive_mask_mode,
        )
        self.normalizer = TextNormalizer()
        self.quality = QualityFilter(
            min_length=quality_min_length,
            max_length=quality_max_length,
            max_dup_char_ratio=quality_max_dup_char,
            language=quality_language,
        )
        self.decisions: List[Dict[str, Any]] = []

    def clean_single(self, text: str) -> Tuple[Optional[str], str]:
        if not text:
            return None, "空文本"

        text = self.html.clean(text)
        text = self.garbled.clean(text)

        if self.garbled.is_garbled(text):
            return None, "乱码"

        text = self.sensitive.clean(text)
        text = self.normalizer.clean(text)

        passed, reason = self.quality.check(text)
        if not passed:
            return None, reason

        return text, "通过"

    def clean_batch(
        self, texts: List[str], verbose: bool = True
    ) -> Tuple[List[str], CleaningStats]:
        stats = CleaningStats(total=len(texts))
        results: List[str] = []
        self.decisions = []

        iterator = tqdm(texts, desc="清洗进度") if verbose else texts

        for index, text in enumerate(iterator):
            cleaned, reason = self.clean_single(text)
            self.decisions.append(
                {
                    "index": index,
                    "kept": cleaned is not None,
                    "reason": reason,
                }
            )

            if cleaned is not None:
                results.append(cleaned)
                stats.passed += 1
            elif "乱码" in reason:
                stats.removed_garbled += 1
            elif "过短" in reason:
                stats.removed_short += 1
            else:
                stats.removed_quality += 1

        stats.sensitive_counts = dict(self.sensitive.stats)
        return results, stats

    def clean_records(
        self,
        records: List[Dict[str, Any]],
        text_key: str = "text",
        verbose: bool = True,
    ) -> Tuple[List[Dict[str, Any]], CleaningStats]:
        texts = [record.get(text_key, "") for record in records]
        cleaned_texts, stats = self.clean_batch(texts, verbose=verbose)

        cleaned_records: List[Dict[str, Any]] = []
        cleaned_iterator = iter(cleaned_texts)

        for record, decision in zip(records, self.decisions):
            if decision["kept"]:
                item = dict(record)
                item[text_key] = next(cleaned_iterator)
                cleaned_records.append(item)

        return cleaned_records, stats

    @staticmethod
    def print_report(stats: CleaningStats) -> None:
        removed = stats.total - stats.passed
        ratio = removed / max(stats.total, 1) * 100

        print(f"\n{'=' * 60}")
        print("    数据清洗报告")
        print(f"{'=' * 60}")
        print(f"    总文档数:      {stats.total:>10,}")
        print(f"    保留文档数:    {stats.passed:>10,}")
        print(f"    移除文档数:    {removed:>10,}    ({ratio:.1f}%)")
        print(f"    ├─ 乱码过滤:   {stats.removed_garbled:>10,}")
        print(f"    ├─ 过短过滤:   {stats.removed_short:>10,}")
        print(f"    └─ 质量过滤:   {stats.removed_quality:>10,}")

        if stats.sensitive_counts:
            print("\n    敏感信息脱敏统计:")
            for name, count in sorted(
                stats.sensitive_counts.items(), key=lambda item: -item[1]
            ):
                print(f"        {name:10s}: {count:>8,} 处")

        print(f"{'=' * 60}")
