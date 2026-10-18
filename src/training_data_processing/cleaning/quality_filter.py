import unicodedata
from typing import Optional, Tuple

from langdetect import LangDetectException, detect


class QualityFilter:
    def __init__(
        self,
        min_length: int = 50,
        max_length: int = 100000,
        min_alpha_ratio: float = 0.3,
        max_special_ratio: float = 0.3,
        max_dup_line_ratio: float = 0.5,
        max_dup_char_ratio: float = 0.3,
        language: Optional[str] = None,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.min_alpha_ratio = min_alpha_ratio
        self.max_special_ratio = max_special_ratio
        self.max_dup_line_ratio = max_dup_line_ratio
        self.max_dup_char_ratio = max_dup_char_ratio
        self.language = language

    def check(self, text: str) -> Tuple[bool, str]:
        if not text:
            return False, "空文本"

        if len(text) < self.min_length:
            return False, f"过短: {len(text)} < {self.min_length}"

        if len(text) > self.max_length:
            return False, f"过长: {len(text)} > {self.max_length}"

        alpha_count = sum(
            1
            for character in text
            if character.isalpha() or ("\u4e00" <= character <= "\u9fff")
        )
        if alpha_count / len(text) < self.min_alpha_ratio:
            return False, f"字母比例过低: {alpha_count / len(text):.2f}"

        special_count = sum(
            1 for character in text if unicodedata.category(character)[0] == "S"
        )
        if special_count / len(text) > self.max_special_ratio:
            return False, f"特殊符号过多: {special_count / len(text):.2f}"

        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if lines:
            duplicate_line_ratio = 1 - len(set(lines)) / len(lines)
            if duplicate_line_ratio > self.max_dup_line_ratio:
                return False, f"重复行过多: {duplicate_line_ratio:.2f}"

        if len(text) > 20:
            bigrams = [text[index : index + 2] for index in range(len(text) - 1)]
            duplicate_bigram_ratio = 1 - len(set(bigrams)) / max(len(bigrams), 1)
            if duplicate_bigram_ratio > self.max_dup_char_ratio:
                return False, f"重复 bigram 过多: {duplicate_bigram_ratio:.2f}"

        if self.language:
            try:
                detected = detect(text[:1000])
                if detected != self.language:
                    return False, f"语言不匹配: detected={detected}"
            except LangDetectException:
                pass

        return True, "通过"
