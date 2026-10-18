import re
from collections import Counter
from typing import List, Optional, Tuple


class SensitiveDataCleaner:
    DEFAULT_RULES: List[Tuple[str, str, str]] = [
        ("手机号", r"(?<!\d)1[3-9]\d{9}(?!\d)", r"[手机号]"),
        ("固定电话", r"(?<!\d)0\d{2,3}-?\d{7,8}(?!\d)", r"[固定电话]"),
        (
            "邮箱",
            r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
            r"[邮箱]",
        ),
        (
            "IP 地址",
            r"(?<!\d)(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)(?!\d)",
            r"[IP 地址]",
        ),
        (
            "身份证号",
            r"(?<![A-Za-z0-9])"
            r"[1-9]\d{5}"
            r"(?:19|20)\d{2}"
            r"(?:0[1-9]|1[0-2])"
            r"(?:0[1-9]|[12]\d|3[01])"
            r"\d{3}"
            r"[\dXx]"
            r"(?![A-Za-z0-9])",
            r"[身份证号]",
        ),
        (
            "银行卡号",
            r"(?<![A-Za-z0-9])"
            r"(?:62|4[0-9]|5[1-5])\d{14,17}"
            r"(?![A-Za-z0-9])",
            r"[银行卡号]",
        ),
        (
            "信用代码",
            r"(?<![A-Za-z0-9])"
            r"[0-9A-HJ-NP-RT-UW-Y]{2}\d{6}[0-9A-HJ-NP-RT-UW-Y]{10}"
            r"(?![A-Za-z0-9])",
            r"[信用代码]",
        ),
        (
            "密码",
            r"(?:password|passwd|pwd|密码|口令)\s*[：:=]\s*\S+",
            r"[密码已脱敏]",
        ),
    ]

    def __init__(
        self,
        extra_rules: Optional[List[Tuple[str, str, str]]] = None,
        mask_mode: str = "replace",
    ):
        self.mask_mode = mask_mode
        self.rules = list(self.DEFAULT_RULES) + (extra_rules or [])
        self.compiled = [
            (name, re.compile(pattern), template)
            for name, pattern, template in self.rules
        ]
        self.stats: Counter = Counter()

    @staticmethod
    def _mask_phone(match: re.Match) -> str:
        number = match.group()
        return number[:3] + "****" + number[7:] if len(number) == 11 else "[手机号]"

    @staticmethod
    def _mask_id_card(match: re.Match) -> str:
        number = match.group()
        return (
            number[:3] + "*" * 11 + number[-4:]
            if len(number) == 18
            else "[身份证号]"
        )

    @staticmethod
    def _mask_bank(match: re.Match) -> str:
        number = match.group()
        return number[:4] + "*" * (len(number) - 8) + number[-4:]

    @staticmethod
    def _mask_email(match: re.Match) -> str:
        address = match.group()
        parts = address.split("@")
        if len(parts) == 2:
            local = parts[0]
            masked = local[0] + "***" if len(local) > 3 else "***"
            return masked + "@" + parts[1]
        return "[邮箱]"

    def clean(self, text: str) -> str:
        if not text:
            return ""

        for name, pattern, template in self.compiled:
            matches = pattern.findall(text)
            if matches:
                self.stats[name] += len(matches)

            if self.mask_mode == "mask":
                if name == "手机号":
                    text = pattern.sub(self._mask_phone, text)
                    continue
                if name == "身份证号":
                    text = pattern.sub(self._mask_id_card, text)
                    continue
                if name == "银行卡号":
                    text = pattern.sub(self._mask_bank, text)
                    continue
                if name == "邮箱":
                    text = pattern.sub(self._mask_email, text)
                    continue

            text = pattern.sub(template, text)

        return text
