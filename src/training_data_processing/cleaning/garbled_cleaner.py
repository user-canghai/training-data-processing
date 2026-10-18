import re
import unicodedata

import ftfy


class GarbledTextCleaner:
    KEEP_WHITESPACE = frozenset(" \t\n\r")

    ZERO_WIDTH_CHARS = (
        "\u200b",
        "\u200c",
        "\u200d",
        "\ufeff",
        "\u00ad",
        "\u200e",
        "\u200f",
        "\u202a",
        "\u202b",
        "\u202c",
        "\u202d",
        "\u202e",
        "\u2060",
        "\u2061",
        "\u2062",
        "\u2063",
        "\u2064",
    )

    @staticmethod
    def clean(text: str) -> str:
        if not text:
            return ""

        text = ftfy.fix_text(text)

        for zero_width_char in GarbledTextCleaner.ZERO_WIDTH_CHARS:
            text = text.replace(zero_width_char, "")

        text = "".join(
            character
            for character in text
            if character in GarbledTextCleaner.KEEP_WHITESPACE
            or (ord(character) >= 32 and unicodedata.category(character)[0] != "C")
        )

        text = unicodedata.normalize("NFC", text)
        return text

    @staticmethod
    def is_garbled(text: str, max_garbled_ratio: float = 0.3) -> bool:
        if not text or len(text.strip()) < 5:
            return True

        printable_count = sum(
            1
            for character in text
            if character.isspace()
            or unicodedata.category(character)[0] in ("L", "N", "P", "S")
        )
        if printable_count / len(text) < (1.0 - max_garbled_ratio):
            return True

        if re.search(r"(.)\1{10,}", text):
            return True

        return False
