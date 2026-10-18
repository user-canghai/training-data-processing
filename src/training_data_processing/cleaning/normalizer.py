import re
import unicodedata


class TextNormalizer:
    @staticmethod
    def clean(text: str) -> str:
        if not text:
            return ""

        result = []
        for character in text:
            code = ord(character)
            if code == 0x3000:
                result.append(" ")
            elif 0xFF01 <= code <= 0xFF5E:
                result.append(chr(code - 0xFEE0))
            else:
                result.append(character)
        text = "".join(result)

        quote_map = {
            "\u201c": '"',
            "\u201d": '"',
            "\u2018": "'",
            "\u2019": "'",
            "\u300c": '"',
            "\u300d": '"',
            "\u300e": '"',
            "\u300f": '"',
        }
        for old, new in quote_map.items():
            text = text.replace(old, new)

        text = text.replace("\u2014", "--")
        text = text.replace("\u2013", "-")
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
