import html
import re

import regex
from bs4 import BeautifulSoup


class HTMLCleaner:
    REMOVE_TAGS = frozenset(
        {
            "script",
            "style",
            "noscript",
            "iframe",
            "svg",
            "math",
            "nav",
            "header",
            "footer",
            "form",
        }
    )

    BLOCK_TAGS = frozenset(
        {
            "p",
            "div",
            "br",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "li",
            "tr",
            "hr",
            "blockquote",
            "pre",
            "table",
            "ul",
            "ol",
            "section",
            "article",
            "main",
            "aside",
            "figure",
            "figcaption",
        }
    )

    INLINE_TAGS = frozenset(
        {
            "a",
            "span",
            "strong",
            "b",
            "em",
            "i",
            "u",
            "code",
            "sub",
            "sup",
            "font",
            "center",
            "small",
            "mark",
            "abbr",
            "del",
            "ins",
            "kbd",
            "var",
            "cite",
        }
    )

    @staticmethod
    def clean(text: str) -> str:
        if not text:
            return ""
        soup = BeautifulSoup(text, "lxml")

        for tag_name in HTMLCleaner.REMOVE_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        for tag_name in HTMLCleaner.BLOCK_TAGS:
            for tag in soup.find_all(tag_name):
                tag.replace_with("\n" + tag.get_text() + "\n")

        for tag_name in HTMLCleaner.INLINE_TAGS:
            for tag in soup.find_all(tag_name):
                tag.unwrap()

        result = soup.get_text(separator=" ")
        result = html.unescape(result)
        result = regex.sub(r"<!--.*?-->", "", result, flags=regex.DOTALL)
        result = re.sub(r"[ \t]+", " ", result)
        result = re.sub(r"\n{3,}", "\n\n", result)
        return result.strip()
