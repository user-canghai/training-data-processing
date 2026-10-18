import hashlib
import re
from typing import Set

import numpy as np


class SimHashFunc:
    def __init__(self, num_hashes: int = 128):
        self.num_hashes = num_hashes

    def hash_shingle(self, shingle: str, hash_idx: int) -> int:
        data = f"{shingle}:{hash_idx}".encode("utf-8")
        digest = hashlib.sha256(data).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=True)


class MinHasher:
    def __init__(self, num_hashes: int = 128, ngram_size: int = 5):
        self.num_hashes = num_hashes
        self.ngram_size = ngram_size
        self.hash_func = SimHashFunc(num_hashes)

    def _get_shingles(self, text: str) -> Set[str]:
        text = re.sub(r"\s+", " ", text.lower().strip())
        if len(text) < self.ngram_size:
            return {text}
        return {
            text[index : index + self.ngram_size]
            for index in range(len(text) - self.ngram_size + 1)
        }

    def compute_signature(self, text: str) -> np.ndarray:
        shingles = self._get_shingles(text)
        if not shingles:
            return np.full(self.num_hashes, np.iinfo(np.int64).max)

        signature = np.full(
            self.num_hashes, np.iinfo(np.int64).max, dtype=np.int64
        )
        for shingle in shingles:
            for index in range(self.num_hashes):
                value = self.hash_func.hash_shingle(shingle, index)
                if value < signature[index]:
                    signature[index] = value
        return signature

    @staticmethod
    def jaccard_similarity(signature1: np.ndarray, signature2: np.ndarray) -> float:
        if len(signature1) != len(signature2):
            raise ValueError("签名维度不一致")
        return float(np.mean(signature1 == signature2))
