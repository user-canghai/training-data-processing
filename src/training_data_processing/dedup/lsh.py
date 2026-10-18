from collections import defaultdict
from typing import Dict, List, Set

import numpy as np


class LSHIndex:
    def __init__(self, num_hashes: int = 128, num_bands: int = 16, seed: int = 42):
        self.num_hashes = num_hashes
        self.num_bands = num_bands
        self.rows_per_band = num_hashes // num_bands

        assert num_hashes % num_bands == 0

        np.random.seed(seed)
        self.band_params = [
            (
                int(np.random.randint(1, (1 << 61) - 1, dtype=np.int64)),
                int(np.random.randint(0, (1 << 61) - 1, dtype=np.int64)),
            )
            for _ in range(num_bands)
        ]
        self.buckets: List[Dict[int, Set[int]]] = [
            defaultdict(set) for _ in range(num_bands)
        ]

    def _hash_band(self, band_values: np.ndarray, band_idx: int) -> int:
        a, b = self.band_params[band_idx]
        prime = (1 << 61) - 1
        value = 0
        for band_value in band_values:
            value = (a * value + int(band_value) + b) % prime
        return value

    def insert(self, doc_id: int, signature: np.ndarray) -> None:
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_hash = self._hash_band(signature[start:end], band_idx)
            self.buckets[band_idx][band_hash].add(doc_id)

    def query_candidates(self, signature: np.ndarray) -> Set[int]:
        candidates: Set[int] = set()
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_hash = self._hash_band(signature[start:end], band_idx)
            if band_hash in self.buckets[band_idx]:
                candidates.update(self.buckets[band_idx][band_hash])
        return candidates
