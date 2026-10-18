import time
from typing import Dict, List, Set

import numpy as np
from tqdm import tqdm

from .lsh import LSHIndex
from .minhash import MinHasher


class TextDeduplicator:
    def __init__(
        self,
        num_hashes: int = 128,
        num_bands: int = 16,
        ngram_size: int = 5,
        similarity_threshold: float = 0.8,
        seed: int = 42,
    ):
        self.minhasher = MinHasher(num_hashes=num_hashes, ngram_size=ngram_size)
        self.lsh = LSHIndex(
            num_hashes=num_hashes, num_bands=num_bands, seed=seed
        )
        self.similarity_threshold = similarity_threshold
        self.signatures: Dict[int, np.ndarray] = {}

    def build_index(self, documents: List[str]) -> None:
        print(f"[文本去重] 为 {len(documents)} 篇文档计算 MinHash 签名...")
        for doc_id, text in enumerate(tqdm(documents, desc="MinHash 签名")):
            signature = self.minhasher.compute_signature(text)
            self.signatures[doc_id] = signature
            self.lsh.insert(doc_id, signature)

    def find_duplicates(self) -> Dict[int, List[int]]:
        total_documents = len(self.signatures)
        print(
            f"[文本去重] 查找重复 (Jaccard 阈值={self.similarity_threshold})..."
        )

        duplicate_groups: Dict[int, List[int]] = {}
        seen: Set[int] = set()

        for doc_id in tqdm(range(total_documents), desc="重复检测"):
            if doc_id in seen:
                continue

            candidates = self.lsh.query_candidates(self.signatures[doc_id])
            duplicates = []

            for candidate_id in candidates:
                if candidate_id <= doc_id:
                    continue
                similarity = self.minhasher.jaccard_similarity(
                    self.signatures[doc_id], self.signatures[candidate_id]
                )
                if similarity >= self.similarity_threshold:
                    duplicates.append(candidate_id)
                    seen.add(candidate_id)

            if duplicates:
                duplicate_groups[doc_id] = duplicates

        total_duplicates = sum(len(group) for group in duplicate_groups.values())
        print(
            f"[文本去重] 发现 {len(duplicate_groups)} 组重复, "
            f"共 {total_duplicates} 篇"
        )
        return duplicate_groups

    def deduplicate(self, documents: List[str]) -> List[str]:
        start_time = time.time()
        self.build_index(documents)
        duplicate_groups = self.find_duplicates()

        remove_ids: Set[int] = set()
        for duplicate_ids in duplicate_groups.values():
            remove_ids.update(duplicate_ids)

        deduplicated = [
            document
            for index, document in enumerate(documents)
            if index not in remove_ids
        ]
        elapsed = time.time() - start_time
        print(
            f"[文本去重] {len(documents)} → {len(deduplicated)} 篇 "
            f"(耗时 {elapsed:.2f}s)"
        )
        return deduplicated
