import time
from typing import Dict, List, Optional, Set

import numpy as np
from tqdm import tqdm


class SemanticDeduplicator:
    def __init__(
        self,
        backend: str = "tfidf",
        similarity_threshold: float = 0.85,
        max_features: int = 50000,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_length: int = 256,
        batch_size: int = 128,
    ):
        self.backend = backend
        self.similarity_threshold = similarity_threshold
        self.embeddings: Optional[np.ndarray] = None

        if backend == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer

            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                analyzer="word",
                sublinear_tf=True,
            )
            print(f"[语义去重] TF-IDF 后端 (max_features={max_features})")
        elif backend == "transformer":
            import torch
            from transformers import AutoModel, AutoTokenizer

            self.max_length = max_length
            self.batch_size = batch_size
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[语义去重] 加载模型: {model_name} (设备: {self.device})")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
        else:
            raise ValueError(
                f"不支持的后端: {backend}, 请选 'tfidf' 或 'transformer'"
            )

    def _encode_batch_transformer(self, texts: List[str]) -> np.ndarray:
        import torch

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoded)
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

        return pooled.cpu().numpy()

    def encode_all(self, documents: List[str]) -> np.ndarray:
        print(f"[语义去重] 编码 {len(documents)} 篇文档...")
        start_time = time.time()

        if not documents:
            self.embeddings = np.zeros((0, 0), dtype=np.float32)
            print(f"[语义去重] 编码完成 {self.embeddings.shape}, 耗时 0.00s")
            return self.embeddings

        if self.backend == "tfidf":
            from sklearn.preprocessing import normalize

            matrix = self.vectorizer.fit_transform(documents).astype(np.float32)
            normalized = normalize(matrix, norm="l2", axis=1)
            self.embeddings = np.asarray(normalized.todense(), dtype=np.float32)
        else:
            embeddings = []
            for index in tqdm(
                range(0, len(documents), self.batch_size), desc="编码"
            ):
                embeddings.append(
                    self._encode_batch_transformer(
                        documents[index : index + self.batch_size]
                    )
                )
            self.embeddings = np.concatenate(embeddings)

        elapsed = time.time() - start_time
        print(
            f"[语义去重] 编码完成 {self.embeddings.shape}, 耗时 {elapsed:.2f}s"
        )
        return self.embeddings

    def find_duplicates(self, documents: List[str]) -> Dict[int, List[int]]:
        if self.embeddings is None:
            self.encode_all(documents)

        total_documents = len(documents)
        if total_documents == 0:
            return {}

        print(f"[语义去重] 计算相似度 (阈值={self.similarity_threshold})...")
        start_time = time.time()

        duplicate_groups: Dict[int, List[int]] = {}
        seen: Set[int] = set()
        chunk_size = 1024

        for start in tqdm(
            range(0, total_documents, chunk_size), desc="相似度"
        ):
            end = min(start + chunk_size, total_documents)
            similarity_matrix = self.embeddings[start:end] @ self.embeddings.T

            for local_index in range(end - start):
                global_index = start + local_index
                if global_index in seen:
                    continue

                duplicates = []
                for other_index in range(global_index + 1, total_documents):
                    if (
                        other_index not in seen
                        and similarity_matrix[local_index, other_index]
                        >= self.similarity_threshold
                    ):
                        duplicates.append(other_index)
                        seen.add(other_index)

                if duplicates:
                    duplicate_groups[global_index] = duplicates

        total_duplicates = sum(len(group) for group in duplicate_groups.values())
        elapsed = time.time() - start_time
        print(
            f"[语义去重] {len(duplicate_groups)} 组, {total_duplicates} 篇, "
            f"耗时 {elapsed:.2f}s"
        )
        return duplicate_groups

    def deduplicate(self, documents: List[str]) -> List[str]:
        if not documents:
            print("[语义去重] 0 → 0 篇 (耗时 0.00s)")
            return []

        start_time = time.time()
        duplicate_groups = self.find_duplicates(documents)

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
            f"[语义去重] {len(documents)} → {len(deduplicated)} 篇 "
            f"(耗时 {elapsed:.2f}s)"
        )
        return deduplicated
