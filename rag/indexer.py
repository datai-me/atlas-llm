"""
Vector Indexer
--------------
Builds FAISS index for document retrieval.
"""

import faiss
import numpy as np


class VectorIndexer:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)

    def build(self, embeddings: np.ndarray):
        self.index.add(embeddings)

    def search(self, query: np.ndarray, k: int = 5):
        distances, indices = self.index.search(query, k)
        return indices
