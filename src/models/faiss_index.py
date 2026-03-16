"""
FAISS IVF (Inverted File) index wrapper for approximate nearest neighbor search.
Uses inner product similarity on L2-normalized embeddings, equivalent to cosine similarity.
"""
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger_tmp = logging.getLogger(__name__)
    logger_tmp.warning("faiss not available — install faiss-cpu or faiss-gpu")

logger = logging.getLogger(__name__)


class FAISSIndex:
    """
    Wrapper around a FAISS IVFFlat index for approximate nearest neighbor retrieval.

    The index operates in inner-product space on L2-normalized vectors,
    so scores are equivalent to cosine similarities in [-1, 1].
    """

    def __init__(self, embed_dim: int = 64, n_lists: int = 100, n_probe: int = 10):
        if not FAISS_AVAILABLE:
            raise ImportError("faiss-cpu is required. Install with: pip install faiss-cpu")
        self.embed_dim = embed_dim
        self.n_lists = n_lists
        self.n_probe = n_probe
        self.index: Optional[faiss.IndexIVFFlat] = None
        self.item_ids: Optional[np.ndarray] = None  # Maps FAISS internal idx → item_id
        self._item_id_to_faiss_idx: Dict[int, int] = {}

    # ------------------------------------------------------------------ #
    # Index Construction                                                   #
    # ------------------------------------------------------------------ #

    def build_ivf_index(
        self,
        embeddings: np.ndarray,
        item_ids: List[int],
    ) -> None:
        """
        Build an IVFFlat index from item embeddings.

        Args:
            embeddings: (n_items, embed_dim) float32 array of L2-normalized vectors
            item_ids: list of item IDs corresponding to each row of embeddings
        """
        assert embeddings.dtype == np.float32, "Embeddings must be float32"
        assert embeddings.shape[1] == self.embed_dim, (
            f"Expected embed_dim={self.embed_dim}, got {embeddings.shape[1]}"
        )
        n = embeddings.shape[0]

        # Ensure vectors are L2 normalized (critical for IP to equal cosine similarity)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-8)

        # IVFFlat index with inner product metric
        quantizer = faiss.IndexFlatIP(self.embed_dim)
        self.index = faiss.IndexIVFFlat(quantizer, self.embed_dim, self.n_lists, faiss.METRIC_INNER_PRODUCT)
        self.index.nprobe = self.n_probe

        logger.info("Training FAISS IVF index on %d vectors (n_lists=%d)...", n, self.n_lists)
        self.index.train(embeddings)
        self.index.add(embeddings)

        self.item_ids = np.array(item_ids, dtype=np.int64)
        self._item_id_to_faiss_idx = {int(iid): idx for idx, iid in enumerate(item_ids)}

        logger.info(
            "FAISS index built: %d vectors, %d lists, probe=%d",
            self.index.ntotal, self.n_lists, self.n_probe,
        )

    # ------------------------------------------------------------------ #
    # Search                                                               #
    # ------------------------------------------------------------------ #

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 500,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for the k nearest items to the query vector.

        Args:
            query_vector: (embed_dim,) or (1, embed_dim) float32 query embedding
            k: number of nearest neighbors to return

        Returns:
            (distances, item_ids) — both shape (k,)
            distances are inner product scores (higher = more similar)
            item_ids are original item IDs from the catalog
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_ivf_index() first.")

        query = np.atleast_2d(query_vector).astype(np.float32)
        norm = np.linalg.norm(query, axis=1, keepdims=True)
        query = query / np.maximum(norm, 1e-8)

        k = min(k, self.index.ntotal)
        distances, faiss_indices = self.index.search(query, k)

        distances = distances[0]
        faiss_indices = faiss_indices[0]

        # Filter out invalid indices (-1 means no result found in some edge cases)
        valid_mask = faiss_indices >= 0
        distances = distances[valid_mask]
        faiss_indices = faiss_indices[valid_mask]

        retrieved_item_ids = self.item_ids[faiss_indices]
        return distances, retrieved_item_ids

    def batch_search(
        self,
        query_vectors: np.ndarray,
        k: int = 500,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch search for multiple queries.

        Returns:
            (distances, item_ids) — both shape (n_queries, k)
        """
        if self.index is None:
            raise RuntimeError("Index not built.")

        queries = query_vectors.astype(np.float32)
        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        queries = queries / np.maximum(norms, 1e-8)

        k = min(k, self.index.ntotal)
        distances, faiss_indices = self.index.search(queries, k)

        # Map FAISS indices to item IDs
        item_id_results = np.where(
            faiss_indices >= 0,
            self.item_ids[np.clip(faiss_indices, 0, len(self.item_ids) - 1)],
            -1,
        )
        return distances, item_id_results

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        """Save FAISS index and item ID mapping to disk."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(save_path))

        meta_path = save_path.with_suffix(".meta.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump(
                {
                    "item_ids": self.item_ids,
                    "item_id_to_faiss_idx": self._item_id_to_faiss_idx,
                    "embed_dim": self.embed_dim,
                    "n_lists": self.n_lists,
                    "n_probe": self.n_probe,
                },
                f,
            )
        logger.info("Saved FAISS index to %s (meta: %s)", save_path, meta_path)

    @classmethod
    def load(cls, path: str) -> "FAISSIndex":
        """Load a saved FAISS index from disk."""
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {load_path}")

        meta_path = load_path.with_suffix(".meta.pkl")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        obj = cls(
            embed_dim=meta["embed_dim"],
            n_lists=meta["n_lists"],
            n_probe=meta["n_probe"],
        )
        obj.index = faiss.read_index(str(load_path))
        obj.index.nprobe = meta["n_probe"]
        obj.item_ids = meta["item_ids"]
        obj._item_id_to_faiss_idx = meta["item_id_to_faiss_idx"]

        logger.info(
            "Loaded FAISS index from %s: %d vectors, dim=%d",
            load_path, obj.index.ntotal, obj.embed_dim,
        )
        return obj

    # ------------------------------------------------------------------ #
    # Utilities                                                            #
    # ------------------------------------------------------------------ #

    def stats(self) -> Dict:
        """Return index statistics."""
        if self.index is None:
            return {"status": "not built"}
        return {
            "n_vectors": int(self.index.ntotal),
            "embed_dim": self.embed_dim,
            "n_lists": self.n_lists,
            "n_probe": self.n_probe,
            "metric": "inner_product",
            "n_item_ids": len(self.item_ids) if self.item_ids is not None else 0,
        }

    def set_n_probe(self, n_probe: int) -> None:
        """Adjust the search-time accuracy/speed tradeoff."""
        self.n_probe = n_probe
        if self.index is not None:
            self.index.nprobe = n_probe
