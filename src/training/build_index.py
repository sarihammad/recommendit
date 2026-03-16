"""
Builds the FAISS ANN index from trained two-tower item embeddings.
Generates embeddings for every item in the catalog and writes the IVF index to disk.
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from src.config import settings
from src.features.feature_engineering import FeatureEngineer, N_GENRES, GENRE_TO_IDX
from src.models.faiss_index import FAISSIndex
from src.models.two_tower import TwoTowerModel

logger = logging.getLogger(__name__)


class IndexBuilder:
    """
    Orchestrates loading of the trained two-tower model, computing all item
    embeddings, and constructing the FAISS IVF index.
    """

    def __init__(
        self,
        model_path: str = None,
        index_output_path: str = None,
        data_dir: str = None,
        embed_dim: int = None,
        n_lists: int = None,
        n_probe: int = None,
        device: Optional[str] = None,
    ):
        self.model_path = model_path or settings.EMBEDDING_MODEL_PATH
        self.index_output_path = index_output_path or settings.FAISS_INDEX_PATH
        self.data_dir = data_dir or settings.DATA_DIR
        self.embed_dim = embed_dim or settings.EMBEDDING_DIM
        self.n_lists = n_lists or settings.FAISS_N_LISTS
        self.n_probe = n_probe or settings.FAISS_N_PROBE

        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def _build_genre_matrix(self, movies_df: pd.DataFrame) -> tuple:
        """Build (item_ids, genre_matrix) from movies DataFrame."""
        item_ids = sorted(movies_df["item_id"].unique().tolist())
        id_to_genres = {
            int(row["item_id"]): str(row["genres"])
            for _, row in movies_df.iterrows()
        }
        genre_matrix = np.zeros((len(item_ids), N_GENRES), dtype=np.float32)
        for i, iid in enumerate(item_ids):
            genres_str = id_to_genres.get(iid, "")
            for genre in genres_str.split("|"):
                idx = GENRE_TO_IDX.get(genre)
                if idx is not None:
                    genre_matrix[i, idx] = 1.0
        return item_ids, genre_matrix

    def build(self) -> FAISSIndex:
        """
        Full pipeline:
        1. Load trained two-tower model
        2. Load movie catalog
        3. Compute item embeddings for all items
        4. Build FAISS IVF index
        5. Save index to disk

        Returns:
            Built and saved FAISSIndex instance
        """
        # Load model
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Two-tower model not found at {self.model_path}. "
                "Run train_embeddings first."
            )
        logger.info("Loading two-tower model from %s ...", self.model_path)
        model = TwoTowerModel.load(self.model_path, device=self.device)
        model.eval()

        # Load movie catalog
        logger.info("Loading movie catalog from %s ...", self.data_dir)
        fe = FeatureEngineer(self.data_dir)
        fe.load_data()
        movies_df = fe.movies_df

        item_ids, genre_matrix = self._build_genre_matrix(movies_df)
        n_items = len(item_ids)
        logger.info("Generating embeddings for %d catalog items ...", n_items)

        # Generate item embeddings in batches
        item_embeddings = model.get_item_embeddings(
            item_ids,
            genre_matrix,
            device=self.device,
            batch_size=512,
        )
        logger.info(
            "Generated embeddings: shape=%s, dtype=%s",
            item_embeddings.shape, item_embeddings.dtype,
        )

        # Verify L2 norms (should be ~1.0 after normalization in tower)
        norms = np.linalg.norm(item_embeddings, axis=1)
        logger.info(
            "Embedding norms: mean=%.4f, min=%.4f, max=%.4f",
            norms.mean(), norms.min(), norms.max(),
        )

        # Build FAISS index
        # IVF requires at least 39 * n_lists training vectors
        min_required = 39 * self.n_lists
        effective_n_lists = self.n_lists if n_items >= min_required else max(1, n_items // 39)
        if effective_n_lists != self.n_lists:
            logger.warning(
                "Reduced n_lists from %d to %d (need >= %d training vectors)",
                self.n_lists, effective_n_lists, min_required,
            )

        faiss_index = FAISSIndex(
            embed_dim=self.embed_dim,
            n_lists=effective_n_lists,
            n_probe=self.n_probe,
        )
        faiss_index.build_ivf_index(item_embeddings, item_ids)

        # Save
        faiss_index.save(self.index_output_path)

        stats = faiss_index.stats()
        logger.info("FAISS index built and saved. Stats: %s", stats)
        return faiss_index
