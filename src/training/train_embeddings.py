"""
Training script for the Two-Tower embedding model on MovieLens 1M.
Uses BPR loss with in-batch negatives and saves checkpoint after each epoch.
"""
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.config import settings
from src.features.feature_engineering import FeatureEngineer, N_GENRES
from src.models.two_tower import TwoTowerModel

logger = logging.getLogger(__name__)


class UserItemDataset(Dataset):
    """
    PyTorch Dataset for two-tower training.
    Each sample is a (user_id, pos_item_id, pos_genre_vec, neg_item_id, neg_genre_vec) tuple.
    Negative items are sampled at construction time.
    """

    def __init__(
        self,
        ratings_df: pd.DataFrame,
        item_genre_dict: Dict[int, np.ndarray],
        all_item_ids: List[int],
        n_negatives: int = 4,
        min_rating: float = 4.0,
    ):
        self.item_genre_dict = item_genre_dict
        self.all_item_ids = all_item_ids
        self.n_negatives = n_negatives

        # Build positive pairs from ratings >= threshold
        positives = ratings_df[ratings_df["rating"] >= min_rating][["user_id", "item_id"]].copy()
        self.user_ids = positives["user_id"].values
        self.item_ids = positives["item_id"].values

        # Precompute set of rated items per user for negative sampling
        self.user_rated: Dict[int, set] = (
            ratings_df.groupby("user_id")["item_id"].apply(set).to_dict()
        )
        self.all_items_array = np.array(all_item_ids)

        logger.info("Dataset: %d positive pairs", len(self.user_ids))

    def __len__(self) -> int:
        return len(self.user_ids)

    def _sample_negative(self, user_id: int) -> int:
        rated = self.user_rated.get(user_id, set())
        while True:
            neg_id = int(np.random.choice(self.all_items_array))
            if neg_id not in rated:
                return neg_id

    def __getitem__(self, idx: int) -> Tuple:
        user_id = int(self.user_ids[idx])
        pos_item_id = int(self.item_ids[idx])
        neg_item_id = self._sample_negative(user_id)

        pos_genre = self.item_genre_dict.get(pos_item_id, np.zeros(N_GENRES, dtype=np.float32))
        neg_genre = self.item_genre_dict.get(neg_item_id, np.zeros(N_GENRES, dtype=np.float32))

        return (
            torch.tensor(user_id, dtype=torch.long),
            torch.tensor(pos_item_id, dtype=torch.long),
            torch.tensor(pos_genre, dtype=torch.float32),
            torch.tensor(neg_item_id, dtype=torch.long),
            torch.tensor(neg_genre, dtype=torch.float32),
        )


class EmbeddingTrainer:
    """Manages the full two-tower training loop."""

    def __init__(
        self,
        data_dir: str = None,
        model_output_path: str = None,
        embed_dim: int = None,
        epochs: int = None,
        batch_size: int = None,
        learning_rate: float = None,
        device: Optional[str] = None,
    ):
        self.data_dir = data_dir or settings.DATA_DIR
        self.model_output_path = model_output_path or settings.EMBEDDING_MODEL_PATH
        self.embed_dim = embed_dim or settings.EMBEDDING_DIM
        self.epochs = epochs or settings.TRAIN_EPOCHS
        self.batch_size = batch_size or settings.BATCH_SIZE
        self.learning_rate = learning_rate or settings.LEARNING_RATE

        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        logger.info("Using device: %s", self.device)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load MovieLens data files."""
        fe = FeatureEngineer(self.data_dir)
        fe.load_data()
        return fe.ratings_df, fe.users_df, fe.movies_df

    def _build_item_genre_dict(self, movies_df: pd.DataFrame) -> Dict[int, np.ndarray]:
        """Build a dict mapping item_id → genre vector."""
        from src.features.feature_engineering import GENRE_TO_IDX
        result = {}
        for _, row in movies_df.iterrows():
            vec = np.zeros(N_GENRES, dtype=np.float32)
            for genre in str(row["genres"]).split("|"):
                idx = GENRE_TO_IDX.get(genre)
                if idx is not None:
                    vec[idx] = 1.0
            result[int(row["item_id"])] = vec
        return result

    def train(self) -> TwoTowerModel:
        """Run the full training pipeline and return the trained model."""
        logger.info("Loading data from %s ...", self.data_dir)
        ratings_df, users_df, movies_df = self.load_data()

        n_users = ratings_df["user_id"].max()
        n_items = ratings_df["item_id"].max()
        all_item_ids = sorted(movies_df["item_id"].unique().tolist())

        item_genre_dict = self._build_item_genre_dict(movies_df)
        logger.info("n_users=%d, n_items=%d", n_users, n_items)

        dataset = UserItemDataset(ratings_df, item_genre_dict, all_item_ids)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=(self.device.type == "cuda"),
            drop_last=True,
        )

        model = TwoTowerModel(
            n_users=n_users,
            n_items=n_items,
            embed_dim=self.embed_dim,
            hidden_dim=128,
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        Path(self.model_output_path).parent.mkdir(parents=True, exist_ok=True)
        best_loss = float("inf")
        training_history = []

        logger.info(
            "Starting training: %d epochs, batch_size=%d, lr=%.4f",
            self.epochs, self.batch_size, self.learning_rate,
        )

        for epoch in range(1, self.epochs + 1):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            t_start = time.time()

            for batch in dataloader:
                user_ids, pos_ids, pos_genres, neg_ids, neg_genres = [
                    b.to(self.device) for b in batch
                ]

                user_emb = model.user_tower(user_ids)
                pos_item_emb = model.item_tower(pos_ids, pos_genres)
                neg_item_emb = model.item_tower(neg_ids, neg_genres)

                loss = model.bpr_loss(user_emb, pos_item_emb, neg_item_emb)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)
            elapsed = time.time() - t_start
            training_history.append({"epoch": epoch, "loss": avg_loss})

            logger.info(
                "Epoch %d/%d — loss: %.4f — time: %.1fs — lr: %.6f",
                epoch, self.epochs, avg_loss, elapsed,
                scheduler.get_last_lr()[0],
            )

            if avg_loss < best_loss:
                best_loss = avg_loss
                model.save(self.model_output_path)
                logger.info("  -> Saved best model (loss=%.4f)", best_loss)

        # After training, precompute and cache item embeddings in model
        logger.info("Precomputing item embeddings for all %d items...", len(all_item_ids))
        genre_matrix = np.stack([
            item_genre_dict.get(iid, np.zeros(N_GENRES, dtype=np.float32))
            for iid in all_item_ids
        ])
        model.precompute_item_embeddings(all_item_ids, genre_matrix, self.device)
        model.save(self.model_output_path)

        logger.info("Training complete. Best loss: %.4f", best_loss)
        return model
