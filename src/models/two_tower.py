"""
Two-Tower neural embedding model for candidate generation.
Uses BPR (Bayesian Personalized Ranking) loss with in-batch negatives.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

N_GENRES = 18  # MovieLens 1M has 18 genres


class UserTower(nn.Module):
    """
    User embedding tower.
    Input: user_id (int) → learned embedding → MLP → L2 normalized output.
    """

    def __init__(self, n_users: int, embed_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(n_users + 1, embed_dim, padding_idx=0)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, user_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(user_ids)
        x = self.mlp(x)
        return F.normalize(x, p=2, dim=-1)


class ItemTower(nn.Module):
    """
    Item embedding tower.
    Input: item_id (int) + genre_vector (18-dim multi-hot) → MLP → L2 normalized output.
    The genre vector enriches the item representation with content features.
    """

    def __init__(self, n_items: int, embed_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(n_items + 1, embed_dim, padding_idx=0)
        # Input dimension = embed_dim + n_genres
        input_dim = embed_dim + N_GENRES
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, item_ids: torch.Tensor, genre_vectors: torch.Tensor) -> torch.Tensor:
        item_emb = self.embedding(item_ids)
        x = torch.cat([item_emb, genre_vectors], dim=-1)
        x = self.mlp(x)
        return F.normalize(x, p=2, dim=-1)


class TwoTowerModel(nn.Module):
    """
    Two-Tower model combining user and item towers.
    Trained with BPR loss using in-batch negatives for efficient training.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim

        self.user_tower = UserTower(n_users, embed_dim, hidden_dim, dropout)
        self.item_tower = ItemTower(n_items, embed_dim, hidden_dim, dropout)

        # Cache for inference
        self._item_embeddings: Optional[torch.Tensor] = None
        self._item_id_to_idx: Optional[Dict[int, int]] = None
        self._idx_to_item_id: Optional[Dict[int, int]] = None

    def forward(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        pos_genre_vectors: torch.Tensor,
        neg_item_ids: Optional[torch.Tensor] = None,
        neg_genre_vectors: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning (user_embeddings, item_embeddings) for positive items.
        """
        user_emb = self.user_tower(user_ids)
        item_emb = self.item_tower(pos_item_ids, pos_genre_vectors)
        return user_emb, item_emb

    def bpr_loss(
        self,
        user_emb: torch.Tensor,
        pos_item_emb: torch.Tensor,
        neg_item_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Bayesian Personalized Ranking loss.
        Maximizes the margin between positive and negative item scores.
        """
        pos_scores = (user_emb * pos_item_emb).sum(dim=-1)
        neg_scores = (user_emb * neg_item_emb).sum(dim=-1)
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        return loss

    def in_batch_bpr_loss(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        In-batch BPR loss: treats all other items in the batch as negatives.
        More efficient than explicit negative sampling for large batches.
        Score matrix: (batch_size, batch_size) — diagonal is positives.
        """
        # scores[i, j] = dot(user_i, item_j)
        scores = torch.matmul(user_emb, item_emb.T)  # (B, B)
        batch_size = user_emb.size(0)

        # Positive scores are on the diagonal
        pos_scores = scores.diag()  # (B,)

        # For each user, subtract positive score from all item scores
        # to get margins; negatives should have negative margins
        loss = 0.0
        count = 0
        for i in range(batch_size):
            neg_mask = torch.ones(batch_size, dtype=torch.bool, device=user_emb.device)
            neg_mask[i] = False
            neg_scores_i = scores[i][neg_mask]
            margins = pos_scores[i] - neg_scores_i
            loss = loss + (-F.logsigmoid(margins)).mean()
            count += 1
        return loss / count

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def get_user_embedding(self, user_id: int, device: torch.device = torch.device("cpu")) -> np.ndarray:
        """Get the normalized embedding for a single user."""
        self.eval()
        uid_tensor = torch.tensor([user_id], dtype=torch.long, device=device)
        emb = self.user_tower(uid_tensor)
        return emb.cpu().numpy()[0]

    @torch.no_grad()
    def get_item_embeddings(
        self,
        item_ids: List[int],
        genre_vectors: np.ndarray,
        device: torch.device = torch.device("cpu"),
        batch_size: int = 512,
    ) -> np.ndarray:
        """
        Get normalized embeddings for a list of items.
        genre_vectors: (n_items, 18) numpy array
        """
        self.eval()
        all_embeddings = []
        for start in range(0, len(item_ids), batch_size):
            end = start + batch_size
            batch_ids = torch.tensor(item_ids[start:end], dtype=torch.long, device=device)
            batch_genres = torch.tensor(
                genre_vectors[start:end], dtype=torch.float32, device=device
            )
            emb = self.item_tower(batch_ids, batch_genres)
            all_embeddings.append(emb.cpu().numpy())
        return np.vstack(all_embeddings)

    def precompute_item_embeddings(
        self,
        item_ids: List[int],
        genre_vectors: np.ndarray,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Precompute and cache all item embeddings for fast inference."""
        logger.info("Precomputing embeddings for %d items...", len(item_ids))
        embs = self.get_item_embeddings(item_ids, genre_vectors, device)
        self._item_embeddings = torch.tensor(embs, dtype=torch.float32)
        self._item_id_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}
        self._idx_to_item_id = {idx: iid for idx, iid in enumerate(item_ids)}
        logger.info("Precomputed %d item embeddings (dim=%d)", len(item_ids), embs.shape[1])

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        """Save model weights and metadata."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "n_users": self.n_users,
                "n_items": self.n_items,
                "embed_dim": self.embed_dim,
                "item_id_to_idx": self._item_id_to_idx,
                "idx_to_item_id": self._idx_to_item_id,
            },
            save_path,
        )
        logger.info("Saved two-tower model to %s", save_path)

    @classmethod
    def load(cls, path: str, device: torch.device = torch.device("cpu")) -> "TwoTowerModel":
        """Load a saved two-tower model."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            n_users=checkpoint["n_users"],
            n_items=checkpoint["n_items"],
            embed_dim=checkpoint["embed_dim"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model._item_id_to_idx = checkpoint.get("item_id_to_idx")
        model._idx_to_item_id = checkpoint.get("idx_to_item_id")
        model.to(device)
        model.eval()
        logger.info(
            "Loaded two-tower model from %s (users=%d, items=%d, dim=%d)",
            path, model.n_users, model.n_items, model.embed_dim,
        )
        return model
