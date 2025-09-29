"""Content-based recommendation using SBERT embeddings and Faiss."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import structlog
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from pathlib import Path
from ..service.config import config

logger = structlog.get_logger(__name__)

class ContentEmbeddingRecommender:
    """Content-based recommender using SBERT embeddings and Faiss."""
    
    def __init__(self, domain: str):
        """Initialize content embedding recommender."""
        self.domain = domain
        self.model = None
        self.faiss_index = None
        self.item_embeddings = None
        self.item_mapping = {}
        self.reverse_item_mapping = {}
        self.item_metadata = {}
        
    def fit(self, items_df: pd.DataFrame):
        """Fit content embedding model."""
        logger.info("Fitting content embedding model", domain=self.domain)
        
        # Create item mappings
        self._create_item_mappings(items_df)
        
        # Load SBERT model
        self._load_sbert_model()
        
        # Generate embeddings
        self._generate_embeddings(items_df)
        
        # Build Faiss index
        self._build_faiss_index()
        
        logger.info("Content embedding model fitted", 
                   domain=self.domain,
                   items=len(self.item_mapping),
                   embedding_dim=self.item_embeddings.shape[1])
    
    def _create_item_mappings(self, items_df: pd.DataFrame):
        """Create item ID mappings."""
        unique_items = sorted(items_df['item_id'].unique())
        self.item_mapping = {item_id: idx for idx, item_id in enumerate(unique_items)}
        self.reverse_item_mapping = {idx: item_id for item_id, idx in self.item_mapping.items()}
        
        # Store item metadata
        for _, row in items_df.iterrows():
            item_id = row['item_id']
            self.item_metadata[item_id] = {
                'title': row.get('title', ''),
                'description': row.get('description', ''),
                'genres': row.get('genres', []),
                'author': row.get('author', ''),
                'year': row.get('year', None)
            }
        
        logger.info("Item mappings created", 
                   domain=self.domain,
                   num_items=len(self.item_mapping))
    
    def _load_sbert_model(self):
        """Load SBERT model."""
        logger.info("Loading SBERT model", model=config.SBERT_MODEL)
        
        try:
            self.model = SentenceTransformer(config.SBERT_MODEL)
            logger.info("SBERT model loaded successfully")
        except Exception as e:
            logger.error("Failed to load SBERT model", error=str(e))
            raise
    
    def _generate_embeddings(self, items_df: pd.DataFrame):
        """Generate embeddings for all items."""
        logger.info("Generating embeddings", domain=self.domain)
        
        # Prepare text data
        texts = []
        for _, row in items_df.iterrows():
            # Combine title and description
            title = row.get('title', '')
            description = row.get('description', '')
            
            # Add genre information if available
            genres = row.get('genres', [])
            genre_text = ''
            if isinstance(genres, list):
                genre_text = ' '.join(genres)
            elif isinstance(genres, str):
                genre_text = genres
            
            # Combine all text
            text = f"{title} {description} {genre_text}".strip()
            
            # Truncate if too long
            if len(text) > config.MAX_TEXT_LENGTH:
                text = text[:config.MAX_TEXT_LENGTH]
            
            texts.append(text)
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts, 
            batch_size=32, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        self.item_embeddings = embeddings
        
        logger.info("Embeddings generated", 
                   domain=self.domain,
                   shape=embeddings.shape)
    
    def _build_faiss_index(self):
        """Build Faiss index for fast similarity search."""
        logger.info("Building Faiss index", domain=self.domain)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.item_embeddings)
        
        # Create index
        dimension = self.item_embeddings.shape[1]
        
        # Use IVF index for better performance with large datasets
        nlist = min(config.FAISS_NLIST, len(self.item_mapping) // 10)
        nlist = max(nlist, 1)  # At least 1 cluster
        
        quantizer = faiss.IndexFlatIP(dimension)
        self.faiss_index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        
        # Train the index
        self.faiss_index.train(self.item_embeddings)
        
        # Add vectors to index
        self.faiss_index.add(self.item_embeddings)
        
        # Set search parameters
        self.faiss_index.nprobe = config.FAISS_NPROBE
        
        logger.info("Faiss index built", 
                   domain=self.domain,
                   dimension=dimension,
                   nlist=nlist)
    
    def get_similar_items(self, item_id: str, n_similar: int = 20) -> List[Dict[str, Any]]:
        """Get similar items for a given item."""
        if self.faiss_index is None or item_id not in self.item_mapping:
            logger.warning("Model not fitted or item not found", item_id=item_id, domain=self.domain)
            return []
        
        # Get item index
        item_idx = self.item_mapping[item_id]
        
        # Get embedding for the item
        item_embedding = self.item_embeddings[item_idx:item_idx+1]
        
        # Search for similar items
        scores, indices = self.faiss_index.search(item_embedding, n_similar + 1)
        
        # Convert to list (skip first result which is the item itself)
        candidates = []
        for i in range(1, len(indices[0])):
            if indices[0][i] != -1:  # Valid index
                similar_item_id = self.reverse_item_mapping[indices[0][i]]
                candidates.append({
                    'item_id': similar_item_id,
                    'score': float(scores[0][i]),
                    'source': 'content_embed'
                })
        
        return candidates
    
    def get_user_candidates_from_history(self, user_history: List[str], 
                                       n_candidates: int = 100) -> List[Dict[str, Any]]:
        """Get candidates based on user's interaction history."""
        if self.faiss_index is None or not user_history:
            return []
        
        # Get embeddings for items in user history
        history_embeddings = []
        valid_items = []
        
        for item_id in user_history:
            if item_id in self.item_mapping:
                item_idx = self.item_mapping[item_id]
                history_embeddings.append(self.item_embeddings[item_idx])
                valid_items.append(item_id)
        
        if not history_embeddings:
            return []
        
        # Average the embeddings
        avg_embedding = np.mean(history_embeddings, axis=0)
        avg_embedding = avg_embedding.reshape(1, -1)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(avg_embedding)
        
        # Search for similar items
        scores, indices = self.faiss_index.search(avg_embedding, n_candidates)
        
        # Convert to list
        candidates = []
        for i in range(len(indices[0])):
            if indices[0][i] != -1:  # Valid index
                item_id = self.reverse_item_mapping[indices[0][i]]
                # Skip items already in history
                if item_id not in valid_items:
                    candidates.append({
                        'item_id': item_id,
                        'score': float(scores[0][i]),
                        'source': 'content_embed_history'
                    })
        
        return candidates
    
    def get_user_candidates(self, user_id: str, user_history: List[str], 
                          n_candidates: int = 100) -> List[Dict[str, Any]]:
        """Get candidates for a user based on their interaction history."""
        return self.get_user_candidates_from_history(user_history, n_candidates)
    
    def predict_score(self, user_history: List[str], item_id: str) -> float:
        """Predict interaction score based on user history and item."""
        if self.faiss_index is None or item_id not in self.item_mapping:
            return 0.0
        
        if not user_history:
            return 0.0
        
        # Get embeddings for items in user history
        history_embeddings = []
        for item_id_history in user_history:
            if item_id_history in self.item_mapping:
                item_idx = self.item_mapping[item_id_history]
                history_embeddings.append(self.item_embeddings[item_idx])
        
        if not history_embeddings:
            return 0.0
        
        # Get embedding for target item
        target_idx = self.item_mapping[item_id]
        target_embedding = self.item_embeddings[target_idx]
        
        # Calculate average similarity
        similarities = []
        for hist_embedding in history_embeddings:
            # Cosine similarity (embeddings are already normalized)
            similarity = np.dot(hist_embedding, target_embedding)
            similarities.append(similarity)
        
        return float(np.mean(similarities))
    
    def get_item_embedding(self, item_id: str) -> Optional[np.ndarray]:
        """Get embedding for a specific item."""
        if item_id not in self.item_mapping:
            return None
        
        item_idx = self.item_mapping[item_id]
        return self.item_embeddings[item_idx]
    
    def get_item_metadata(self, item_id: str) -> Dict[str, Any]:
        """Get metadata for a specific item."""
        return self.item_metadata.get(item_id, {})
    
    def save(self, domain: str):
        """Save content embedding model."""
        artifacts_dir = Path(config.ARTIFACTS_DIR) / domain
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Save SBERT model name
        with open(artifacts_dir / "sbert_model_name.txt", "w") as f:
            f.write(config.SBERT_MODEL)
        
        # Save Faiss index
        if self.faiss_index:
            faiss.write_index(self.faiss_index, str(artifacts_dir / "faiss.index"))
        
        # Save embeddings
        if self.item_embeddings is not None:
            np.save(artifacts_dir / "item_embeddings.npy", self.item_embeddings)
        
        # Save mappings and metadata
        mappings = {
            'item_mapping': self.item_mapping,
            'reverse_item_mapping': self.reverse_item_mapping,
            'item_metadata': self.item_metadata
        }
        
        with open(artifacts_dir / "content_mappings.pkl", "wb") as f:
            pickle.dump(mappings, f)
        
        logger.info("Content embedding model saved", domain=domain)
    
    def load(self, domain: str):
        """Load content embedding model."""
        artifacts_dir = Path(config.ARTIFACTS_DIR) / domain
        
        # Load SBERT model
        self._load_sbert_model()
        
        # Load Faiss index
        index_path = artifacts_dir / "faiss.index"
        if index_path.exists():
            self.faiss_index = faiss.read_index(str(index_path))
        
        # Load embeddings
        embeddings_path = artifacts_dir / "item_embeddings.npy"
        if embeddings_path.exists():
            self.item_embeddings = np.load(embeddings_path)
        
        # Load mappings and metadata
        mappings_path = artifacts_dir / "content_mappings.pkl"
        if mappings_path.exists():
            with open(mappings_path, "rb") as f:
                mappings = pickle.load(f)
                self.item_mapping = mappings['item_mapping']
                self.reverse_item_mapping = mappings['reverse_item_mapping']
                self.item_metadata = mappings['item_metadata']
        
        logger.info("Content embedding model loaded", domain=domain) 