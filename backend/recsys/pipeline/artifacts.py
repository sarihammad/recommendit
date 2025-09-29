"""Artifacts management for saving and loading models, indices, and encoders."""

import pickle
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import structlog
from pathlib import Path
import faiss
import lightgbm as lgb
from ..service.config import config

logger = structlog.get_logger(__name__)

class ArtifactManager:
    """Manager for saving and loading model artifacts."""
    
    def __init__(self, domain: str):
        """Initialize artifact manager."""
        self.domain = domain
        self.artifacts_dir = Path(config.ARTIFACTS_DIR) / domain
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
    def save_als_model(self, model, mappings: Dict[str, Any], factors: Tuple[np.ndarray, np.ndarray]):
        """Save ALS model artifacts."""
        logger.info("Saving ALS model artifacts", domain=self.domain)
        
        # Save model
        model_path = self.artifacts_dir / "als_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        # Save mappings
        mappings_path = self.artifacts_dir / "als_mappings.pkl"
        with open(mappings_path, "wb") as f:
            pickle.dump(mappings, f)
        
        # Save factors
        user_factors, item_factors = factors
        factors_path = self.artifacts_dir / "als_factors.npz"
        np.savez_compressed(factors_path, user_factors=user_factors, item_factors=item_factors)
        
        logger.info("ALS model artifacts saved", domain=self.domain)
    
    def load_als_model(self):
        """Load ALS model artifacts."""
        logger.info("Loading ALS model artifacts", domain=self.domain)
        
        artifacts = {}
        
        # Load model
        model_path = self.artifacts_dir / "als_model.pkl"
        if model_path.exists():
            with open(model_path, "rb") as f:
                artifacts['model'] = pickle.load(f)
        
        # Load mappings
        mappings_path = self.artifacts_dir / "als_mappings.pkl"
        if mappings_path.exists():
            with open(mappings_path, "rb") as f:
                artifacts['mappings'] = pickle.load(f)
        
        # Load factors
        factors_path = self.artifacts_dir / "als_factors.npz"
        if factors_path.exists():
            factors_data = np.load(factors_path)
            artifacts['factors'] = (factors_data['user_factors'], factors_data['item_factors'])
        
        logger.info("ALS model artifacts loaded", domain=self.domain)
        return artifacts
    
    def save_item_item_model(self, similarity_matrix: np.ndarray, mappings: Dict[str, Any], user_item_matrix):
        """Save item-item model artifacts."""
        logger.info("Saving item-item model artifacts", domain=self.domain)
        
        # Save similarity matrix
        similarity_path = self.artifacts_dir / "item_similarity_matrix.npy"
        np.save(similarity_path, similarity_matrix)
        
        # Save mappings
        mappings_path = self.artifacts_dir / "item_item_mappings.pkl"
        with open(mappings_path, "wb") as f:
            pickle.dump(mappings, f)
        
        # Save user-item matrix
        matrix_path = self.artifacts_dir / "user_item_matrix.pkl"
        with open(matrix_path, "wb") as f:
            pickle.dump(user_item_matrix, f)
        
        logger.info("Item-item model artifacts saved", domain=self.domain)
    
    def load_item_item_model(self):
        """Load item-item model artifacts."""
        logger.info("Loading item-item model artifacts", domain=self.domain)
        
        artifacts = {}
        
        # Load similarity matrix
        similarity_path = self.artifacts_dir / "item_similarity_matrix.npy"
        if similarity_path.exists():
            artifacts['similarity_matrix'] = np.load(similarity_path)
        
        # Load mappings
        mappings_path = self.artifacts_dir / "item_item_mappings.pkl"
        if mappings_path.exists():
            with open(mappings_path, "rb") as f:
                artifacts['mappings'] = pickle.load(f)
        
        # Load user-item matrix
        matrix_path = self.artifacts_dir / "user_item_matrix.pkl"
        if matrix_path.exists():
            with open(matrix_path, "rb") as f:
                artifacts['user_item_matrix'] = pickle.load(f)
        
        logger.info("Item-item model artifacts loaded", domain=self.domain)
        return artifacts
    
    def save_content_model(self, sbert_model_name: str, faiss_index, embeddings: np.ndarray, 
                          mappings: Dict[str, Any], metadata: Dict[str, Any]):
        """Save content model artifacts."""
        logger.info("Saving content model artifacts", domain=self.domain)
        
        # Save SBERT model name
        model_name_path = self.artifacts_dir / "sbert_model_name.txt"
        with open(model_name_path, "w") as f:
            f.write(sbert_model_name)
        
        # Save Faiss index
        if faiss_index:
            index_path = self.artifacts_dir / "faiss.index"
            faiss.write_index(faiss_index, str(index_path))
        
        # Save embeddings
        if embeddings is not None:
            embeddings_path = self.artifacts_dir / "item_embeddings.npy"
            np.save(embeddings_path, embeddings)
        
        # Save mappings and metadata
        content_data = {
            'mappings': mappings,
            'metadata': metadata
        }
        content_path = self.artifacts_dir / "content_mappings.pkl"
        with open(content_path, "wb") as f:
            pickle.dump(content_data, f)
        
        logger.info("Content model artifacts saved", domain=self.domain)
    
    def load_content_model(self):
        """Load content model artifacts."""
        logger.info("Loading content model artifacts", domain=self.domain)
        
        artifacts = {}
        
        # Load SBERT model name
        model_name_path = self.artifacts_dir / "sbert_model_name.txt"
        if model_name_path.exists():
            with open(model_name_path, "r") as f:
                artifacts['sbert_model_name'] = f.read().strip()
        
        # Load Faiss index
        index_path = self.artifacts_dir / "faiss.index"
        if index_path.exists():
            artifacts['faiss_index'] = faiss.read_index(str(index_path))
        
        # Load embeddings
        embeddings_path = self.artifacts_dir / "item_embeddings.npy"
        if embeddings_path.exists():
            artifacts['embeddings'] = np.load(embeddings_path)
        
        # Load mappings and metadata
        content_path = self.artifacts_dir / "content_mappings.pkl"
        if content_path.exists():
            with open(content_path, "rb") as f:
                content_data = pickle.load(f)
                artifacts['mappings'] = content_data['mappings']
                artifacts['metadata'] = content_data['metadata']
        
        logger.info("Content model artifacts loaded", domain=self.domain)
        return artifacts
    
    def save_ranking_model(self, lgb_model, feature_names: List[str], 
                          scalers: Dict[str, Any], encoders: Dict[str, Any], 
                          feature_stats: Dict[str, Any]):
        """Save ranking model artifacts."""
        logger.info("Saving ranking model artifacts", domain=self.domain)
        
        # Save LightGBM model
        if lgb_model:
            model_path = self.artifacts_dir / "lgbm_model.txt"
            lgb_model.save_model(str(model_path))
        
        # Save feature information
        feature_info = {
            'feature_names': feature_names,
            'scalers': scalers,
            'encoders': encoders,
            'feature_stats': feature_stats
        }
        feature_path = self.artifacts_dir / "lgbm_features.pkl"
        with open(feature_path, "wb") as f:
            pickle.dump(feature_info, f)
        
        logger.info("Ranking model artifacts saved", domain=self.domain)
    
    def load_ranking_model(self):
        """Load ranking model artifacts."""
        logger.info("Loading ranking model artifacts", domain=self.domain)
        
        artifacts = {}
        
        # Load LightGBM model
        model_path = self.artifacts_dir / "lgbm_model.txt"
        if model_path.exists():
            artifacts['model'] = lgb.Booster(model_file=str(model_path))
        
        # Load feature information
        feature_path = self.artifacts_dir / "lgbm_features.pkl"
        if feature_path.exists():
            with open(feature_path, "rb") as f:
                feature_info = pickle.load(f)
                artifacts['feature_names'] = feature_info['feature_names']
                artifacts['scalers'] = feature_info['scalers']
                artifacts['encoders'] = feature_info['encoders']
                artifacts['feature_stats'] = feature_info['feature_stats']
        
        logger.info("Ranking model artifacts loaded", domain=self.domain)
        return artifacts
    
    def save_feature_extractors(self, content_extractor, user_extractor):
        """Save feature extractor artifacts."""
        logger.info("Saving feature extractor artifacts", domain=self.domain)
        
        # Save content feature extractor
        if content_extractor:
            content_path = self.artifacts_dir / "content_feature_extractor.pkl"
            with open(content_path, "wb") as f:
                pickle.dump(content_extractor, f)
        
        # Save user feature extractor
        if user_extractor:
            user_path = self.artifacts_dir / "user_feature_extractor.pkl"
            with open(user_path, "wb") as f:
                pickle.dump(user_extractor, f)
        
        logger.info("Feature extractor artifacts saved", domain=self.domain)
    
    def load_feature_extractors(self):
        """Load feature extractor artifacts."""
        logger.info("Loading feature extractor artifacts", domain=self.domain)
        
        artifacts = {}
        
        # Load content feature extractor
        content_path = self.artifacts_dir / "content_feature_extractor.pkl"
        if content_path.exists():
            with open(content_path, "rb") as f:
                artifacts['content_extractor'] = pickle.load(f)
        
        # Load user feature extractor
        user_path = self.artifacts_dir / "user_feature_extractor.pkl"
        if user_path.exists():
            with open(user_path, "rb") as f:
                artifacts['user_extractor'] = pickle.load(f)
        
        logger.info("Feature extractor artifacts loaded", domain=self.domain)
        return artifacts
    
    def save_popularity_items(self, popularity_items: List[Dict[str, Any]]):
        """Save popularity items."""
        logger.info("Saving popularity items", domain=self.domain)
        
        popularity_path = self.artifacts_dir / "popularity_items.pkl"
        with open(popularity_path, "wb") as f:
            pickle.dump(popularity_items, f)
        
        logger.info("Popularity items saved", domain=self.domain)
    
    def load_popularity_items(self) -> List[Dict[str, Any]]:
        """Load popularity items."""
        logger.info("Loading popularity items", domain=self.domain)
        
        popularity_path = self.artifacts_dir / "popularity_items.pkl"
        if popularity_path.exists():
            with open(popularity_path, "rb") as f:
                popularity_items = pickle.load(f)
            logger.info("Popularity items loaded", domain=self.domain)
            return popularity_items
        else:
            logger.warning("Popularity items not found", domain=self.domain)
            return []
    
    def save_training_report(self, report: Dict[str, Any]):
        """Save training report."""
        logger.info("Saving training report", domain=self.domain)
        
        report_path = self.artifacts_dir / "training_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info("Training report saved", domain=self.domain)
    
    def load_training_report(self) -> Optional[Dict[str, Any]]:
        """Load training report."""
        logger.info("Loading training report", domain=self.domain)
        
        report_path = self.artifacts_dir / "training_report.json"
        if report_path.exists():
            with open(report_path, "r") as f:
                report = json.load(f)
            logger.info("Training report loaded", domain=self.domain)
            return report
        else:
            logger.warning("Training report not found", domain=self.domain)
            return None
    
    def save_evaluation_results(self, results: Dict[str, Any], evaluation_name: str = "evaluation"):
        """Save evaluation results."""
        logger.info("Saving evaluation results", domain=self.domain)
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        results_path = self.artifacts_dir / f"{evaluation_name}_{timestamp}.json"
        
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info("Evaluation results saved", domain=self.domain, path=str(results_path))
        return str(results_path)
    
    def list_artifacts(self) -> Dict[str, List[str]]:
        """List all available artifacts."""
        artifacts = {
            'models': [],
            'data': [],
            'reports': []
        }
        
        for file_path in self.artifacts_dir.glob("*"):
            if file_path.is_file():
                file_name = file_path.name
                if any(keyword in file_name for keyword in ['model', 'als', 'lgbm', 'faiss']):
                    artifacts['models'].append(file_name)
                elif any(keyword in file_name for keyword in ['data', 'features', 'mappings']):
                    artifacts['data'].append(file_name)
                elif any(keyword in file_name for keyword in ['report', 'evaluation']):
                    artifacts['reports'].append(file_name)
        
        return artifacts
    
    def check_artifacts_completeness(self) -> Dict[str, bool]:
        """Check if all required artifacts are present."""
        required_artifacts = {
            'als_model': self.artifacts_dir / "als_model.pkl",
            'als_mappings': self.artifacts_dir / "als_mappings.pkl",
            'als_factors': self.artifacts_dir / "als_factors.npz",
            'item_similarity': self.artifacts_dir / "item_similarity_matrix.npy",
            'faiss_index': self.artifacts_dir / "faiss.index",
            'item_embeddings': self.artifacts_dir / "item_embeddings.npy",
            'lgbm_model': self.artifacts_dir / "lgbm_model.txt",
            'lgbm_features': self.artifacts_dir / "lgbm_features.pkl",
            'popularity_items': self.artifacts_dir / "popularity_items.pkl",
            'training_report': self.artifacts_dir / "training_report.json"
        }
        
        completeness = {}
        for artifact_name, artifact_path in required_artifacts.items():
            completeness[artifact_name] = artifact_path.exists()
        
        return completeness
    
    def cleanup_old_artifacts(self, keep_latest: int = 3):
        """Clean up old artifacts, keeping only the latest versions."""
        logger.info("Cleaning up old artifacts", domain=self.domain)
        
        # Find evaluation files
        evaluation_files = list(self.artifacts_dir.glob("evaluation_*.json"))
        evaluation_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Remove old evaluation files
        for old_file in evaluation_files[keep_latest:]:
            old_file.unlink()
            logger.info("Removed old artifact", file=str(old_file))
        
        logger.info("Artifact cleanup completed", domain=self.domain) 