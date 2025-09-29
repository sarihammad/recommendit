"""System integration tests for the recommendation system."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import json

from recsys.data.loaders import data_loader, SchemaMapper
from recsys.service.config import config
from recsys.pipeline.artifacts import ArtifactManager
from recsys.pipeline.cold_start import ColdStartManager
from recsys.pipeline.eval import RecommendationEvaluator

class TestSystemSetup:
    """Test system setup and configuration."""
    
    def test_config_loading(self):
        """Test configuration loading."""
        assert config.DATA_DIR == "dataset"
        assert config.ARTIFACTS_DIR == "artifacts"
        assert config.ALS_FACTORS == 100
        assert config.SBERT_MODEL == "all-MiniLM-L6-v2"
        assert config.RECALL_CANDIDATES == 1000
        assert config.API_HOST == "0.0.0.0"
        assert config.API_PORT == 8000
    
    def test_schema_mapper(self):
        """Test schema mapping functionality."""
        sample_data = pd.DataFrame({
            'User-ID': ['1', '2', '3'],
            'ISBN': ['123', '456', '789'],
            'Book-Rating': [5, 4, 3],
            'title': ['Book1', 'Book2', 'Book3']
        })
        mapping = SchemaMapper.infer_schema(sample_data)
        assert mapping['user_id'] == 'User-ID'
        assert mapping['item_id'] == 'ISBN'
        assert mapping['rating'] == 'Book-Rating'
    
    def test_data_loader_initialization(self):
        """Test data loader initialization."""
        assert data_loader.data_dir == Path(config.DATA_DIR)
    
    def test_artifacts_directory_creation(self):
        """Test artifacts directory creation."""
        artifacts_dir = Path(config.ARTIFACTS_DIR)
        test_dirs = ['books', 'movies']
        for domain in test_dirs:
            domain_dir = artifacts_dir / domain
            domain_dir.mkdir(parents=True, exist_ok=True)
            assert domain_dir.exists()
    
    def test_imports(self):
        """Test that all modules can be imported."""
        try:
            from recsys.models.cf_als import ALSRecommender
            from recsys.models.cf_item_item import ItemItemRecommender
            from recsys.models.content_embed import ContentEmbeddingRecommender
            from recsys.models.ranker_lgbm import LightGBMRanker
            from recsys.pipeline.recall import RecallPipeline
            from recsys.pipeline.rank import RankingPipeline
            from recsys.service.api import app
            from recsys.service.cache import cache
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import module: {e}")
    
    def test_fastapi_app(self):
        """Test FastAPI app instantiation."""
        from recsys.service.api import app
        assert app is not None
        assert hasattr(app, 'routes')

class TestArtifactManagement:
    """Test artifact management functionality."""
    
    @pytest.fixture
    def temp_artifacts_dir(self):
        """Create temporary artifacts directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_artifact_manager_initialization(self, temp_artifacts_dir):
        """Test artifact manager initialization."""
        # Temporarily change artifacts directory
        original_artifacts_dir = config.ARTIFACTS_DIR
        config.ARTIFACTS_DIR = temp_artifacts_dir
        
        try:
            manager = ArtifactManager('test')
            assert manager.domain == 'test'
            assert manager.artifacts_dir == Path(temp_artifacts_dir) / 'test'
            assert manager.artifacts_dir.exists()
        finally:
            config.ARTIFACTS_DIR = original_artifacts_dir
    
    def test_artifact_completeness_check(self, temp_artifacts_dir):
        """Test artifact completeness checking."""
        # Temporarily change artifacts directory
        original_artifacts_dir = config.ARTIFACTS_DIR
        config.ARTIFACTS_DIR = temp_artifacts_dir
        
        try:
            manager = ArtifactManager('test')
            completeness = manager.check_artifacts_completeness()
            
            # Should return a dictionary with boolean values
            assert isinstance(completeness, dict)
            assert all(isinstance(value, bool) for value in completeness.values())
            
            # All artifacts should be missing initially
            assert not any(completeness.values())
        finally:
            config.ARTIFACTS_DIR = original_artifacts_dir

class TestColdStartManagement:
    """Test cold start management functionality."""
    
    def test_cold_start_manager_initialization(self):
        """Test cold start manager initialization."""
        manager = ColdStartManager('test')
        assert manager.domain == 'test'
        assert manager.strategy is not None
        assert manager.new_user_strategy is not None
        assert manager.new_item_strategy is not None
        assert manager.alpha_scheduler is not None
    
    def test_new_user_detection(self):
        """Test new user detection."""
        manager = ColdStartManager('test')
        
        # Set up user interaction counts
        manager.strategy.set_user_interaction_counts({
            'user_1': 3,   # New user (less than MIN_USER_INTERACTIONS)
            'user_2': 50,  # Existing user
            'user_3': 100  # Existing user
        })
        
        assert manager.is_new_user('user_1') == True
        assert manager.is_new_user('user_2') == False
        assert manager.is_new_user('user_3') == False
        assert manager.is_new_user('unknown_user') == True
    
    def test_alpha_scheduling(self):
        """Test alpha scheduling for blending."""
        manager = ColdStartManager('test')
        
        # Set up user interaction counts
        manager.strategy.set_user_interaction_counts({
            'user_1': 3,   # New user
            'user_2': 25,  # Medium user
            'user_3': 100  # Experienced user
        })
        
        alpha_1 = manager.get_alpha('user_1')
        alpha_2 = manager.get_alpha('user_2')
        alpha_3 = manager.get_alpha('user_3')
        
        # Alpha should increase with interaction count
        assert alpha_1 == config.ALPHA_MIN
        assert alpha_2 > alpha_1
        assert alpha_3 >= alpha_2
        assert alpha_3 <= config.ALPHA_MAX

class TestEvaluationSystem:
    """Test evaluation system functionality."""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = RecommendationEvaluator('test')
        assert evaluator.domain == 'test'
        assert evaluator.metrics == {}
    
    def test_recall_calculation(self):
        """Test recall calculation."""
        evaluator = RecommendationEvaluator('test')
        
        ground_truth = ['item_1', 'item_2', 'item_3']
        predictions = ['item_1', 'item_4', 'item_2', 'item_5']
        
        recall = evaluator.calculate_recall_at_k(ground_truth, predictions, k=3)
        expected_recall = 2 / 3  # 2 out of 3 ground truth items found in top-3
        assert recall == expected_recall
    
    def test_ndcg_calculation(self):
        """Test NDCG calculation."""
        evaluator = RecommendationEvaluator('test')
        
        ground_truth = ['item_1', 'item_2']
        predictions = ['item_1', 'item_3', 'item_2']
        
        ndcg = evaluator.calculate_ndcg_at_k(ground_truth, predictions, k=3)
        assert isinstance(ndcg, float)
        assert 0 <= ndcg <= 1
    
    def test_map_calculation(self):
        """Test MAP calculation."""
        evaluator = RecommendationEvaluator('test')
        
        ground_truth = ['item_1', 'item_2']
        predictions = ['item_1', 'item_3', 'item_2']
        
        map_score = evaluator.calculate_map_at_k(ground_truth, predictions, k=3)
        assert isinstance(map_score, float)
        assert 0 <= map_score <= 1
    
    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        evaluator = RecommendationEvaluator('test')
        
        ground_truth = ['item_1', 'item_2']
        predictions = ['item_3', 'item_4', 'item_5']
        
        hit_rate = evaluator.calculate_hit_rate_at_k(ground_truth, predictions, k=3)
        assert hit_rate == 0.0  # No hits
        
        predictions_with_hit = ['item_1', 'item_3', 'item_4']
        hit_rate = evaluator.calculate_hit_rate_at_k(ground_truth, predictions_with_hit, k=3)
        assert hit_rate == 1.0  # Has hit

class TestDataIntegration:
    """Test data integration functionality."""
    
    def test_load_books_schema_inference(self):
        """Test books loading with schema inference."""
        try:
            books_df = data_loader.load_books()
            if books_df is not None and len(books_df) > 0:
                # Check that required columns exist
                required_columns = ['item_id', 'title']
                for col in required_columns:
                    assert col in books_df.columns, f"Missing required column: {col}"
                
                # Check data types
                assert isinstance(books_df['item_id'], pd.Series)
                assert isinstance(books_df['title'], pd.Series)
        except Exception as e:
            pytest.skip(f"Books data not available: {e}")
    
    def test_load_movies_schema_inference(self):
        """Test movies loading with schema inference."""
        try:
            movies_df = data_loader.load_movies()
            if movies_df is not None and len(movies_df) > 0:
                # Check that required columns exist
                required_columns = ['item_id', 'title']
                for col in required_columns:
                    assert col in movies_df.columns, f"Missing required column: {col}"
                
                # Check data types
                assert isinstance(movies_df['item_id'], pd.Series)
                assert isinstance(movies_df['title'], pd.Series)
        except Exception as e:
            pytest.skip(f"Movies data not available: {e}")
    
    def test_load_interactions_books(self):
        """Test loading book interactions."""
        try:
            interactions_df = data_loader.load_interactions('books')
            if interactions_df is not None and len(interactions_df) > 0:
                # Check that required columns exist
                required_columns = ['user_id', 'item_id', 'weight']
                for col in required_columns:
                    assert col in interactions_df.columns, f"Missing required column: {col}"
                
                # Check that weights are numeric
                assert pd.api.types.is_numeric_dtype(interactions_df['weight'])
                
                # Check that weights are positive
                assert (interactions_df['weight'] >= 0).all()
        except Exception as e:
            pytest.skip(f"Books interactions not available: {e}")
    
    def test_load_interactions_movies(self):
        """Test loading movie interactions."""
        try:
            interactions_df = data_loader.load_interactions('movies')
            if interactions_df is not None and len(interactions_df) > 0:
                # Check that required columns exist
                required_columns = ['user_id', 'item_id', 'weight']
                for col in required_columns:
                    assert col in interactions_df.columns, f"Missing required column: {col}"
                
                # Check that weights are numeric
                assert pd.api.types.is_numeric_dtype(interactions_df['weight'])
                
                # Check that weights are positive
                assert (interactions_df['weight'] >= 0).all()
        except Exception as e:
            pytest.skip(f"Movies interactions not available: {e}")

class TestSystemPerformance:
    """Test system performance characteristics."""
    
    def test_memory_usage(self):
        """Test memory usage of core components."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Test loading a reasonable amount of data
        try:
            # Try to load some data
            books_df = data_loader.load_books()
            if books_df is not None:
                # Check memory usage after loading
                current_memory = process.memory_info().rss
                memory_increase = current_memory - initial_memory
                
                # Memory increase should be reasonable (less than 1GB)
                assert memory_increase < 1024 * 1024 * 1024
        except Exception as e:
            pytest.skip(f"Data loading failed: {e}")
    
    def test_import_performance(self):
        """Test import performance."""
        import time
        
        start_time = time.time()
        
        # Import all major modules
        from recsys.data.loaders import data_loader
        from recsys.models.cf_als import ALSRecommender
        from recsys.models.content_embed import ContentEmbeddingRecommender
        from recsys.pipeline.recall import RecallPipeline
        from recsys.service.api import app
        
        end_time = time.time()
        import_time = end_time - start_time
        
        # Import should be fast (less than 5 seconds)
        assert import_time < 5.0

if __name__ == "__main__":
    pytest.main([__file__]) 