"""Tests for content-based recommendation models."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from recsys.models.content_embed import ContentEmbeddingRecommender
from recsys.service.config import config

class TestContentEmbeddingRecommender:
    """Test content embedding recommendation model."""
    
    @pytest.fixture
    def sample_items(self):
        """Create sample items data for testing."""
        return pd.DataFrame({
            'item_id': ['1', '2', '3', '4', '5'],
            'title': [
                'The Great Adventure',
                'Mystery of the Night',
                'Science Fiction Novel',
                'Romance in Paris',
                'Thriller at Midnight'
            ],
            'description': [
                'An exciting adventure story about exploration and discovery.',
                'A mysterious tale that keeps you guessing until the end.',
                'A futuristic story about space travel and technology.',
                'A beautiful love story set in the romantic city of Paris.',
                'A heart-pounding thriller that will keep you on the edge of your seat.'
            ],
            'genres': [
                'Adventure|Fiction',
                'Mystery|Thriller',
                'Science Fiction|Adventure',
                'Romance|Drama',
                'Thriller|Mystery'
            ],
            'author': [
                'John Smith',
                'Jane Doe',
                'Bob Johnson',
                'Alice Brown',
                'Charlie Wilson'
            ],
            'year': [2020, 2019, 2021, 2018, 2022]
        })
    
    @pytest.fixture
    def temp_artifacts_dir(self):
        """Create temporary artifacts directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_content_embedding_initialization(self):
        """Test content embedding model initialization."""
        model = ContentEmbeddingRecommender('test')
        
        assert model.domain == 'test'
        assert model.model is None
        assert model.faiss_index is None
        assert model.item_embeddings is None
        assert isinstance(model.item_mapping, dict)
        assert isinstance(model.item_metadata, dict)
    
    def test_content_embedding_fit(self, sample_items):
        """Test content embedding model fitting."""
        model = ContentEmbeddingRecommender('test')
        
        # Fit the model
        model.fit(sample_items)
        
        # Check that model was trained
        assert model.model is not None
        assert model.item_embeddings is not None
        assert model.faiss_index is not None
        
        # Check mappings
        assert len(model.item_mapping) == len(sample_items)
        assert len(model.item_metadata) == len(sample_items)
        
        # Check embeddings shape
        assert model.item_embeddings.shape[0] == len(sample_items)
        assert model.item_embeddings.shape[1] > 0  # Should have embedding dimension
        
        # Check that all items are mapped
        for item_id in sample_items['item_id']:
            assert item_id in model.item_mapping
            assert item_id in model.item_metadata
    
    def test_content_embedding_get_similar_items(self, sample_items):
        """Test getting similar items from content embedding model."""
        model = ContentEmbeddingRecommender('test')
        model.fit(sample_items)
        
        # Get similar items
        item_id = '1'
        similar_items = model.get_similar_items(item_id, n_similar=3)
        
        assert isinstance(similar_items, list)
        assert len(similar_items) <= 3
        
        if similar_items:
            # Check similar item structure
            similar_item = similar_items[0]
            assert 'item_id' in similar_item
            assert 'score' in similar_item
            assert 'source' in similar_item
            assert similar_item['source'] == 'content'
            
            # Check that scores are numeric and between 0 and 1
            assert isinstance(similar_item['score'], (int, float))
            assert 0 <= similar_item['score'] <= 1
            
            # Check that the item itself is not in similar items (or has score 1.0)
            if similar_item['item_id'] == item_id:
                assert similar_item['score'] == 1.0
    
    def test_content_embedding_get_user_candidates(self, sample_items):
        """Test getting user candidates from content embedding model."""
        model = ContentEmbeddingRecommender('test')
        model.fit(sample_items)
        
        # Get candidates for a user with history
        user_history = ['1', '2']  # User has interacted with items 1 and 2
        candidates = model.get_user_candidates_from_history(user_history, n_candidates=3)
        
        assert isinstance(candidates, list)
        assert len(candidates) <= 3
        
        if candidates:
            # Check candidate structure
            candidate = candidates[0]
            assert 'item_id' in candidate
            assert 'score' in candidate
            assert 'source' in candidate
            assert candidate['source'] == 'content'
            
            # Check that scores are numeric
            assert isinstance(candidate['score'], (int, float))
    
    def test_content_embedding_get_user_candidates_no_history(self, sample_items):
        """Test getting user candidates when user has no history."""
        model = ContentEmbeddingRecommender('test')
        model.fit(sample_items)
        
        # Get candidates for a user with no history
        candidates = model.get_user_candidates('new_user', n_candidates=3)
        
        assert isinstance(candidates, list)
        # Should return some candidates even without history (popularity-based)
        assert len(candidates) <= 3
    
    def test_content_embedding_predict_score(self, sample_items):
        """Test content embedding score prediction."""
        model = ContentEmbeddingRecommender('test')
        model.fit(sample_items)
        
        # Predict score for a user-item pair
        user_history = ['1', '2']  # User has interacted with items 1 and 2
        item_id = '3'
        score = model.predict_score(user_history, item_id)
        
        assert isinstance(score, (int, float))
        assert score >= 0  # Scores should be non-negative
    
    def test_content_embedding_save_and_load(self, sample_items, temp_artifacts_dir):
        """Test content embedding model saving and loading."""
        model = ContentEmbeddingRecommender('test')
        model.fit(sample_items)
        
        # Temporarily change artifacts directory
        original_artifacts_dir = config.ARTIFACTS_DIR
        config.ARTIFACTS_DIR = temp_artifacts_dir
        
        try:
            # Save model
            model.save('test')
            
            # Create new model and load
            loaded_model = ContentEmbeddingRecommender('test')
            loaded_model.load('test')
            
            # Check that loaded model has same properties
            assert loaded_model.domain == model.domain
            assert loaded_model.model is not None
            assert loaded_model.item_embeddings is not None
            assert loaded_model.faiss_index is not None
            
            # Check mappings
            assert len(loaded_model.item_mapping) == len(model.item_mapping)
            assert len(loaded_model.item_metadata) == len(model.item_metadata)
            
            # Check that embeddings are the same
            np.testing.assert_array_almost_equal(
                loaded_model.item_embeddings,
                model.item_embeddings
            )
            
            # Check that predictions are similar
            user_history = ['1', '2']
            item_id = '3'
            original_score = model.predict_score(user_history, item_id)
            loaded_score = loaded_model.predict_score(user_history, item_id)
            
            # Scores should be close (allowing for small numerical differences)
            assert abs(original_score - loaded_score) < 1e-6
            
        finally:
            # Restore original artifacts directory
            config.ARTIFACTS_DIR = original_artifacts_dir

class TestContentEmbeddingEdgeCases:
    """Test content embedding model with edge cases."""
    
    def test_content_embedding_empty_data(self):
        """Test content embedding with empty data."""
        model = ContentEmbeddingRecommender('test')
        
        empty_data = pd.DataFrame(columns=['item_id', 'title', 'description'])
        
        with pytest.raises(Exception):
            model.fit(empty_data)
    
    def test_content_embedding_missing_columns(self):
        """Test content embedding with missing columns."""
        model = ContentEmbeddingRecommender('test')
        
        # Data with only item_id and title
        minimal_data = pd.DataFrame({
            'item_id': ['1', '2'],
            'title': ['Book1', 'Book2']
        })
        
        # Should handle missing description gracefully
        model.fit(minimal_data)
        assert model.model is not None
        assert model.item_embeddings is not None
    
    def test_content_embedding_single_item(self):
        """Test content embedding with single item."""
        model = ContentEmbeddingRecommender('test')
        
        single_item_data = pd.DataFrame({
            'item_id': ['1'],
            'title': ['Single Book'],
            'description': ['A single book description.']
        })
        
        # Should handle single item gracefully
        model.fit(single_item_data)
        assert model.model is not None
        assert model.item_embeddings is not None
        assert len(model.item_mapping) == 1
    
    def test_content_embedding_duplicate_items(self):
        """Test content embedding with duplicate item IDs."""
        model = ContentEmbeddingRecommender('test')
        
        duplicate_data = pd.DataFrame({
            'item_id': ['1', '1', '2'],  # Duplicate item_id
            'title': ['Book1', 'Book1 Updated', 'Book2'],
            'description': ['Desc1', 'Desc1 Updated', 'Desc2']
        })
        
        # Should handle duplicates gracefully (keep last occurrence)
        model.fit(duplicate_data)
        assert model.model is not None
        assert len(model.item_mapping) == 2  # Should have 2 unique items
    
    def test_content_embedding_long_text(self):
        """Test content embedding with very long text."""
        model = ContentEmbeddingRecommender('test')
        
        long_text_data = pd.DataFrame({
            'item_id': ['1'],
            'title': ['A' * 1000],  # Very long title
            'description': ['B' * 2000]  # Very long description
        })
        
        # Should handle long text gracefully
        model.fit(long_text_data)
        assert model.model is not None
        assert model.item_embeddings is not None

class TestContentEmbeddingIntegration:
    """Integration tests for content embedding model."""
    
    def test_content_embedding_with_genres(self):
        """Test content embedding with genre information."""
        model = ContentEmbeddingRecommender('test')
        
        genre_data = pd.DataFrame({
            'item_id': ['1', '2', '3'],
            'title': ['Action Movie', 'Romance Novel', 'Sci-Fi Book'],
            'description': ['Action packed movie', 'Romantic story', 'Science fiction'],
            'genres': ['Action|Adventure', 'Romance|Drama', 'Science Fiction|Adventure']
        })
        
        model.fit(genre_data)
        
        # Test that similar items respect genre boundaries
        similar_items = model.get_similar_items('1', n_similar=2)
        
        # Should find similar items (though exact similarity depends on embeddings)
        assert isinstance(similar_items, list)
    
    def test_content_embedding_with_authors(self):
        """Test content embedding with author information."""
        model = ContentEmbeddingRecommender('test')
        
        author_data = pd.DataFrame({
            'item_id': ['1', '2', '3'],
            'title': ['Book1', 'Book2', 'Book3'],
            'description': ['Description1', 'Description2', 'Description3'],
            'author': ['Author A', 'Author A', 'Author B']  # Same author for first two
        })
        
        model.fit(author_data)
        
        # Test that items by same author might be similar
        similar_items = model.get_similar_items('1', n_similar=2)
        
        assert isinstance(similar_items, list)
    
    def test_content_embedding_performance(self):
        """Test content embedding performance with larger dataset."""
        model = ContentEmbeddingRecommender('test')
        
        # Create larger dataset
        n_items = 100
        large_data = pd.DataFrame({
            'item_id': [f'item_{i}' for i in range(n_items)],
            'title': [f'Title {i}' for i in range(n_items)],
            'description': [f'Description for item {i}' for i in range(n_items)]
        })
        
        # Should handle larger dataset efficiently
        model.fit(large_data)
        
        assert model.model is not None
        assert model.item_embeddings.shape[0] == n_items
        
        # Test search performance
        similar_items = model.get_similar_items('item_0', n_similar=10)
        assert len(similar_items) <= 10

if __name__ == "__main__":
    pytest.main([__file__]) 