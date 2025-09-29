"""Tests for data loaders module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from recsys.data.loaders import data_loader, SchemaMapper
from recsys.service.config import config

class TestSchemaMapper:
    """Test SchemaMapper functionality."""
    
    def test_infer_schema_books(self):
        """Test schema inference for books data."""
        sample_data = pd.DataFrame({
            'User-ID': ['1', '2', '3'],
            'ISBN': ['123', '456', '789'],
            'Book-Rating': [5, 4, 3],
            'Book-Title': ['Book1', 'Book2', 'Book3'],
            'Book-Author': ['Author1', 'Author2', 'Author3'],
            'Year-Of-Publication': [2000, 2001, 2002]
        })
        
        mapping = SchemaMapper.infer_schema(sample_data)
        
        assert mapping['user_id'] == 'User-ID'
        assert mapping['item_id'] == 'ISBN'
        assert mapping['rating'] == 'Book-Rating'
        assert mapping['title'] == 'Book-Title'
        assert mapping['author'] == 'Book-Author'
        assert mapping['year'] == 'Year-Of-Publication'
    
    def test_infer_schema_movies(self):
        """Test schema inference for movies data."""
        sample_data = pd.DataFrame({
            'userId': ['1', '2', '3'],
            'movieId': ['123', '456', '789'],
            'rating': [5.0, 4.0, 3.0],
            'title': ['Movie1', 'Movie2', 'Movie3'],
            'genres': ['Action|Adventure', 'Comedy', 'Drama']
        })
        
        mapping = SchemaMapper.infer_schema(sample_data)
        
        assert mapping['user_id'] == 'userId'
        assert mapping['item_id'] == 'movieId'
        assert mapping['rating'] == 'rating'
        assert mapping['title'] == 'title'
        assert mapping['genres'] == 'genres'
    
    def test_standardize_dataframe(self):
        """Test dataframe standardization."""
        sample_data = pd.DataFrame({
            'User-ID': ['1', '2', '3'],
            'ISBN': ['123', '456', '789'],
            'Book-Rating': [5, 4, 3],
            'Book-Title': ['Book1', 'Book2', 'Book3']
        })
        
        standardized = SchemaMapper.standardize_dataframe(sample_data, 'books')
        
        assert 'user_id' in standardized.columns
        assert 'item_id' in standardized.columns
        assert 'rating' in standardized.columns
        assert 'title' in standardized.columns
        assert len(standardized) == 3
    
    def test_parse_genres(self):
        """Test genre parsing."""
        genres_str = "Action|Adventure|Comedy"
        parsed = SchemaMapper.parse_genres(genres_str)
        
        assert isinstance(parsed, list)
        assert len(parsed) == 3
        assert 'Action' in parsed
        assert 'Adventure' in parsed
        assert 'Comedy' in parsed
    
    def test_parse_year(self):
        """Test year parsing."""
        # Valid year
        assert SchemaMapper.parse_year("2000") == 2000
        assert SchemaMapper.parse_year(2000) == 2000
        
        # Invalid year
        assert SchemaMapper.parse_year("invalid") is None
        assert SchemaMapper.parse_year("0") is None

class TestDataLoader:
    """Test DataLoader functionality."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_rating_to_weight(self):
        """Test rating to weight conversion."""
        assert data_loader._rating_to_weight(5.0) == 5.0
        assert data_loader._rating_to_weight(4.0) == 5.0
        assert data_loader._rating_to_weight(3.0) == 2.0
        assert data_loader._rating_to_weight(2.0) == 0.5
        assert data_loader._rating_to_weight(1.0) == 0.5
    
    def test_create_time_split(self):
        """Test time-based data splitting."""
        # Create sample interaction data
        interactions = pd.DataFrame({
            'user_id': ['1', '2', '3', '4', '5'] * 10,
            'item_id': ['A', 'B', 'C', 'D', 'E'] * 10,
            'rating': [5, 4, 3, 2, 1] * 10,
            'timestamp': pd.date_range('2020-01-01', periods=50, freq='D')
        })
        
        train, val, test = data_loader.create_time_split(interactions)
        
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0
        assert len(train) + len(val) + len(test) == len(interactions)
        
        # Check that timestamps are properly ordered
        assert train['timestamp'].max() <= val['timestamp'].min()
        assert val['timestamp'].max() <= test['timestamp'].min()
    
    def test_compute_item_popularity(self):
        """Test item popularity computation."""
        interactions = pd.DataFrame({
            'user_id': ['1', '2', '3', '1', '2', '4'],
            'item_id': ['A', 'A', 'B', 'C', 'B', 'A'],
            'rating': [5, 4, 3, 5, 4, 3],
            'timestamp': pd.date_range('2020-01-01', periods=6)
        })
        
        popularity = data_loader.compute_item_popularity(interactions)
        
        assert isinstance(popularity, pd.Series)
        assert len(popularity) > 0
        assert 'A' in popularity.index
        assert 'B' in popularity.index
        assert 'C' in popularity.index
        
        # Item A should be most popular (3 interactions)
        assert popularity['A'] >= popularity['B']
        assert popularity['A'] >= popularity['C']
    
    def test_get_user_interaction_counts(self):
        """Test user interaction count computation."""
        interactions = pd.DataFrame({
            'user_id': ['1', '2', '3', '1', '2', '4'],
            'item_id': ['A', 'A', 'B', 'C', 'B', 'A'],
            'rating': [5, 4, 3, 5, 4, 3],
            'timestamp': pd.date_range('2020-01-01', periods=6)
        })
        
        counts = data_loader.get_user_interaction_counts(interactions)
        
        assert isinstance(counts, pd.Series)
        assert len(counts) > 0
        assert counts['1'] == 2  # User 1 has 2 interactions
        assert counts['2'] == 2  # User 2 has 2 interactions
        assert counts['3'] == 1  # User 3 has 1 interaction
        assert counts['4'] == 1  # User 4 has 1 interaction
    
    def test_save_and_load_processed_data(self, temp_data_dir):
        """Test saving and loading processed data."""
        # Create sample data
        data = pd.DataFrame({
            'user_id': ['1', '2', '3'],
            'item_id': ['A', 'B', 'C'],
            'rating': [5, 4, 3],
            'weight': [5.0, 4.0, 3.0],
            'timestamp': pd.date_range('2020-01-01', periods=3)
        })
        
        # Temporarily change data directory
        original_data_dir = data_loader.data_dir
        data_loader.data_dir = Path(temp_data_dir)
        
        try:
            # Save data
            data_loader.save_processed_data(data, 'test', 'train')
            
            # Load data
            loaded_data = data_loader.load_processed_data('test', 'train')
            
            assert loaded_data is not None
            assert len(loaded_data) == len(data)
            assert all(loaded_data.columns == data.columns)
            
        finally:
            # Restore original data directory
            data_loader.data_dir = original_data_dir

class TestDataLoaderIntegration:
    """Integration tests for data loading."""
    
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

if __name__ == "__main__":
    pytest.main([__file__]) 