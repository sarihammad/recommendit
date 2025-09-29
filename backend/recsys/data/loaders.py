"""Data loading and preprocessing utilities."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import structlog
from pathlib import Path
import json
from datetime import datetime
import ast
from .config import config

logger = structlog.get_logger(__name__)

class SchemaMapper:
    """Maps different CSV schemas to standard format."""
    
    # Standard column names
    STANDARD_COLUMNS = {
        'user_id': ['user_id', 'userid', 'user-id', 'userid', 'User-ID'],
        'item_id': ['item_id', 'itemid', 'item-id', 'bookid', 'movieid', 'id', 'isbn'],
        'rating': ['rating', 'book-rating', 'Book-Rating', 'score'],
        'timestamp': ['timestamp', 'ts', 'time', 'date'],
        'title': ['title', 'name', 'book_title', 'movie_title'],
        'author': ['author', 'director', 'creator'],
        'genres': ['genres', 'genre', 'categories', 'category'],
        'description': ['description', 'overview', 'summary', 'plot', 'desc'],
        'year': ['year', 'publish_date', 'release_date', 'date'],
        'popularity': ['popularity', 'num_ratings', 'vote_count', 'rating_count']
    }
    
    @classmethod
    def infer_schema(cls, df: pd.DataFrame) -> Dict[str, str]:
        """Infer schema mapping from DataFrame columns."""
        mapping = {}
        df_columns = [col.lower() for col in df.columns]
        
        for standard_col, possible_names in cls.STANDARD_COLUMNS.items():
            for possible_name in possible_names:
                if possible_name.lower() in df_columns:
                    # Find the actual column name (preserve case)
                    for col in df.columns:
                        if col.lower() == possible_name.lower():
                            mapping[standard_col] = col
                            break
                    break
        
        return mapping
    
    @classmethod
    def standardize_dataframe(cls, df: pd.DataFrame, domain: str) -> pd.DataFrame:
        """Standardize DataFrame columns and data types."""
        mapping = cls.infer_schema(df)
        logger.info("Schema mapping", domain=domain, mapping=mapping)
        
        # Create standardized DataFrame
        std_df = pd.DataFrame()
        
        # Map columns
        for std_col, orig_col in mapping.items():
            if orig_col in df.columns:
                std_df[std_col] = df[orig_col]
        
        # Handle special cases for different domains
        if domain == 'books':
            std_df = cls._standardize_books(std_df, df, mapping)
        elif domain == 'movies':
            std_df = cls._standardize_movies(std_df, df, mapping)
        
        return std_df
    
    @classmethod
    def _standardize_books(cls, std_df: pd.DataFrame, orig_df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
        """Standardize books data."""
        # Handle ISBN as item_id if not already mapped
        if 'item_id' not in std_df.columns and 'isbn' in orig_df.columns:
            std_df['item_id'] = orig_df['isbn']
        
        # Parse genres if they're in string format
        if 'genres' in std_df.columns:
            std_df['genres'] = std_df['genres'].apply(cls._parse_genres)
        
        # Parse publish date
        if 'year' in std_df.columns:
            std_df['year'] = std_df['year'].apply(cls._parse_year)
        
        return std_df
    
    @classmethod
    def _standardize_movies(cls, std_df: pd.DataFrame, orig_df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
        """Standardize movies data."""
        # Parse genres from JSON-like string
        if 'genres' in std_df.columns:
            std_df['genres'] = std_df['genres'].apply(cls._parse_movie_genres)
        
        # Parse release date
        if 'year' in std_df.columns:
            std_df['year'] = std_df['year'].apply(cls._parse_movie_year)
        
        return std_df
    
    @staticmethod
    def _parse_genres(genre_str: str) -> List[str]:
        """Parse genres from string format."""
        if pd.isna(genre_str) or genre_str == '':
            return []
        
        try:
            # Try to parse as list
            if genre_str.startswith('[') and genre_str.endswith(']'):
                return ast.literal_eval(genre_str)
            # Split by comma
            return [g.strip() for g in genre_str.split(',') if g.strip()]
        except:
            return [genre_str.strip()]
    
    @staticmethod
    def _parse_movie_genres(genre_str: str) -> List[str]:
        """Parse movie genres from JSON-like string."""
        if pd.isna(genre_str) or genre_str == '':
            return []
        
        try:
            genres = ast.literal_eval(genre_str)
            if isinstance(genres, list):
                return [g.get('name', '') for g in genres if isinstance(g, dict)]
            return []
        except:
            return []
    
    @staticmethod
    def _parse_year(date_str: str) -> Optional[int]:
        """Parse year from date string."""
        if pd.isna(date_str) or date_str == '':
            return None
        
        try:
            # Try different date formats
            for fmt in ['%Y', '%Y-%m-%d', '%m/%d/%Y', '%Y-%m']:
                try:
                    return pd.to_datetime(date_str, format=fmt).year
                except:
                    continue
            return None
        except:
            return None
    
    @staticmethod
    def _parse_movie_year(date_str: str) -> Optional[int]:
        """Parse movie release year."""
        if pd.isna(date_str) or date_str == '':
            return None
        
        try:
            return pd.to_datetime(date_str).year
        except:
            return None

class DataLoader:
    """Data loader for recommendation system."""
    
    def __init__(self, data_dir: str = None):
        """Initialize data loader."""
        self.data_dir = Path(data_dir or config.DATA_DIR)
        self.schema_cache = {}
    
    def load_books(self) -> pd.DataFrame:
        """Load and standardize books data."""
        logger.info("Loading books data")
        
        books_path = self.data_dir / "books.csv"
        if not books_path.exists():
            raise FileNotFoundError(f"Books file not found: {books_path}")
        
        # Load with error handling for different encodings
        try:
            df = pd.read_csv(books_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(books_path, encoding='latin-1')
        
        # Standardize schema
        std_df = SchemaMapper.standardize_dataframe(df, 'books')
        
        # Clean and validate
        std_df = self._clean_books_data(std_df)
        
        logger.info("Books data loaded", shape=std_df.shape, columns=list(std_df.columns))
        return std_df
    
    def load_movies(self) -> pd.DataFrame:
        """Load and standardize movies data."""
        logger.info("Loading movies data")
        
        movies_path = self.data_dir / "movies.csv"
        if not movies_path.exists():
            raise FileNotFoundError(f"Movies file not found: {movies_path}")
        
        df = pd.read_csv(movies_path)
        
        # Standardize schema
        std_df = SchemaMapper.standardize_dataframe(df, 'movies')
        
        # Clean and validate
        std_df = self._clean_movies_data(std_df)
        
        logger.info("Movies data loaded", shape=std_df.shape, columns=list(std_df.columns))
        return std_df
    
    def load_interactions(self, domain: str) -> pd.DataFrame:
        """Load user-item interactions."""
        logger.info("Loading interactions", domain=domain)
        
        if domain == 'books':
            ratings_path = self.data_dir / "book_ratings.csv"
        elif domain == 'movies':
            ratings_path = self.data_dir / "movie_user_ratings.csv"
        else:
            raise ValueError(f"Unsupported domain: {domain}")
        
        if not ratings_path.exists():
            raise FileNotFoundError(f"Ratings file not found: {ratings_path}")
        
        # Load with different separators
        try:
            df = pd.read_csv(ratings_path, sep=',')
        except:
            df = pd.read_csv(ratings_path, sep=';')
        
        # Standardize schema
        std_df = SchemaMapper.standardize_dataframe(df, domain)
        
        # Clean and validate
        std_df = self._clean_interactions_data(std_df, domain)
        
        logger.info("Interactions loaded", domain=domain, shape=std_df.shape)
        return std_df
    
    def _clean_books_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean books data."""
        # Remove duplicates
        df = df.drop_duplicates(subset=['item_id'])
        
        # Fill missing values
        df['title'] = df['title'].fillna('Unknown Title')
        df['author'] = df['author'].fillna('Unknown Author')
        df['genres'] = df['genres'].fillna('[]')
        df['description'] = df['description'].fillna('')
        
        # Convert item_id to string
        df['item_id'] = df['item_id'].astype(str)
        
        return df
    
    def _clean_movies_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean movies data."""
        # Remove duplicates
        df = df.drop_duplicates(subset=['item_id'])
        
        # Fill missing values
        df['title'] = df['title'].fillna('Unknown Title')
        df['description'] = df['description'].fillna('')
        df['genres'] = df['genres'].fillna('[]')
        
        # Convert item_id to string
        df['item_id'] = df['item_id'].astype(str)
        
        return df
    
    def _clean_interactions_data(self, df: pd.DataFrame, domain: str) -> pd.DataFrame:
        """Clean interactions data."""
        # Remove rows with missing user_id or item_id
        df = df.dropna(subset=['user_id', 'item_id'])
        
        # Convert to string
        df['user_id'] = df['user_id'].astype(str)
        df['item_id'] = df['item_id'].astype(str)
        
        # Handle ratings
        if 'rating' in df.columns:
            # Convert to numeric, fill missing with 0
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0)
            
            # Create implicit weights
            df['weight'] = df['rating'].apply(self._rating_to_weight)
        else:
            # No ratings, assume implicit feedback
            df['weight'] = 1.0
        
        # Handle timestamps
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
            df = df.dropna(subset=['timestamp'])
        else:
            # Create dummy timestamps
            df['timestamp'] = pd.Timestamp.now()
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['user_id', 'item_id'])
        
        return df
    
    def _rating_to_weight(self, rating: float) -> float:
        """Convert explicit rating to implicit weight."""
        if rating >= 4:
            return 5.0  # Positive interaction
        elif rating >= 3:
            return 2.0  # Neutral interaction
        else:
            return 0.5  # Negative interaction
    
    def create_time_split(self, interactions: pd.DataFrame, 
                         train_ratio: float = 0.8,
                         val_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create time-based train/validation/test split."""
        logger.info("Creating time-based split")
        
        # Sort by timestamp
        interactions = interactions.sort_values('timestamp')
        
        # Calculate split points
        total_rows = len(interactions)
        train_end = int(total_rows * train_ratio)
        val_end = int(total_rows * (train_ratio + val_ratio))
        
        # Split
        train_data = interactions.iloc[:train_end]
        val_data = interactions.iloc[train_end:val_end]
        test_data = interactions.iloc[val_end:]
        
        logger.info("Time split created", 
                   train_size=len(train_data), 
                   val_size=len(val_data), 
                   test_size=len(test_data))
        
        return train_data, val_data, test_data
    
    def compute_item_popularity(self, interactions: pd.DataFrame) -> pd.Series:
        """Compute item popularity scores."""
        item_counts = interactions['item_id'].value_counts()
        
        # Log-scale popularity
        popularity = np.log1p(item_counts)
        
        return popularity
    
    def get_user_interaction_counts(self, interactions: pd.DataFrame) -> pd.Series:
        """Get user interaction counts."""
        return interactions['user_id'].value_counts()
    
    def save_processed_data(self, data: pd.DataFrame, domain: str, split: str):
        """Save processed data to artifacts directory."""
        artifacts_dir = Path(config.ARTIFACTS_DIR) / domain
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = artifacts_dir / f"{split}_data.parquet"
        data.to_parquet(output_path, index=False)
        
        logger.info("Processed data saved", domain=domain, split=split, path=str(output_path))
    
    def load_processed_data(self, domain: str, split: str) -> pd.DataFrame:
        """Load processed data from artifacts directory."""
        artifacts_dir = Path(config.ARTIFACTS_DIR) / domain
        input_path = artifacts_dir / f"{split}_data.parquet"
        
        if not input_path.exists():
            raise FileNotFoundError(f"Processed data not found: {input_path}")
        
        return pd.read_parquet(input_path)

# Global data loader instance
data_loader = DataLoader() 