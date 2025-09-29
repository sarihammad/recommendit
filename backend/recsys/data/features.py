"""Feature extraction utilities for content-based recommendation."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import structlog
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
from pathlib import Path
from .config import config

logger = structlog.get_logger(__name__)

class ContentFeatureExtractor:
    """Extract content-based features from item metadata."""
    
    def __init__(self, domain: str):
        """Initialize feature extractor."""
        self.domain = domain
        self.tfidf_vectorizer = None
        self.genre_encoder = LabelEncoder()
        self.author_encoder = LabelEncoder()
        self.year_scaler = StandardScaler()
        
    def fit_transform(self, items_df: pd.DataFrame) -> pd.DataFrame:
        """Fit feature extractors and transform data."""
        logger.info("Fitting content feature extractors", domain=self.domain)
        
        # Extract text features
        text_features = self._extract_text_features(items_df)
        
        # Extract categorical features
        categorical_features = self._extract_categorical_features(items_df)
        
        # Extract numerical features
        numerical_features = self._extract_numerical_features(items_df)
        
        # Combine all features
        features_df = pd.concat([
            items_df[['item_id']],
            text_features,
            categorical_features,
            numerical_features
        ], axis=1)
        
        logger.info("Content features extracted", 
                   domain=self.domain, 
                   shape=features_df.shape,
                   text_features=text_features.shape[1],
                   categorical_features=categorical_features.shape[1],
                   numerical_features=numerical_features.shape[1])
        
        return features_df
    
    def transform(self, items_df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted extractors."""
        if self.tfidf_vectorizer is None:
            raise ValueError("Feature extractor not fitted. Call fit_transform first.")
        
        # Extract text features
        text_features = self._extract_text_features(items_df, fitted=True)
        
        # Extract categorical features
        categorical_features = self._extract_categorical_features(items_df, fitted=True)
        
        # Extract numerical features
        numerical_features = self._extract_numerical_features(items_df, fitted=True)
        
        # Combine all features
        features_df = pd.concat([
            items_df[['item_id']],
            text_features,
            categorical_features,
            numerical_features
        ], axis=1)
        
        return features_df
    
    def _extract_text_features(self, items_df: pd.DataFrame, fitted: bool = False) -> pd.DataFrame:
        """Extract text-based features using TF-IDF."""
        # Combine title and description
        text_data = items_df['title'].fillna('') + ' ' + items_df['description'].fillna('')
        
        if not fitted:
            # Fit TF-IDF vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                stop_words='english',
                min_df=2,
                max_df=0.95
            )
            text_features = self.tfidf_vectorizer.fit_transform(text_data)
        else:
            # Transform using fitted vectorizer
            text_features = self.tfidf_vectorizer.transform(text_data)
        
        # Convert to DataFrame
        feature_names = [f'text_{i}' for i in range(text_features.shape[1])]
        text_df = pd.DataFrame(
            text_features.toarray(),
            columns=feature_names,
            index=items_df.index
        )
        
        return text_df
    
    def _extract_categorical_features(self, items_df: pd.DataFrame, fitted: bool = False) -> pd.DataFrame:
        """Extract categorical features."""
        features = {}
        
        # Genre features
        if 'genres' in items_df.columns:
            genre_features = self._extract_genre_features(items_df['genres'], fitted)
            features.update(genre_features)
        
        # Author/Director features
        if 'author' in items_df.columns:
            author_features = self._extract_author_features(items_df['author'], fitted)
            features.update(author_features)
        
        return pd.DataFrame(features, index=items_df.index)
    
    def _extract_genre_features(self, genres_series: pd.Series, fitted: bool = False) -> Dict[str, pd.Series]:
        """Extract genre-based features."""
        features = {}
        
        # Flatten genres and get unique list
        all_genres = []
        for genres in genres_series:
            if isinstance(genres, list):
                all_genres.extend(genres)
            elif isinstance(genres, str):
                all_genres.extend([g.strip() for g in genres.split(',') if g.strip()])
        
        unique_genres = list(set(all_genres))
        
        # Create binary features for each genre
        for genre in unique_genres:
            feature_name = f'genre_{genre.lower().replace(" ", "_")}'
            features[feature_name] = genres_series.apply(
                lambda x: 1 if isinstance(x, list) and genre in x 
                else (1 if isinstance(x, str) and genre in x else 0)
            )
        
        # Genre count feature
        features['genre_count'] = genres_series.apply(
            lambda x: len(x) if isinstance(x, list) 
            else (len([g.strip() for g in x.split(',') if g.strip()]) if isinstance(x, str) else 0)
        )
        
        return features
    
    def _extract_author_features(self, author_series: pd.Series, fitted: bool = False) -> Dict[str, pd.Series]:
        """Extract author/director features."""
        features = {}
        
        # Clean author names
        clean_authors = author_series.fillna('Unknown').astype(str)
        
        if not fitted:
            # Fit label encoder
            self.author_encoder.fit(clean_authors)
        
        # Encode authors
        author_encoded = self.author_encoder.transform(clean_authors)
        features['author_encoded'] = author_encoded
        
        # Author popularity (count of books/movies by same author)
        author_counts = clean_authors.value_counts()
        features['author_popularity'] = clean_authors.map(author_counts)
        
        return features
    
    def _extract_numerical_features(self, items_df: pd.DataFrame, fitted: bool = False) -> pd.DataFrame:
        """Extract numerical features."""
        features = {}
        
        # Year features
        if 'year' in items_df.columns:
            year_features = self._extract_year_features(items_df['year'], fitted)
            features.update(year_features)
        
        # Popularity features (if available)
        if 'popularity' in items_df.columns:
            features['popularity'] = items_df['popularity'].fillna(0)
        
        return pd.DataFrame(features, index=items_df.index)
    
    def _extract_year_features(self, year_series: pd.Series, fitted: bool = False) -> Dict[str, pd.Series]:
        """Extract year-based features."""
        features = {}
        
        # Fill missing years with median
        median_year = year_series.median()
        year_filled = year_series.fillna(median_year)
        
        if not fitted:
            # Fit scaler
            self.year_scaler.fit(year_filled.values.reshape(-1, 1))
        
        # Scale years
        year_scaled = self.year_scaler.transform(year_filled.values.reshape(-1, 1)).flatten()
        features['year_scaled'] = year_scaled
        
        # Year categories
        features['year_category'] = pd.cut(
            year_filled, 
            bins=[0, 1950, 1980, 2000, 2020, 9999],
            labels=['very_old', 'old', 'recent', 'new', 'very_new']
        ).cat.codes
        
        return features
    
    def save(self, domain: str):
        """Save fitted feature extractors."""
        artifacts_dir = Path(config.ARTIFACTS_DIR) / domain
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Save TF-IDF vectorizer
        if self.tfidf_vectorizer:
            with open(artifacts_dir / "tfidf_vectorizer.pkl", "wb") as f:
                pickle.dump(self.tfidf_vectorizer, f)
        
        # Save encoders
        with open(artifacts_dir / "genre_encoder.pkl", "wb") as f:
            pickle.dump(self.genre_encoder, f)
        
        with open(artifacts_dir / "author_encoder.pkl", "wb") as f:
            pickle.dump(self.author_encoder, f)
        
        # Save scaler
        with open(artifacts_dir / "year_scaler.pkl", "wb") as f:
            pickle.dump(self.year_scaler, f)
        
        logger.info("Feature extractors saved", domain=domain)
    
    def load(self, domain: str):
        """Load fitted feature extractors."""
        artifacts_dir = Path(config.ARTIFACTS_DIR) / domain
        
        # Load TF-IDF vectorizer
        tfidf_path = artifacts_dir / "tfidf_vectorizer.pkl"
        if tfidf_path.exists():
            with open(tfidf_path, "rb") as f:
                self.tfidf_vectorizer = pickle.load(f)
        
        # Load encoders
        with open(artifacts_dir / "genre_encoder.pkl", "rb") as f:
            self.genre_encoder = pickle.load(f)
        
        with open(artifacts_dir / "author_encoder.pkl", "rb") as f:
            self.author_encoder = pickle.load(f)
        
        # Load scaler
        with open(artifacts_dir / "year_scaler.pkl", "rb") as f:
            self.year_scaler = pickle.load(f)
        
        logger.info("Feature extractors loaded", domain=domain)

class UserFeatureExtractor:
    """Extract user-based features."""
    
    def __init__(self):
        """Initialize user feature extractor."""
        self.user_interaction_counts = None
        self.user_avg_ratings = None
        self.user_genre_preferences = None
    
    def fit(self, interactions_df: pd.DataFrame, items_df: pd.DataFrame):
        """Fit user feature extractors."""
        logger.info("Fitting user feature extractors")
        
        # User interaction counts
        self.user_interaction_counts = interactions_df['user_id'].value_counts()
        
        # User average ratings
        if 'rating' in interactions_df.columns:
            self.user_avg_ratings = interactions_df.groupby('user_id')['rating'].mean()
        
        # User genre preferences
        self.user_genre_preferences = self._compute_user_genre_preferences(interactions_df, items_df)
        
        logger.info("User feature extractors fitted")
    
    def _compute_user_genre_preferences(self, interactions_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
        """Compute user genre preferences."""
        # Merge interactions with item genres
        merged = interactions_df.merge(
            items_df[['item_id', 'genres']], 
            on='item_id', 
            how='left'
        )
        
        # Flatten genres and compute preferences
        user_genres = []
        for _, row in merged.iterrows():
            user_id = row['user_id']
            genres = row['genres']
            weight = row.get('weight', 1.0)
            
            if isinstance(genres, list):
                for genre in genres:
                    user_genres.append({
                        'user_id': user_id,
                        'genre': genre,
                        'weight': weight
                    })
        
        if not user_genres:
            return pd.DataFrame()
        
        user_genres_df = pd.DataFrame(user_genres)
        
        # Aggregate by user and genre
        preferences = user_genres_df.groupby(['user_id', 'genre'])['weight'].sum().reset_index()
        
        # Pivot to wide format
        preferences_wide = preferences.pivot(
            index='user_id', 
            columns='genre', 
            values='weight'
        ).fillna(0)
        
        # Normalize by user total
        user_totals = preferences_wide.sum(axis=1)
        preferences_normalized = preferences_wide.div(user_totals, axis=0)
        
        return preferences_normalized
    
    def get_user_features(self, user_id: str) -> Dict[str, float]:
        """Get features for a specific user."""
        features = {}
        
        # Interaction count
        features['interaction_count'] = self.user_interaction_counts.get(user_id, 0)
        
        # Average rating
        if self.user_avg_ratings is not None:
            features['avg_rating'] = self.user_avg_ratings.get(user_id, 0.0)
        
        # Genre preferences
        if self.user_genre_preferences is not None and user_id in self.user_genre_preferences.index:
            user_prefs = self.user_genre_preferences.loc[user_id]
            for genre, pref in user_prefs.items():
                features[f'genre_pref_{genre.lower().replace(" ", "_")}'] = pref
        
        return features
    
    def save(self, domain: str):
        """Save user feature extractors."""
        artifacts_dir = Path(config.ARTIFACTS_DIR) / domain
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Save user features
        with open(artifacts_dir / "user_features.pkl", "wb") as f:
            pickle.dump({
                'interaction_counts': self.user_interaction_counts,
                'avg_ratings': self.user_avg_ratings,
                'genre_preferences': self.user_genre_preferences
            }, f)
        
        logger.info("User feature extractors saved", domain=domain)
    
    def load(self, domain: str):
        """Load user feature extractors."""
        artifacts_dir = Path(config.ARTIFACTS_DIR) / domain
        features_path = artifacts_dir / "user_features.pkl"
        
        if features_path.exists():
            with open(features_path, "rb") as f:
                features = pickle.load(f)
                self.user_interaction_counts = features['interaction_counts']
                self.user_avg_ratings = features['avg_ratings']
                self.user_genre_preferences = features['genre_preferences']
            
            logger.info("User feature extractors loaded", domain=domain) 