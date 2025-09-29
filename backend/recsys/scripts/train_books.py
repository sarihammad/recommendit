#!/usr/bin/env python3
"""Training script for books domain."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import structlog
from pathlib import Path
import pandas as pd
import numpy as np

from recsys.data.loaders import data_loader
from recsys.data.features import ContentFeatureExtractor, UserFeatureExtractor
from recsys.models.cf_als import ALSRecommender
from recsys.models.cf_item_item import ItemItemRecommender
from recsys.models.content_embed import ContentEmbeddingRecommender
from recsys.models.ranker_lgbm import LightGBMRanker, FeatureEngineer
from recsys.service.config import config

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

def main():
    """Main training function for books domain."""
    domain = 'books'
    logger.info("Starting training for books domain")
    
    try:
        # Step 1: Load and preprocess data
        logger.info("Step 1: Loading and preprocessing data")
        
        # Load books data
        books_df = data_loader.load_books()
        logger.info("Books data loaded", shape=books_df.shape)
        
        # Load interactions
        interactions_df = data_loader.load_interactions(domain)
        logger.info("Interactions data loaded", shape=interactions_df.shape)
        
        # Create time-based split
        train_data, val_data, test_data = data_loader.create_time_split(interactions_df)
        
        # Save processed data
        data_loader.save_processed_data(train_data, domain, 'train')
        data_loader.save_processed_data(val_data, domain, 'val')
        data_loader.save_processed_data(test_data, domain, 'test')
        
        # Compute item popularity
        item_popularity = data_loader.compute_item_popularity(train_data)
        
        # Step 2: Train content embedding model
        logger.info("Step 2: Training content embedding model")
        
        content_model = ContentEmbeddingRecommender(domain)
        content_model.fit(books_df)
        content_model.save(domain)
        
        # Step 3: Train collaborative filtering models
        logger.info("Step 3: Training collaborative filtering models")
        
        # ALS model
        als_model = ALSRecommender(domain)
        als_model.fit(train_data)
        als_model.save(domain)
        
        # Item-item model
        item_item_model = ItemItemRecommender(domain)
        item_item_model.fit(train_data)
        item_item_model.save(domain)
        
        # Step 4: Extract content features
        logger.info("Step 4: Extracting content features")
        
        content_feature_extractor = ContentFeatureExtractor(domain)
        content_features = content_feature_extractor.fit_transform(books_df)
        content_feature_extractor.save(domain)
        
        # Step 5: Extract user features
        logger.info("Step 5: Extracting user features")
        
        user_feature_extractor = UserFeatureExtractor()
        user_feature_extractor.fit(train_data, books_df)
        user_feature_extractor.save(domain)
        
        # Step 6: Prepare training data for ranking model
        logger.info("Step 6: Preparing ranking training data")
        
        # Create positive examples from training data
        positive_examples = train_data[train_data['weight'] >= 3.0].copy()
        positive_examples['label'] = 1
        
        # Create negative examples (random sampling)
        negative_count = len(positive_examples) * 4  # 4:1 negative to positive ratio
        negative_examples = train_data[train_data['weight'] < 2.0].sample(
            n=min(negative_count, len(train_data[train_data['weight'] < 2.0])),
            random_state=42
        ).copy()
        negative_examples['label'] = 0
        
        # Combine positive and negative examples
        ranking_train_data = pd.concat([positive_examples, negative_examples])
        ranking_train_data = ranking_train_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Step 7: Train ranking model
        logger.info("Step 7: Training ranking model")
        
        # Create feature engineer
        feature_engineer = FeatureEngineer(domain)
        
        # Create ranking features for training data
        ranking_features_list = []
        
        # Sample a subset for training (to avoid memory issues)
        sample_size = min(10000, len(ranking_train_data))
        sample_data = ranking_train_data.sample(n=sample_size, random_state=42)
        
        for _, row in sample_data.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            
            # Get user features
            user_features = user_feature_extractor.get_user_features(user_id)
            
            # Create candidate
            candidate = {
                'item_id': item_id,
                'score': row['weight'],
                'source': 'training'
            }
            
            # Create features
            features = feature_engineer.create_ranking_features(
                user_id=user_id,
                candidates=[candidate],
                als_model=als_model,
                item_item_model=item_item_model,
                content_model=content_model,
                user_features=user_features,
                items_df=books_df
            )
            
            # Add label
            features['label'] = row['label']
            features['user_id'] = user_id
            
            ranking_features_list.append(features.iloc[0])
        
        ranking_features_df = pd.DataFrame(ranking_features_list)
        
        # Train LightGBM ranker
        ranker = LightGBMRanker(domain)
        ranker.fit(ranking_features_df)
        ranker.save(domain)
        
        # Save feature engineer
        feature_engineer.save(domain)
        
        # Step 8: Save popularity items
        logger.info("Step 8: Saving popularity items")
        
        popularity_items = []
        for item_id, popularity_score in item_popularity.head(1000).items():
            popularity_items.append({
                'item_id': item_id,
                'score': float(popularity_score)
            })
        
        # Save popularity items
        import pickle
        artifacts_dir = Path(config.ARTIFACTS_DIR) / domain
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        with open(artifacts_dir / "popularity_items.pkl", "wb") as f:
            pickle.dump(popularity_items, f)
        
        # Step 9: Generate training report
        logger.info("Step 9: Generating training report")
        
        report = {
            'domain': domain,
            'training_stats': {
                'books_count': len(books_df),
                'interactions_count': len(interactions_df),
                'users_count': interactions_df['user_id'].nunique(),
                'items_count': interactions_df['item_id'].nunique(),
                'train_size': len(train_data),
                'val_size': len(val_data),
                'test_size': len(test_data),
                'positive_examples': len(positive_examples),
                'negative_examples': len(negative_examples),
                'ranking_features': len(ranking_features_df)
            },
            'models_trained': [
                'content_embedding',
                'als_collaborative_filtering',
                'item_item_collaborative_filtering',
                'lightgbm_ranker'
            ],
            'artifacts_saved': [
                'content_model',
                'als_model',
                'item_item_model',
                'ranking_model',
                'feature_extractors',
                'popularity_items'
            ]
        }
        
        # Save report
        import json
        with open(artifacts_dir / "training_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info("Training completed successfully", domain=domain, report=report)
        
        return True
        
    except Exception as e:
        logger.error("Training failed", domain=domain, error=str(e))
        raise

if __name__ == "__main__":
    main() 