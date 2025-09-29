#!/usr/bin/env python3
"""Evaluation script for comparing recommendation models."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import structlog
from pathlib import Path
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any

from recsys.data.loaders import data_loader
from recsys.pipeline.eval import RecommendationEvaluator, BaselineEvaluator
from recsys.pipeline.recall import RecallPipeline
from recsys.pipeline.rank import RankingPipeline
from recsys.models.cf_als import ALSRecommender
from recsys.models.cf_item_item import ItemItemRecommender
from recsys.models.content_embed import ContentEmbeddingRecommender
from recsys.models.ranker_lgbm import LightGBMRanker
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

def evaluate_domain(domain: str):
    """Evaluate recommendation models for a specific domain."""
    logger.info("Starting evaluation for domain", domain=domain)
    
    try:
        # Load test data
        test_data = data_loader.load_processed_data(domain, 'test')
        if test_data is None or len(test_data) == 0:
            logger.warning("No test data available", domain=domain)
            return None
        
        # Load items data
        if domain == 'books':
            items_df = data_loader.load_books()
        else:
            items_df = data_loader.load_movies()
        
        # Create ground truth
        ground_truth = create_ground_truth(test_data)
        
        # Load models
        models = load_models(domain)
        if not models:
            logger.warning("No models loaded", domain=domain)
            return None
        
        # Evaluate different approaches
        results = {}
        
        # 1. Popularity baseline
        logger.info("Evaluating popularity baseline", domain=domain)
        popularity_items = load_popularity_items(domain)
        if popularity_items:
            popularity_item_ids = [item['item_id'] for item in popularity_items]
            baseline_evaluator = BaselineEvaluator(domain)
            popularity_results = baseline_evaluator.evaluate_popularity_baseline(
                popularity_item_ids, ground_truth, k=20
            )
            results['popularity'] = popularity_results
        
        # 2. Content-based only
        logger.info("Evaluating content-based model", domain=domain)
        if 'content_model' in models:
            content_results = evaluate_content_only(models['content_model'], ground_truth, domain)
            results['content_only'] = content_results
        
        # 3. Collaborative filtering only
        logger.info("Evaluating collaborative filtering models", domain=domain)
        if 'als_model' in models:
            cf_results = evaluate_cf_only(models['als_model'], ground_truth, domain)
            results['cf_only'] = cf_results
        
        # 4. Hybrid approach
        logger.info("Evaluating hybrid model", domain=domain)
        if all(key in models for key in ['als_model', 'item_item_model', 'content_model', 'ranker']):
            hybrid_results = evaluate_hybrid(models, ground_truth, domain, items_df)
            results['hybrid'] = hybrid_results
        
        # 5. Random baseline
        logger.info("Evaluating random baseline", domain=domain)
        all_item_ids = items_df['item_id'].tolist()
        random_results = baseline_evaluator.evaluate_random_baseline(
            all_item_ids, ground_truth, k=20, num_runs=3
        )
        results['random'] = random_results
        
        # Generate comparison report
        comparison_report = generate_comparison_report(results, domain)
        
        # Save results
        save_evaluation_results(results, comparison_report, domain)
        
        logger.info("Evaluation completed", domain=domain, num_models=len(results))
        return results
        
    except Exception as e:
        logger.error("Evaluation failed", domain=domain, error=str(e))
        raise

def create_ground_truth(test_data: pd.DataFrame) -> Dict[str, List[str]]:
    """Create ground truth from test data."""
    logger.info("Creating ground truth from test data")
    
    # Group by user and get items with positive interactions
    positive_interactions = test_data[test_data['weight'] >= 3.0]
    
    ground_truth = {}
    for user_id, group in positive_interactions.groupby('user_id'):
        ground_truth[user_id] = group['item_id'].tolist()
    
    logger.info("Ground truth created", num_users=len(ground_truth))
    return ground_truth

def load_models(domain: str) -> Dict[str, Any]:
    """Load trained models for evaluation."""
    logger.info("Loading models", domain=domain)
    
    models = {}
    
    try:
        # Load ALS model
        als_model = ALSRecommender(domain)
        als_model.load(domain)
        models['als_model'] = als_model
        
        # Load item-item model
        item_item_model = ItemItemRecommender(domain)
        item_item_model.load(domain)
        models['item_item_model'] = item_item_model
        
        # Load content model
        content_model = ContentEmbeddingRecommender(domain)
        content_model.load(domain)
        models['content_model'] = content_model
        
        # Load ranker
        ranker = LightGBMRanker(domain)
        ranker.load(domain)
        models['ranker'] = ranker
        
        logger.info("Models loaded successfully", domain=domain, models=list(models.keys()))
        
    except Exception as e:
        logger.warning("Failed to load some models", domain=domain, error=str(e))
    
    return models

def load_popularity_items(domain: str) -> List[Dict[str, Any]]:
    """Load popularity items."""
    try:
        import pickle
        artifacts_dir = Path(config.ARTIFACTS_DIR) / domain
        popularity_path = artifacts_dir / "popularity_items.pkl"
        
        if popularity_path.exists():
            with open(popularity_path, "rb") as f:
                popularity_items = pickle.load(f)
            return popularity_items
    except Exception as e:
        logger.warning("Failed to load popularity items", domain=domain, error=str(e))
    
    return []

def evaluate_content_only(content_model, ground_truth: Dict[str, List[str]], domain: str) -> Dict[str, Any]:
    """Evaluate content-based model only."""
    logger.info("Evaluating content-only model", domain=domain)
    
    evaluator = RecommendationEvaluator(domain)
    user_recommendations = {}
    
    # Sample users for evaluation (to avoid memory issues)
    sample_users = list(ground_truth.keys())[:1000]
    
    for user_id in sample_users:
        try:
            # Get content-based recommendations
            candidates = content_model.get_user_candidates(user_id, 100)
            recommendations = [candidate['item_id'] for candidate in candidates[:20]]
            user_recommendations[user_id] = recommendations
        except Exception as e:
            logger.warning("Failed to get recommendations for user", user_id=user_id, error=str(e))
            continue
    
    return evaluator.evaluate_recommendations(user_recommendations, ground_truth, [5, 10, 20])

def evaluate_cf_only(als_model, ground_truth: Dict[str, List[str]], domain: str) -> Dict[str, Any]:
    """Evaluate collaborative filtering model only."""
    logger.info("Evaluating CF-only model", domain=domain)
    
    evaluator = RecommendationEvaluator(domain)
    user_recommendations = {}
    
    # Sample users for evaluation
    sample_users = list(ground_truth.keys())[:1000]
    
    for user_id in sample_users:
        try:
            # Get CF recommendations
            candidates = als_model.get_user_candidates(user_id, 100)
            recommendations = [candidate['item_id'] for candidate in candidates[:20]]
            user_recommendations[user_id] = recommendations
        except Exception as e:
            logger.warning("Failed to get CF recommendations for user", user_id=user_id, error=str(e))
            continue
    
    return evaluator.evaluate_recommendations(user_recommendations, ground_truth, [5, 10, 20])

def evaluate_hybrid(models: Dict[str, Any], ground_truth: Dict[str, List[str]], 
                   domain: str, items_df: pd.DataFrame) -> Dict[str, Any]:
    """Evaluate hybrid model."""
    logger.info("Evaluating hybrid model", domain=domain)
    
    evaluator = RecommendationEvaluator(domain)
    user_recommendations = {}
    
    # Create recall pipeline
    recall_pipeline = RecallPipeline(
        domain=domain,
        als_model=models['als_model'],
        item_item_model=models['item_item_model'],
        content_model=models['content_model']
    )
    
    # Sample users for evaluation
    sample_users = list(ground_truth.keys())[:1000]
    
    for user_id in sample_users:
        try:
            # Get candidates from recall pipeline
            candidates = recall_pipeline.get_candidates(user_id, 100)
            
            # Get top-20 recommendations
            recommendations = [candidate['item_id'] for candidate in candidates[:20]]
            user_recommendations[user_id] = recommendations
        except Exception as e:
            logger.warning("Failed to get hybrid recommendations for user", user_id=user_id, error=str(e))
            continue
    
    return evaluator.evaluate_recommendations(user_recommendations, ground_truth, [5, 10, 20])

def generate_comparison_report(results: Dict[str, Any], domain: str) -> str:
    """Generate comparison report in markdown format."""
    logger.info("Generating comparison report", domain=domain)
    
    report = f"""# Recommendation System Evaluation Report

## Domain: {domain}
## Timestamp: {pd.Timestamp.now().isoformat()}

## Model Comparison

| Model | Recall@5 | Recall@10 | Recall@20 | NDCG@5 | NDCG@10 | NDCG@20 | MAP@5 | MAP@10 | MAP@20 |
|-------|----------|-----------|-----------|--------|---------|---------|-------|--------|--------|
"""
    
    for model_name, model_results in results.items():
        if 'metrics' in model_results:
            metrics = model_results['metrics']
            
            # Extract metrics for each k value
            recall_5 = metrics.get('k_5', {}).get('recall', 0.0)
            recall_10 = metrics.get('k_10', {}).get('recall', 0.0)
            recall_20 = metrics.get('k_20', {}).get('recall', 0.0)
            
            ndcg_5 = metrics.get('k_5', {}).get('ndcg', 0.0)
            ndcg_10 = metrics.get('k_10', {}).get('ndcg', 0.0)
            ndcg_20 = metrics.get('k_20', {}).get('ndcg', 0.0)
            
            map_5 = metrics.get('k_5', {}).get('map', 0.0)
            map_10 = metrics.get('k_10', {}).get('map', 0.0)
            map_20 = metrics.get('k_20', {}).get('map', 0.0)
            
            report += f"| {model_name} | {recall_5:.4f} | {recall_10:.4f} | {recall_20:.4f} | {ndcg_5:.4f} | {ndcg_10:.4f} | {ndcg_20:.4f} | {map_5:.4f} | {map_10:.4f} | {map_20:.4f} |\n"
    
    report += "\n## Summary\n\n"
    
    # Find best performing model for each metric
    best_models = {}
    for metric in ['recall', 'ndcg', 'map']:
        for k in [5, 10, 20]:
            metric_key = f'{metric}_{k}'
            best_score = 0.0
            best_model = None
            
            for model_name, model_results in results.items():
                if 'metrics' in model_results:
                    score = model_results['metrics'].get(f'k_{k}', {}).get(metric, 0.0)
                    if score > best_score:
                        best_score = score
                        best_model = model_name
            
            if best_model:
                best_models[metric_key] = {'model': best_model, 'score': best_score}
    
    report += "### Best Performing Models\n\n"
    for metric_key, info in best_models.items():
        report += f"- **{metric_key}**: {info['model']} ({info['score']:.4f})\n"
    
    return report

def save_evaluation_results(results: Dict[str, Any], report: str, domain: str):
    """Save evaluation results and report."""
    logger.info("Saving evaluation results", domain=domain)
    
    # Create artifacts directory
    artifacts_dir = Path(config.ARTIFACTS_DIR) / domain
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results as JSON
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    results_path = artifacts_dir / f"evaluation_results_{timestamp}.json"
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Save report as markdown
    report_path = artifacts_dir / f"evaluation_report_{timestamp}.md"
    
    with open(report_path, "w") as f:
        f.write(report)
    
    logger.info("Evaluation results saved", 
               domain=domain,
               results_path=str(results_path),
               report_path=str(report_path))

def main():
    """Main evaluation function."""
    logger.info("Starting evaluation process")
    
    try:
        # Evaluate books domain
        logger.info("Evaluating books domain")
        books_results = evaluate_domain('books')
        
        # Evaluate movies domain
        logger.info("Evaluating movies domain")
        movies_results = evaluate_domain('movies')
        
        # Generate overall summary
        generate_overall_summary(books_results, movies_results)
        
        logger.info("Evaluation process completed successfully")
        
    except Exception as e:
        logger.error("Evaluation process failed", error=str(e))
        raise

def generate_overall_summary(books_results: Dict[str, Any], movies_results: Dict[str, Any]):
    """Generate overall summary across domains."""
    logger.info("Generating overall summary")
    
    summary = """# Overall Evaluation Summary

## Cross-Domain Comparison

### Books Domain Performance
"""
    
    if books_results:
        for model_name, model_results in books_results.items():
            if 'metrics' in model_results:
                metrics = model_results['metrics']
                recall_20 = metrics.get('k_20', {}).get('recall', 0.0)
                ndcg_20 = metrics.get('k_20', {}).get('ndcg', 0.0)
                summary += f"- **{model_name}**: Recall@20={recall_20:.4f}, NDCG@20={ndcg_20:.4f}\n"
    
    summary += "\n### Movies Domain Performance\n"
    
    if movies_results:
        for model_name, model_results in movies_results.items():
            if 'metrics' in model_results:
                metrics = model_results['metrics']
                recall_20 = metrics.get('k_20', {}).get('recall', 0.0)
                ndcg_20 = metrics.get('k_20', {}).get('ndcg', 0.0)
                summary += f"- **{model_name}**: Recall@20={recall_20:.4f}, NDCG@20={ndcg_20:.4f}\n"
    
    # Save overall summary
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    summary_path = Path(config.ARTIFACTS_DIR) / f"overall_evaluation_summary_{timestamp}.md"
    
    with open(summary_path, "w") as f:
        f.write(summary)
    
    logger.info("Overall summary generated", summary_path=str(summary_path))

if __name__ == "__main__":
    main() 