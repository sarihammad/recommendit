"""Evaluation module for recommendation system metrics."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import structlog
from pathlib import Path
import json
from datetime import datetime
from sklearn.metrics import ndcg_score
from ..service.config import config

logger = structlog.get_logger(__name__)

class RecommendationEvaluator:
    """Evaluator for recommendation system metrics."""
    
    def __init__(self, domain: str):
        """Initialize evaluator."""
        self.domain = domain
        self.metrics = {}
        
    def calculate_recall_at_k(self, 
                            ground_truth: List[str], 
                            predictions: List[str], 
                            k: int) -> float:
        """Calculate Recall@K."""
        if not ground_truth:
            return 0.0
        
        # Get top-k predictions
        top_k_predictions = predictions[:k]
        
        # Count relevant items in top-k
        relevant_in_top_k = len(set(ground_truth) & set(top_k_predictions))
        
        # Calculate recall
        recall = relevant_in_top_k / len(ground_truth)
        
        return recall
    
    def calculate_ndcg_at_k(self, 
                           ground_truth: List[str], 
                           predictions: List[str], 
                           k: int,
                           relevance_scores: Optional[Dict[str, float]] = None) -> float:
        """Calculate NDCG@K."""
        if not ground_truth:
            return 0.0
        
        # Get top-k predictions
        top_k_predictions = predictions[:k]
        
        # Create relevance vector for predictions
        y_true = []
        y_score = []
        
        for pred in top_k_predictions:
            if pred in ground_truth:
                relevance = relevance_scores.get(pred, 1.0) if relevance_scores else 1.0
                y_true.append(relevance)
            else:
                y_true.append(0.0)
            y_score.append(1.0)  # Assuming uniform ranking
        
        # Calculate NDCG
        try:
            ndcg = ndcg_score([y_true], [y_score], k=k)
            return ndcg
        except:
            return 0.0
    
    def calculate_map_at_k(self, 
                          ground_truth: List[str], 
                          predictions: List[str], 
                          k: int) -> float:
        """Calculate MAP@K (Mean Average Precision)."""
        if not ground_truth:
            return 0.0
        
        # Get top-k predictions
        top_k_predictions = predictions[:k]
        
        # Calculate precision at each position
        precisions = []
        relevant_count = 0
        
        for i, pred in enumerate(top_k_predictions):
            if pred in ground_truth:
                relevant_count += 1
                precision = relevant_count / (i + 1)
                precisions.append(precision)
        
        # Calculate MAP
        if precisions:
            map_score = sum(precisions) / len(ground_truth)
            return map_score
        else:
            return 0.0
    
    def calculate_hit_rate_at_k(self, 
                               ground_truth: List[str], 
                               predictions: List[str], 
                               k: int) -> float:
        """Calculate Hit Rate@K (whether any relevant item is in top-k)."""
        if not ground_truth:
            return 0.0
        
        # Get top-k predictions
        top_k_predictions = predictions[:k]
        
        # Check if any relevant item is in top-k
        has_hit = bool(set(ground_truth) & set(top_k_predictions))
        
        return 1.0 if has_hit else 0.0
    
    def calculate_diversity(self, predictions: List[str], 
                          item_features: Dict[str, Dict[str, Any]]) -> float:
        """Calculate diversity of recommendations."""
        if len(predictions) < 2:
            return 0.0
        
        # Calculate pairwise diversity based on features
        diversity_scores = []
        
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                item1 = predictions[i]
                item2 = predictions[j]
                
                # Get features for both items
                features1 = item_features.get(item1, {})
                features2 = item_features.get(item2, {})
                
                # Calculate feature diversity (simplified)
                diversity = self._calculate_item_diversity(features1, features2)
                diversity_scores.append(diversity)
        
        return np.mean(diversity_scores) if diversity_scores else 0.0
    
    def _calculate_item_diversity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """Calculate diversity between two items based on features."""
        # Simplified diversity calculation
        # In practice, you'd use more sophisticated metrics
        
        # Genre diversity
        genres1 = set(features1.get('genres', []))
        genres2 = set(features2.get('genres', []))
        
        if genres1 and genres2:
            genre_overlap = len(genres1 & genres2) / len(genres1 | genres2)
            genre_diversity = 1.0 - genre_overlap
        else:
            genre_diversity = 0.5  # Neutral diversity
        
        # Author diversity
        author1 = features1.get('author', '')
        author2 = features2.get('author', '')
        
        if author1 and author2:
            author_diversity = 0.0 if author1 == author2 else 1.0
        else:
            author_diversity = 0.5
        
        # Year diversity
        year1 = features1.get('year')
        year2 = features2.get('year')
        
        if year1 and year2:
            year_diff = abs(year1 - year2)
            year_diversity = min(year_diff / 50.0, 1.0)  # Normalize to 50 years
        else:
            year_diversity = 0.5
        
        # Combine diversities
        overall_diversity = (genre_diversity + author_diversity + year_diversity) / 3.0
        
        return overall_diversity
    
    def evaluate_recommendations(self, 
                               user_recommendations: Dict[str, List[str]], 
                               ground_truth: Dict[str, List[str]], 
                               k_values: List[int] = [5, 10, 20, 50],
                               item_features: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Evaluate recommendations for multiple users."""
        logger.info("Starting evaluation", domain=self.domain, num_users=len(user_recommendations))
        
        results = {
            'domain': self.domain,
            'timestamp': datetime.now().isoformat(),
            'k_values': k_values,
            'metrics': {}
        }
        
        # Calculate metrics for each k value
        for k in k_values:
            k_results = {
                'recall': [],
                'ndcg': [],
                'map': [],
                'hit_rate': []
            }
            
            for user_id, predictions in user_recommendations.items():
                if user_id in ground_truth:
                    user_ground_truth = ground_truth[user_id]
                    
                    # Calculate metrics
                    recall = self.calculate_recall_at_k(user_ground_truth, predictions, k)
                    ndcg = self.calculate_ndcg_at_k(user_ground_truth, predictions, k)
                    map_score = self.calculate_map_at_k(user_ground_truth, predictions, k)
                    hit_rate = self.calculate_hit_rate_at_k(user_ground_truth, predictions, k)
                    
                    k_results['recall'].append(recall)
                    k_results['ndcg'].append(ndcg)
                    k_results['map'].append(map_score)
                    k_results['hit_rate'].append(hit_rate)
            
            # Calculate averages
            results['metrics'][f'k_{k}'] = {
                'recall': np.mean(k_results['recall']) if k_results['recall'] else 0.0,
                'ndcg': np.mean(k_results['ndcg']) if k_results['ndcg'] else 0.0,
                'map': np.mean(k_results['map']) if k_results['map'] else 0.0,
                'hit_rate': np.mean(k_results['hit_rate']) if k_results['hit_rate'] else 0.0,
                'num_users': len(k_results['recall'])
            }
        
        # Calculate diversity if item features are provided
        if item_features:
            diversity_scores = []
            for user_id, predictions in user_recommendations.items():
                diversity = self.calculate_diversity(predictions, item_features)
                diversity_scores.append(diversity)
            
            results['metrics']['diversity'] = {
                'mean': np.mean(diversity_scores) if diversity_scores else 0.0,
                'std': np.std(diversity_scores) if diversity_scores else 0.0
            }
        
        self.metrics = results
        return results
    
    def compare_models(self, 
                      model_results: Dict[str, Dict[str, List[str]]], 
                      ground_truth: Dict[str, List[str]], 
                      k_values: List[int] = [5, 10, 20]) -> Dict[str, Any]:
        """Compare multiple models."""
        logger.info("Comparing models", models=list(model_results.keys()))
        
        comparison = {
            'domain': self.domain,
            'timestamp': datetime.now().isoformat(),
            'k_values': k_values,
            'models': {}
        }
        
        for model_name, user_recommendations in model_results.items():
            model_metrics = self.evaluate_recommendations(
                user_recommendations, ground_truth, k_values
            )
            comparison['models'][model_name] = model_metrics['metrics']
        
        return comparison
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """Generate evaluation report."""
        if not self.metrics:
            logger.warning("No metrics available for report generation")
            return ""
        
        # Create report directory
        artifacts_dir = Path(config.ARTIFACTS_DIR) / self.domain
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        if output_path is None:
            output_path = artifacts_dir / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Save metrics to file
        with open(output_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Generate markdown report
        markdown_report = self._generate_markdown_report()
        markdown_path = str(output_path).replace('.json', '.md')
        
        with open(markdown_path, 'w') as f:
            f.write(markdown_report)
        
        logger.info("Evaluation report generated", 
                   json_path=str(output_path),
                   markdown_path=markdown_path)
        
        return markdown_report
    
    def _generate_markdown_report(self) -> str:
        """Generate markdown format evaluation report."""
        if not self.metrics:
            return "# Evaluation Report\n\nNo metrics available."
        
        report = f"""# Recommendation System Evaluation Report

## Domain: {self.domain}
## Timestamp: {self.metrics.get('timestamp', 'Unknown')}

## Metrics Summary

"""
        
        # Add metrics for each k value
        for k_key, k_metrics in self.metrics.get('metrics', {}).items():
            if k_key.startswith('k_'):
                k = k_key.replace('k_', '')
                report += f"### K = {k}\n\n"
                report += f"- **Recall@{k}**: {k_metrics['recall']:.4f}\n"
                report += f"- **NDCG@{k}**: {k_metrics['ndcg']:.4f}\n"
                report += f"- **MAP@{k}**: {k_metrics['map']:.4f}\n"
                report += f"- **Hit Rate@{k}**: {k_metrics['hit_rate']:.4f}\n"
                report += f"- **Number of Users**: {k_metrics['num_users']}\n\n"
        
        # Add diversity metrics if available
        if 'diversity' in self.metrics.get('metrics', {}):
            diversity = self.metrics['metrics']['diversity']
            report += f"### Diversity\n\n"
            report += f"- **Mean Diversity**: {diversity['mean']:.4f}\n"
            report += f"- **Std Diversity**: {diversity['std']:.4f}\n\n"
        
        # Add model comparison if available
        if 'models' in self.metrics:
            report += "## Model Comparison\n\n"
            for model_name, model_metrics in self.metrics['models'].items():
                report += f"### {model_name}\n\n"
                for k_key, k_metrics in model_metrics.items():
                    if k_key.startswith('k_'):
                        k = k_key.replace('k_', '')
                        report += f"- **Recall@{k}**: {k_metrics['recall']:.4f}\n"
                        report += f"- **NDCG@{k}**: {k_metrics['ndcg']:.4f}\n"
                        report += f"- **MAP@{k}**: {k_metrics['map']:.4f}\n"
                report += "\n"
        
        return report

class BaselineEvaluator:
    """Evaluator for baseline models (popularity, random, etc.)."""
    
    def __init__(self, domain: str):
        """Initialize baseline evaluator."""
        self.domain = domain
        self.evaluator = RecommendationEvaluator(domain)
    
    def evaluate_popularity_baseline(self, 
                                   popularity_items: List[str], 
                                   ground_truth: Dict[str, List[str]], 
                                   k: int = 20) -> Dict[str, Any]:
        """Evaluate popularity baseline."""
        logger.info("Evaluating popularity baseline", domain=self.domain)
        
        # Create popularity recommendations for all users
        user_recommendations = {}
        for user_id in ground_truth.keys():
            user_recommendations[user_id] = popularity_items[:k]
        
        return self.evaluator.evaluate_recommendations(user_recommendations, ground_truth, [k])
    
    def evaluate_random_baseline(self, 
                               all_items: List[str], 
                               ground_truth: Dict[str, List[str]], 
                               k: int = 20, 
                               num_runs: int = 5) -> Dict[str, Any]:
        """Evaluate random baseline with multiple runs."""
        logger.info("Evaluating random baseline", domain=self.domain, num_runs=num_runs)
        
        all_results = []
        
        for run in range(num_runs):
            # Create random recommendations for all users
            user_recommendations = {}
            for user_id in ground_truth.keys():
                random_items = np.random.choice(all_items, size=k, replace=False)
                user_recommendations[user_id] = random_items.tolist()
            
            run_results = self.evaluator.evaluate_recommendations(user_recommendations, ground_truth, [k])
            all_results.append(run_results['metrics'][f'k_{k}'])
        
        # Average results across runs
        avg_results = {}
        for metric in ['recall', 'ndcg', 'map', 'hit_rate']:
            values = [result[metric] for result in all_results]
            avg_results[metric] = np.mean(values)
            avg_results[f'{metric}_std'] = np.std(values)
        
        return {
            'domain': self.domain,
            'baseline': 'random',
            'k': k,
            'num_runs': num_runs,
            'metrics': avg_results
        } 