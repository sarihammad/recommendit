#!/usr/bin/env python3
"""Script for building and optimizing Faiss indices."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import structlog
from pathlib import Path
import pandas as pd
import numpy as np
import faiss
import pickle
import json
from typing import Dict, List, Any

from recsys.data.loaders import data_loader
from recsys.models.content_embed import ContentEmbeddingRecommender
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

def build_books_index():
    """Build Faiss index for books domain."""
    logger.info("Building books index")
    
    try:
        # Load books data
        books_df = data_loader.load_books()
        logger.info("Books data loaded", shape=books_df.shape)
        
        # Create content embedding model
        content_model = ContentEmbeddingRecommender('books')
        content_model.fit(books_df)
        
        # Build optimized index
        optimized_index = build_optimized_index(
            embeddings=content_model.item_embeddings,
            domain='books',
            index_type='IVFFlat'
        )
        
        # Save optimized index
        artifacts_dir = Path(config.ARTIFACTS_DIR) / 'books'
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(optimized_index, str(artifacts_dir / "faiss_optimized.index"))
        
        # Save index metadata
        index_metadata = {
            'domain': 'books',
            'index_type': 'IVFFlat',
            'num_items': len(books_df),
            'embedding_dim': content_model.item_embeddings.shape[1],
            'nlist': config.FAISS_NLIST,
            'nprobe': config.FAISS_NPROBE
        }
        
        with open(artifacts_dir / "faiss_metadata.json", "w") as f:
            json.dump(index_metadata, f, indent=2)
        
        logger.info("Books index built successfully", metadata=index_metadata)
        return True
        
    except Exception as e:
        logger.error("Failed to build books index", error=str(e))
        raise

def build_movies_index():
    """Build Faiss index for movies domain."""
    logger.info("Building movies index")
    
    try:
        # Load movies data
        movies_df = data_loader.load_movies()
        logger.info("Movies data loaded", shape=movies_df.shape)
        
        # Create content embedding model
        content_model = ContentEmbeddingRecommender('movies')
        content_model.fit(movies_df)
        
        # Build optimized index
        optimized_index = build_optimized_index(
            embeddings=content_model.item_embeddings,
            domain='movies',
            index_type='IVFFlat'
        )
        
        # Save optimized index
        artifacts_dir = Path(config.ARTIFACTS_DIR) / 'movies'
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(optimized_index, str(artifacts_dir / "faiss_optimized.index"))
        
        # Save index metadata
        index_metadata = {
            'domain': 'movies',
            'index_type': 'IVFFlat',
            'num_items': len(movies_df),
            'embedding_dim': content_model.item_embeddings.shape[1],
            'nlist': config.FAISS_NLIST,
            'nprobe': config.FAISS_NPROBE
        }
        
        with open(artifacts_dir / "faiss_metadata.json", "w") as f:
            json.dump(index_metadata, f, indent=2)
        
        logger.info("Movies index built successfully", metadata=index_metadata)
        return True
        
    except Exception as e:
        logger.error("Failed to build movies index", error=str(e))
        raise

def build_optimized_index(embeddings: np.ndarray, domain: str, index_type: str = 'IVFFlat') -> faiss.Index:
    """Build an optimized Faiss index."""
    logger.info("Building optimized index", domain=domain, index_type=index_type, shape=embeddings.shape)
    
    # Normalize embeddings
    faiss.normalize_L2(embeddings)
    
    # Determine number of clusters
    nlist = min(config.FAISS_NLIST, embeddings.shape[0] // 30)  # At least 30 vectors per cluster
    
    if index_type == 'IVFFlat':
        # Create IVF index with Flat quantizer
        quantizer = faiss.IndexFlatIP(embeddings.shape[1])
        index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1], nlist, faiss.METRIC_INNER_PRODUCT)
        
        # Train the index
        index.train(embeddings)
        
        # Add vectors to the index
        index.add(embeddings)
        
        # Set nprobe for search
        index.nprobe = min(config.FAISS_NPROBE, nlist)
        
    elif index_type == 'HNSW':
        # Create HNSW index
        index = faiss.IndexHNSWFlat(embeddings.shape[1], 32)  # 32 neighbors per layer
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 100
        
        # Add vectors to the index
        index.add(embeddings)
        
    else:
        # Default to Flat index
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
    
    logger.info("Index built successfully", 
               domain=domain, 
               index_type=index_type, 
               num_vectors=index.ntotal,
               nlist=nlist if hasattr(index, 'nlist') else None)
    
    return index

def benchmark_index_performance(domain: str, index_path: str, num_queries: int = 1000):
    """Benchmark index performance."""
    logger.info("Benchmarking index performance", domain=domain)
    
    try:
        # Load index
        index = faiss.read_index(index_path)
        
        # Load embeddings for queries
        artifacts_dir = Path(config.ARTIFACTS_DIR) / domain
        embeddings_path = artifacts_dir / "item_embeddings.npy"
        
        if not embeddings_path.exists():
            logger.warning("Embeddings not found for benchmarking", domain=domain)
            return
        
        embeddings = np.load(embeddings_path)
        faiss.normalize_L2(embeddings)
        
        # Generate random queries
        np.random.seed(42)
        query_indices = np.random.choice(len(embeddings), num_queries, replace=False)
        queries = embeddings[query_indices]
        
        # Benchmark search performance
        import time
        
        # Warm up
        for _ in range(10):
            index.search(queries[:10], 20)
        
        # Benchmark
        start_time = time.time()
        for i in range(0, num_queries, 100):  # Batch queries
            batch_queries = queries[i:i+100]
            D, I = index.search(batch_queries, 20)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_query = total_time / num_queries * 1000  # Convert to milliseconds
        
        # Calculate recall
        recall_scores = []
        for i, query_idx in enumerate(query_indices):
            # Get ground truth (top-20 most similar items)
            query_embedding = embeddings[query_idx:query_idx+1]
            D_gt, I_gt = faiss.IndexFlatIP(embeddings.shape[1]).search(query_embedding, 20)
            
            # Get predicted results
            D_pred, I_pred = index.search(query_embedding, 20)
            
            # Calculate recall@20
            ground_truth = set(I_gt[0])
            predicted = set(I_pred[0])
            recall = len(ground_truth & predicted) / len(ground_truth)
            recall_scores.append(recall)
        
        avg_recall = np.mean(recall_scores)
        
        # Save benchmark results
        benchmark_results = {
            'domain': domain,
            'index_type': 'optimized',
            'num_queries': num_queries,
            'avg_time_per_query_ms': avg_time_per_query,
            'total_time_seconds': total_time,
            'avg_recall_at_20': avg_recall,
            'index_size_mb': Path(index_path).stat().st_size / (1024 * 1024)
        }
        
        with open(artifacts_dir / "index_benchmark.json", "w") as f:
            json.dump(benchmark_results, f, indent=2)
        
        logger.info("Index benchmark completed", results=benchmark_results)
        
    except Exception as e:
        logger.error("Failed to benchmark index", domain=domain, error=str(e))
        raise

def main():
    """Main function to build indices for all domains."""
    logger.info("Starting index building process")
    
    try:
        # Build books index
        if build_books_index():
            logger.info("Books index built successfully")
            
            # Benchmark books index
            books_index_path = Path(config.ARTIFACTS_DIR) / 'books' / "faiss_optimized.index"
            if books_index_path.exists():
                benchmark_index_performance('books', str(books_index_path))
        
        # Build movies index
        if build_movies_index():
            logger.info("Movies index built successfully")
            
            # Benchmark movies index
            movies_index_path = Path(config.ARTIFACTS_DIR) / 'movies' / "faiss_optimized.index"
            if movies_index_path.exists():
                benchmark_index_performance('movies', str(movies_index_path))
        
        logger.info("Index building process completed successfully")
        
    except Exception as e:
        logger.error("Index building process failed", error=str(e))
        raise

if __name__ == "__main__":
    main() 