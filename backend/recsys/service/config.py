"""Configuration settings for the recommendation system."""

import os
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class Config:
    """Main configuration class."""
    
    # Data paths
    DATA_DIR: str = "dataset"
    ARTIFACTS_DIR: str = "artifacts"
    
    # Model parameters
    ALS_FACTORS: int = 100
    ALS_ITERATIONS: int = 20
    ALS_REGULARIZATION: float = 0.01
    ALS_ALPHA: float = 0.01
    
    # Content embedding
    SBERT_MODEL: str = "all-MiniLM-L6-v2"
    MAX_TEXT_LENGTH: int = 512
    
    # Faiss parameters
    FAISS_NPROBE: int = 10
    FAISS_NLIST: int = 100
    
    # Candidate generation
    RECALL_CANDIDATES: int = 1000
    MAX_CANDIDATES_PER_SOURCE: int = 300
    
    # Cold start
    MIN_USER_INTERACTIONS: int = 5
    ALPHA_MIN: float = 0.2
    ALPHA_MAX: float = 0.9
    ALPHA_THRESHOLD: int = 50
    
    # LightGBM parameters
    LGBM_OBJECTIVE: str = "lambdarank"
    LGBM_METRIC: str = "ndcg"
    LGBM_NDCG_EVAL_AT: str = "5,10,20"
    LGBM_NUM_LEAVES: int = 31
    LGBM_LEARNING_RATE: float = 0.05
    LGBM_NUM_ITERATIONS: int = 100
    
    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_TTL: int = 3600  # 1 hour
    
    # API settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Evaluation
    TEST_SPLIT_RATIO: float = 0.2
    VALIDATION_SPLIT_RATIO: float = 0.1
    TIME_SPLIT_THRESHOLD: float = 0.8  # Use 80% of data for training
    
    # Performance targets
    RECALL_LATENCY_P95_MS: int = 60
    RANK_LATENCY_P95_MS: int = 100
    E2E_LATENCY_P95_MS: int = 150
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables."""
        config = cls()
        
        # Override with environment variables
        for field in config.__dataclass_fields__:
            env_var = f"RECSYS_{field}"
            if env_var in os.environ:
                value = os.environ[env_var]
                # Try to convert to appropriate type
                field_type = type(getattr(config, field))
                if field_type == int:
                    setattr(config, field, int(value))
                elif field_type == float:
                    setattr(config, field, float(value))
                elif field_type == bool:
                    setattr(config, field, value.lower() == "true")
                else:
                    setattr(config, field, value)
        
        return config

# Global config instance
config = Config.from_env() 