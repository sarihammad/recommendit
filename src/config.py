from pydantic_settings import BaseSettings
from pydantic import Field
import os


class Settings(BaseSettings):
    REDIS_URL: str = Field(default="redis://localhost:6379", description="Redis connection URL")
    FAISS_INDEX_PATH: str = Field(default="models/faiss.index", description="Path to FAISS index file")
    RANKER_MODEL_PATH: str = Field(default="models/ranker.lgbm", description="Path to LightGBM ranker model")
    EMBEDDING_MODEL_PATH: str = Field(default="models/two_tower.pt", description="Path to two-tower PyTorch model")
    TOP_K_CANDIDATES: int = Field(default=500, description="Number of ANN candidates to retrieve")
    TOP_K_RESULTS: int = Field(default=20, description="Final number of recommendations to return")
    EMBEDDING_DIM: int = Field(default=64, description="Embedding dimension for two-tower model")
    DATA_DIR: str = Field(default="data/ml-1m", description="Directory containing MovieLens data files")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    API_HOST: str = Field(default="0.0.0.0", description="FastAPI host")
    API_PORT: int = Field(default=8000, description="FastAPI port")
    MODEL_VERSION: str = Field(default="1.0.0", description="Model version string")
    CACHE_TTL_SECONDS: int = Field(default=300, description="Redis cache TTL for recommendation results")
    FEATURE_CACHE_TTL_SECONDS: int = Field(default=3600, description="Redis cache TTL for feature data")
    N_NEGATIVES: int = Field(default=4, description="Number of negative samples per positive for training")
    FAISS_N_LISTS: int = Field(default=100, description="Number of IVF lists for FAISS index")
    FAISS_N_PROBE: int = Field(default=10, description="Number of IVF lists to probe at search time")
    TRAIN_EPOCHS: int = Field(default=10, description="Number of training epochs for two-tower model")
    BATCH_SIZE: int = Field(default=1024, description="Training batch size")
    LEARNING_RATE: float = Field(default=1e-3, description="Learning rate for two-tower training")
    LGBM_NUM_LEAVES: int = Field(default=63, description="LightGBM number of leaves")
    LGBM_N_ESTIMATORS: int = Field(default=500, description="LightGBM number of boosting rounds")
    LGBM_LEARNING_RATE: float = Field(default=0.05, description="LightGBM learning rate")
    SKEW_KL_THRESHOLD: float = Field(default=0.1, description="KL divergence threshold for skew detection")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()
