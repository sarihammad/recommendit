#!/usr/bin/env python3
"""Script for serving the recommendation system API."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import structlog
import uvicorn
from pathlib import Path
import argparse
import signal
import time
from typing import Optional

from recsys.service.config import config
from recsys.service.api import app

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

class GracefulShutdown:
    """Handle graceful shutdown of the service."""
    
    def __init__(self):
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        logger.info("Shutdown signal received", signal=signum)
        self.shutdown_requested = True
    
    def is_shutdown_requested(self):
        return self.shutdown_requested

def check_artifacts(domain: str) -> bool:
    """Check if required artifacts exist for a domain."""
    artifacts_dir = Path(config.ARTIFACTS_DIR) / domain
    
    required_files = [
        "als_model.pkl",
        "item_similarity_matrix.npy", 
        "faiss.index",
        "lgbm_model.txt",
        "popularity_items.pkl"
    ]
    
    missing_files = []
    for file_name in required_files:
        file_path = artifacts_dir / file_name
        if not file_path.exists():
            missing_files.append(file_name)
    
    if missing_files:
        logger.warning("Missing artifacts for domain", 
                      domain=domain, 
                      missing_files=missing_files)
        return False
    
    logger.info("All artifacts found for domain", domain=domain)
    return True

def check_redis_connection() -> bool:
    """Check Redis connection."""
    try:
        from recsys.service.cache import cache
        if cache.health_check():
            logger.info("Redis connection successful")
            return True
        else:
            logger.error("Redis health check failed")
            return False
    except Exception as e:
        logger.error("Failed to connect to Redis", error=str(e))
        return False

def validate_environment():
    """Validate the environment before starting the service."""
    logger.info("Validating environment")
    
    # Check artifacts for both domains
    books_ok = check_artifacts('books')
    movies_ok = check_artifacts('movies')
    
    if not books_ok and not movies_ok:
        logger.error("No valid artifacts found for any domain")
        return False
    
    # Check Redis connection
    redis_ok = check_redis_connection()
    if not redis_ok:
        logger.warning("Redis connection failed - some features may not work")
    
    # Check data directories
    data_dir = Path(config.DATA_DIR)
    if not data_dir.exists():
        logger.warning("Data directory not found", data_dir=str(data_dir))
    
    # Check artifacts directory
    artifacts_dir = Path(config.ARTIFACTS_DIR)
    if not artifacts_dir.exists():
        logger.warning("Artifacts directory not found", artifacts_dir=str(artifacts_dir))
    
    logger.info("Environment validation completed")
    return True

def start_server(host: str = None, port: int = None, reload: bool = False, workers: int = 1):
    """Start the FastAPI server."""
    host = host or config.API_HOST
    port = port or config.API_PORT
    
    logger.info("Starting recommendation service", 
               host=host, 
               port=port, 
               reload=reload, 
               workers=workers)
    
    # Validate environment
    if not validate_environment():
        logger.error("Environment validation failed")
        sys.exit(1)
    
    # Configure uvicorn
    uvicorn_config = {
        "app": "recsys.service.api:app",
        "host": host,
        "port": port,
        "reload": reload,
        "workers": workers if not reload else 1,
        "log_level": config.LOG_LEVEL.lower(),
        "access_log": True,
        "use_colors": True
    }
    
    # Add SSL configuration if needed
    ssl_keyfile = os.environ.get("SSL_KEYFILE")
    ssl_certfile = os.environ.get("SSL_CERTFILE")
    
    if ssl_keyfile and ssl_certfile:
        uvicorn_config["ssl_keyfile"] = ssl_keyfile
        uvicorn_config["ssl_certfile"] = ssl_certfile
        logger.info("SSL enabled", keyfile=ssl_keyfile, certfile=ssl_certfile)
    
    try:
        # Start server
        uvicorn.run(**uvicorn_config)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error("Server failed to start", error=str(e))
        sys.exit(1)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Start the recommendation system API")
    
    parser.add_argument(
        "--host", 
        type=str, 
        default=config.API_HOST,
        help=f"Host to bind to (default: {config.API_HOST})"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=config.API_PORT,
        help=f"Port to bind to (default: {config.API_PORT})"
    )
    
    parser.add_argument(
        "--reload", 
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    parser.add_argument(
        "--workers", 
        type=int, 
        default=1,
        help="Number of worker processes (default: 1)"
    )
    
    parser.add_argument(
        "--check-only", 
        action="store_true",
        help="Only check environment and artifacts, don't start server"
    )
    
    parser.add_argument(
        "--validate", 
        action="store_true",
        help="Validate environment and exit"
    )
    
    args = parser.parse_args()
    
    if args.validate:
        logger.info("Running environment validation")
        if validate_environment():
            logger.info("Environment validation passed")
            sys.exit(0)
        else:
            logger.error("Environment validation failed")
            sys.exit(1)
    
    if args.check_only:
        logger.info("Running artifact check")
        books_ok = check_artifacts('books')
        movies_ok = check_artifacts('movies')
        
        if books_ok or movies_ok:
            logger.info("Artifact check passed")
            sys.exit(0)
        else:
            logger.error("Artifact check failed")
            sys.exit(1)
    
    # Start the server
    start_server(
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers
    )

if __name__ == "__main__":
    main() 