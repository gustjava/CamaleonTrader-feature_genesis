"""
Configuration management for Dynamic Stage 0 pipeline.

This module handles loading and validation of configuration from YAML files
with environment variable substitution.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration container for the Dynamic Stage 0 pipeline."""
    
    database: Dict[str, Any]
    r2: Dict[str, Any]
    dask: Dict[str, Any]
    features: Dict[str, Any]
    processing: Dict[str, Any]
    logging: Dict[str, Any]
    monitoring: Dict[str, Any]
    output: Dict[str, Any]
    development: Dict[str, Any]
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create Config instance from dictionary."""
        return cls(
            database=config_dict.get('database', {}),
            r2=config_dict.get('r2', {}),
            dask=config_dict.get('dask', {}),
            features=config_dict.get('features', {}),
            processing=config_dict.get('processing', {}),
            logging=config_dict.get('logging', {}),
            monitoring=config_dict.get('monitoring', {}),
            output=config_dict.get('output', {}),
            development=config_dict.get('development', {})
        )
    
    def get_database_url(self) -> str:
        """Generate SQLAlchemy database URL."""
        db = self.database
        return (
            f"mysql+pymysql://{db['username']}:{db['password']}@"
            f"{db['host']}:{db['port']}/{db['database']}?charset={db['charset']}"
        )
    
    def get_r2_storage_options(self) -> Dict[str, Any]:
        """Generate storage options for R2/S3 connectivity."""
        return {
            'key': self.r2['access_key'],
            'secret': self.r2['secret_key'],
            'endpoint_url': self.r2['endpoint_url'],
            'region': self.r2['region']
        }


def substitute_environment_variables(value: Any) -> Any:
    """Recursively substitute environment variables in configuration values."""
    if isinstance(value, str):
        # Handle ${VAR:default} pattern
        if value.startswith('${') and value.endswith('}'):
            var_part = value[2:-1]  # Remove ${ and }
            if ':' in var_part:
                var_name, default = var_part.split(':', 1)
                return os.getenv(var_name, default)
            else:
                return os.getenv(var_part, '')
        return value
    elif isinstance(value, dict):
        return {k: substitute_environment_variables(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [substitute_environment_variables(item) for item in value]
    else:
        return value


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file with environment variable substitution.
    
    Args:
        config_path: Path to configuration file. If None, uses default location.
        
    Returns:
        Config instance with loaded configuration.
        
    Raises:
        FileNotFoundError: If configuration file doesn't exist.
        yaml.YAMLError: If configuration file is invalid YAML.
    """
    if config_path is None:
        # Default to config/config.yaml relative to this file
        config_path = Path(__file__).parent / "config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # Substitute environment variables
        config_dict = substitute_environment_variables(config_dict)
        
        # Create Config instance
        config = Config.from_dict(config_dict)
        
        logger.info(f"Configuration loaded from {config_path}")
        return config
        
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file {config_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        raise


def validate_config(config: Config) -> bool:
    """
    Validate configuration for required fields and values.
    
    Args:
        config: Configuration to validate.
        
    Returns:
        True if configuration is valid.
        
    Raises:
        ValueError: If configuration is invalid.
    """
    errors = []
    
    # Validate database configuration
    required_db_fields = ['host', 'port', 'database', 'username', 'password']
    for field in required_db_fields:
        if not config.database.get(field):
            errors.append(f"Database configuration missing required field: {field}")
    
    # Validate R2 configuration
    required_r2_fields = ['account_id', 'access_key', 'secret_key', 'bucket_name']
    for field in required_r2_fields:
        if not config.r2.get(field):
            errors.append(f"R2 configuration missing required field: {field}")
    
    # Validate Dask configuration
    if config.dask.get('gpus_per_worker', 0) <= 0:
        errors.append("Dask configuration: gpus_per_worker must be > 0")
    
    # Validate feature configuration
    if not config.features.get('rolling_corr', {}).get('windows'):
        errors.append("Feature configuration: rolling_corr.windows is required")
    
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info("Configuration validation passed")
    return True


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance, loading if necessary."""
    global _config
    if _config is None:
        _config = load_config()
        validate_config(_config)
    return _config


def set_config(config: Config) -> None:
    """Set global configuration instance."""
    global _config
    _config = config
