"""
Settings module for Dynamic Stage 0 pipeline.

This module provides type-safe access to configuration values with sensible defaults.
"""

from typing import List, Optional
from dataclasses import dataclass
from .config import get_config


@dataclass
class DatabaseSettings:
    """Database connection settings."""
    host: str
    port: int
    database: str
    username: str
    password: str
    charset: str
    pool_size: int
    max_overflow: int
    pool_timeout: int
    pool_recycle: int


@dataclass
class R2Settings:
    """Cloudflare R2 storage settings."""
    account_id: str
    access_key: str
    secret_key: str
    bucket_name: str
    endpoint_url: str
    region: str


@dataclass
class DaskSettings:
    """Dask-CUDA cluster settings."""
    gpus_per_worker: int
    threads_per_worker: int
    memory_limit: str
    rmm_pool_size: str
    rmm_initial_pool_size: str
    rmm_maximum_pool_size: str
    spilling_enabled: bool
    spilling_target: float
    spilling_max_spill: str


@dataclass
class FeatureSettings:
    """Feature engineering settings."""
    rolling_windows: List[int]
    rolling_min_periods: int
    frac_diff_values: List[float]
    frac_diff_threshold: float
    frac_diff_max_lag: int
    baxter_king_low_freq: int
    baxter_king_high_freq: int
    baxter_king_k: int
    garch_p: int
    garch_q: int
    garch_max_iter: int
    garch_tolerance: float
    distance_corr_max_samples: int
    emd_max_imfs: int
    emd_tolerance: float
    emd_max_iterations: int


@dataclass
class ProcessingSettings:
    """Processing pipeline settings."""
    batch_size: int
    chunk_size: str
    max_workers: int
    timeout: int


@dataclass
class LoggingSettings:
    """Logging configuration settings."""
    level: str
    format: str
    file: str
    max_size: str
    backup_count: int


@dataclass
class MonitoringSettings:
    """Monitoring and metrics settings."""
    metrics_enabled: bool
    dashboard_port: int
    health_check_interval: int


@dataclass
class OutputSettings:
    """Output configuration settings."""
    format: str
    compression: str
    partition_size: str
    output_path: str


@dataclass
class DevelopmentSettings:
    """Development and debugging settings."""
    debug: bool
    profile: bool
    test_mode: bool


@dataclass
class Settings:
    """Complete settings container for the Dynamic Stage 0 pipeline."""
    
    database: DatabaseSettings
    r2: R2Settings
    dask: DaskSettings
    features: FeatureSettings
    processing: ProcessingSettings
    logging: LoggingSettings
    monitoring: MonitoringSettings
    output: OutputSettings
    development: DevelopmentSettings
    
    @classmethod
    def from_config(cls) -> 'Settings':
        """Create Settings instance from loaded configuration."""
        config = get_config()
        
        # Database settings
        db_config = config.database
        database = DatabaseSettings(
            host=db_config.get('host', 'localhost'),
            port=db_config.get('port', 3306),
            database=db_config.get('database', 'dynamic_stage0_db'),
            username=db_config.get('username', 'root'),
            password=db_config.get('password', ''),
            charset=db_config.get('charset', 'utf8mb4'),
            pool_size=db_config.get('pool_size', 10),
            max_overflow=db_config.get('max_overflow', 20),
            pool_timeout=db_config.get('pool_timeout', 30),
            pool_recycle=db_config.get('pool_recycle', 3600)
        )
        
        # R2 settings
        r2_config = config.r2
        r2 = R2Settings(
            account_id=r2_config.get('account_id', ''),
            access_key=r2_config.get('access_key', ''),
            secret_key=r2_config.get('secret_key', ''),
            bucket_name=r2_config.get('bucket_name', ''),
            endpoint_url=r2_config.get('endpoint_url', ''),
            region=r2_config.get('region', 'auto')
        )
        
        # Dask settings
        dask_config = config.dask
        rmm_config = dask_config.get('rmm', {})
        spilling_config = dask_config.get('spilling', {})
        dask = DaskSettings(
            gpus_per_worker=dask_config.get('gpus_per_worker', 1),
            threads_per_worker=dask_config.get('threads_per_worker', 1),
            memory_limit=dask_config.get('memory_limit', 'auto'),
            rmm_pool_size=rmm_config.get('pool_size', '24GB'),
            rmm_initial_pool_size=rmm_config.get('initial_pool_size', '12GB'),
            rmm_maximum_pool_size=rmm_config.get('maximum_pool_size', '48GB'),
            spilling_enabled=spilling_config.get('enabled', True),
            spilling_target=spilling_config.get('target', 0.8),
            spilling_max_spill=spilling_config.get('max_spill', '32GB')
        )
        
        # Feature settings
        features_config = config.features
        rolling_config = features_config.get('rolling_corr', {})
        frac_diff_config = features_config.get('frac_diff', {})
        bk_config = features_config.get('baxter_king', {})
        garch_config = features_config.get('garch', {})
        dc_config = features_config.get('distance_corr', {})
        emd_config = features_config.get('emd', {})
        
        features = FeatureSettings(
            rolling_windows=rolling_config.get('windows', [20, 50, 100, 200]),
            rolling_min_periods=rolling_config.get('min_periods', 10),
            frac_diff_values=frac_diff_config.get('d_values', [0.1, 0.3, 0.5, 0.7, 0.9]),
            frac_diff_threshold=frac_diff_config.get('threshold', 1e-5),
            frac_diff_max_lag=frac_diff_config.get('max_lag', 1000),
            baxter_king_low_freq=bk_config.get('low_freq', 6),
            baxter_king_high_freq=bk_config.get('high_freq', 32),
            baxter_king_k=bk_config.get('k', 12),
            garch_p=garch_config.get('p', 1),
            garch_q=garch_config.get('q', 1),
            garch_max_iter=garch_config.get('max_iter', 1000),
            garch_tolerance=garch_config.get('tolerance', 1e-6),
            distance_corr_max_samples=dc_config.get('max_samples', 10000),
            emd_max_imfs=emd_config.get('max_imfs', 10),
            emd_tolerance=emd_config.get('tolerance', 1e-10),
            emd_max_iterations=emd_config.get('max_iterations', 100)
        )
        
        # Processing settings
        processing_config = config.processing
        processing = ProcessingSettings(
            batch_size=processing_config.get('batch_size', 10000),
            chunk_size=processing_config.get('chunk_size', '256MB'),
            max_workers=processing_config.get('max_workers', 4),
            timeout=processing_config.get('timeout', 3600)
        )
        
        # Logging settings
        logging_config = config.logging
        logging = LoggingSettings(
            level=logging_config.get('level', 'INFO'),
            format=logging_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            file=logging_config.get('file', 'logs/dynamic_stage0.log'),
            max_size=logging_config.get('max_size', '100MB'),
            backup_count=logging_config.get('backup_count', 5)
        )
        
        # Monitoring settings
        monitoring_config = config.monitoring
        monitoring = MonitoringSettings(
            metrics_enabled=monitoring_config.get('metrics_enabled', True),
            dashboard_port=monitoring_config.get('dashboard_port', 8787),
            health_check_interval=monitoring_config.get('health_check_interval', 30)
        )
        
        # Output settings
        output_config = config.output
        output = OutputSettings(
            format=output_config.get('format', 'parquet'),
            compression=output_config.get('compression', 'snappy'),
            partition_size=output_config.get('partition_size', '1GB'),
            output_path=output_config.get('output_path', 'data/processed_features/')
        )
        
        # Development settings
        development_config = config.development
        development = DevelopmentSettings(
            debug=development_config.get('debug', False),
            profile=development_config.get('profile', False),
            test_mode=development_config.get('test_mode', False)
        )
        
        return cls(
            database=database,
            r2=r2,
            dask=dask,
            features=features,
            processing=processing,
            logging=logging,
            monitoring=monitoring,
            output=output,
            development=development
        )


def get_settings() -> Settings:
    """Get global settings instance."""
    return Settings.from_config()
