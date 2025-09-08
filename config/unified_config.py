"""
Unified Configuration Management for Dynamic Stage 0 Pipeline

This module provides a consolidated configuration system that combines
the functionality of config.py and settings.py into a single, well-organized
configuration management system.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    host: str = "localhost"
    port: int = 3306
    database: str = "camaleon"
    username: str = "root"
    password: str = ""
    charset: str = "utf8mb4"
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    
    def get_url(self) -> str:
        """Generate SQLAlchemy database URL."""
        return (
            f"mysql+pymysql://{self.username}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}?charset={self.charset}"
        )


@dataclass
class R2Config:
    """Cloudflare R2 storage configuration."""
    account_id: str = ""
    access_key: str = ""
    secret_key: str = ""
    bucket_name: str = "camaleon-data"
    endpoint_url: str = "https://account_id.r2.cloudflarestorage.com"
    region: str = "auto"
    
    def get_storage_options(self) -> Dict[str, Any]:
        """Generate storage options for R2/S3 connectivity."""
        return {
            'key': self.access_key,
            'secret': self.secret_key,
            'endpoint_url': self.endpoint_url,
            'region': self.region
        }


@dataclass
class DaskConfig:
    """Dask-CUDA cluster configuration."""
    gpus_per_worker: int = 1
    threads_per_worker: int = 1
    memory_limit: str = "8GB"
    rmm_pool_size: str = "8GB"
    rmm_initial_pool_size: str = "8GB"
    rmm_maximum_pool_size: str = "16GB"
    # New: proportional pool sizing (fractions of device total). If > 0, these override fixed sizes.
    rmm_pool_fraction: float = 0.0                 # e.g., 0.60 -> 60% of device total
    rmm_initial_pool_fraction: float = 0.0         # e.g., 0.50 -> 50% of rmm_pool (or total)
    rmm_maximum_pool_fraction: float = 0.0         # optional cap; if 0.0, default safety cap is used
    spilling_enabled: bool = True
    spilling_target: float = 0.9
    spilling_max_spill: str = "32GB"
    memory_target_fraction: float = 0.8
    memory_spill_fraction: float = 0.9
    local_directory: str = "/tmp/dask-worker-space"
    protocol: str = "ucx"
    enable_tcp_over_ucx: bool = True
    enable_infiniband: bool = False
    enable_nvlink: bool = True


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    rolling_windows: List[int] = field(default_factory=lambda: [10, 20, 50, 100, 200])
    rolling_min_periods: int = 1
    rolling_corr: Dict[str, Any] = field(default_factory=lambda: {
        'windows': [15, 30, 60],
        'min_periods': 1
    })
    frac_diff: Dict[str, Any] = field(default_factory=lambda: {
        'd_values': [0.1, 0.2, 0.3, 0.4, 0.5],
        'threshold': 1e-5,
        'max_lag': 1000
    })
    baxter_king: Dict[str, Any] = field(default_factory=lambda: {
        'low_freq': 6,
        'high_freq': 32,
        'k': 12
    })
    # New nested feature engineering section
    feature_engineering: Dict[str, Any] = field(default_factory=dict)
    garch: Dict[str, Any] = field(default_factory=lambda: {
        'p': 1,
        'q': 1,
        'max_iter': 1000,
        'tolerance': 1e-6,
        'max_samples': 10000,
        'min_price_rows': 200,
        'min_return_rows': 100,
        'log_price': True,
    })
    distance_corr: Dict[str, Any] = field(default_factory=lambda: {
        'max_samples': 10000
    })
    emd: Dict[str, Any] = field(default_factory=lambda: {
        'max_imfs': 10,
        'tolerance': 1e-8,
        'max_iterations': 100
    })
    # Legacy fields for backward compatibility
    frac_diff_values: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5])
    frac_diff_threshold: float = 1e-5
    frac_diff_max_lag: int = 1000
    baxter_king_low_freq: int = 6
    baxter_king_high_freq: int = 32
    baxter_king_k: int = 12
    garch_p: int = 1
    garch_q: int = 1
    garch_max_iter: int = 1000
    garch_tolerance: float = 1e-6
    garch_max_samples: int = 10000
    garch_min_price_rows: int = 200
    garch_min_return_rows: int = 100
    garch_log_price: bool = True
    distance_corr_max_samples: int = 10000
    distance_corr_tile_size: int = 2048
    # Selection stage (Stage 1) and dCor extras
    selection_target_column: str = "y_ret_1m"
    selection_target_columns: List[str] = field(default_factory=list)
    dcor_top_k: int = 50
    dcor_include_permutation: bool = True
    dcor_permutations: int = 100
    selection_max_rows: int = 100000
    vif_threshold: float = 5.0
    mi_threshold: float = 0.3
    stage3_top_n: int = 50
    # Stage 3 wrappers (LightGBM tuning)
    stage3_task: str = "auto"  # auto|regression|classification
    stage3_random_state: int = 42
    stage3_lgbm_enabled: bool = True
    stage3_lgbm_num_leaves: int = 31
    stage3_lgbm_max_depth: int = -1
    stage3_lgbm_n_estimators: int = 200
    stage3_lgbm_learning_rate: float = 0.05
    stage3_lgbm_feature_fraction: float = 0.8
    stage3_lgbm_bagging_fraction: float = 0.8
    stage3_lgbm_bagging_freq: int = 0
    stage3_lgbm_early_stopping_rounds: int = 0
    stage3_use_gpu: bool = True
    stage3_wrapper_backend: str = "xgb_gpu"  # lgbm|xgb_gpu
    # Stage 3 embedded selector (SelectFromModel)
    stage3_selector_method: str = "wrappers"  # wrappers|selectfrommodel
    stage3_importance_type: str = "gain"      # gain|split|weight (backend-dependent)
    stage3_importance_threshold: str = "median"  # 'median' or float as string
    stage3_save_importances_format: str = "json"  # json|parquet
    # Stage 3 CV (leakage-safe) selection
    stage3_cv_splits: int = 3               # TimeSeriesSplit folds for aggregated importances (>=2 -> enabled)
    stage3_cv_min_train: int = 200          # Minimum train rows per split
    # Stage 1 retention controls
    dcor_min_threshold: float = 0.0
    dcor_min_percentile: float = 0.0  # 0.0..1.0
    stage1_top_n: int = 0  # 0 = no cap
    # Additional Stage 1 gates
    correlation_min_threshold: float = 0.0  # abs(Pearson)
    pvalue_max_alpha: float = 1.0           # F-test p-value
    # Stage 0 ADF
    adf_alpha: float = 0.05
    # dCor fast 1D approximation and permutation stage
    dcor_fast_1d_enabled: bool = True
    dcor_fast_1d_bins: int = 2048
    dcor_permutation_top_k: int = 20   # 0 = disabled, else apply on top-K by dCor
    dcor_pvalue_alpha: float = 0.05
    # Stage 1 batching for progress logging
    dcor_batch_size: int = 64
    # Stage 1 rolling dCor (new)
    stage1_rolling_enabled: bool = True
    stage1_rolling_window: int = 2000
    stage1_rolling_step: int = 500
    stage1_rolling_min_periods: int = 200
    # New: minimum pairwise valid (non-NaN) observations required per rolling window
    stage1_rolling_min_valid_pairs: int = 200
    stage1_rolling_max_rows: int = 20000
    stage1_rolling_max_windows: int = 20
    stage1_agg: str = "median"  # one of: mean, median, min, max, p25, p75
    stage1_use_rolling_scores: bool = True
    # Logging controls for Stage 1
    stage1_log_top_k: int = 20
    stage1_log_all_scores: bool = False
    # Stage 1 quality gates
    stage1_min_coverage_ratio: float = 0.30   # min fraction of valid (target & feature finite) pairs
    stage1_min_variance: float = 1e-12        # variance threshold to avoid constant features
    stage1_min_unique_values: int = 2         # require at least 2 unique finite values
    stage1_min_rolling_windows: int = 5       # min finite rolling windows required per feature
    # Dataset schema/feature gating (leakage control)
    dataset_target_columns: List[str] = field(default_factory=list)
    dataset_target_prefixes: List[str] = field(default_factory=list)
    feature_allowlist: List[str] = field(default_factory=list)
    feature_allow_prefixes: List[str] = field(default_factory=list)
    feature_denylist: List[str] = field(default_factory=lambda: [
        "y_tick_volume",
        "y_total_volume",
        "y_minutes_since_open",
    ])
    feature_deny_prefixes: List[str] = field(default_factory=lambda: ['y_ret_fwd_'])
    feature_deny_regex: List[str] = field(default_factory=list)
    metrics_prefixes: List[str] = field(default_factory=lambda: ['dcor_', 'dcor_roll_', 'dcor_pvalue_', 'stage1_', 'cpcv_'])
    # Selection protection (always keep)
    always_keep_features: List[str] = field(default_factory=list)
    always_keep_prefixes: List[str] = field(default_factory=list)
    # Stage 1 visibility and debugging
    stage1_broadcast_scores: bool = False     # add dcor_* columns to the frame
    stage1_broadcast_rolling: bool = False    # add dcor_roll_* and cnt_* columns
    debug_write_artifacts: bool = True        # persist JSON artifacts per stage
    artifacts_dir: str = "artifacts"         # subfolder under output_path
    # Stage 2 MI clustering (scalable)
    mi_cluster_enabled: bool = True
    mi_cluster_method: str = "agglo"  # only 'agglo' supported now
    mi_cluster_threshold: float = 0.3
    mi_max_candidates: int = 400
    mi_chunk_size: int = 128
    # CPCV controls
    cpcv_enabled: bool = True
    cpcv_n_groups: int = 6
    cpcv_k_leave_out: int = 2
    cpcv_purge: int = 0
    cpcv_embargo: int = 0
    # Stage 4 stability selection
    stage4_enabled: bool = False
    stage4_n_bootstrap: int = 30
    stage4_block_size: int = 5000
    stage4_stability_threshold: float = 0.7
    stage4_plot: bool = True
    stage4_random_state: int = 42
    stage4_bootstrap_method: str = "block"  # block|tssplit
    emd_max_imfs: int = 10
    emd_tolerance: float = 1e-8
    emd_max_iterations: int = 100
    # Fractional diff cache/tuning (new)
    fracdiff_cache_max_entries: int = 32
    fracdiff_partition_threshold: int = 4096
    # Stationarization helpers (basic rolling features)
    station_basic_rolling_enabled: bool = False
    # Rolling correlation pair selection strategy: 'first' | 'dcor'
    rolling_corr_pair_selection: str = "first"
    # Column filtering controls
    drop_metric_columns_on_save: bool = True
    drop_metric_columns_on_intermediate: bool = True
    # Explicit candidate selection for Stage 1 (no regex)
    station_candidates_include: List[str] = field(default_factory=list)
    station_candidates_exclude: List[str] = field(default_factory=list)
    # Drop original column after creating its stationary counterpart (FFD)
    drop_original_after_transform: bool = False
    # Drop non-retained candidates after Stage 1 (based on dCor gates)
    drop_nonretained_after_stage1: bool = False
    # GPU usage control
    force_gpu_usage: bool = True  # Force GPU usage for all stages
    gpu_fallback_enabled: bool = False  # Disable CPU fallbacks to force GPU usage
    # Session auto-mask configuration
    session_auto_mask: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'window_rows': 120,
        'min_valid': 90
    })
    # Session configuration for external drivers
    sessions: Dict[str, Any] = field(default_factory=dict)
    # Index gap imputation configuration (Kalman session-aware)
    index_imputation: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class SelectionStage3Config:
    """Stage 3 multivariate selection configuration."""
    model: str = "lgbm"  # lgbm
    task: str = "auto"   # auto|regression|classification  
    importance_threshold: str = "median"  # median|float
    use_gpu: bool = True
    random_state: int = 42
    n_estimators: int = 300
    learning_rate: float = 0.05
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    early_stopping_rounds: int = 0

@dataclass
class SelectionConfig:
    """Feature selection configuration organized by stage."""
    stage3: SelectionStage3Config = field(default_factory=SelectionStage3Config)

@dataclass
class ProcessingConfig:
    """Processing pipeline configuration."""
    batch_size: int = 1000
    chunk_size: str = "100MB"
    max_workers: int = 4
    timeout: int = 1800  # 30 minutes
    continue_on_error: bool = False
    fail_fast: bool = True


@dataclass
class ValidationConfig:
    """Data validation configuration (input/output/series quality)."""
    validate_input_data: bool = True
    required_columns: List[str] = field(default_factory=lambda: ["timestamp", "open", "high", "low", "close", "volume"])
    expected_dtypes: Dict[str, str] = field(default_factory=lambda: {
        "timestamp": "datetime64[ns]",
        "open": "float32",
        "high": "float32",
        "low": "float32",
        "close": "float32",
        "volume": "float32",
    })
    validate_output_data: bool = True
    max_nan_percentage: float = 50.0
    check_infinite_values: bool = True
    # Series-quality thresholds used by StationarizationEngine
    min_rows: int = 100
    max_missing_percentage: float = 20.0
    outlier_threshold: float = 3.0


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "pipeline.log"
    max_size: str = "100MB"
    backup_count: int = 5


@dataclass
class MonitoringConfig:
    """Monitoring and metrics configuration."""
    metrics_enabled: bool = True
    dashboard_port: int = 8787
    health_check_interval: int = 30


@dataclass
class MemoryConfig:
    """Advanced memory management configuration."""
    # RMM configuration (GB)
    rmm_pool_size_gb: float = 8.0
    rmm_initial_pool_size_gb: float = 4.0
    rmm_maximum_pool_size_gb: float = 16.0
    rmm_memory_target_fraction: float = 0.8
    rmm_memory_spill_threshold: float = 0.9

    # Chunked processing (rows)
    chunk_size: int = 10000
    chunk_overlap: int = 1000
    max_memory_gb: float = 8.0

    # Spilling
    enable_spilling: bool = True
    spill_to_disk: bool = True
    spill_directory: str = "/tmp/gpu_spill"

    # Monitoring
    monitor_memory: bool = True
    alert_threshold: float = 0.9
    check_interval: int = 10


@dataclass
class OutputConfig:
    """Output configuration."""
    output_path: str = "./output"
    compression: str = "lz4"
    format: str = "feather"
    version: int = 2
    # Intermediate checkpoints
    save_intermediate_per_engine: bool = False
    intermediate_format: str = "parquet"  # parquet|feather
    intermediate_compression: str = "zstd"
    intermediate_version: int = 2


@dataclass
class DevelopmentConfig:
    """Development and debugging configuration."""
    debug_mode: bool = False
    clean_existing_output: bool = False
    force_reprocessing: bool = False
    enable_profiling: bool = False
    log_memory_usage: bool = True
    validate_data: bool = True


@dataclass
class PipelineEngineConfig:
    """Configuration for individual pipeline engines."""
    enabled: bool = True
    order: int = 999
    description: str = ""
    timeout: int = 300
    retry_count: int = 3


@dataclass
class PipelineConfig:
    """Pipeline engine configuration."""
    engines: Dict[str, PipelineEngineConfig] = field(default_factory=lambda: {
        'index_gap_imputation': PipelineEngineConfig(enabled=False, order=0, description="Kalman session-aware index gap imputation"),
        'stationarization': PipelineEngineConfig(enabled=True, order=1, description="Stationarization techniques"),
        'feature_engineering': PipelineEngineConfig(enabled=True, order=2, description="Early feature engineering (e.g., BK)"),
        'garch_models': PipelineEngineConfig(enabled=True, order=3, description="GARCH volatility modeling"),
        'statistical_tests': PipelineEngineConfig(enabled=True, order=4, description="Statistical tests and analysis"),
    })


@dataclass
class ErrorHandlingConfig:
    """Error handling configuration."""
    continue_on_error: bool = False
    max_retries: int = 3
    retry_delay: int = 5
    critical_error_threshold: int = 1


@dataclass
class UnifiedConfig:
    """
    Unified configuration container for the Dynamic Stage 0 pipeline.
    
    This class consolidates all configuration settings into a single,
    well-organized structure with proper type hints and validation.
    """
    
    # Core configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    r2: R2Config = field(default_factory=R2Config)
    dask: DaskConfig = field(default_factory=DaskConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    development: DevelopmentConfig = field(default_factory=DevelopmentConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    error_handling: ErrorHandlingConfig = field(default_factory=ErrorHandlingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    
    def validate(self) -> bool:
        """
        Validate the configuration.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Validate database configuration
            if not self.database.host or not self.database.database:
                logger.error("Database host and database name are required")
                return False
            
            # Validate R2 configuration
            if not self.r2.access_key or not self.r2.secret_key:
                logger.warning("R2 access key and secret key are not configured")
            
            # Validate Dask configuration
            if self.dask.gpus_per_worker <= 0:
                logger.error("GPUs per worker must be greater than 0")
                return False
            
            # Validate feature configuration
            if not self.features.rolling_windows:
                logger.error("At least one rolling window must be specified")
                return False
            
            # Validate output configuration
            if not self.output.output_path:
                logger.error("Output path must be specified")
                return False

            # Validate memory configuration
            if self.memory.chunk_size <= 0:
                logger.error("Memory.chunk_size must be > 0")
                return False
            if self.memory.chunk_overlap < 0:
                logger.error("Memory.chunk_overlap must be >= 0")
                return False
            
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'database': self.database.__dict__,
            'r2': self.r2.__dict__,
            'dask': self.dask.__dict__,
            'features': self.features.__dict__,
            'processing': self.processing.__dict__,
            'logging': self.logging.__dict__,
            'monitoring': self.monitoring.__dict__,
            'memory': self.memory.__dict__,
            'output': self.output.__dict__,
            'development': self.development.__dict__,
            'pipeline': {k: v.__dict__ for k, v in self.pipeline.engines.items()},
            'error_handling': self.error_handling.__dict__
        }


def substitute_environment_variables(value: Any) -> Any:
    """
    Recursively substitute environment variables in configuration values.
    
    Supports the pattern ${VAR:default} where default is optional.
    """
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


def load_config_from_dict(config_dict: Dict[str, Any]) -> UnifiedConfig:
    """
    Load configuration from dictionary with environment variable substitution.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        UnifiedConfig instance
    """
    # Substitute environment variables
    config_dict = substitute_environment_variables(config_dict)
    
    # Create configuration objects
    database = DatabaseConfig(**config_dict.get('database', {}))
    r2 = R2Config(**config_dict.get('r2', {}))
    dask = DaskConfig(**config_dict.get('dask', {}))
    features = FeatureConfig(**config_dict.get('features', {}))
    processing = ProcessingConfig(**config_dict.get('processing', {}))
    logging = LoggingConfig(**config_dict.get('logging', {}))
    monitoring = MonitoringConfig(**config_dict.get('monitoring', {}))
    memory = MemoryConfig(**config_dict.get('memory', {}))
    output = OutputConfig(**config_dict.get('output', {}))
    development = DevelopmentConfig(**config_dict.get('development', {}))
    error_handling = ErrorHandlingConfig(**config_dict.get('error_handling', {}))
    validation = ValidationConfig(**config_dict.get('validation', {}))
    
    # Handle pipeline engines configuration
    pipeline_engines = {}
    engines_config = config_dict.get('pipeline', {}).get('engines', {})
    for engine_name, engine_config in engines_config.items():
        pipeline_engines[engine_name] = PipelineEngineConfig(**engine_config)
    
    pipeline = PipelineConfig(engines=pipeline_engines)
    
    return UnifiedConfig(
        database=database,
        r2=r2,
        dask=dask,
        features=features,
        processing=processing,
        logging=logging,
        monitoring=monitoring,
        memory=memory,
        output=output,
        development=development,
        pipeline=pipeline,
        error_handling=error_handling,
        validation=validation
    )


def load_config_from_file(config_path: Optional[str] = None) -> UnifiedConfig:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file. If None, uses default location.
        
    Returns:
        UnifiedConfig instance
        
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
        
        if config_dict is None:
            raise ValueError("Configuration file is empty")
        
        config = load_config_from_dict(config_dict)
        
        # Validate configuration
        if not config.validate():
            raise ValueError("Configuration validation failed")
        
        logger.info(f"Configuration loaded successfully from {config_path}")
        return config
        
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise


@lru_cache(maxsize=1)
def get_unified_config() -> UnifiedConfig:
    """
    Get the unified configuration instance.
    
    This function is cached to ensure the configuration is loaded only once.
    
    Returns:
        UnifiedConfig instance
    """
    return load_config_from_file()


# Backward compatibility functions
def get_config():
    """Backward compatibility function for get_config()."""
    return get_unified_config()


def get_settings():
    """Backward compatibility function for get_settings()."""
    return get_unified_config()
