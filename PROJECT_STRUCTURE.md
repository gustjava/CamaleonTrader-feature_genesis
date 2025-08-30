# Dynamic Stage 0 - Project Structure

This document describes the complete project structure for the GPU-accelerated feature stationarization pipeline.

## Directory Structure

```
feature_selection/feature_genesis/
├── config/                     # Configuration management
│   ├── __init__.py
│   ├── config.yaml            # Main configuration file
│   ├── config.py              # Configuration loading and validation
│   └── settings.py            # Type-safe settings classes
├── orchestration/             # Pipeline orchestration
│   └── __init__.py
├── data_io/                   # Data input/output operations
│   └── __init__.py
├── features/                  # Feature engineering implementations
│   └── __init__.py
├── utils/                     # Utility functions and helpers
│   └── __init__.py
├── docker/                    # Docker and database files
│   └── init.sql              # Database schema initialization
├── Dockerfile                 # Main Docker image definition
├── docker-compose.yml         # Development environment setup
├── environment.yml            # Conda environment specification
├── requirements.txt           # Python dependencies
├── build.sh                   # Automated build script
├── test_gpu_setup.py          # GPU environment test
├── test_database.py           # Database connection test
├── README.md                  # Main project documentation
├── .dockerignore              # Docker build exclusions
└── PROJECT_STRUCTURE.md       # This file
```

## Module Descriptions

### 1. Configuration (`config/`)

**Purpose**: Centralized configuration management with environment variable support.

**Key Components**:
- `config.yaml`: Main configuration file with all pipeline parameters
- `config.py`: Configuration loading with environment variable substitution
- `settings.py`: Type-safe dataclasses for configuration access

**Features**:
- Environment variable substitution (e.g., `${MYSQL_HOST:localhost}`)
- Configuration validation
- Type-safe access to settings
- Support for database, R2, Dask, and feature parameters

### 2. Orchestration (`orchestration/`)

**Purpose**: Main pipeline orchestration and task management.

**Planned Components**:
- `pipeline.py`: Main DynamicStage0Pipeline class
- `cluster_manager.py`: Dask-CUDA cluster management
- `task_manager.py`: Task scheduling and state management

**Features**:
- Master-slave orchestration pattern
- Sequential processing of currency pairs
- Parallel feature computation within pairs
- Fail-fast error handling

### 3. Data I/O (`data_io/`)

**Purpose**: Data input/output operations and connectivity.

**Planned Components**:
- `r2_client.py`: Cloudflare R2 storage client
- `database.py`: MySQL database operations
- `data_loader.py`: High-performance data loading

**Features**:
- Optimized R2/S3 data loading
- Database state management
- Parquet file handling with GPU acceleration

### 4. Features (`features/`)

**Purpose**: GPU-accelerated feature engineering implementations.

**Planned Components**:
- `stationarization.py`: Stationarization techniques
- `statistical_tests.py`: Statistical tests (ADF, etc.)
- `signal_processing.py`: Signal processing filters
- `garch_models.py`: GARCH model implementations

**Features**:
- Native GPU implementations using cuDF/CuPy
- Fractional differentiation
- Baxter-King filters
- Distance correlation
- Empirical Mode Decomposition

### 5. Utilities (`utils/`)

**Purpose**: Common utilities and helper functions.

**Planned Components**:
- `logging.py`: Structured logging setup
- `gpu_utils.py`: GPU utility functions
- `validation.py`: Data validation utilities
- `metrics.py`: Performance metrics collection

**Features**:
- Structured logging with JSON format
- GPU memory management utilities
- Data validation and integrity checks
- Performance monitoring

### 6. Docker (`docker/`)

**Purpose**: Containerization and database setup.

**Components**:
- `init.sql`: Database schema initialization script

**Features**:
- Complete database schema with indexes
- Views for easy querying
- Sample data insertion (commented)
- Proper foreign key constraints

## Database Schema

### Tables

#### 1. `processing_tasks`
- **Purpose**: Source of truth for work to be done
- **Key Fields**:
  - `task_id`: Unique identifier (AUTO_INCREMENT)
  - `currency_pair`: Currency pair (e.g., 'EURUSD')
  - `r2_path`: Base path in R2 for data
  - `added_timestamp`: When task was added

#### 2. `feature_status`
- **Purpose**: Execution status and audit log
- **Key Fields**:
  - `status_id`: Unique identifier (AUTO_INCREMENT)
  - `task_id`: Foreign key to processing_tasks
  - `status`: ENUM('PENDING', 'RUNNING', 'COMPLETED', 'FAILED')
  - `start_time`, `end_time`: Execution timestamps
  - `hostname`: Instance identifier
  - `error_message`: Detailed error information

### Views

#### 1. `pending_tasks`
- Shows tasks that are pending, failed, or stuck running
- Includes status information and error messages

#### 2. `completed_tasks`
- Shows successfully completed tasks
- Includes processing time calculations

## Configuration Parameters

### Database Configuration
```yaml
database:
  host: ${MYSQL_HOST:localhost}
  port: ${MYSQL_PORT:3306}
  database: ${MYSQL_DATABASE:dynamic_stage0_db}
  username: ${MYSQL_USERNAME:root}
  password: ${MYSQL_PASSWORD:}
```

### R2 Configuration
```yaml
r2:
  account_id: ${R2_ACCOUNT_ID:}
  access_key: ${R2_ACCESS_KEY:}
  secret_key: ${R2_SECRET_KEY:}
  bucket_name: ${R2_BUCKET_NAME:}
```

### Dask-CUDA Configuration
```yaml
dask:
  gpus_per_worker: 1
  rmm:
    pool_size: "24GB"
    initial_pool_size: "12GB"
    maximum_pool_size: "48GB"
  spilling:
    enabled: true
    target: 0.8
    max_spill: "32GB"
```

### Feature Engineering Configuration
```yaml
features:
  rolling_corr:
    windows: [20, 50, 100, 200]
  frac_diff:
    d_values: [0.1, 0.3, 0.5, 0.7, 0.9]
  baxter_king:
    low_freq: 6
    high_freq: 32
    k: 12
```

## Environment Variables

### Required Environment Variables
- `MYSQL_HOST`: MySQL server hostname
- `MYSQL_PORT`: MySQL server port
- `MYSQL_DATABASE`: Database name
- `MYSQL_USERNAME`: Database username
- `MYSQL_PASSWORD`: Database password
- `R2_ACCOUNT_ID`: Cloudflare R2 account ID
- `R2_ACCESS_KEY`: R2 access key
- `R2_SECRET_KEY`: R2 secret key
- `R2_BUCKET_NAME`: R2 bucket name

### Optional Environment Variables
- `LOG_LEVEL`: Logging level (default: INFO)
- `DEBUG`: Enable debug mode (default: false)
- `TEST_MODE`: Enable test mode (default: false)

## Testing

### GPU Environment Test
```bash
python test_gpu_setup.py
```
Tests all GPU libraries (CuPy, cuDF, Dask-CUDA, cuSignal, cuML) and Python dependencies.

### Database Test
```bash
python test_database.py
```
Tests database connection, schema validation, and basic CRUD operations.

## Development Workflow

1. **Setup Environment**:
   ```bash
   ./build.sh
   docker-compose up -d
   ```

2. **Test Environment**:
   ```bash
   python test_gpu_setup.py
   python test_database.py
   ```

3. **Initialize Database**:
   ```bash
   mysql -h <host> -P <port> -u <user> -p < docker/init.sql
   ```

4. **Development**:
   - Access Jupyter Lab at `http://localhost:8888`
   - Edit code in mounted volumes
   - Monitor Dask dashboard at `http://localhost:8787`

## Production Deployment

### vast.ai Deployment
1. Build production Docker image
2. Configure environment variables
3. Deploy to vast.ai with GPU support
4. Initialize database schema
5. Run pipeline orchestration

### Monitoring
- Structured logging to files
- Dask dashboard for cluster monitoring
- Database views for task status
- Performance metrics collection

## Next Steps

1. **Implement Core Modules**:
   - Complete orchestration pipeline
   - Implement feature engineering algorithms
   - Add data I/O operations

2. **Add Monitoring**:
   - Structured logging implementation
   - Performance metrics collection
   - Health check endpoints

3. **Production Readiness**:
   - Error handling and recovery
   - Configuration validation
   - Performance optimization

4. **Testing**:
   - Unit tests for each module
   - Integration tests
   - Performance benchmarks
