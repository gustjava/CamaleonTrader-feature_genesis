# Dynamic Stage 0 - GPU-Accelerated Feature Stationarization Pipeline

This directory contains the Docker environment for the "Dynamic Stage 0" pipeline, which provides GPU-accelerated feature stationarization for quantitative trading and high-frequency research.

## Prerequisites

- Docker with NVIDIA Container Runtime support
- NVIDIA GPU with CUDA 11.8+ support
- Docker Compose

## Quick Start

### 1. Build the Docker Image

```bash
docker build -t dynamic-stage0 .
```

### 2. Run with Docker Compose (Recommended for Development)

```bash
docker-compose up -d
```

This will start the container with:
- Jupyter Lab accessible at `http://localhost:8888`
- Dask Dashboard accessible at `http://localhost:8787`
- GPU support enabled
- Volume mounts for development

### 3. Run Interactive Shell

```bash
docker run --gpus all -it --rm dynamic-stage0 /bin/bash
```

Then activate the conda environment:
```bash
conda activate dynamic-stage0
```

## Environment Details

### Conda Environment (`dynamic-stage0`)

The environment includes:
- **RAPIDS 23.12**: cuDF, dask-cudf, dask-cuda for GPU-accelerated data processing
- **CuPy 12.2.0**: GPU-accelerated numerical computing
- **cuSignal 23.12**: GPU-accelerated signal processing
- **cuML 23.12**: GPU-accelerated machine learning
- **Scientific Python**: NumPy, Pandas, SciPy, scikit-learn

### Python Dependencies

Additional Python packages via pip:
- **Cloud Storage**: boto3, s3fs, pyarrow for R2/S3 connectivity
- **Database**: SQLAlchemy, PyMySQL for MySQL connectivity
- **Configuration**: PyYAML, python-dotenv
- **Logging**: structlog, rich
- **Development**: pytest, black, flake8

## GPU Configuration

The environment is configured for multi-GPU support:
- Uses `dask-cuda.LocalCUDACluster` for GPU worker management
- RMM (RAPIDS Memory Manager) for efficient GPU memory management
- Support for out-of-core processing with GPU spilling

## Development Workflow

1. **Start the environment**: `docker-compose up -d`
2. **Access Jupyter Lab**: Open `http://localhost:8888`
3. **Edit code**: Files are mounted from host to container
4. **Test GPU functionality**: Run GPU verification scripts
5. **Stop environment**: `docker-compose down`

## Testing GPU Setup

Create a test script to verify GPU functionality:

```python
import cupy as cp
import cudf
import dask_cudf
from dask_cuda import LocalCUDACluster

# Test CuPy
print(f"CuPy version: {cp.__version__}")
print(f"Available GPUs: {cp.cuda.runtime.getDeviceCount()}")

# Test cuDF
df = cudf.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
print(f"cuDF DataFrame:\n{df}")

# Test Dask-CUDA
cluster = LocalCUDACluster()
print(f"Dask-CUDA cluster: {cluster}")
cluster.close()
```

## Troubleshooting

### GPU Not Detected
- Ensure NVIDIA Container Runtime is installed
- Check `nvidia-smi` output on host
- Verify `--gpus all` flag is used

### Memory Issues
- Adjust RMM pool size in configuration
- Enable GPU spilling for large datasets
- Monitor GPU memory usage with `nvidia-smi`

### Build Issues
- Clear Docker cache: `docker system prune -a`
- Check CUDA version compatibility
- Verify conda environment creation

## Next Steps

After setting up the environment:
1. Configure database connections
2. Set up Cloudflare R2 credentials
3. Implement the feature stationarization pipeline
4. Add monitoring and logging
5. Deploy to vast.ai for production
