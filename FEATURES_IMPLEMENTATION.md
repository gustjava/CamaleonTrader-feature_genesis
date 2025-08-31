# Dynamic Stage 0 - Feature Engineering Implementation

This document describes the complete implementation of GPU-accelerated feature engineering modules for the Dynamic Stage 0 pipeline.

## Overview

The feature engineering pipeline consists of four main engines, each implementing advanced techniques for financial time series analysis:

1. **StationarizationEngine** - Fractional differentiation and stationarization techniques
2. **StatisticalTests** - ADF tests and distance correlation analysis
3. **SignalProcessor** - Baxter-King filters for cycle isolation
4. **GARCHModels** - Volatility modeling with GARCH(1,1)

## 1. StationarizationEngine (`features/stationarization.py`)

### Purpose
Achieves stationarity while preserving memory properties of financial time series using fractional differentiation and other techniques.

### Key Features

#### Fractional Differentiation
- **Implementation**: GPU-accelerated using CuPy
- **Method**: Binomial weights computation with convolution
- **Parameters**: 
  - `d_values`: [0.1, 0.3, 0.5, 0.7, 0.9] (configurable)
  - `threshold`: 1e-5 (stationarity threshold)
  - `max_lag`: 1000 (maximum lag for computation)

#### Stationarity Testing
- **Method**: Variance ratio test
- **Periods**: [2, 4, 8, 16] for variance comparison
- **Criterion**: Variance ratio close to 1.0 indicates stationarity

#### Additional Techniques
- **Rolling Stationarization**: Window-based normalization
- **Variance Stabilization**: Log, sqrt, and Box-Cox transformations

### Generated Features
- `frac_diff_close`: Fractionally differentiated close prices
- `frac_diff_d`: Optimal differentiation order
- `frac_diff_stationary`: Stationarity flag
- `frac_diff_variance_ratio`: Variance ratio metric
- `frac_diff_mean/std/skewness/kurtosis`: Statistical moments
- `rolling_stationary_close`: Rolling window stationarized prices
- `log_stabilized_close`: Log-transformed prices

## 2. StatisticalTests (`features/statistical_tests.py`)

### Purpose
Provides quantitative validation of feature properties through statistical testing and non-linear relationship discovery.

### Key Features

#### Augmented Dickey-Fuller (ADF) Test
- **Implementation**: GPU-accelerated OLS using CuPy
- **Method**: Batch computation with automatic lag selection
- **Parameters**:
  - `max_lag`: 10 (default)
  - `regression`: 'c' (constant), 'ct' (constant+trend), 'nc' (no constant)
  - `autolag`: 'AIC' for optimal lag selection

#### Distance Correlation (dCor)
- **Implementation**: GPU-accelerated distance matrix computation
- **Method**: Pairwise Euclidean distances with CuPy
- **Features**:
  - Non-linear relationship detection
  - Permutation-based significance testing
  - Automatic subsampling for large datasets

### Generated Features
- `adf_test_statistic`: ADF test statistic
- `adf_p_value`: P-value for stationarity test
- `is_stationary`: Stationarity determination
- `adf_lag_order`: Optimal lag order used
- `dcor_returns_volume`: Distance correlation between returns and volume
- `dcor_significance`: Statistical significance of dCor

## 3. SignalProcessor (`features/signal_processing.py`)

### Purpose
Isolates specific market cycles using advanced filtering techniques, removing high-frequency noise and long-term trends.

### Key Features

#### Baxter-King Filter
- **Implementation**: GPU-accelerated using cuSignal
- **Method**: Band-pass filter with FFT convolution
- **Parameters**:
  - `low_freq`: 6 periods (lower frequency bound)
  - `high_freq`: 32 periods (upper frequency bound)
  - `k`: 12 (truncation parameter)

#### Filter Characteristics
- **Band-pass**: Isolates medium-term cycles (6-32 periods)
- **Trend removal**: Weights sum to zero
- **Edge handling**: NaN values at boundaries

### Generated Features
- `bk_filtered_close`: Baxter-King filtered close prices
- `bk_volatility`: Volatility of filtered series
- `bk_zero_crossings`: Zero-crossing rate (cycle frequency indicator)
- `bk_peak_trough_ratio`: Peak-to-trough ratio

## 4. GARCHModels (`features/garch_models.py`)

### Purpose
Models volatility clustering and provides predictive features for risk management and position sizing.

### Key Features

#### GARCH(1,1) Model
- **Implementation**: Hybrid CPU-GPU approach
- **CPU**: Parameter optimization using scipy.optimize
- **GPU**: Log-likelihood computation using CuPy
- **Parameters**:
  - `p=1, q=1`: GARCH orders
  - `max_iter`: 1000 (optimization iterations)
  - `tolerance`: 1e-6 (convergence tolerance)

#### Model Features
- **Constraints**: Positivity and stationarity constraints
- **Diagnostics**: AIC, BIC, log-likelihood
- **Forecasting**: Multi-step volatility forecasts

### Generated Features
- `garch_omega/alpha/beta`: GARCH parameters
- `garch_persistence`: Persistence parameter (α + β)
- `garch_log_likelihood/aic/bic`: Model fit diagnostics
- `garch_is_stationary`: Model stationarity
- `garch_conditional_variance`: Fitted conditional variance
- `volatility_mean/std/min/max`: Volatility statistics
- `volatility_skewness/kurtosis`: Volatility distribution moments
- `residual_mean/std/skewness/kurtosis`: Residual statistics
- `volatility_autocorr_lag1/lag5`: Volatility clustering
- `leverage_effect`: Asymmetry measure
- `garch_volatility_forecast`: Mean volatility forecast

## Pipeline Integration

### Orchestration (`orchestration/main.py`)
The feature engines are integrated into the main pipeline in the following sequence:

1. **Data Loading**: Load currency pair data using LocalDataLoader
2. **Stationarization**: Apply fractional differentiation and stationarization
3. **Statistical Tests**: Perform ADF tests and distance correlation analysis
4. **Signal Processing**: Apply Baxter-King filters for cycle isolation
5. **GARCH Modeling**: Fit volatility models and generate forecasts

### Error Handling
- Each engine includes comprehensive error handling
- Failed operations are logged with detailed error messages
- Pipeline continues with partial results when possible
- Database status updates reflect specific failure points

## Configuration

### Settings (`config/settings.py`)
All feature parameters are configurable through the settings system:

```python
@dataclass
class FeatureSettings:
    # Fractional differentiation
    frac_diff_values: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9]
    frac_diff_threshold: float = 1e-5
    frac_diff_max_lag: int = 1000
    
    # Baxter-King filter
    baxter_king_low_freq: int = 6
    baxter_king_high_freq: int = 32
    baxter_king_k: int = 12
    
    # GARCH model
    garch_p: int = 1
    garch_q: int = 1
    garch_max_iter: int = 1000
    garch_tolerance: float = 1e-6
    
    # Distance correlation
    distance_corr_max_samples: int = 10000
```

## Testing

### Test Script (`test_features.py`)
Comprehensive test suite covering:

- **Individual Engine Tests**: Each engine tested in isolation
- **Integration Tests**: Full pipeline testing
- **Synthetic Data**: Realistic OHLCV data generation
- **Error Handling**: Edge cases and failure scenarios

### Running Tests
```bash
python test_features.py
```

## Performance Considerations

### GPU Memory Management
- Automatic subsampling for large datasets
- Efficient CuPy operations for matrix computations
- Memory cleanup after operations

### Computational Complexity
- **Fractional Differentiation**: O(n × max_lag)
- **ADF Test**: O(n × lag_order)
- **Distance Correlation**: O(n²) with subsampling
- **Baxter-King Filter**: O(n × k) with FFT
- **GARCH Fitting**: O(n × iterations)

### Optimization Strategies
- Vectorized operations using CuPy
- Parallel processing where applicable
- Early termination for convergence
- Caching of intermediate results

## Usage Example

```python
from features import (
    StationarizationEngine, 
    StatisticalTests, 
    SignalProcessor, 
    GARCHModels
)

# Initialize engines
stationarization = StationarizationEngine()
statistical_tests = StatisticalTests()
signal_processor = SignalProcessor()
garch_models = GARCHModels()

# Process currency pair data
df = load_currency_data("EURUSD")

# Apply feature engineering pipeline
df = stationarization.process_currency_pair(df)
df = statistical_tests.process_currency_pair(df)
df = signal_processor.process_currency_pair(df)
df = garch_models.process_currency_pair(df)

# Access generated features
print(f"Generated {len(df.columns)} features")
print(f"Stationarity: {df['is_stationary'].iloc[-1]}")
print(f"GARCH persistence: {df['garch_persistence'].iloc[-1]:.4f}")
```

## Future Enhancements

### Planned Features
1. **EMD (Empirical Mode Decomposition)**: Automatic signal decomposition
2. **Advanced GARCH Models**: EGARCH, GJR-GARCH for asymmetry
3. **Wavelet Analysis**: Multi-resolution signal analysis
4. **Regime Detection**: Hidden Markov Models for regime identification

### Performance Improvements
1. **Multi-GPU Support**: Distributed computation across GPUs
2. **Streaming Processing**: Real-time feature computation
3. **Caching Layer**: Persistent storage of intermediate results
4. **Optimization**: Further GPU kernel optimizations

## Conclusion

The Dynamic Stage 0 feature engineering pipeline provides a comprehensive suite of GPU-accelerated techniques for financial time series analysis. The modular design allows for easy extension and customization, while the robust error handling ensures reliable operation in production environments.

Each engine contributes unique insights:
- **StationarizationEngine**: Memory-preserving stationarity
- **StatisticalTests**: Quantitative validation and non-linear relationships
- **SignalProcessor**: Cycle isolation and noise reduction
- **GARCHModels**: Volatility modeling and forecasting

Together, these engines generate a rich set of features that capture various aspects of market dynamics, providing a solid foundation for advanced trading strategies and risk management systems.
