"""
GARCH Models Module for Dynamic Stage 0 Pipeline

This module implements GPU-accelerated GARCH model fitting for volatility modeling.
Uses a hybrid CPU-GPU approach with CuPy for log-likelihood computation.
"""

import logging  # Logging para debug e monitoramento
import numpy as np  # Computação numérica CPU para otimização
import cupy as cp  # Computação numérica GPU para log-likelihood
import dask_cudf  # DataFrames distribuídos com cuDF
import cudf  # DataFrames GPU para processamento rápido
from typing import Optional, Dict, Any  # Type hints para melhor documentação
from scipy.optimize import minimize  # Otimização para fitting de parâmetros GARCH

from .base_engine import BaseFeatureEngine  # Classe base para engines de features
from config.unified_config import UnifiedConfig  # Configuração unificada do sistema
from dask.distributed import Client  # Cliente Dask para computação distribuída

logger = logging.getLogger(__name__)


def _garch_default_row() -> Dict[str, float]:
    """Return a default row (NaNs) for meta schema compliance."""
    return {
        'garch_omega': np.nan,  # Parâmetro omega (constante) do modelo GARCH
        'garch_alpha': np.nan,  # Parâmetro alpha (choque) do modelo GARCH
        'garch_beta': np.nan,  # Parâmetro beta (persistência) do modelo GARCH
        'garch_persistence': np.nan,  # Persistência total (alpha + beta)
        'garch_log_likelihood': np.nan,  # Log-likelihood do modelo ajustado
        'garch_aic': np.nan,  # Critério de informação de Akaike
        'garch_bic': np.nan,  # Critério de informação bayesiano
        'garch_is_stationary': np.nan,  # Flag indicando se modelo é estacionário
    }


def _garch_fit_partition_np(part: cudf.DataFrame, price_col: str, max_samples: int, max_iter: int, tolerance: float) -> cudf.DataFrame:
    """Pure module-level function for Dask map_partitions (deterministic hashing).

    Computes a small set of GARCH(1,1) metrics on a single partition using NumPy/CPU
    and returns a 1-row cuDF DataFrame matching the meta schema.
    """
    try:
        if price_col not in part.columns:  # Verifica se coluna de preço existe
            return cudf.DataFrame([_garch_default_row()])

        x = part[price_col].to_pandas().to_numpy()  # Converte para NumPy para processamento CPU
        if x.size < 100:  # Precisa de pelo menos 100 observações para GARCH
            return cudf.DataFrame([_garch_default_row()])

        x = x.astype(np.float64, copy=False)  # Converte para float64 para precisão
        if np.isnan(x).any() or np.isinf(x).any():  # Verifica valores inválidos
            # Replace non-finite with previous value or small positive
            x = np.where(np.isfinite(x), x, np.nan)  # Marca valores inválidos como NaN
            # forward fill then back fill then fill with small
            pd = __import__('pandas')  # Importa pandas para preenchimento
            x = pd.Series(x).fillna(method='ffill').fillna(method='bfill').fillna(1e-8).to_numpy()  # Preenche gaps com forward/backward fill

        # Truncate tail for bounded work
        if x.size > int(max_samples):  # Limita tamanho para controlar tempo de processamento
            x = x[-int(max_samples):]  # Pega apenas as últimas observações

        # Log returns
        logx = np.log(np.maximum(x, 1e-8))  # Log dos preços (com proteção contra zero)
        r = np.diff(logx)  # Calcula retornos logarítmicos
        r = r[np.isfinite(r)]  # Remove valores inválidos
        if r.size < 50:  # Precisa de pelo menos 50 retornos para GARCH
            return cudf.DataFrame([_garch_default_row()])

        # GARCH(1,1) negative log-likelihood (CPU)
        def nll(params):  # Função de log-likelihood negativa para otimização
            omega, alpha, beta = params  # Parâmetros do modelo GARCH(1,1)
            # basic constraints
            if omega <= 0 or alpha < 0 or beta < 0 or (alpha + beta) >= 1.0:  # Verifica restrições básicas
                return 1e12  # Retorna valor alto se parâmetros inválidos
            n = r.size  # Número de observações
            h = np.empty(n, dtype=np.float64)  # Array para variâncias condicionais
            # unconditional variance init
            h0 = np.var(r) if (1 - alpha - beta) <= 1e-8 else omega / (1 - alpha - beta)  # Variância incondicional
            h[0] = max(h0, 1e-9)  # Inicializa primeira variância
            for t in range(1, n):  # Loop para calcular variâncias condicionais
                h[t] = omega + alpha * r[t-1] * r[t-1] + beta * h[t-1]  # Equação GARCH(1,1)
                if h[t] <= 0:  # Verifica se variância é positiva
                    return 1e12  # Retorna valor alto se variância inválida
            ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(h) + (r * r) / h)  # Log-likelihood gaussiana
            return -float(ll)  # Retorna log-likelihood negativa

        x0 = np.array([0.01, 0.1, 0.8], dtype=np.float64)  # Valores iniciais para otimização
        bounds = [(1e-10, None), (0.0, 1.0), (0.0, 1.0)]  # Limites para parâmetros (omega, alpha, beta)
        res = minimize(nll, x0=x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': int(max_iter), 'ftol': float(tolerance)})  # Otimização L-BFGS-B
        if not res.success:  # Verifica se otimização convergiu
            return cudf.DataFrame([_garch_default_row()])

        omega, alpha, beta = map(float, res.x)  # Extrai parâmetros otimizados
        persistence = float(alpha + beta)  # Calcula persistência (alpha + beta)
        ll = -float(res.fun)  # Log-likelihood final
        n_obs = int(r.size)  # Número de observações
        aic = 2 * 3 - 2 * ll  # Critério de informação de Akaike (3 parâmetros)
        bic = 3 * np.log(n_obs) - 2 * ll  # Critério de informação bayesiano
        out = {
            'garch_omega': omega, 'garch_alpha': alpha, 'garch_beta': beta,  # Parâmetros do modelo
            'garch_persistence': persistence, 'garch_log_likelihood': ll,  # Persistência e log-likelihood
            'garch_aic': float(aic), 'garch_bic': float(bic), 'garch_is_stationary': float(persistence < 1.0),  # Critérios de informação e estacionariedade
        }
        return cudf.DataFrame([out])  # Retorna DataFrame com resultados
    except Exception:
        return cudf.DataFrame([_garch_default_row()])  # Retorna valores padrão em caso de erro


class GARCHModels(BaseFeatureEngine):
    """
    GPU-accelerated GARCH model fitting engine for volatility modeling.
    """

    def __init__(self, settings: UnifiedConfig, client: Client):
        """Initialize the GARCH models engine with configuration."""
        super().__init__(settings, client)  # Inicializa classe base com configurações e cliente Dask
        # Access individual GARCH settings instead of nested object
        self.p = self.settings.features.garch_p  # Ordem do termo ARCH (choques)
        self.q = self.settings.features.garch_q  # Ordem do termo GARCH (persistência)
        self.max_iter = self.settings.features.garch_max_iter  # Máximo de iterações para otimização
        self.tolerance = self.settings.features.garch_tolerance  # Tolerância para convergência
        # Usar chave dedicada para GARCH (não reutilizar Stage 1)
        try:
            self.max_samples = int(getattr(self.settings.features, 'garch_max_samples', 10000))
        except Exception:
            self.max_samples = 10000
        try:
            self.min_price_rows = int(getattr(self.settings.features, 'garch_min_price_rows', 200))
            self.min_return_rows = int(getattr(self.settings.features, 'garch_min_return_rows', 100))
            self.log_price = bool(getattr(self.settings.features, 'garch_log_price', True))
        except Exception:
            self.min_price_rows, self.min_return_rows, self.log_price = 200, 100, True

    def _log_likelihood_gpu(self, params: np.ndarray, returns: np.ndarray) -> float:
        """
        Compute log-likelihood using numpy (for scipy compatibility).
        """
        try:
            # Ensure params are float64 for scipy compatibility
            params = np.asarray(params, dtype=np.float64)  # Converte parâmetros para float64
            omega, alpha, beta = params  # Extrai parâmetros omega, alpha, beta
            
            # Ensure returns are float64
            returns = np.asarray(returns, dtype=np.float64)  # Converte retornos para float64
            n = len(returns)  # Número de observações
            h = np.zeros(n, dtype=np.float64)  # Array para variâncias condicionais
            
            # Initialize with unconditional variance if possible, otherwise simple variance
            uncond_var = float(np.var(returns))  # Variância incondicional (simples)
            if (1 - alpha - beta) > 1e-8:  # Se modelo é estacionário
                uncond_var = omega / (1 - alpha - beta)  # Variância incondicional teórica
            h[0] = uncond_var  # Inicializa primeira variância

            # GARCH(1,1) recursion
            for t in range(1, n):  # Loop para calcular variâncias condicionais
                h[t] = omega + alpha * returns[t-1]**2 + beta * h[t-1]  # Equação GARCH(1,1)

            h = np.maximum(h, 1e-9) # Ensure positive variance  # Garante variâncias positivas

            # Compute log-likelihood
            log_likelihood = -0.5 * np.sum(np.log(2 * np.pi) + np.log(h) + returns**2 / h)  # Log-likelihood gaussiana
            
            # Return negative for minimization
            result = -float(log_likelihood) if np.isfinite(log_likelihood) else np.inf  # Retorna negativo para minimização
            return result

        except Exception as e:
            self._log_error(f"Error in log likelihood computation: {e}")
            return np.inf  # Retorna infinito em caso de erro

    def _garch_log_likelihood_gpu(self, returns: cp.ndarray, omega: float, alpha: float, beta: float) -> float:
        """
        GPU-accelerated log-likelihood for GARCH(1,1) without Python loops.
        """
        try:
            n = len(returns)  # Número de observações
            h0 = cp.var(returns)  # Variância inicial (unconditional)
            r2 = returns[:-1]**2  # Retornos ao quadrado (n-1 elementos)

            # calcula recursivamente: h_t = omega + alpha*r_{t-1}^2 + beta*h_{t-1}
            # podemos obter h[1:] = omega/(1-beta) + (alpha*r2) ⊗ K + beta^i*h0,
            # onde K é um kernel de convolução decrescente beta^i.
            betas = beta ** cp.arange(n, dtype=cp.float64)  # Potências de beta para convolução
            # soma ponderada de alphas*r2 com potências de beta (convolução)
            # use cp.signal.convolve se cusignal não estiver disponível
            from cupyx.scipy.signal import lfilter  # Importa filtro linear para recursão
            # lfilter aplica soma recursiva: y[i] = alpha*r2[i] + beta*y[i-1]
            h_rec = lfilter([alpha], [1, -beta], r2)  # Aplica filtro recursivo para variâncias
            h = omega / (1 - beta) + cp.concatenate([cp.array([h0]), h_rec])  # Concatena variância inicial com recursivas
            h = cp.maximum(h, 1e-9)  # Garante variâncias positivas
            
            log_likelihood = -0.5 * cp.sum(cp.log(2*cp.pi) + cp.log(h) + returns**2 / h)  # Log-likelihood gaussiana
            return float(log_likelihood)  # Converte para float Python
        except Exception as e:
            self._critical_error(f"Error in vectorized GARCH log-likelihood: {e}")  # Log de erro crítico
    
    def _garch_log_likelihood_cpu(self, returns: np.ndarray, omega: float, alpha: float, beta: float) -> float:
        """
        CPU fallback for GARCH log-likelihood computation.
        """
        try:
            n = len(returns)  # Número de observações
            
            # Initialize variance series
            variance = np.zeros(n)  # Array para variâncias condicionais
            variance[0] = np.var(returns)  # Initial variance  # Variância inicial (unconditional)
            
            # Compute variance series using GARCH recursion
            for t in range(1, n):  # Loop para calcular variâncias condicionais
                variance[t] = omega + alpha * returns[t-1]**2 + beta * variance[t-1]  # Equação GARCH(1,1)
            
            # Compute log-likelihood (normal distribution assumption)
            log_likelihood = -0.5 * np.sum(
                np.log(2 * np.pi * variance) + returns**2 / variance  # Log-likelihood gaussiana
            )
            
            return float(log_likelihood)  # Converte para float Python
            
        except Exception as e:
            self._critical_error(f"Error in CPU GARCH log-likelihood: {e}")  # Log de erro crítico

    def fit_garch_gpu(self, returns: cp.ndarray, max_iter: int = 1000, tolerance: float = 1e-6) -> Dict[str, Any]:
        """
        Fit GARCH(1,1) model using GPU-accelerated log-likelihood computation.
        
        Args:
            returns: Return series on GPU
            max_iter: Maximum optimization iterations
            tolerance: Convergence tolerance
            
        Returns:
            Dictionary with GARCH parameters and fit statistics
        """
        try:
            # Convert to numpy for optimization (only the optimization loop runs on CPU)
            returns_np = cp.asnumpy(returns)  # Converte para NumPy para otimização CPU
            
            # Initial parameter guesses
            initial_params = np.array([0.01, 0.1, 0.8])  # omega, alpha, beta  # Valores iniciais típicos
            
            # Parameter bounds (ensure stationarity and positivity)
            bounds = [(1e-6, None), (0, 1), (0, 1)]  # omega > 0, 0 <= alpha, beta <= 1  # Limites para parâmetros
            
            # Constraint: alpha + beta < 1 for stationarity
            def constraint(params):  # Função de restrição para estacionariedade
                return 1 - params[1] - params[2]  # alpha + beta < 1  # Restrição de estacionariedade
            
            # Objective function (negative log-likelihood for minimization)
            def objective(params):  # Função objetivo para minimização
                omega, alpha, beta = params  # Extrai parâmetros
                
                # Check stationarity constraint
                if alpha + beta >= 1:  # Verifica restrição de estacionariedade
                    return np.inf  # Retorna infinito se não estacionário
                
                # Use GPU log-likelihood computation
                try:
                    log_lik = self._garch_log_likelihood_gpu(returns, omega, alpha, beta)  # Tenta GPU primeiro
                    return -log_lik  # Negative for minimization  # Retorna negativo para minimização
                except Exception:
                    # Fallback to CPU
                    log_lik = self._garch_log_likelihood_cpu(returns_np, omega, alpha, beta)  # Fallback para CPU
                    return -log_lik  # Retorna negativo para minimização
            
            # Optimize using scipy
            from scipy.optimize import minimize  # Importa otimizador
            
            result = minimize(
                objective,  # Função objetivo
                initial_params,  # Parâmetros iniciais
                method='SLSQP',  # Método SLSQP para otimização com restrições
                bounds=bounds,  # Limites dos parâmetros
                constraints={'type': 'ineq', 'fun': constraint},  # Restrições de desigualdade
                options={'maxiter': max_iter, 'ftol': tolerance}  # Opções de convergência
            )
            
            if result.success:  # Verifica se otimização convergiu
                omega, alpha, beta = result.x  # Extrai parâmetros otimizados
                
                # Compute final log-likelihood
                final_log_lik = -result.fun  # Log-likelihood final
                
                # Compute fitted variance series
                variance = cp.zeros(len(returns), dtype=cp.float64)  # Array para variâncias
                variance[0] = cp.var(returns)  # Variância inicial
                for t in range(1, len(returns)):  # Loop para calcular variâncias
                    variance[t] = omega + alpha * returns[t-1]**2 + beta * variance[t-1]  # Equação GARCH(1,1)
                
                return {
                    'omega': float(omega),  # Parâmetro omega
                    'alpha': float(alpha),  # Parâmetro alpha
                    'beta': float(beta),  # Parâmetro beta
                    'log_likelihood': float(final_log_lik),  # Log-likelihood final
                    'variance': cp.asnumpy(variance),  # Série de variâncias (converte para NumPy)
                    'converged': True,  # Flag de convergência
                    'iterations': result.nit  # Número de iterações
                }
            else:
                logger.warning(f"GARCH optimization failed: {result.message}")  # Log de aviso se falhou
                return {
                    'omega': None,  # Retorna None se falhou
                    'alpha': None,
                    'beta': None,
                    'log_likelihood': None,
                    'variance': None,
                    'converged': False,
                    'iterations': result.nit
                }
                
        except Exception as e:
            self._critical_error(f"Error in GPU GARCH fitting: {e}")

    def fit_garch11(self, series: cudf.Series) -> Optional[Dict[str, Any]]:
        """
        Fit GARCH(1,1) model to a single time series (partition).
        """
        try:
            # Debug: Check input data type and shape
            self._log_info(f"GARCH input data type: {type(series)}, shape: {series.shape}")  # Log de debug para tipo e forma
            
            data = series.to_cupy()  # Converte para CuPy para processamento GPU
            self._log_info(f"GARCH cupy data type: {type(data)}, shape: {data.shape}")  # Log de debug para dados CuPy

            if len(data) < 100:  # Verifica se há dados suficientes
                self._log_warn("Insufficient data for GARCH, skipping.", series_len=len(data))  # Log de aviso se insuficiente
                return None
            
            if len(data) > self.max_samples:  # Limita tamanho se muito grande
                data = data[-self.max_samples:]  # Pega apenas as últimas amostras
                self._log_info("Truncated data for GARCH fitting.", samples=len(data))  # Log de truncamento
            
            # Ensure data is float64 and handle any NaN values
            data = cp.asarray(data, dtype=cp.float64)  # Converte para float64
            data = cp.nan_to_num(data, nan=0.0)  # Substitui NaN por zero
            self._log_info(f"GARCH data after conversion: type={type(data)}, min={float(cp.min(data))}, max={float(cp.max(data))}")  # Log de dados após conversão
            
            # Compute returns with proper type handling
            log_data = cp.log(cp.maximum(data, 1e-8))  # Avoid log(0)  # Log dos dados com proteção contra zero
            returns = cp.diff(log_data)  # Calcula retornos logarítmicos
            self._log_info(f"GARCH returns after diff: type={type(returns)}, min={float(cp.min(returns))}, max={float(cp.max(returns))}")  # Log de retornos
            
            # Remove any remaining NaN or infinite values
            returns = returns[cp.isfinite(returns)]  # Remove valores inválidos
            self._log_info(f"GARCH returns after filtering: type={type(returns)}, len={len(returns)}")  # Log de retornos filtrados
            
            if len(returns) < 50:  # Verifica se há retornos suficientes
                self._log_warn("Insufficient valid returns for GARCH, skipping.", valid_returns=len(returns))  # Log de aviso se insuficiente
                return None
            
            # Use more conservative initial parameters without strict constraints
            returns_var = float(cp.var(returns))  # Variância dos retornos
            initial_params = np.array([returns_var * 0.01, 0.05, 0.85], dtype=np.float64)  # Parâmetros iniciais conservadores
            bounds = [(1e-8, None), (0.0, 1.0), (0.0, 1.0)]  # Limites para parâmetros
            
            self._log_info(f"GARCH initial params: {initial_params}, returns_var: {returns_var}")  # Log de parâmetros iniciais
            
            # Convert to numpy for scipy compatibility
            returns_np = cp.asnumpy(returns).astype(np.float64)  # Converte para NumPy para scipy
            self._log_info(f"GARCH returns_np: type={type(returns_np)}, shape={returns_np.shape}, min={float(np.min(returns_np))}, max={float(np.max(returns_np))}")  # Log de dados NumPy
            
            result = minimize(
                fun=self._log_likelihood_gpu,  # Função de log-likelihood
                x0=initial_params,  # Parâmetros iniciais
                args=(returns_np,),  # Argumentos (retornos)
                method='L-BFGS-B',  # Use L-BFGS-B which is more robust  # Método L-BFGS-B mais robusto
                bounds=bounds,  # Limites dos parâmetros
                options={'maxiter': int(self.max_iter), 'ftol': float(self.tolerance)}  # Opções de convergência
            )

            if not result.success:  # Verifica se otimização convergiu
                self._log_warn("GARCH optimization failed.", message=result.message)  # Log de aviso se falhou
                return None

            # Return fitted parameters and diagnostics
            omega, alpha, beta = result.x  # Extrai parâmetros otimizados
            persistence = alpha + beta  # Calcula persistência
            log_likelihood = -result.fun  # Log-likelihood final
            n_obs = int(len(returns))  # Convert to int explicitly  # Número de observações
            aic = 2 * 3 - 2 * log_likelihood  # Critério de informação de Akaike
            bic = 3 * np.log(n_obs) - 2 * log_likelihood  # Critério de informação bayesiano
            is_stationary = persistence < 1.0  # Verifica estacionariedade
            
            self._log_info("GARCH(1,1) fitted successfully.", alpha=f"{alpha:.4f}", beta=f"{beta:.4f}")  # Log de sucesso

            return {
                'garch_omega': omega, 'garch_alpha': alpha, 'garch_beta': beta,  # Parâmetros do modelo
                'garch_persistence': persistence, 'garch_log_likelihood': log_likelihood,  # Persistência e log-likelihood
                'garch_aic': aic, 'garch_bic': bic, 'garch_is_stationary': float(is_stationary)  # Critérios de informação e estacionariedade
            }
        except Exception as e:
            self._critical_error(f"Error during GARCH fitting: {e}")

    def _fit_on_partition(self, part: cudf.DataFrame) -> Dict[str, Any]:
        """
        Execute GARCH fitting inside GPU worker (without leaving cluster).
        
        Args:
            part: DataFrame partition with 'y_close' column
            
        Returns:
            Dictionary with GARCH parameters
        """
        close_series = part["y_close"]  # Extrai série de preços de fechamento
        res = self.fit_garch11(close_series)  # Ajusta modelo GARCH(1,1)
        # fit_garch11 now always returns a dict (either fitted or default values)
        return res  # Retorna dicionário com parâmetros

    def _fit_on_partition_wrapper(self, part: cudf.DataFrame) -> cudf.DataFrame:
        """
        Wrapper function for Dask map_partitions to avoid lambda hashing issues.
        
        Args:
            part: DataFrame partition with 'y_close' column
            
        Returns:
            cuDF DataFrame with GARCH parameters
        """
        result_dict = self._fit_on_partition(part)  # Executa fitting GARCH
        return cudf.DataFrame([result_dict])  # Converte para DataFrame cuDF

    def process(self, df: dask_cudf.DataFrame) -> dask_cudf.DataFrame:
        """
        Executes the GARCH modeling pipeline (Dask version).
        """
        self._log_info("Starting GARCH (Dask)...")  # Log de início do processamento

        # (optional) ensure temporal order
        # df = self._ensure_sorted(df, by="ts")

        # Resolve a robust price source (y_close preferred)
        cols = list(df.columns)  # Lista colunas disponíveis
        price_col = None  # Inicializa coluna de preço
        for pref in ("y_close", "log_stabilized_y_close"):  # Preferências de colunas de preço
            if pref in cols:  # Verifica se coluna preferida existe
                price_col = pref  # Define coluna de preço
                break  # Para na primeira encontrada
        if price_col is None:  # Se não encontrou coluna preferida
            for c in cols:  # Procura por colunas com 'close'
                if 'close' in str(c).lower():  # Verifica se contém 'close'
                    price_col = str(c)  # Define coluna de preço
                    break  # Para na primeira encontrada
        if price_col is None:  # Se não encontrou nenhuma coluna de preço
            self._log_warn("GARCH (Dask): no close-like column found; skipping")  # Log de aviso
            return df  # Retorna DataFrame original

        # Bring series to single partition in one GPU worker
        one = self._single_partition(df, cols=[price_col])  # Consolida em uma partição

        # Diagnostics on a bounded head (same cap as fit)
        try:
            price_sample = one[price_col].head(self.max_samples)
            price_series = price_sample.compute().to_pandas()
            n_total = int(len(price_series))
            n_nonnull = int(price_series.notna().sum())
            n_null = int(n_total - n_nonnull)
            # Returns diagnostic
            px = price_series.fillna(method='ffill').fillna(method='bfill')
            if self.log_price:
                px = np.log(np.maximum(px.to_numpy(dtype=float), 1e-8))
            else:
                px = px.to_numpy(dtype=float)
            r = np.diff(px)
            r = r[np.isfinite(r)]
            n_ret = int(r.size)
            v_px = float(np.nanvar(price_series.to_numpy(dtype=float)))
            v_r = float(np.nanvar(r)) if n_ret > 0 else float('nan')
            self._log_info(
                "GARCH diagnostics",
                price_rows=n_total,
                nonnull=n_nonnull,
                null=n_null,
                returns_rows=n_ret,
                var_price=v_px,
                var_returns=v_r,
                max_samples=int(self.max_samples),
            )
            if n_total < self.min_price_rows or n_ret < self.min_return_rows:
                self._log_warn(
                    "GARCH insufficient data",
                    min_price_rows=int(self.min_price_rows),
                    min_return_rows=int(self.min_return_rows),
                )
        except Exception as _e_diag:
            self._log_warn("GARCH diagnostics failed", error=str(_e_diag))

        # Compute model ONCE inside worker and materialize dict in driver
        # (it's small - just scalars)
        meta = {  # Metadados para schema do DataFrame
            'garch_omega': 'f8', 'garch_alpha': 'f8', 'garch_beta': 'f8',  # Parâmetros do modelo
            'garch_persistence': 'f8', 'garch_log_likelihood': 'f8',  # Persistência e log-likelihood
            'garch_aic': 'f8', 'garch_bic': 'f8', 'garch_is_stationary': 'f8'  # Critérios de informação e estacionariedade
        }

        # Generate a Dask DataFrame with 1 row containing metrics using module-level function
        params_ddf = one.map_partitions(  # Aplica função em cada partição
            _garch_fit_partition_np,  # Função de fitting GARCH
            price_col,  # Coluna de preço
            int(self.max_samples),  # Máximo de amostras
            int(self.max_iter),  # Máximo de iterações
            float(self.tolerance),  # Tolerância
            meta=cudf.DataFrame({k: cudf.Series([], dtype=v) for k, v in meta.items()})  # Schema de metadados
        )

        params_pdf = params_ddf.compute().to_pandas()  # 1 row, cheap  # Computa e converte para pandas
        params = params_pdf.iloc[0].to_dict()  # Extrai parâmetros como dicionário

        self._log_info("GARCH fitted.", **{k: float(params[k]) if params[k] == params[k] else None for k in params})  # Log de parâmetros ajustados
        # Emit reason hint if NaNs returned
        try:
            vals = [params.get('garch_omega'), params.get('garch_alpha'), params.get('garch_beta')]
            if all([(v != v) for v in vals]):  # all NaN
                self._log_warn(
                    "GARCH returned NaNs",
                    hint="Check price coverage/NaNs and return length vs min thresholds",
                    min_price_rows=int(self.min_price_rows),
                    min_return_rows=int(self.min_return_rows),
                    used_max_samples=int(self.max_samples),
                )
        except Exception:
            pass

        # Record metrics and optional artifact summary
        try:
            self._record_metrics('garch', params)  # Registra métricas GARCH
            if bool(getattr(self.settings.features, 'debug_write_artifacts', True)):  # Verifica se deve escrever artefatos
                from pathlib import Path  # Importa Path para manipulação de diretórios
                out_root = Path(getattr(self.settings.output, 'output_path', './output'))  # Diretório raiz de saída
                subdir = str(getattr(self.settings.features, 'artifacts_dir', 'artifacts'))  # Subdiretório de artefatos
                out_dir = out_root / subdir / 'garch'  # Diretório específico para GARCH
                out_dir.mkdir(parents=True, exist_ok=True)  # Cria diretório se não existir
                import json as _json  # Importa JSON para serialização
                summary_path = out_dir / 'summary.json'  # Caminho do arquivo de resumo
                with open(summary_path, 'w') as f:  # Abre arquivo para escrita
                    _json.dump(params, f, indent=2)  # Escreve parâmetros em JSON
                self._record_artifact('garch', str(summary_path), kind='json')  # Registra artefato
        except Exception:
            pass  # Ignora erros na gravação de artefatos

        # Broadcast scalars to original DataFrame (all rows)
        df = self._broadcast_scalars(df, params)  # Transmite escalares para todas as linhas
        self._log_info("GARCH features attached.")  # Log de features anexadas
        return df  # Retorna DataFrame com features GARCH

    def process_cudf(self, gdf: cudf.DataFrame) -> cudf.DataFrame:
        """
        Executes the comprehensive GARCH modeling pipeline (cuDF version).
        
        Implements the complete GARCH modeling pipeline as specified in the technical plan:
        - GARCH(1,1) parameter estimation
        - Conditional variance series estimation
        - Volatility statistics and autocorrelations
        - Residual statistics
        - Volatility forecasting
        
        Args:
            gdf: Input cuDF DataFrame
            
        Returns:
            DataFrame with comprehensive GARCH features
        """
        self._log_info("Starting comprehensive GARCH (cuDF)...")  # Log de início do processamento cuDF

        # Fit GARCH directly on cuDF DataFrame
        garch_result = self._fit_comprehensive_garch(gdf)  # Ajusta modelo GARCH abrangente
        
        if garch_result is None:  # Verifica se fitting falhou
            # Return default values to maintain schema
            garch_result = self._get_default_garch_result()  # Retorna valores padrão para manter schema

        self._log_info("Comprehensive GARCH fitted successfully")

        # Record metrics and optional artifact summary
        try:
            if garch_result:
                metrics = {
                    'omega': garch_result.get('garch_omega'),
                    'alpha': garch_result.get('garch_alpha'),
                    'beta': garch_result.get('garch_beta'),
                    'persistence': garch_result.get('garch_persistence'),
                    'log_likelihood': garch_result.get('garch_log_likelihood'),
                    'aic': garch_result.get('garch_aic'),
                    'bic': garch_result.get('garch_bic'),
                    'converged': garch_result.get('garch_converged'),
                    'iterations': garch_result.get('garch_iterations'),
                }
                self._record_metrics('garch', metrics)
                if bool(getattr(self.settings.features, 'debug_write_artifacts', True)):
                    from pathlib import Path
                    out_root = Path(getattr(self.settings.output, 'output_path', './output'))
                    subdir = str(getattr(self.settings.features, 'artifacts_dir', 'artifacts'))
                    out_dir = out_root / subdir / 'garch'
                    out_dir.mkdir(parents=True, exist_ok=True)
                    import json as _json
                    summary_path = out_dir / 'summary.json'
                    with open(summary_path, 'w') as f:
                        _json.dump(metrics, f, indent=2)
                    self._record_artifact('garch', str(summary_path), kind='json')
        except Exception:
            pass

        # Add all GARCH features to DataFrame
        gdf = self._add_comprehensive_garch_features(gdf, garch_result)

        self._log_info("Comprehensive GARCH features attached.")
        return gdf
    
    def _fit_comprehensive_garch(self, df: cudf.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Fit comprehensive GARCH model with all additional features.
        """
        try:
            if 'y_ret_1m' not in df.columns:
                self._log_warn("y_ret_1m column not found for GARCH fitting")
                return None
            
            # Get returns data
            returns = df['y_ret_1m'].to_cupy()
            
            # Remove NaN values
            valid_mask = ~cp.isnan(returns)
            if cp.sum(valid_mask) < 100:
                self._log_warn("Insufficient valid returns for GARCH fitting")
                return None
            
            clean_returns = returns[valid_mask]
            
            # Fit GARCH model using GPU-accelerated method
            garch_result = self.fit_garch_gpu(clean_returns, max_iter=self.max_iter, tolerance=self.tolerance)
            
            if garch_result is None or not garch_result.get('converged', False):
                self._log_warn("GARCH fitting failed or did not converge")
                return None
            
            # Add comprehensive additional features
            comprehensive_result = self._add_comprehensive_garch_features_to_result(garch_result, clean_returns)
            
            return comprehensive_result
            
        except Exception as e:
            self._critical_error(f"Error in comprehensive GARCH fitting: {e}")
    
    def _add_comprehensive_garch_features_to_result(self, garch_result: Dict[str, Any], returns: cp.ndarray) -> Dict[str, Any]:
        """
        Add comprehensive GARCH features to the result dictionary.
        """
        try:
            # Extract basic GARCH parameters
            omega = garch_result['omega']
            alpha = garch_result['alpha']
            beta = garch_result['beta']
            variance_series = garch_result['variance']
            
            # 1. CONDITIONAL VARIANCE SERIES STATISTICS
            volatility_stats = self._calculate_volatility_statistics(variance_series)
            
            # 2. VOLATILITY AUTOCORRELATIONS
            volatility_autocorr = self._calculate_volatility_autocorrelations(variance_series)
            
            # 3. RESIDUAL STATISTICS
            residual_stats = self._calculate_residual_statistics(returns, variance_series)
            
            # 4. VOLATILITY FORECASTING
            volatility_forecast = self._calculate_volatility_forecast(omega, alpha, beta, variance_series)
            
            # 5. ADDITIONAL GARCH METRICS
            additional_metrics = self._calculate_additional_garch_metrics(omega, alpha, beta, returns, variance_series)
            
            # Combine all results
            comprehensive_result = {
                # Basic GARCH parameters
                'garch_omega': omega,
                'garch_alpha': alpha,
                'garch_beta': beta,
                'garch_persistence': alpha + beta,
                'garch_log_likelihood': garch_result['log_likelihood'],
                'garch_aic': garch_result.get('aic', np.nan),
                'garch_bic': garch_result.get('bic', np.nan),
                'garch_is_stationary': float(alpha + beta < 1.0),
                'garch_converged': garch_result['converged'],
                'garch_iterations': garch_result['iterations'],
                
                # Volatility statistics
                **volatility_stats,
                
                # Volatility autocorrelations
                **volatility_autocorr,
                
                # Residual statistics
                **residual_stats,
                
                # Volatility forecasting
                **volatility_forecast,
                
                # Additional metrics
                **additional_metrics
            }
            
            return comprehensive_result
            
        except Exception as e:
            self._critical_error(f"Error adding comprehensive GARCH features: {e}")
    
    def _calculate_volatility_statistics(self, variance_series: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive volatility statistics.
        """
        try:
            volatility = np.sqrt(variance_series)
            
            stats = {
                'garch_vol_mean': float(np.mean(volatility)),
                'garch_vol_std': float(np.std(volatility)),
                'garch_vol_skew': float(self._calculate_skewness(volatility)),
                'garch_vol_kurt': float(self._calculate_kurtosis(volatility)),
                'garch_vol_min': float(np.min(volatility)),
                'garch_vol_max': float(np.max(volatility)),
                'garch_vol_median': float(np.median(volatility)),
                'garch_vol_q25': float(np.percentile(volatility, 25)),
                'garch_vol_q75': float(np.percentile(volatility, 75)),
                'garch_vol_range': float(np.max(volatility) - np.min(volatility)),
                'garch_vol_cv': float(np.std(volatility) / np.mean(volatility)) if np.mean(volatility) > 0 else 0.0
            }
            
            return stats
            
        except Exception as e:
            self._critical_error(f"Error calculating volatility statistics: {e}")
    
    def _calculate_volatility_autocorrelations(self, variance_series: np.ndarray) -> Dict[str, float]:
        """
        Calculate volatility autocorrelations at different lags.
        """
        try:
            volatility = np.sqrt(variance_series)
            
            autocorr = {}
            lags = [1, 5, 10, 20, 50]
            
            for lag in lags:
                if len(volatility) > lag:
                    autocorr_value = self._calculate_autocorrelation_numpy(volatility, lag)
                    autocorr[f'garch_vol_autocorr_lag{lag}'] = autocorr_value
                else:
                    autocorr[f'garch_vol_autocorr_lag{lag}'] = np.nan
            
            return autocorr
            
        except Exception as e:
            self._critical_error(f"Error calculating volatility autocorrelations: {e}")
    
    def _calculate_residual_statistics(self, returns: cp.ndarray, variance_series: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive residual statistics.
        """
        try:
            # Calculate standardized residuals
            volatility = np.sqrt(variance_series)
            standardized_residuals = returns[1:] / volatility  # Skip first observation
            
            # Remove any infinite or NaN values
            valid_mask = np.isfinite(standardized_residuals)
            clean_residuals = standardized_residuals[valid_mask]
            
            if len(clean_residuals) == 0:
                return {f'garch_residual_{stat}': np.nan for stat in ['mean', 'std', 'skew', 'kurt', 'jarque_bera']}
            
            stats = {
                'garch_residual_mean': float(np.mean(clean_residuals)),
                'garch_residual_std': float(np.std(clean_residuals)),
                'garch_residual_skew': float(self._calculate_skewness(clean_residuals)),
                'garch_residual_kurt': float(self._calculate_kurtosis(clean_residuals)),
                'garch_residual_jarque_bera': float(self._calculate_jarque_bera_statistic(clean_residuals))
            }
            
            return stats
            
        except Exception as e:
            self._critical_error(f"Error calculating residual statistics: {e}")
    
    def _calculate_volatility_forecast(self, omega: float, alpha: float, beta: float, variance_series: np.ndarray) -> Dict[str, float]:
        """
        Calculate volatility forecasts at different horizons.
        """
        try:
            # Get the last variance value
            last_variance = variance_series[-1]
            
            # Calculate forecasts at different horizons
            forecasts = {}
            horizons = [1, 5, 10, 20]
            
            for h in horizons:
                # GARCH(1,1) forecast formula: E[σ²_{t+h}] = ω + (α + β)^h * (σ²_t - ω/(1-α-β))
                if alpha + beta < 1.0:
                    long_run_variance = omega / (1 - alpha - beta)
                    forecast_variance = long_run_variance + (alpha + beta)**h * (last_variance - long_run_variance)
                else:
                    # If not stationary, use simple extrapolation
                    forecast_variance = last_variance
                
                forecasts[f'garch_vol_forecast_h{h}'] = float(np.sqrt(forecast_variance))
            
            return forecasts
            
        except Exception as e:
            self._critical_error(f"Error calculating volatility forecast: {e}")
    
    def _calculate_additional_garch_metrics(self, omega: float, alpha: float, beta: float, 
                                          returns: cp.ndarray, variance_series: np.ndarray) -> Dict[str, float]:
        """
        Calculate additional GARCH metrics and diagnostics.
        """
        try:
            metrics = {}
            
            # Leverage effect (asymmetric response)
            negative_returns = returns[returns < 0]
            positive_returns = returns[returns > 0]
            
            if len(negative_returns) > 0 and len(positive_returns) > 0:
                neg_vol = float(cp.std(negative_returns))
                pos_vol = float(cp.std(positive_returns))
                leverage_effect = neg_vol / pos_vol if pos_vol > 0 else 1.0
                metrics['garch_leverage_effect'] = leverage_effect
            else:
                metrics['garch_leverage_effect'] = 1.0
            
            # Volatility clustering measure
            volatility = np.sqrt(variance_series)
            volatility_changes = np.diff(volatility)
            volatility_clustering = float(np.corrcoef(volatility[:-1], volatility[1:])[0, 1]) if len(volatility) > 1 else 0.0
            metrics['garch_volatility_clustering'] = volatility_clustering
            
            # Mean reversion speed
            if alpha + beta < 1.0:
                mean_reversion_speed = 1 - (alpha + beta)
                metrics['garch_mean_reversion_speed'] = float(mean_reversion_speed)
            else:
                metrics['garch_mean_reversion_speed'] = 0.0
            
            # Half-life of volatility shocks
            if alpha + beta < 1.0:
                half_life = np.log(0.5) / np.log(alpha + beta)
                metrics['garch_volatility_half_life'] = float(half_life)
            else:
                metrics['garch_volatility_half_life'] = np.inf
            
            return metrics
            
        except Exception as e:
            self._critical_error(f"Error calculating additional GARCH metrics: {e}")
    
    def _add_comprehensive_garch_features(self, df: cudf.DataFrame, garch_result: Dict[str, Any]) -> cudf.DataFrame:
        """
        Add all comprehensive GARCH features to the DataFrame.
        """
        try:
            # Add all GARCH features as columns
            for key, value in garch_result.items():
                df[key] = value
            
            return df
            
        except Exception as e:
            self._critical_error(f"Error adding GARCH features to DataFrame: {e}")
    
    def _get_default_garch_result(self) -> Dict[str, Any]:
        """
        Get default GARCH result with NaN values.
        """
        return {
            'garch_omega': np.nan, 'garch_alpha': np.nan, 'garch_beta': np.nan,
            'garch_persistence': np.nan, 'garch_log_likelihood': np.nan,
            'garch_aic': np.nan, 'garch_bic': np.nan, 'garch_is_stationary': np.nan,
            'garch_converged': False, 'garch_iterations': 0,
            'garch_vol_mean': np.nan, 'garch_vol_std': np.nan, 'garch_vol_skew': np.nan,
            'garch_vol_kurt': np.nan, 'garch_vol_min': np.nan, 'garch_vol_max': np.nan,
            'garch_vol_median': np.nan, 'garch_vol_q25': np.nan, 'garch_vol_q75': np.nan,
            'garch_vol_range': np.nan, 'garch_vol_cv': np.nan,
            'garch_vol_autocorr_lag1': np.nan, 'garch_vol_autocorr_lag5': np.nan,
            'garch_vol_autocorr_lag10': np.nan, 'garch_vol_autocorr_lag20': np.nan,
            'garch_vol_autocorr_lag50': np.nan,
            'garch_residual_mean': np.nan, 'garch_residual_std': np.nan,
            'garch_residual_skew': np.nan, 'garch_residual_kurt': np.nan,
            'garch_residual_jarque_bera': np.nan,
            'garch_vol_forecast_h1': np.nan, 'garch_vol_forecast_h5': np.nan,
            'garch_vol_forecast_h10': np.nan, 'garch_vol_forecast_h20': np.nan,
            'garch_leverage_effect': np.nan, 'garch_volatility_clustering': np.nan,
            'garch_mean_reversion_speed': np.nan, 'garch_volatility_half_life': np.nan
        }
    
    # Helper methods for statistical calculations
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 3))
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 4))
    
    def _calculate_autocorrelation_numpy(self, data: np.ndarray, lag: int) -> float:
        """Calculate autocorrelation at given lag."""
        if lag >= len(data):
            return 0.0
        
        data_lagged = data[lag:]
        data_original = data[:-lag]
        
        mean_original = np.mean(data_original)
        mean_lagged = np.mean(data_lagged)
        
        numerator = np.sum((data_original - mean_original) * (data_lagged - mean_lagged))
        denominator = np.sqrt(np.sum((data_original - mean_original)**2) * np.sum((data_lagged - mean_lagged)**2))
        
        if denominator > 1e-9:
            return float(numerator / denominator)
        else:
            return 0.0
    
    def _calculate_jarque_bera_statistic(self, data: np.ndarray) -> float:
        """Calculate Jarque-Bera test statistic."""
        n = len(data)
        skewness = self._calculate_skewness(data)
        kurtosis = self._calculate_kurtosis(data)
        
        jb_stat = n * (skewness**2 / 6 + (kurtosis - 3)**2 / 24)
        return float(jb_stat)
