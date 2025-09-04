"""
Main Orchestration Script for Dynamic Stage 0 Pipeline

This script manages the Dask-CUDA cluster lifecycle and provides the foundation
for the GPU-accelerated feature engineering pipeline using the new modular architecture.
"""

import logging
import sys
import os
import signal
import time
import threading
from typing import Optional, Dict, Any, List
from contextlib import contextmanager

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dask_cuda import LocalCUDACluster
    from dask.distributed import Client
    import cupy as cp
    import cudf
except ImportError as e:
    print(f"Error importing Dask-CUDA libraries: {e}")
    print("Make sure the GPU environment is properly set up.")
    sys.exit(1)

from config.unified_config import get_unified_config
from orchestration.pipeline_orchestrator import PipelineOrchestrator
from orchestration.data_processor import process_currency_pair_worker
from features.base_engine import CriticalPipelineError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global flag for emergency shutdown
EMERGENCY_SHUTDOWN = threading.Event()


def emergency_shutdown_handler(signum, frame):
    """Handle emergency shutdown signals."""
    logger.critical("🚨 EMERGENCY SHUTDOWN SIGNAL RECEIVED")
    logger.critical("🛑 Initiating immediate shutdown of all processes...")
    EMERGENCY_SHUTDOWN.set()
    sys.exit(1)


# Register signal handlers for emergency shutdown
signal.signal(signal.SIGINT, emergency_shutdown_handler)
signal.signal(signal.SIGTERM, emergency_shutdown_handler)


class DaskClusterManager:
    """Manages the Dask-CUDA cluster lifecycle for Dynamic Stage 0 pipeline."""

    def __init__(self):
        """Initialize the cluster manager with unified configuration."""
        self.config = get_unified_config()
        self.cluster: Optional[LocalCUDACluster] = None
        self.client: Optional[Client] = None
        self.hostname = os.uname().nodename if hasattr(os, 'uname') else 'unknown'

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown()
        sys.exit(0)

    def _get_gpu_count(self) -> int:
        """Get the number of available GPUs."""
        try:
            return cp.cuda.runtime.getDeviceCount()  # Detecta número de GPUs NVIDIA disponíveis
        except Exception as e:
            logger.warning(f"Could not detect GPU count: {e}")
            return 1

    def _configure_rmm(self):
        """Configure RMM (RAPIDS Memory Manager) for optimal memory management."""
        try:
            from rmm import reinitialize  # Importa gerenciador de memória RAPIDS
            
            # Determine safe pool sizes based on actual device memory
            def parse_size_gb(val: str) -> float:  # Converte strings de tamanho para GB
                v = str(val).strip().upper()
                if v.endswith('GB'):
                    return float(v[:-2])
                if v.endswith('MB'):
                    return float(v[:-2]) / 1024.0
                return float(v)

            try:
                free_b, total_b = cp.cuda.runtime.memGetInfo()  # Obtém memória livre e total da GPU
                total_gb = total_b / (1024 ** 3)  # Converte bytes para GB
            except Exception:
                total_gb = 8.0  # Fallback conservador se não conseguir detectar memória

            # Proportional sizing if configured; fallback to fixed sizes
            pool_frac = float(getattr(self.config.dask, 'rmm_pool_fraction', 0.0) or 0.0)
            init_frac = float(getattr(self.config.dask, 'rmm_initial_pool_fraction', 0.0) or 0.0)
            max_frac = float(getattr(self.config.dask, 'rmm_maximum_pool_fraction', 0.0) or 0.0)

            if pool_frac > 0.0:
                desired_pool_gb = max(0.25, total_gb * pool_frac)
            else:
                desired_pool_gb = parse_size_gb(self.config.dask.rmm_pool_size)  # Tamanho desejado do pool em GB

            if init_frac > 0.0:
                desired_init_gb = max(0.25, total_gb * init_frac)
            else:
                desired_init_gb = parse_size_gb(self.config.dask.rmm_initial_pool_size)  # Tamanho inicial desejado em GB

            # Limita a fração máxima da memória total da GPU para evitar overallocation do pool
            if max_frac > 0.0:
                cap_gb = max(0.25, total_gb * max_frac)
            else:
                cap_gb = max(0.25, total_gb * 0.60)  # Máximo de 60% da memória total, mínimo 0.25GB
            safe_pool_gb = max(0.25, min(desired_pool_gb, cap_gb))  # Pool seguro (não excede o limite)
            safe_init_gb = max(0.25, min(desired_init_gb, safe_pool_gb))  # Inicial seguro (não excede o pool)

            # Persist safe values for cluster kwargs
            self._safe_rmm_pool_size_str = f"{safe_pool_gb:.2f}GB"  # String formatada para cluster kwargs
            initial_pool_size = int(safe_init_gb * (1024 ** 3))  # Converte GB para bytes para RMM

            # Tenta pool CUDA malloc async primeiro (se disponível), senão pool clássico
            try:
                reinitialize(
                    pool_allocator=True,  # Habilita pool allocator (mais eficiente)
                    initial_pool_size=initial_pool_size,  # Tamanho inicial do pool em bytes
                    managed_memory=False  # Desabilita managed memory (mais previsível)
                )
                logger.info(
                    "✅ RMM configured (pool) with %.2fGB initial, pool=%.2fGB (cap=%.2fGB of %.2fGB)",
                    safe_init_gb, safe_pool_gb, cap_gb, total_gb
                )
            except Exception as e_pool:
                logger.warning("RMM pool init failed (%s). Falling back to default CUDA allocator.", e_pool)
                os.environ.setdefault("RMM_ALLOCATOR", "cuda_malloc")  # Fallback para allocator padrão CUDA
            
        except ImportError:
            logger.warning("⚠️ RMM not available, using default CUDA memory management")
            os.environ.setdefault("RMM_ALLOCATOR", "cuda_malloc")  # Usa allocator padrão CUDA se RMM não disponível
        except Exception as e:
            logger.error(f"Failed to configure RMM: {e}")
            os.environ.setdefault("RMM_ALLOCATOR", "cuda_malloc")  # Fallback para allocator padrão em caso de erro

    def start_cluster(self) -> bool:
        """
        Start the Dask-CUDA cluster with proper RMM configuration.
        Returns:
            bool: True if cluster started successfully, False otherwise
        """
        try:
            logger.info("Starting Dask-CUDA cluster...")
            gpu_count = self._get_gpu_count()  # Detecta número de GPUs disponíveis
            logger.info(f"Detected {gpu_count} GPU(s)")

            # Configure RMM before creating cluster
            self._configure_rmm()  # Configura gerenciamento de memória GPU antes de criar cluster

            # Configure worker memory via dask config to avoid deprecation warnings
            try:
                import dask
                dask.config.set({
                    'distributed.worker.memory.target': float(self.config.dask.memory_target_fraction),  # Alvo de utilização de RAM (80%)
                    'distributed.worker.memory.spill': float(self.config.dask.memory_spill_fraction),  # Inicia spill quando RAM atinge 90%
                })
                logger.info("Dask memory config set (target=%.2f, spill=%.2f)",
                            float(self.config.dask.memory_target_fraction),
                            float(self.config.dask.memory_spill_fraction))
            except Exception as e:
                logger.warning(f"Could not set Dask memory config: {e}")

            # Build cluster configuration from unified config (omit deprecated kwargs)
            cluster_kwargs = {
                'n_workers': gpu_count,  # Um worker por GPU (isolamento completo)
                'threads_per_worker': self.config.dask.threads_per_worker,  # Uma thread por worker (evita conflitos GPU)
                # Use safe pool size string if computed during RMM config
                'rmm_pool_size': getattr(self, '_safe_rmm_pool_size_str', self.config.dask.rmm_pool_size),  # Pool de memória GPU calculado dinamicamente
                'local_directory': self.config.dask.local_directory,  # Diretório para spill de dados quando RAM fica cheia
            }

            # Add UCX configuration if enabled
            if self.config.dask.protocol == "ucx":  # Se protocolo UCX estiver habilitado
                cluster_kwargs.update({
                    'protocol': "ucx",  # Protocolo UCX para comunicação GPU-GPU
                    'enable_tcp_over_ucx': self.config.dask.enable_tcp_over_ucx,  # Habilita TCP sobre UCX
                    'enable_infiniband': self.config.dask.enable_infiniband,  # InfiniBand (geralmente desabilitado)
                    'enable_nvlink': self.config.dask.enable_nvlink,  # NVLink para comunicação direta entre GPUs
                })

            try:
                logger.info("Creating LocalCUDACluster with UCX...")
                self.cluster = LocalCUDACluster(**cluster_kwargs)  # Tenta criar cluster com UCX (mais rápido)
            except Exception as ucx_err:
                logger.warning(f"UCX unavailable ({ucx_err}); falling back to TCP.")
                # Remove UCX-specific parameters for TCP fallback
                for key in ['protocol', 'enable_tcp_over_ucx', 'enable_infiniband', 'enable_nvlink']:  # Remove parâmetros UCX
                    cluster_kwargs.pop(key, None)
                self.cluster = LocalCUDACluster(**cluster_kwargs)  # Cria cluster com TCP (fallback mais lento mas funcional)

            logger.info("✓ Cluster created successfully")
            self.client = Client(self.cluster)  # Cria cliente Dask para comunicação com cluster
            logger.info("✓ Client created successfully")

            logger.info("Waiting for workers to be ready...")
            self.client.wait_for_workers(gpu_count, timeout=300)  # Aguarda todos os workers ficarem prontos (timeout 5min)
            logger.info(f"✓ {gpu_count} workers ready")
            logger.info(f"✓ Dashboard URL: {self.client.dashboard_link}")
            logger.info(f"✓ Active workers: {len(self.client.scheduler_info()['workers'])}")
            
            # Set up worker death monitoring
            self._setup_worker_monitoring()  # Configura monitoramento para detectar falhas de workers
            
            return True

        except Exception as e:
            logger.error(f"Failed to start Dask-CUDA cluster: {e}", exc_info=True)
            self.shutdown()  # Limpa recursos em caso de falha
            return False

    def get_client(self) -> Optional[Client]:
        """Get the Dask client instance."""
        return self.client  # Retorna cliente Dask para comunicação com cluster

    def get_cluster(self) -> Optional[LocalCUDACluster]:
        """Get the Dask-CUDA cluster instance."""
        return self.cluster  # Retorna instância do cluster Dask-CUDA

    def is_active(self) -> bool:
        """Check if the cluster is active."""
        return self.cluster is not None and self.client is not None  # Verifica se cluster e cliente estão ativos
        
    def shutdown(self):
        """Shutdown the cluster and client gracefully."""
        logger.info("Shutting down Dask-CUDA cluster...")
        try:
            if self.client:
                self.client.close()  # Fecha cliente Dask
            if self.cluster:
                self.cluster.close()  # Fecha cluster Dask-CUDA
            logger.info("Dask-CUDA cluster shutdown complete")
        except Exception as e:
            logger.error(f"Error during cluster shutdown: {e}")
        finally:
            self.client = None  # Limpa referência ao cliente
            self.cluster = None  # Limpa referência ao cluster
    
    def __enter__(self):
        """Context manager entry."""
        if not self.start_cluster():  # Inicia cluster ao entrar no contexto
            raise RuntimeError("Failed to start Dask-CUDA cluster")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()  # Fecha cluster ao sair do contexto (garantia de limpeza)

    def _setup_worker_monitoring(self):
        """Set up monitoring to detect worker deaths and stop pipeline."""
        try:
            # Get initial worker count
            self.initial_worker_count = len(self.client.scheduler_info()["workers"])  # Conta workers iniciais
            logger.info(f"Monitoring {self.initial_worker_count} workers for failures")
            
            # Set up periodic check
            def monitor_workers():
                while True:
                    try:
                        current_workers = len(self.client.scheduler_info()["workers"])  # Conta workers atuais
                        if current_workers < self.initial_worker_count:  # Se algum worker morreu
                            logger.critical("🚨 WORKER DEATH DETECTED!")
                            logger.critical(f"Workers: {current_workers}/{self.initial_worker_count}")
                            logger.critical("🛑 STOPPING PIPELINE IMMEDIATELY")
                            EMERGENCY_SHUTDOWN.set()  # Sinaliza shutdown de emergência
                            break
                        time.sleep(5)  # Check every 5 seconds
                    except Exception as e:
                        logger.error(f"Error in worker monitoring: {e}")
                        break
            
            # Start monitoring in background thread
            monitor_thread = threading.Thread(target=monitor_workers, daemon=True)  # Thread daemon (não bloqueia shutdown)
            monitor_thread.start()  # Inicia monitoramento em background
            
        except Exception as e:
            logger.error(f"Failed to setup worker monitoring: {e}")


@contextmanager
def managed_dask_cluster():
    """Context manager for Dask-CUDA cluster lifecycle."""
    cluster_manager = DaskClusterManager()  # Cria gerenciador de cluster
    try:
        if not cluster_manager.start_cluster():  # Inicia cluster
            raise RuntimeError("Failed to start Dask-CUDA cluster")
        yield cluster_manager  # Retorna cluster ativo
    finally:
        cluster_manager.shutdown()  # Garante shutdown mesmo em caso de erro


def run_pipeline():
    """Run the complete Dynamic Stage 0 pipeline using the new modular architecture."""
    logger.info("=" * 60)
    logger.info("Dynamic Stage 0 - Pipeline Execution (Modular Architecture)")
    logger.info("=" * 60)

    try:
        # Get unified configuration
        config = get_unified_config()  # Carrega configuração unificada (YAML + variáveis de ambiente)
        
        # Initialize pipeline orchestrator
        orchestrator = PipelineOrchestrator()  # Cria orquestrador do pipeline
        
        # Connect to database (non-fatal; continue without DB if unavailable)
        if not orchestrator.connect_database():  # Tenta conectar ao banco (não é fatal)
            logger.warning("Database unavailable; continuing without task tracking.")

        # Discover tasks
        pending_tasks = orchestrator.discover_tasks()  # Descobre pares de moeda que precisam de processamento
        if not pending_tasks:
            logger.info("No tasks to process. Pipeline complete.")
            return 0

        logger.info(f"Processing {len(pending_tasks)} currency pairs that need feature engineering")  # Log do número de tarefas

        # Execute pipeline with cluster management
        with managed_dask_cluster() as cluster_manager:  # Context manager garante shutdown automático
            if not cluster_manager.is_active():  # Verifica se cluster está ativo
                logger.error("Cluster manager is not active")
                raise RuntimeError("Failed to start or activate Dask cluster.")
            
            client = cluster_manager.get_client()  # Obtém cliente Dask do cluster
            if not client:
                logger.error("Failed to get Dask client")
                raise RuntimeError("Failed to get Dask client.")
            
            # Start DB-backed run lifecycle (best-effort)
            try:
                orchestrator.start_run(
                    dashboard_url=getattr(client, 'dashboard_link', None),  # URL do dashboard Dask
                    hostname=cluster_manager.hostname,  # Hostname da máquina
                )
            except Exception as e:
                logger.warning(f"Run lifecycle tracking unavailable: {e}")  # Não é fatal se falhar

            # Execute the pipeline
            result = orchestrator.execute_pipeline(cluster_manager, client)  # Executa pipeline completo
            
            # Log pipeline summary
            orchestrator.log_pipeline_summary(result, len(pending_tasks))  # Log do resumo da execução
            
            # Clean up
            orchestrator.cleanup()  # Limpa recursos e conexões
            
            # Check if emergency shutdown was triggered
            if result.emergency_shutdown:  # Se shutdown de emergência foi acionado
                logger.critical("🚨 PIPELINE STOPPED DUE TO EMERGENCY SHUTDOWN")
                try:
                    orchestrator.end_run(status='ABORTED')  # Marca execução como abortada
                finally:
                    return 1  # Código de erro para shutdown de emergência
            
            exit_code = 0 if result.failed_tasks == 0 else 1  # 0 = sucesso, 1 = falha
            try:
                orchestrator.end_run(status='COMPLETED' if exit_code == 0 else 'FAILED')  # Marca status final da execução
            finally:
                return exit_code  # Retorna código de saída apropriado

    except Exception as e:
        logger.error(f"FATAL PIPELINE ERROR: {e}", exc_info=True)  # Log de erro fatal com stack trace
        return 1  # Código de erro para falha fatal


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # Suporte para multiprocessing em executáveis congelados
    sys.exit(run_pipeline())  # Executa pipeline e sai com código de retorno apropriado
