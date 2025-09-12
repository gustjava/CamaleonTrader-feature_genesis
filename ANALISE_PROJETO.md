
# Análise Completa do Pipeline de Orquestração - orchestration/main.py

## 1. Configurações Disponíveis e Suas Funções

### 1.1 Configurações de Banco de Dados (`database`)
- **`host`**: Endereço do servidor MySQL (padrão: localhost)
- **`port`**: Porta do MySQL (padrão: 3010)
- **`database`**: Nome do banco de dados (padrão: feature_genesis_db)
- **`username/password`**: Credenciais de acesso
- **`pool_size`**: Tamanho do pool de conexões (padrão: 10)
- **`max_overflow`**: Conexões extras permitidas (padrão: 20)
- **`pool_timeout`**: Timeout para obter conexão (padrão: 30s)
- **`pool_recycle`**: Tempo para reciclar conexões (padrão: 3600s)

### 1.2 Configurações de Armazenamento R2 (`r2`)
- **`account_id`**: ID da conta Cloudflare R2
- **`access_key/secret_key`**: Chaves de acesso para R2
- **`bucket_name`**: Nome do bucket de armazenamento
- **`endpoint_url`**: URL do endpoint R2
- **`region`**: Região de armazenamento (padrão: auto)

### 1.3 Configurações do Cluster Dask-CUDA (`dask`)
- **`gpus_per_worker`**: GPUs por worker (padrão: 1)
- **`threads_per_worker`**: Threads por worker (padrão: 1)
- **`memory_limit`**: Limite fixo de RAM por worker (padrão: 0GB - desabilitado)
- **`memory_limit_fraction`**: Fração da RAM do sistema por worker (padrão: 0.25 = 25%)
- **`rmm_pool_fraction`**: Fração da GPU para pool RMM (padrão: 0.0 - desabilitado)
- **`rmm_initial_pool_fraction`**: Fração inicial do pool RMM (padrão: 0.0)
- **`spilling_enabled`**: Habilita spill para RAM (padrão: true)
- **`spilling_target`**: Alvo de utilização antes do spill (padrão: 0.9)
- **`memory_target_fraction`**: Alvo de utilização de RAM (padrão: 0.8)
- **`protocol`**: Protocolo de rede (tcp/ucx, padrão: tcp)
- **`enable_nvlink`**: Habilita NVLink (padrão: false)

### 1.4 Configurações de Feature Engineering (`features`)
- **`rolling_windows`**: Janelas móveis para cálculos (padrão: [10, 20, 50, 100, 200])
- **`frac_diff.d_values`**: Valores de diferenciação fracionária (padrão: [0.1, 0.2, 0.3, 0.4, 0.5])
- **`baxter_king`**: Configurações do filtro Baxter-King
  - `low_freq`: Frequência baixa (padrão: 6)
  - `high_freq`: Frequência alta (padrão: 32)
  - `k`: Tamanho do kernel (padrão: 12)
- **`garch`**: Configurações do modelo GARCH
  - `p/q`: Ordem do modelo GARCH (padrão: 1,1)
  - `max_iter`: Máximo de iterações (padrão: 1000)
  - `max_samples`: Máximo de amostras (padrão: 10000)
- **`selection_target_column`**: Coluna alvo para seleção (padrão: y_ret_fwd_15m)
- **`dcor_top_k`**: Top K features por correlação de distância (padrão: 50)
- **`stage1_rolling_enabled`**: Habilita correlação móvel (padrão: true)
- **`force_gpu_usage`**: Força uso de GPU (padrão: true)

### 1.5 Configurações de Pipeline (`pipeline.engines`)
- **`stationarization`**: Engine de estacionarização (ordem: 1)
- **`feature_engineering`**: Engine de feature engineering (ordem: 2)
- **`garch_models`**: Engine de modelos GARCH (ordem: 3)
- **`statistical_tests`**: Engine de testes estatísticos (ordem: 4)

### 1.6 Configurações de Desenvolvimento (`development`)
- **`debug_mode`**: Modo de debug (padrão: false)
- **`clean_existing_output`**: Limpa saídas existentes (padrão: false)
- **`force_reprocessing`**: Força reprocessamento (padrão: false)
- **`log_memory_usage`**: Log de uso de memória (padrão: true)

## 2. Tecnologias e Algoritmos por Passo


[1 tool called]

### 2.1 Inicialização e Configuração
**Tecnologias:**
- **YAML**: Formato de configuração hierárquica
- **Environment Variables**: Substituição dinâmica de variáveis (padrão: `${VAR:default}`)
- **Python dataclasses**: Estruturação de configurações com validação de tipos
- **LRU Cache**: Cache de configuração para evitar recarregamento

**Algoritmos:**
- **Configuração Unificada**: Sistema de configuração consolidado que combina múltiplas fontes
- **Validação de Configuração**: Verificação de parâmetros obrigatórios e limites

### 2.2 Gerenciamento de Cluster Dask-CUDA
**Tecnologias:**
- **Dask-CUDA**: Framework de computação distribuída para GPUs
- **LocalCUDACluster**: Cluster local com workers GPU
- **RMM (RAPIDS Memory Manager)**: Gerenciamento avançado de memória GPU
- **CuPy**: Computação numérica GPU
- **UCX**: Protocolo de comunicação de alta performance

**Algoritmos:**
- **Detecção Automática de GPU**: Contagem dinâmica de dispositivos CUDA disponíveis
- **Configuração Proporcional de Memória**: Alocação de memória baseada em frações da capacidade total
- **Fallback para CPU**: Sistema de fallback quando GPU não está disponível
- **Monitoramento de Workers**: Detecção de falhas de workers com shutdown automático

### 2.3 Engine 1: Estacionarização (`stationarization`)
**Tecnologias:**
- **Diferenciação Fracionária**: Algoritmo para tornar séries não-estacionárias em estacionárias
- **Convolução GPU**: Operações de convolução aceleradas por GPU
- **FFT (Fast Fourier Transform)**: Transformada rápida de Fourier para kernels grandes
- **Z-Score Rolling**: Normalização estatística móvel

**Algoritmos:**
- **Fractional Differentiation**: 
  - Usa pesos binomials para diferenciação de ordem fracionária
  - Preserva memória de longo prazo das séries temporais
  - Implementação GPU com CuPy para performance
- **Rolling Z-Score**: Normalização estatística móvel para estabilização de variância
- **Variance Stabilization**: Transformação logarítmica com shift para estabilizar variância
- **Rolling Correlation**: Correlação móvel entre pares de séries

### 2.4 Engine 2: Feature Engineering (`feature_engineering`)
**Tecnologias:**
- **Baxter-King Filter**: Filtro passa-banda para séries temporais
- **FFT Convolution**: Convolução via transformada de Fourier
- **GPU Acceleration**: Implementação CuPy para processamento paralelo

**Algoritmos:**
- **Baxter-King Bandpass Filter**:
  - Remove tendências de longo prazo (low_freq) e ruído de curto prazo (high_freq)
  - Kernel simétrico com pesos calculados via funções trigonométricas
  - Implementação causal para evitar look-ahead bias
  - Preenchimento inteligente de bordas com forward/backward fill

### 2.5 Engine 3: Modelos GARCH (`garch_models`)
**Tecnologias:**
- **GARCH(1,1)**: Modelo de volatilidade condicional
- **Maximum Likelihood Estimation**: Estimação de parâmetros via máxima verossimilhança
- **L-BFGS-B**: Algoritmo de otimização com restrições
- **Hybrid CPU-GPU**: CPU para otimização, GPU para computação numérica

**Algoritmos:**
- **GARCH(1,1) Model**:
  - Modela volatilidade condicional: σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁
  - ω: constante, α: choque, β: persistência
  - Log-likelihood gaussiana para estimação de parâmetros
  - Critérios de informação (AIC/BIC) para seleção de modelo
- **Stationarity Constraints**: Verificação de estacionariedade (α + β < 1)

### 2.6 Engine 4: Testes Estatísticos (`statistical_tests`)
**Tecnologias:**
- **Distance Correlation**: Correlação de distância para relações não-lineares
- **Permutation Tests**: Testes de permutação para significância estatística
- **Rolling Analysis**: Análise móvel de correlações
- **GPU Acceleration**: Implementação CuPy para cálculos paralelos

**Algoritmos:**
- **Distance Correlation (dCor)**:
  - Mede dependência não-linear entre variáveis
  - Baseado em distâncias euclidianas entre observações
  - Implementação 1D rápida com binning para eficiência
- **Permutation Testing**: Teste de significância via permutação de dados
- **Rolling dCor**: Correlação de distância móvel para análise temporal

### 2.7 Processamento de Dados
**Tecnologias:**
- **cuDF**: DataFrames GPU para processamento rápido
- **Dask-cuDF**: DataFrames distribuídos para escalabilidade
- **Feather v2**: Formato de serialização eficiente
- **Arrow**: Formato de dados colunar para interoperabilidade

**Algoritmos:**
- **Lazy Evaluation**: Computação sob demanda com Dask
- **Memory Management**: Gerenciamento inteligente de memória GPU
- **Partitioned Storage**: Armazenamento particionado para datasets grandes
- **Compression**: Compressão LZ4 para eficiência de I/O

### 2.8 Monitoramento e Logging
**Tecnologias:**
- **Structured Logging**: Logging estruturado com contexto
- **Pipeline Dashboard**: Dashboard web para monitoramento
- **Smart Alerts**: Sistema de alertas inteligentes
- **Memory Monitoring**: Monitoramento de uso de memória

**Algoritmos:**
- **Context-Aware Logging**: Logging com contexto de execução
- **Performance Metrics**: Coleta de métricas de performance
- **Anomaly Detection**: Detecção de anomalias em execução
- **Resource Monitoring**: Monitoramento de recursos computacionais


[1 tool called]

## 3. Passo a Passo Detalhado do Pipeline

### 3.1 Inicialização (`run_pipeline()`)
1. **Setup de Contexto de Execução**
   - Gera ID único de execução baseado em timestamp
   - Obtém hostname da máquina
   - Configura contexto de logging com run_id

2. **Carregamento de Configuração**
   - Carrega configuração unificada de `config.yaml`
   - Substitui variáveis de ambiente (padrão: `${VAR:default}`)
   - Valida configuração obrigatória

3. **Inicialização do Orchestrator**
   - Cria instância de `PipelineOrchestrator`
   - Inicializa handlers de banco de dados e carregamento local
   - Configura sistema de alertas e dashboard

### 3.2 Descoberta de Tarefas (`discover_tasks()`)
1. **Descoberta de Pares de Moeda**
   - Escaneia diretório de dados local
   - Identifica arquivos de dados disponíveis
   - Extrai metadados (tamanho, tipo, caminho)

2. **Verificação de Idempotência**
   - Verifica se arquivos de saída já existem
   - Aplica lógica de skip baseada em configuração
   - Remove arquivos existentes se `force_reprocessing=true`

3. **Criação de Tarefas**
   - Cria objetos `ProcessingTask` para cada par de moeda
   - Define caminhos de entrada e saída
   - Registra tarefas no dashboard

### 3.3 Gerenciamento de Cluster (`DaskClusterManager`)
1. **Detecção de GPU**
   - Conta GPUs disponíveis via CUDA runtime
   - Configura número de workers baseado em GPUs
   - Fallback para CPU se GPU não disponível

2. **Configuração RMM**
   - Calcula tamanhos de pool baseados em frações da GPU
   - Configura RMM para gerenciamento de memória
   - Fallback para CUDA malloc se RMM falhar

3. **Criação do Cluster**
   - Inicia `LocalCUDACluster` com configurações otimizadas
   - Cria cliente Dask distribuído
   - Aguarda workers ficarem prontos (timeout: 300s)

4. **Monitoramento de Workers**
   - Inicia thread de monitoramento
   - Detecta falhas de workers
   - Dispara shutdown de emergência se worker morrer

### 3.4 Execução do Pipeline (`execute_pipeline()`)
1. **Verificação de Cluster**
   - Valida se cluster está ativo
   - Verifica disponibilidade do cliente Dask
   - Loga diagnósticos do cluster

2. **Processamento Sequencial**
   - Processa cada par de moeda sequencialmente
   - Usa todos os GPUs por tarefa (driver-side processing)
   - Aplica fail-fast em caso de erro crítico

### 3.5 Processamento de Par de Moeda (`process_currency_pair_dask()`)
1. **Carregamento de Dados**
   - Carrega dados como `dask_cuDF` DataFrame
   - Suporta formatos Parquet e Feather
   - Aplica fallback entre formatos se necessário

2. **Validação Inicial**
   - Verifica se DataFrame não está vazio
   - Valida schema de colunas
   - Remove colunas negadas (deny list)

3. **Execução de Engines (Ordem Configurável)**
   - **Engine 1 - Estacionarização**: Aplica diferenciação fracionária e filtros
   - **Engine 2 - Feature Engineering**: Aplica filtro Baxter-King
   - **Engine 3 - GARCH Models**: Modela volatilidade condicional
   - **Engine 4 - Statistical Tests**: Calcula correlações de distância

4. **Persistência e Limpeza**
   - Persiste dados entre engines para estabilidade
   - Libera memória GPU entre engines
   - Valida dados após cada engine

### 3.6 Salvamento de Dados (`_save_processed_data_dask()`)
1. **Preparação para Salvamento**
   - Remove colunas de métricas se configurado
   - Cria diretório de saída
   - Define ordem de colunas

2. **Salvamento Particionado**
   - Converte para tarefas delayed
   - Salva cada partição como arquivo Feather separado
   - Usa compressão LZ4 para eficiência

3. **Verificação de Integridade**
   - Verifica arquivos salvos
   - Calcula tamanho total
   - Loga estatísticas de salvamento

### 3.7 Finalização e Limpeza
1. **Log de Resumo**
   - Gera resumo de execução
   - Cria diagramas de pipeline
   - Gera relatório de evolução de features

2. **Limpeza de Recursos**
   - Fecha conexões de banco de dados
   - Shutdown do cluster Dask
   - Libera memória GPU

3. **Tratamento de Erros**
   - Registra falhas no banco de dados
   - Aplica shutdown de emergência se necessário
   - Retorna código de saída apropriado


[1 tool called]

## Resumo Executivo

O pipeline de orquestração `orchestration/main.py` é um sistema sofisticado de processamento de dados financeiros que combina:

**Arquitetura:**
- **Computação Distribuída**: Dask-CUDA para processamento paralelo em múltiplas GPUs
- **Gerenciamento de Memória**: RMM para otimização de memória GPU
- **Processamento Sequencial**: Driver-side processing com uso de todas as GPUs por tarefa

**Engines de Feature Engineering:**
1. **Estacionarização**: Diferenciação fracionária e filtros para tornar séries estacionárias
2. **Feature Engineering**: Filtro Baxter-King para análise de frequências
3. **GARCH Models**: Modelagem de volatilidade condicional
4. **Statistical Tests**: Correlação de distância e testes de significância

**Características Técnicas:**
- **GPU-First**: Otimizado para processamento GPU com fallbacks inteligentes
- **Idempotente**: Suporte a reprocessamento e skip de tarefas já processadas
- **Fault-Tolerant**: Monitoramento de workers e shutdown de emergência
- **Configurável**: Sistema de configuração unificado e flexível
- **Observável**: Logging estruturado e dashboard de monitoramento

O sistema é projetado para processar grandes volumes de dados financeiros de forma eficiente, aplicando transformações estatísticas avançadas para extrair features relevantes para modelos de machine learning.