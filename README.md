# Módulo Feature Genesis (GPU) — Estágio 0 e Pré‑Seleção

Este módulo implementa a geração e preparação de features com aceleração em GPU para séries temporais de Forex. O foco atual é o Estágio 0 (estacionarização e engenharia inicial) e testes estatísticos básicos, usando um cluster Dask‑CUDA para processar cada par de moedas utilizando todas as GPUs disponíveis (processamento por moeda, sequencial entre moedas).

O design e o roadmap seguem o documento “Um Pipeline Robusto de Seleção de Features em Múltiplos Estágios…”. Tudo o que ainda não estiver coerente com o documento está marcado neste README como “A IMPLEMENTAR”.

## Visão Geral

- Cluster multi‑GPU com `dask‑cuda.LocalCUDACluster` e `dask.distributed.Client`.
- Processamento por moeda na máquina “driver”, alavancando o cluster (uma moeda por vez; usa todas as GPUs por moeda).
- Dados carregados com `dask_cudf` (Feather/Parquet), processamento em `map_partitions` nos ENGINES e `persist()/wait()` entre estágios; ao final, `compute()` para `cudf.DataFrame` e persistência em Feather v2.
- Saída consolidada por par em formato Feather v2, com compressão leve.

## Arquitetura do Módulo

- `orchestration/main.py`: gerencia ciclo de vida do cluster Dask‑CUDA, RMM e sinalização de encerramento.
- `orchestration/pipeline_orchestrator.py`: descobre tarefas (pares disponíveis), dispara o processamento por moeda no driver e aplica fail‑fast em caso de erro.
- `orchestration/data_processor.py`: orquestra o fluxo por moeda (carregamento → ENGINES → validação → salvamento).
- `data_io/local_loader.py` e `data_io/r2_loader.py`: leitura de Feather/Parquet como `dask_cudf.DataFrame` (e fallback síncrono para `cudf` quando necessário).
- `features/base_engine.py`: base comum aos ENGINES (validações leves compatíveis com Dask, monitoramento de memória, logging e utilidades).
- ENGINES (em `features/`):
  - `stationarization.py` (StationarizationEngine): Diferenciação Fracionária (FFD), variantes de estacionarização e correl. rolantes, com implementações vetorizadas em GPU.
  - `statistical_tests.py` (StatisticalTests): ADF e correlação de distância (dCor) em GPU (ver limitações abaixo).
  - `signal_processing.py` (SignalProcessor): filtros e transformações de sinal no domínio do tempo/estacionário.
  - `garch_models.py` (GARCHModels): ajuste GARCH(1,1) com log‑verossimilhança vetorizada em GPU e extração de métricas.

## Fluxo de Execução (por moeda)

1. Descoberta de pares e verificação de saída existente.
2. Carregamento (`dask_cudf.read_feather`/`read_parquet`) e `persist()`.
3. Execução dos ENGINES na ordem definida em `config` (cada engine retorna o DataFrame com novas colunas):
   - StationarizationEngine → StatisticalTests → SignalProcessor → GARCHModels.
   - Entre motores: `persist()` + `client.wait(...)` para estabilizar o grafo/memória.
4. `compute()` para `cudf.DataFrame` e salvamento como Feather v2 (um arquivo por par).
5. Registro de status em base (quando habilitado).

## ENGINES — O que cada um faz

- StationarizationEngine (`features/stationarization.py`)
  - FFD: pesos vetorizados em GPU; busca de d ótimo com teste de estacionariedade simples (variância por períodos).
  - Séries geradas: `frac_diff_*` e estatísticas associadas; estabilizações de variância (log/sqrt); correl. rolantes entre categorias relevantes detectadas dinamicamente (preço, retornos, volume, spread, volatilidade, OFI).
  - Rotas Dask usam detecção dinâmica de colunas (e.g., `y_close`, `y_ret_1m`) para evitar dependência de nomes fixos.

- StatisticalTests (`features/statistical_tests.py`)
  - ADF vetorizado em GPU usando decomposição QR; versão rolling disponível para séries fracionadas.
  - Correlação de distância (dCor) vetorizada com “centering trick”. Para datasets grandes é usada amostra de cauda (<= 10k). Existe fallback simplificado (correlação Pearson) no caminho single‑fit para contornar limites de memória.

- SignalProcessor (`features/signal_processing.py`)
  - Transformações/indicadores no domínio da série já estabilizada (e.g., filtros, métricas derivadas). Descrições específicas nas funções do arquivo.

- GARCHModels (`features/garch_models.py`)
  - Ajuste GARCH(1,1) com log‑likelihood vetorizado em GPU (`cupyx.scipy.signal.lfilter`), parâmetros (ω, α, β), variância condicional, estatísticas de volatilidade e métricas associadas.

## Entradas e Saídas

- Entrada: Feather/Parquet com colunas mínimas esperadas (detectadas dinamicamente). Exemplos comuns: `y_close`, `y_ret_1m`, `y_tick_volume`, `timestamp`.
- Saída: arquivo Feather v2 por par, contendo as colunas originais + novas features produzidas pelos ENGINES.
- Caminhos de leitura/gravação: configuráveis em `config` (ver adiante).

## Configuração

- Dask/Cluster: número de workers (igual ao número de GPUs), `threads_per_worker`, RMM (`rmm_pool_size`), protocolo (UCX/TCP), diretório local de trabalho.
- Pipeline/Engines: ordem de execução, enable/disable por engine e parâmetros específicos (janelas de rolling, limites de amostra etc.).
- Output: diretório base para salvar os Feather consolidados por par.

- Config única: toda a configuração é lida de `config/config.yaml` via `config/unified_config.py`. O caminho legacy (`config/settings.py`/`config/config.py`) foi removido. Use apenas o unified.

Parâmetros relevantes (em `features` do `config.yaml`):
- `distance_corr_max_samples`: número máximo de amostras (cauda) para dCor em séries longas. Default: 10000.
- `distance_corr_tile_size`: tamanho do bloco (tile) para processamento por blocos na GPU. Default: 2048.
- `selection_target_column`: coluna alvo para ranking dCor (ex.: `y_ret_1m`).
- `selection_max_rows`: amostra máxima (CPU) para estágios 2–3.
- `dcor_min_threshold`, `dcor_min_percentile`, `stage1_top_n`: controlam retenção no Estágio 1.
- `vif_threshold`, `mi_threshold`: controlam redundância no Estágio 2.
- `stage3_top_n`: top‑N no Estágio 3 (wrappers).

Parâmetros adicionais (via unified_config, com defaults seguros):
- `fracdiff_cache_max_entries`: tamanho do cache LRU de pesos FFD (default: 32).
- `fracdiff_partition_threshold`: limiar de tamanho do kernel para tratamentos especiais (default: 4096).

## Implementado x A Implementar (alinhado ao documento)

Estágio 0 — Estacionarização
- OK: FFD vetorizado (GPU), correlações rolantes, estabilização de variância; seleção do d ótimo guiada por ADF (significância `features.adf_alpha`, fallback por razão de variâncias).
- OK: validações adicionais de qualidade de série (janelas e limites dinâmicos) e cache de pesos da FFD (com LRU) para d values muito longos. As séries passam por checagens de comprimento mínimo, percentuais de NaNs, baixa variância rolling e outliers (z‑score), com janelas adaptadas ao tamanho da série. Pesos da FFD agora são reutilizados entre tentativas de d, reduzindo recomputo e pressão de memória.

Estágio 1 — Filtragem Univariada (Relevância)
- OK: dCor em GPU (chunked) + caminho rápido 1D (opcional); ranking global (dcor_<feature>), retenção automática (threshold/percentil/top‑N), estágio opcional de permutação para Top‑K com filtro por p‑valor.
- OK: dCor “rolling por janela” com agregação temporal do ranking (média/mediana/pXX). Quando habilitado, o ranking pode usar os escores agregados por janela (`dcor_roll_<feature>`) em vez do global, conforme configuração.
- Parâmetros: `distance_corr_*`, `selection_target_column`, `selection_max_rows`, `dcor_min_threshold`, `dcor_min_percentile`, `stage1_top_n`, `dcor_permutation_top_k`, `dcor_permutations`, `dcor_pvalue_alpha`, `dcor_top_k` (logging).
  - Rolling (via unified_config): `stage1_rolling_enabled`, `stage1_rolling_window`, `stage1_rolling_step`, `stage1_rolling_min_periods`, `stage1_rolling_max_rows`, `stage1_rolling_max_windows`, `stage1_agg` (mean|median|min|max|p25|p75), `stage1_use_rolling_scores`.

Estágio 2 — Redundância (Linear e Não‑Linear)
- OK: VIF iterativo (CPU) seguido de clustering por MI (global) com seleção de representante por maior dCor; fallback para MI par‑a‑par quando clusterização não disponível.
- Escalabilidade: limites configuráveis de candidatos (top‑N por dCor), cálculo de MI por blocos (chunked) e threshold de cluster por MI normalizada.
- Parâmetros: `vif_threshold`, `mi_threshold`; e (via unified_config) `mi_cluster_enabled`, `mi_cluster_method`, `mi_cluster_threshold`, `mi_max_candidates`, `mi_chunk_size`.

Estágio 3 — Wrappers Leves (Consenso)
- OK: LassoCV (TimeSeriesSplit) + LightGBM otimizado para CPU (com seed/early‑stopping) e fallback RandomForest; seleção final por interseção/união top‑N. Suporte a regressão e classificação (auto ou forçado). Opcional: backend GPU via XGBoost GPU.
- Parâmetros: `stage3_top_n`; (via unified_config) `stage3_task` (auto|regression|classification), `stage3_random_state`, `stage3_lgbm_enabled`, `stage3_lgbm_num_leaves`, `stage3_lgbm_max_depth`, `stage3_lgbm_n_estimators`, `stage3_lgbm_learning_rate`, `stage3_lgbm_feature_fraction`, `stage3_lgbm_bagging_fraction`, `stage3_lgbm_bagging_freq`, `stage3_lgbm_early_stopping_rounds`, `stage3_use_gpu` (bool), `stage3_wrapper_backend` (lgbm|xgb_gpu).

Estágio 4 — Validação da Seleção (Robustez)
- OK (integrado): CPCV com purga/embargo após Estágio 3, em amostra CPU. Broadcast: `cpcv_splits`, `cpcv_top_features`.
- OK: persistência detalhada por fold (listas de features do topo por importância por fold e métricas por split) em `output/<PAIR>/cpcv/<TARGET>/` como `summary.json` e `folds.json`.
- OK: purga/embargo automáticos quando não configurados — inferidos a partir do horizonte do alvo (parse de sufixos como `_1m`, `_5m`, `_1h`, `_1d`) e da cadência temporal (`timestamp`).
- Parâmetros: `cpcv_enabled`, `cpcv_n_groups`, `cpcv_k_leave_out`, `cpcv_purge`, `cpcv_embargo`.

Multi‑Target e Personas
- A IMPLEMENTAR: execução do pipeline por combinação (Modelo/Persona, Target) e variação de relevância específica por target; estratégia híbrida com redundância global única (otimização) vs. redundância por target (mais rigorosa).

Implementação dCor
- OK: dCor em GPU com centragem por blocos (chunked), controlando memória e evitando matrizes n×n completas.
- OK: caminho rápido 1D (O(n log n) overall) via ordenação + decimação para `dcor_fast_1d_bins` antes do cálculo chunked, útil para séries muito longas.
- OK: parâmetros expostos em `config.yaml` (`distance_corr_max_samples`, `distance_corr_tile_size`, `dcor_fast_1d_enabled`, `dcor_fast_1d_bins`).
- OK: estágio opcional de permutação para Top‑K do Estágio 1, com filtro por `dcor_pvalue_alpha` e broadcast de `dcor_pvalue_<feature>`.

Prevenção de Data Leakage e Validação Temporal
- OK: CPCV com purga/embargo integrado no Estágio 4; purga/embargo automáticos baseados no horizonte do target quando não especificados.
- A FAZER (futuro): aplicar purga/embargo consistentes também em calibração de hiperparâmetros e treino final, e integrar labeling específicos (e.g., triple‑barrier) de ponta a ponta.

## Execução

- Suba o cluster Dask‑CUDA (vide `orchestration/main.py`).
- Rode o pipeline principal (descoberta, processamento por moeda, salvamento). Você pode usar:
  - `onstart.sh` + `environment.yml`: caminho recomendado. O `onstart` cria/atualiza o ambiente `dynamic-stage0` com as dependências corretas (inclui LightGBM). Recrie a instância se necessário.
  - `deploy_to_vast.sh`/`run_pipeline_vast.sh`: automatizam sync de código/dados e execução remota. O deploy inclui fallback para instalar `lightgbm` caso o ambiente já exista sem ele. A fonte da verdade continua sendo o `environment.yml`.

## Notas de Projeto

- O processamento entre moedas é sequencial por design para maximizar throughput por moeda (todas as GPUs servem uma única moeda por vez). Caso seja necessário reintroduzir concorrência entre moedas, isso deverá ser feito num ramo próprio e fora do escopo atual.
- As validações em `BaseEngine` para Dask evitam computações globais pesadas (e.g., `len(ddf)`), usando amostras com `head()`; isso favorece estabilidade operacional em datasets grandes.

## Pastas Principais

- `orchestration/`: cluster/driver, orquestração e entrypoints Python.
- `data_io/`: readers de dados locais/R2 (Feather/Parquet) com `dask_cudf`.
- `features/`: ENGINES e base (`BaseFeatureEngine`).
- `config/`: parâmetros de Dask, pipeline e paths de I/O.

---

Este README é a referência única deste módulo. Tarefas listadas como “A IMPLEMENTAR” seguem o documento de referência e constituem o roadmap imediato para completar o pipeline multiestágio de seleção de features.
