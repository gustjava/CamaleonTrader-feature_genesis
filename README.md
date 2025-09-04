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
  - **Engine 1**: `stationarization.py` (StationarizationEngine, order: 1): Diferenciação Fracionária (FFD), variantes de estacionarização e correl. rolantes, com implementações vetorizadas em GPU.
  - **Engine 2**: `feature_engineering.py` (FeatureEngineeringEngine, order: 2): transformações iniciais (ex.: Baxter–King generalizado por múltiplas colunas) aceleradas em GPU.
  - **Engine 3**: `garch_models.py` (GARCHModels, order: 3): ajuste GARCH(1,1) com log‑verossimilhança vetorizada em GPU e extração de métricas/colunas de volatilidade e resíduos para seleção.
  - **Engine 4**: `statistical_tests.py` (StatisticalTests, order: 4): ADF e correlação de distância (dCor) em GPU (ver limitações abaixo).

## Fluxo de Execução (por moeda)

1. Descoberta de pares e verificação de saída existente.
2. Carregamento (`dask_cudf.read_feather`/`read_parquet`) e `persist()`.
3. Execução dos ENGINES na ordem definida em `config` (cada engine retorna o DataFrame com novas colunas):
   - **Engine 1**: StationarizationEngine (order: 1) → **Engine 2**: FeatureEngineeringEngine (order: 2) → **Engine 3**: GARCHModels (order: 3) → **Engine 4**: StatisticalTests (order: 4).
   - Entre motores: `persist()` + `client.wait(...)` para estabilizar o grafo/memória.
4. `compute()` para `cudf.DataFrame` e salvamento como Feather v2 (um arquivo por par).
5. Registro de status em base (quando habilitado).

## ENGINES — O que cada um faz

**Engine 1 — StationarizationEngine (`features/stationarization.py`)**
- FFD: pesos vetorizados em GPU; busca de d ótimo com teste de estacionariedade simples (variância por períodos).
- Séries geradas: `frac_diff_*` e estatísticas associadas; estabilizações de variância (log/sqrt); correl. rolantes entre categorias relevantes detectadas dinamicamente (preço, retornos, volume, spread, volatilidade, OFI).
- Rotas Dask usam detecção dinâmica de colunas (e.g., `y_close`, `y_ret_1m`) para evitar dependência de nomes fixos.

**Engine 2 — FeatureEngineeringEngine (`features/feature_engineering.py`)**
- Transformações iniciais de sinal (ex.: Baxter–King generalizado para múltiplas colunas) operando em GPU; gera colunas `bk_filter_<col>`.

**Engine 3 — GARCHModels (`features/garch_models.py`)**
- Ajuste GARCH(1,1) com log‑likelihood vetorizado em GPU (`cupyx.scipy.signal.lfilter`), parâmetros (ω, α, β), variância condicional, estatísticas de volatilidade e métricas associadas.

**Engine 4 — StatisticalTests (`features/statistical_tests.py`)**
- ADF vetorizado em GPU usando decomposição QR; versão rolling disponível para séries fracionadas.
- Correlação de distância (dCor) vetorizada com "centering trick". Para datasets grandes é usada amostra de cauda (<= 10k). Existe fallback simplificado (correlação Pearson) no caminho single‑fit para contornar limites de memória.

## Entradas e Saídas

- Entrada: Feather/Parquet com colunas mínimas esperadas (detectadas dinamicamente). Exemplos comuns: `y_close`, `y_ret_1m`, `y_tickvol_z_15m`, `timestamp`.
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
  - Visibilidade: `stage1_broadcast_scores`/`stage1_broadcast_rolling` controlam se colunas `dcor_*` e `dcor_roll_*` são anexadas ao DataFrame (por padrão desativado para evitar “inchar” o schema).
  - Transparência: `debug_write_artifacts=true` persiste JSONs com escores e seleção em `<output_path>/<par>/artifacts/stage1/<target>/`.
  - Rolling (via unified_config): `stage1_rolling_enabled`, `stage1_rolling_window`, `stage1_rolling_step`, `stage1_rolling_min_periods`, `stage1_rolling_min_valid_pairs`, `stage1_rolling_max_rows`, `stage1_rolling_max_windows`, `stage1_agg` (mean|median|min|max|p25|p75), `stage1_use_rolling_scores`.

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
  - `onstart.sh` + `environment.yml`: caminho recomendado. O `onstart` cria/atualiza o ambiente `feature-genesis` com as dependências corretas (inclui LightGBM). Recrie a instância se necessário.
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

## Pipeline de Seleção de Features (Plano de Refatoração)

Resumo orientado a performance do pipeline proposto (sem alterar código ainda):

- Estágio 0 — Engenharia de Features
  - Mover Baxter–King (BK) para a engenharia inicial e generalizar para múltiplas colunas de origem via configuração.
  - Nomes: `bk_filter_<col_origem>`; casting para `float32`; bordas `NaN` nos `k` extremos.
  - GPU: `cp.convolve` para kernels curtos; FFT (`cusignal.fftconvolve`) quando `2k+1 > 129`; pesos gerados em CPU e cacheados por worker (L1), copiados 1x para GPU por tamanho.
  - Config (exemplo):
    feature_engineering:
      baxter_king:
        k: 12
        low_freq: 32
        high_freq: 6
        source_columns:
          - log_stabilized_y_close
          - ustbondtrusd_vol_60m

- Estágio 1 — Filtro Univariado (Gating)
  - Métricas obrigatórias vs. alvo: dCor, Pearson, MI (`mutual_info_regression`) e F‑test (`f_regression` p‑valor).
  - Regras (YAML): `dcor_min_threshold`, `correlation_min_threshold`, `pvalue_max_alpha`, `stage1_top_n`.
  - Saída: lista retida + tabela de métricas por feature; artefatos Parquet/JSON opcionais.
  - Performance: batching de candidatos; amostragem controlada para CPU‑only métricas; evitar broadcast de colunas de métricas (`stage1_broadcast_scores=false`).

- Estágio 2 — Redundância (VIF/MI)
  - Entrada: lista do Estágio 1; amostrar até `selection_max_rows` em CPU.
  - VIF iterativo até `vif_threshold`; MI por clusterização (preferencial) limitada por `mi_max_candidates` e `mi_chunk_size`.
  - Dedup 2.5: quando existirem pares `x` e `bk_filter_x`, mantém só um (maior dCor; empate → BK).
  - Saída: `stage2_features` enxuta (após VIF/MI + dedup) para wrappers.

- Estágio 3 — Seleção Multivariada (Wrapper/Embutido)
  - `SelectFromModel` com LightGBM (`LGBMRegressor`/`Classifier`), `importance_type=gain` e limiar `median` (ou valor configurável).
  - GPU quando disponível (`device=gpu`); early‑stopping opcional; `TimeSeriesSplit` para validação leve.
  - Artefatos: importâncias por feature e lista final.

- Estágio 4 — Estabilidade (Bootstrap em Blocos)
  - Repetir Estágio 3 em amostras temporais (block bootstrap/`TimeSeriesSplit` janelado) e medir frequência de seleção por feature.
  - Seleção final por limiar de estabilidade (ex.: ≥ 0.7); artefatos com distribuição de frequências.

### Diretrizes de Performance

- Minimizar ida/volta CPU↔GPU: gerar pesos/MI/VIF em CPU sobre amostras; manter convoluções/filtros em GPU.
- `float32` como padrão; controlar `persist()/wait()` entre engines para estabilizar DAG/memória.
- Batching para métricas univariadas e MI; limitar candidatos com `stage1_top_n` e `mi_max_candidates`.
- Em wrappers, limitar `selection_max_rows`, usar `early_stopping_rounds` e `feature_fraction/bagging_fraction` para velocidade.

### Configuração YAML (proposta)

- feature_engineering.baxter_king: parâmetros e `source_columns`.
- selection.stage1: thresholds (dCor, Pearson, MI, F‑test) e `top_n`.
- selection.stage3: `model: lgbm`, `importance_threshold: median|float`, `use_gpu: true`.
- selection.stage4: `n_bootstrap`, `block_size`, `stability_threshold`.

Observação: a estrutura YAML acima será integrada ao `config/config.yaml` numa próxima alteração. Até lá, manter chaves atuais em `features.*` e espelhar novas opções como aliases.

### Artefatos e Persistência

- Salvar métricas de Estágio 1, listas de Estágio 1/2/3 e importâncias do Estágio 3; no Estágio 4, salvar frequências (Parquet) e gráficos de barras (PNG) por par/alvo.

Referências detalhadas: ver `README_ENGINE2_FEATURE_ENGINEERING.md`, `README_STATISTICAL_TESTS.md` (atualizado como Estágio 1), `README_STAGE2_VIF_MI.md`, `README_STAGE3_WRAPPER_EMBEDDED.md` e `README_STAGE4_STABILITY.md`.
