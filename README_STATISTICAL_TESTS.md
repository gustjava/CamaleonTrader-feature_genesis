StatisticalTests — O Que Executa, Logs e Saídas

Este documento explica o comportamento prático do Estágio 2 (módulo StatisticalTests) e seus sub-estágios integrados (ranking, poda de redundância, wrappers e CPCV opcional). Ele mapeia as linhas de log que você vê para as computações reais e lista as chaves de configuração que controlam cada etapa.

Escopo (Caminho Dask)

O módulo StatisticalTests, quando executado em dask_cudf, realiza estas fases:

1) ADF Rolante em colunas fracdiff
- Objetivo: validar estacionaridade para séries fracionariamente diferenciadas.
- O que faz: para cada coluna fracdiff detectada (ex: frac_diff_<col>), aplica um teste ADF em modo rolante e registra a estatística rolante em uma nova coluna: adf_stat_<suffix>.
- Logs: 
  - "ADF rolling on '<col>'…" quando cada coluna fracdiff é processada.
- Config: features.adf_alpha (usado em outros lugares ao tomar decisões sobre resultados ADF).

2) dCor Single-Fit (verificação de sanidade)
- Objetivo: computar uma correlação de distância simples entre uma coluna de retornos e y_tickvol_z_15m como métrica rápida de sanidade.
- Comportamento: 
  - Seleciona uma coluna de retornos heuristicamente (ex: y_ret_1m ou returns) e a coluna de tick volume.
  - Amostra a cauda (<= features.distance_corr.max_samples; padrão 10k) e opcionalmente decima para memória.
  - Computa um escalar dCor único.
- Logs: 
  - "Computing single-fit distance correlation using <ret_col> and y_tickvol_z_15m…"
  - "Using tail sample (<= 10000) rows for dCor calculation"
  - "dCor computed | { 'n': N, 'tile': T, 'dcor': X, 'elapsed': s }"
    - n: número de amostras usadas; tile: tamanho de chunk interno usado.
  - O escalar é transmitido como dcor_returns_volume se bem-sucedido.

3) Estágio 1 — Ranking dCor vs Target (global)
- Objetivo: ranquear features candidatas por correlação de distância com o target (ex: y_ret_1m).
- Seleção de candidatos:
  - Parte de colunas float (por padrão) e exclui o target.
  - Vazamento e elegibilidade são controlados por listas no config (features):
    - dataset_target_columns: todos os rótulos/targets do dataset (excluídos)
    - feature_denylist / feature_deny_prefixes / feature_deny_regex: filtros de exclusão
    - feature_allowlist / feature_allow_prefixes: se definidos, atuam como allowlist (só entram os permitidos)
    - metrics_prefixes: remove métricas broadcastadas (ex.: dcor_*, dcor_roll_*, stage1_*, cpcv_*)
  - O alvo principal (selection_target_column) e quaisquer selection_target_columns são sempre excluídos.
- dCor Rolante (opcional): se features.stage1_rolling_enabled é true, computa dCor com janelas e agrega por feature (agg definido por features.stage1_agg; padrão median). Progresso é limitado por features.stage1_rolling_*.
  - Para lidar com séries esparsas (ex.: ativos off‑hours), use `features.stage1_rolling_min_valid_pairs` para exigir um número mínimo de observações pareadas (sem NaN) por janela ao calcular o dCor. Se o número de pares válidos for menor que esse mínimo, a janela é ignorada.
- Lotes e progresso: o conjunto de candidatos é dividido em lotes (features.dcor_batch_size, padrão 64). Após cada lote, uma linha de log é emitida:
  - "dCor batch completed | { 'processed': P, 'total': N, 'batch': B }"
- Tratamento de features all-NaN (rolante): quando uma feature tem zero janelas com dCor finito, o agregado rolante é NaN. O motor registra e transmite:
  - Log: "Rolling dCor all-NaN features | { 'count': K, 'sample': [...] }"
  - Escalares: dcor_roll_allnan_features (separado por vírgula) e dcor_roll_allnan_count.
- Retenção (três portões aplicados em ordem):
  - Threshold: mantém features com dCor >= features.dcor_min_threshold.
  - Percentil: mantém features acima de features.dcor_min_percentile (0..1) entre as retidas.
  - Top-N: se features.stage1_top_n > 0, mantém apenas os top N por dCor.
- P-valores de permutação (opcional):
  - Se features.dcor_permutation_top_k > 0 e features.dcor_permutations > 0, computa p-valores de permutação nos top-K features por dCor.
  - Filtra por features.dcor_pvalue_alpha.
- Logs:
  - "Computing dCor ranking | { 'target': ..., 'n_candidates': N, 'rolling': True|False }"
  - "Top-K dCor features | { 'top': [ 'feat:score', ... ], 'agg': 'median|min|max|...', 'source': 'dcor'|'dcor_roll' }"
- Escalares transmitidos:
  - stage1_features: lista retida separada por vírgula; stage1_features_count.

4) Estágio 2 — Poda de Redundância (VIF + MI)
- Objetivo: remover features redundantes após ranking do Estágio 1.
- Passos:
  - Amostra até features.selection_max_rows linhas na CPU.
  - Eliminação VIF iterativa usando scikit-learn.
  - Agrupamento MI (preferido) ou poda MI pairwise (threshold features.mi_threshold), com limites features.mi_max_candidates e features.mi_chunk_size.
- Saída (transmitida):
  - stage2_features, stage2_features_count.

5) Estágio 3 — Wrappers Leves (consenso)
- Objetivo: selecionar features finais usando importância baseada em modelo + Lasso.
- Backends: LightGBM (otimizado para CPU) ou XGBoost GPU (features.stage3_use_gpu, features.stage3_wrapper_backend). Fallback gracioso se backend GPU indisponível.
- Tipo de tarefa: auto (classificação se poucas classes discretas, senão regressão), ou forçado via features.stage3_task.
- Early stopping e splits: TimeSeriesSplit usado no Lasso; eval_set LightGBM opcional.
- Saída (transmitida):
  - selected_features, selected_features_count.
- Log: "Stage 3 selection done".

6) Estágio 4 — CPCV Opcional (robustez)
- Se features.cpcv_enabled é true, executa CV purgado combinatório sobre grupos com purge/embargo (auto-derivado do horizonte target quando não configurado; senão features.cpcv_*).
- Persiste detalhes por-fold em output/<PAIR>/cpcv/<TARGET>/{summary.json, folds.json}.
- Saída (transmitida): cpcv_splits e cpcv_top_features.
- Log: "Stage 4 CPCV complete | { splits, top }".

Folha de Cola de Configuração (chaves primárias)

- Target e candidatos
  - features.selection_target_column (ex: y_ret_1m)
  - features.selection_max_rows (amostra CPU para Estágios 2/3)
- Computação dCor
  - features.distance_corr_max_samples (limite single-fit)
  - features.distance_corr_tile_size (tiling)
  - features.dcor_batch_size (tamanho do lote para logs de progresso de ranking)
  - features.dcor_top_k (para logging ou filtragem de permutação)
  - features.dcor_min_threshold, features.dcor_min_percentile, features.stage1_top_n (portões de retenção)
- dCor Rolante
  - features.stage1_rolling_enabled, stage1_rolling_window, stage1_rolling_step
  - stage1_rolling_min_periods, stage1_rolling_max_rows, stage1_rolling_max_windows
- stage1_agg (median|min|max|mean|p25|p75)
- stage1_broadcast_scores / stage1_broadcast_rolling: controlam anexar colunas `dcor_*` e `dcor_roll_*` ao DataFrame. Útil desligar para manter o schema enxuto.
- debug_write_artifacts + artifacts_dir: quando ativado, salva JSONs com escores (globais/rolling), contagens e seleção em `<output_path>/<par>/artifacts/stage1/<target>/`.
- Teste de permutação
  - features.dcor_permutation_top_k, features.dcor_permutations, features.dcor_pvalue_alpha
- Redundância e Wrappers
  - features.vif_threshold, features.mi_threshold
  - features.mi_cluster_enabled, mi_cluster_method, mi_cluster_threshold, mi_max_candidates, mi_chunk_size
  - features.stage3_* (configurações LightGBM/XGBoost)
- CPCV
  - features.cpcv_enabled, cpcv_n_groups, cpcv_k_leave_out, cpcv_purge, cpcv_embargo

Interpretando os Logs de Exemplo (passo-a-passo)

1) "Starting StatisticalTests (Dask)…"
- Entra no módulo; prepara ADF/dCor.

2) "ADF rolling on 'frac_diff_…' …" (duas linhas)
- Executa ADF rolante sobre duas colunas fracdiff detectadas.

3) "Computing single-fit distance correlation …"
- Faz dCor simples entre uma coluna de retornos e y_tickvol_z_15m.

4) "Using tail sample (<= 10000) …" e "dCor computed | { 'n': 2048, 'tile': 2048, 'dcor': … }"
- Usou amostra curta para memória; n e tile iguais indica bloco único; dCor próximo de 0 significa fraca dependência distância nessa amostra.

5) "StatisticalTests complete."
- Concluiu esta sub-fase (ADF + dCor simples). Em seguida parte para o ranking global.

6) "Computing dCor ranking | { target: 'y_ret_1m', n_candidates: 212, rolling: True }"
- Vai calcular dCor para 212 candidatos, com rolling habilitado.

7) "RuntimeWarning: All-NaN slice encountered"
- Algumas features tiveram todas as janelas sem dCor finito. O agregador agora filtra NaNs/infinitos e, quando não há valores, retorna NaN sem warning e loga as features all-NaN (com contadores) para facilitar inspeção.

8) "dCor batch completed | { processed: 64/128/192/212, batch: 64/20 }"
- Progresso do ranking por lotes (features.dcor_batch_size). Cada linha mostra quantas features já foram processadas.

9) "Top-K dCor features | { top: ['feat:score', …], agg: 'median', source: 'dcor_roll' }"
- Lista consolidada das melhores features pelo dCor (ou pelo dCor rolling, se rolling enabled + selected), com a estatística usada.

Saídas (escalares transmitidos)

- Estágio 1: stage1_features, stage1_features_count
- Estágio 2: stage2_features, stage2_features_count
- Estágio 3: selected_features, selected_features_count
- Estágio 4 (opcional): cpcv_splits, cpcv_top_features

Solução de Problemas

- Features rolantes all-NaN: ver dcor_roll_allnan_features e dcor_roll_allnan_count (transmitidos) e os logs "Rolling dCor all-NaN features". Ajustar janelas/min_periods ou revisar cobertura de dados.
- dCor próximo de 1: indica forte dependência distância; perto de 0: fraca ou inexistente.
- Se o ranking demora: reduzir features.dcor_batch_size ou desabilitar rolling (features.stage1_rolling_enabled=false) para teste rápido.
