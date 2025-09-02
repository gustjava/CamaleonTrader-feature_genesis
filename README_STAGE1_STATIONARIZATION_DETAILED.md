Stage 1 — Stationarization (Detalhado)

Este documento descreve apenas o Estágio 1 (StationarizationEngine): o que ele faz, como detecta colunas, quais transformações aplica, quais colunas cria, como valida e que métricas/artefatos persiste.

Visão Geral

- Objetivo: tornar as séries mais adequadas para testes estatísticos e seleção posterior, preservando memória quando possível (FFD) e adicionando sinais rolantes leves.
- Implementação: `features/stationarization.py` (rotas Dask e cuDF) herda de `BaseFeatureEngine` e usa utilitários de validação/log/memória.
- Execução: é o primeiro engine da pipeline (configurável via `pipeline.engines.stationarization`).

Configuração Atual (baseada em MASTER_FEATURES_COLUMNS.md)

- Entradas usadas diretamente no Estágio 1 (detectadas por nome):
  - Preço: `y_close` (e OHLC em geral para detecção)
  - Retornos: `y_ret_1m` (preferido). Anti‑leakage: colunas `y_ret_fwd_*` e prefixos de targets `m1_..m9_` são ignorados como “retornos”.
  - Tick volume: `y_tick_volume` (usado para z‑scores específicos do tick volume)
  - Outras (opcionais para correlações rolantes leves): `y_spread_rel`, `y_ofi_*`, `y_volume`, etc., se existirem no dataset.

- Itens do MASTER mapeados e processados neste estágio:
  - 25. `y_tickvol_z_15m` → Gerado aqui (z‑score 15m do tick volume)
  - 26. `y_tickvol_z_60m` → Gerado aqui (z‑score 60m do tick volume)
  - 27. `y_tickvol_z_l1` → Gerado aqui (lag 1 do z‑score 15m do tick volume)
  - Demais itens do MASTER (RSI, MACD, BB, ATR, etc.) permanecem como “pass‑through” para estágios seguintes (ranking/seleção), não são criados pelo Stationarization.

Seleção Explícita de Candidatas (sem regex)

- O Stage 1 aplica FFD adicional de forma explícita conforme `config.features.station_candidates_include` (e exclui nomes em `station_candidates_exclude`).
- Se a lista estiver vazia, o motor usa apenas as heurísticas básicas (close/retornos + tick volume z‑scores) descritas acima.
- Recomenda-se preencher o include com os “níveis” mais relevantes do MASTER (ex.: `y_avg_spread`, `y_max_spread`, `y_min_spread`, `y_spread_lvl`, `y_sma_*`, `y_ema_*`, `y_weighted_close`, `y_typical_price`).
- A curadoria é guiada pelo `MASTER_FEATURES_COLUMNS.md` (seção “Stage 1 — Mapa de Inclusão/Exclusão”) e refletida no config, sem regex.

Detecção de Colunas (Entrada)

- Estratégia: varrer nomes e agrupar por categorias, priorizando prefixo `y_` quando disponível.
- Categorias (exemplos de match):
  - Preço: `y_close`, `y_open`, `y_high`, `y_low`, qualquer nome contendo `close|open|high|low`.
  - Retornos: `y_ret_1m`, `returns`, nomes contendo `ret|return`.
  - Volume: `y_volume`, `y_tickvol_z_15m`, nomes contendo `volume|tick`.
  - Outras (quando presentes): spread (`spread`), volatilidade (`vol|rv|volatility`), OFI (`ofi`).

Transformações Aplicadas

- FracDiff (Diferenciação Fracionária)
  - Dask: aplica FFD em 1–2 colunas-alvo (1 preço + 1 retorno), usando d padrão `features.frac_diff_values[-1]` para throughput.
  - cuDF: busca d ótimo por coluna (grade `frac_diff_values` e testes simples; fallback por variância), gera estatísticas da série diferenciada.
  - Saídas (por coluna `<col>`): `frac_diff_<col>`, `frac_diff_<col>_d`, `frac_diff_<col>_stationary`, `frac_diff_<col>_variance_ratio`, `frac_diff_<col>_{mean|std|skewness|kurtosis}`.

- Estacionarização Rolante (Z‑score)
  - Aplica z‑score rolante em uma série de retorno (ex.: `y_ret_1m`) com janela e min_periods configuráveis.
  - Saída: `rolling_stationary_<ret_col>`.

- Estabilização de Variância
  - Transformação `log(x + eps)` com deslocamento seguro para garantir positividade.
  - Saída: `log_stabilized_<price_col>` (quando aplicável).

- Correlações Rolantes (Proxy leve)
  - Constrói até poucos pares significativos (ex.: retornos×volume, retornos×OFI, spread×vol, preço×volume).
  - Proxy determinística baseada em médias rolantes limitada em [-1, 1].
  - Saída (quando habilitada): `rolling_corr_<col1>_<col2>_<w>w`.

Colunas Criadas (Padrões)

- FFD: `frac_diff_<col>`, `frac_diff_<col>_d`, `frac_diff_<col>_stationary`, `frac_diff_<col>_variance_ratio`, `frac_diff_<col>_{mean|std|skewness|kurtosis}`.
- Rolantes simples: `rolling_mean_<price_col>_20`, `rolling_std_<price_col>_20` (fallback básico no caminho cuDF).
- Z‑score: `rolling_stationary_<ret_col>`.
- Variância (log): `log_stabilized_<price_col>`.
- Correlação proxy: `rolling_corr_<col1>_<col2>_<window>w` (limitado para evitar explosão de features).
- Tick volume (novo no Estágio 1):
  - `y_tickvol_z_15m`, `y_tickvol_z_60m`, `y_tickvol_z_l1`.

Validações e Segurança de Memória

- Qualidade de série (FFD): checa tamanho mínimo, %NaN, outliers e std rolante ~0 antes de aplicar FFD.
- Monitoramento de memória (CPU/GPU) antes/depois das etapas; limpeza de pools quando necessário.
- Limites operacionais: nº de pares e janelas para correlações rolantes; cache de pesos FFD; fallback para FFT CPU quando `cusignal` indisponível.

Configurações Relevantes (config/config.yaml)

- `features.frac_diff_values`, `features.frac_diff_threshold`, `features.frac_diff_max_lag`: controle do FFD e grade de d.
- `features.rolling_windows`, `features.rolling_min_periods`: janelas rolantes.
- `validation.{min_rows,max_missing_percentage,outlier_threshold}`: gate de qualidade de série.
- `features.artifacts_dir`, `features.debug_write_artifacts`: persistência de artefatos.
- `features.station_basic_rolling_enabled`: quando true, adiciona rolantes básicos (`rolling_mean/std_*_20`) no caminho cuDF (default false para evitar redundância com SMA já existentes no MASTER).
- `features.drop_original_after_transform`: quando true, remove a coluna original após criar `frac_diff_<col>` (reduz colinearidade e tamanho do arquivo final).

Persistência e Observabilidade

- MySQL (task_metrics):
  - Dask: `new_columns`, `new_columns_count`, `fracdiff_default_d`.
  - cuDF: `new_columns`, `new_columns_count`, mapa `fracdiff_optimal_d` (por coluna `_d`).
- Artefatos (task_artifacts):
  - `output/<PAIR?>/artifacts/stationarization/summary.json` contendo novas colunas e d(s) usados.
- Eventos por estágio (engine_stage_events): START/END/ERROR com `cols_before/after`, `new_cols` e lista detalhada de colunas novas em `details`.

Boas Práticas de Retenção no Dataset Final

- Manter: `frac_diff_*` (série e, se útil, `*_d` e `*_stationary`) e um subconjunto de rolantes simples.
- Avaliar remoção: métricas auxiliares muito granulares (kurtosis/skewness por coluna) se não forem usadas a jusante.
- Evitar redundâncias: limitar pares/janelas em correlações rolantes; métricas meramente de diagnóstico devem ir para MySQL/artefatos.

Logs úteis (interpretação)

- "Starting stationarization pipeline…" / "completed successfully": entrada/saída do estágio.
- "Applying fractional differentiation…": início do FFD; no cuDF, a busca por d imprime avisos se a série não passar em qualidade.
- "Applying rolling stationarization…", "Applying variance stabilization…": passos subsequentes.
- "Created N feature pairs…": pares considerados para correlações rolantes.

Relação com Estágios Seguintes

- As séries `frac_diff_*` são insumos preferenciais para ADF rolante e dCor no Estágio 2 (Statistical Tests).
- Reduzir a cardinalidade de colunas rolantes e de diagnóstico aqui ajuda o tamanho do arquivo final e o tempo do ranking posterior.
 - As novas features de tick volume normalizado (`y_tickvol_z_*`) substituem o uso direto de `y_tick_volume` bruto nos estágios de seleção/modelagem, conforme recomendado no MASTER.
