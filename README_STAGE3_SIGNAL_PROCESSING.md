Stage 3 — Signal Processing (Baxter–King)

ATUALIZAÇÃO IMPORTANTE

- O filtro Baxter–King (BK) foi movido para o Estágio 0 (Engenharia de Features) e generalizado para múltiplas colunas de origem. Este documento permanece para referência de comportamento e performance do BK, mas a especificação atualizada encontra‑se em `README_STAGE0_FEATURE_ENGINEERING.md`.
  - Configuração proposta: `feature_engineering.baxter_king.{k, low_freq, high_freq, source_columns}`.
  - Nomes de saída: `bk_filter_<col_origem>`.

Objetivo

- Extrair componente cíclica/“band‑pass” do preço (ou derivado) via filtro Baxter–King (BK), acelerado em GPU, gerando uma série filtrada para uso por estágios seguintes e análises.

O Que Faz (exato)

- Escolha da fonte de preço (em ordem):
  - `y_close`
  - `log_stabilized_y_close`
  - Primeira coluna cujo nome contenha “close” (fallback)
- Aplica filtro Baxter–King (passa‑banda) sobre a coluna fonte, produzindo colunas (na especificação atual, em Estágio 0):
  - `bk_filter_<nome_da_coluna_origem>`
- Bordas do filtro: define `NaN` nos `k` pontos iniciais e finais (propriedade do BK).
- Persistência de métricas/artifacts (opcional): escreve `artifacts/signal/summary.json` com parâmetros usados e colunas geradas.

GPU/Performance (como implementado)

- Convolução em GPU com CuPy:
  - Kernel curto (`2k+1 ≤ 129`): `cp.convolve`
  - Kernel longo: FFT‑based (`cusignal.fftconvolve` quando disponível; fallback para SciPy FFT com ida/volta controlada)
- Pesos BK são gerados em CPU (NumPy) e cacheados por worker (`lru_cache(maxsize=16)`), depois copiados 1x para GPU por tamanho/worker.
- Precisão: processamento em `float32` (cast explícito), equilibrando estabilidade/velocidade.

Parâmetros (config.features)

- `baxter_king_k` (int): semi‑largura do kernel. Determina comprimento `2k+1` e o tamanho da borda inválida. Ex.: 12.
- `baxter_king_low_freq` (float): período de baixa frequência (componente lenta). Ex.: 32.
- `baxter_king_high_freq` (float): período de alta frequência (componente rápida). Ex.: 6.
- `debug_write_artifacts` (bool): se true, escreve sumário em `artifacts/signal`.
- `artifacts_dir` (str): subpasta para artifacts (default `artifacts`).

Entradas e Saídas

- Entrada: `dask_cudf.DataFrame` (ou `cudf.DataFrame`) com colunas de preço (idealmente `y_close`).
- Saída (legado): coluna `bk_filter_<...>`; na arquitetura atual, isso é produzido no Estágio 0.
- Artifacts (opcional): `artifacts/signal/summary.json` com:
  - `new_columns`, `new_columns_count`, `bk_k`, `bk_low_period`, `bk_high_period`.

Logs (esperados)

- “Starting SignalProcessor (Dask)…”
- “Applying Baxter–King filter to '<source_col>'…”
- “SignalProcessor complete.”

Comportamento em Falhas/Edge Cases

- Sem coluna “close”: loga aviso e pula o filtro (não adiciona colunas).
- Kernel muito longo: troca para FFT para manter performance.
- Bordas: os `k` primeiros/últimos pontos do filtro são `NaN` por definição do BK.

Complexidade (intuição)

- Convolução direta: O(n·k) por partição.
- FFT (longos): ~O(n log n) por partição.
- Menor custo incremental graças ao cache de pesos por worker.

Boas Práticas

- Ajuste `k`, `low_freq`, `high_freq` ao seu horizonte:
  - Períodos maiores → ciclos mais longos (mais “suavizado”).
  - Evite janelas muito longas sem necessidade (pico de memória/tempo).
- Considere usar `log_stabilized_y_close` como fonte se a escala do preço variar muito.
- Trate `NaN` de borda antes de métricas derivadas (ex.: corte inicial/final, ou `fillna` com políticas bem definidas).

Integração com Estágios

- Agora roda no Estágio 0 (Engenharia de Features), antes do Filtro Univariado. As colunas `bk_filter_*` tornam‑se candidatas para o Estágio 1.
- Não envia dataset inteiro para CPU; opera em GPU via Dask/cuDF. Somente ao salvar/artefatos podem ocorrer conversões controladas.

Referências de Código (legado)

- Implementação original: `features/signal_processing.py` (BK removido; hoje é pass-through)
- Implementação atual do BK: `features/feature_engineering.py`
  - Dask: `FeatureEngineeringEngine.process`
  - Convolução por partição: `_apply_bk_filter_gpu_partition`
  - Pesos com cache: `_bk_weights_cpu`
