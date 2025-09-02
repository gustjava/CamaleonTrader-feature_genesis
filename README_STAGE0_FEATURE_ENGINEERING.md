Stage 0 — Engenharia de Features (BK Generalizado)

Objetivo

- Centralizar transformações iniciais de sinal (antes de qualquer seleção) e generalizar o filtro Baxter–King (BK) para múltiplas colunas, gerando candidatos cíclicos com aceleração em GPU.

O Que Faz

- Recebe uma lista de colunas de origem (níveis ou séries estabilizadas) e aplica BK em cada uma.
- Nomeia as saídas como `bk_filter_<col_origem>` (dtype `float32`).
- Bordas do filtro: define `NaN` nos `k` pontos iniciais e finais.
- Artefatos (opcional): grava `artifacts/signal/summary.json` com parâmetros e colunas geradas.

Performance (como implementar)

- Kernel curto (`2k+1 ≤ 129`): `cupy.convolve` (domínio do tempo) por partição.
- Kernel longo: FFT‑based (`cusignal.fftconvolve` quando disponível; fallback SciPy FFT com ida/volta controlada para GPU).
- Pesos BK: gerados em CPU (NumPy), `lru_cache(maxsize=16)` por worker; copiados 1× para GPU por tamanho/worker.
- Precisão: `float32` explícito para melhor throughput e uso de memória.

Configuração (YAML proposto)

feature_engineering:
  baxter_king:
    k: 12
    low_freq: 32
    high_freq: 6
    source_columns:
      - log_stabilized_y_close
      - ustbondtrusd_vol_60m

Entradas e Saídas

- Entrada: `dask_cudf.DataFrame` (ou `cudf.DataFrame`) com as colunas listadas em `source_columns` existentes.
- Saída: mesmas partições com novas colunas `bk_filter_*` (uma por fonte existente).
- Artefatos: `artifacts/signal/summary.json` com `new_columns`, `bk_k`, `bk_low_period`, `bk_high_period`.

Logs (esperados)

- "Starting SignalProcessor (Dask)…"
- "Applying Baxter–King filter to '<source_col>'…"
- "SignalProcessor complete."

Integração

- Roda antes do Estágio 1 (Filtro Univariado). Novas colunas `bk_filter_*` entram como candidatas nas métricas univariadas.
- Não mover dataset inteiro para CPU; apenas pesos são criados em CPU e copiados para GPU de forma cacheada.

Boas Práticas

- Ajuste `k`, `low_freq`, `high_freq` ao horizonte de interesse; evite janelas muito longas sem necessidade.
- Prefira fontes estabilizadas (ex.: `log_stabilized_y_close`) quando a escala variar muito.
- Trate `NaN` de borda antes de cálculos subsequentes (corte ou `fillna` com política definida).

Referências de Código

- Implementação base: `features/signal_processing.py`
  - Dask: `SignalProcessor.process`
  - Partição: `_apply_bk_filter_gpu_partition`
  - Pesos: `SignalProcessor._bk_weights_cpu`

