Stage 4 — GARCH Models

Objetivo

- Estimar um modelo GARCH(1,1) para caracterizar a dinâmica de volatilidade da série de preços e derivar métricas agregadas (parâmetros e critérios de informação) para uso analítico/monitoramento.

Visão Geral (Dask path padrão)

- Fonte de preço (ordem de preferência):
  - `y_close`
  - `log_stabilized_y_close`
  - qualquer coluna cujo nome contenha "close" (fallback)
- Colhe a coluna de preço em uma única partição (um worker/GPU) via `_single_partition` para ter a série contínua.
- Constrói retornos logarítmicos na partição e ajusta GARCH(1,1) por máxima verossimilhança.
- Retorna um conjunto pequeno de escalares (1 linha) e os "broadcasta" para todas as linhas do DataFrame.
- Opcionalmente grava um resumo em `artifacts/garch/summary.json`.

Saídas (colunas adicionadas)

- `garch_omega`: termo constante da variância condicional.
- `garch_alpha`: sensibilidade ao choque (retorno^2 anterior).
- `garch_beta`: persistência da variância (variância condicional anterior).
- `garch_persistence`: `alpha + beta` (estaçõesidade se < 1.0).
- `garch_log_likelihood`: log‑verossimilhança no ótimo.
- `garch_aic`: critério de Akaike (2k − 2LL, com k=3).
- `garch_bic`: critério Bayesiano (k ln n − 2LL).
- `garch_is_stationary`: 1.0 se `alpha + beta < 1`, senão 0.0.

Como é implementado (híbrido CPU/GPU)

- Dask/cuDF organiza a série e empurra o cálculo para um worker.
- O ajuste GARCH(1,1) na rota Dask atual (`process`) é feito em CPU (NumPy + SciPy `minimize`) dentro de `map_partitions`, retornando uma linha por partição (na prática usamos 1 partição com toda a série).
- GPU é usada para orquestração (Dask‑cuDF), conversões e possíveis cálculos auxiliares; a rota "comprehensive" (`process_cudf`) inclui utilitários acelerados, mas o ajuste padrão mostrado no log é CPU‑based (estável e simples).

Parâmetros (config.features)

- `garch_p`, `garch_q`: ordens do GARCH (atualmente 1,1 suportado na rota Dask padrão).
- `garch_max_iter`: iterações máximas do otimizador.
- `garch_tolerance`: tolerância (`ftol`) do otimizador.
- `distance_corr_max_samples`: é reutilizado como limite superior de amostras para o ajuste (truncagem da cauda para custo previsível).

Entradas e Pré‑Processo

- Requer uma coluna de preço positiva (para log) — preferir `y_close`.
- Internamente:
  - Faz sanitização de valores não finitos por preenchimentos forward/backward e fallback pequeno (`1e-8`).
  - Usa log‑retornos: ln(P[t]) − ln(P[t−1]).
  - Trunca a série para no máximo `distance_corr_max_samples` observações (parte final) para custo estável.

Logs Esperados

- "Starting GARCH (Dask)…"
- "GARCH fitted." com `{garch_omega, garch_alpha, garch_beta, garch_persistence, garch_log_likelihood, garch_aic, garch_bic, garch_is_stationary}`
- "GARCH features attached."

Comportamento em Edge Cases

- Série curta (< 100 pontos) ou retornos efetivos < 50 → retorna linha default (NaNs) para manter o schema.
- Falha do otimizador → retorna linha default.
- Ausência de coluna de preço → etapa é pulada com aviso.

Complexidade (intuição)

- Montagem da variância condicional: O(n) por avaliação da função de custo.
- Otimização (L‑BFGS‑B) com limites → custo proporcional ao número de avaliações até convergência (depende de dados/`max_iter`).

Boas Práticas

- Garanta `y_close` limpo e positivo.
- Ajuste `distance_corr_max_samples` para estabilizar custo (ex.: 2000–5000).
- Monitore `garch_persistence`: valores muito altos (≈1) indicam persistência elevada; instabilidade se ≥1.
- Use artifacts para auditoria em produção.

Referências de Código

- Implementação: `features/garch_models.py`
  - Rota Dask padrão: `GARCHModels.process`
  - Função de partição (CPU): `_garch_fit_partition_np`
  - Broadcast de escalares: `_broadcast_scalars`

