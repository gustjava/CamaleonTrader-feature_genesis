Stage 4 — Estabilidade (Block Bootstrap)

Objetivo

- Validar a robustez do conjunto de features selecionado no Estágio 3, estimando a frequência de seleção sob reamostragens temporais.

Abordagem

- Reexecutar o Estágio 3 em múltiplas amostras temporais preservando dependência temporal:
  - Block bootstrap ou múltiplas janelas com `TimeSeriesSplit` (aproximação robusta).
- Medir a frequência com que cada feature é selecionada (0..1) e aplicar um limiar de estabilidade.

Configuração (YAML sugerido)

selection:
  stage4:
    n_bootstrap: 50
    block_size: 2048          # em linhas ou minutos, conforme index
    stability_threshold: 0.7   # mantém features selecionadas em >= 70% das amostras
    random_state: 42

Entradas e Saídas

- Entrada: conjunto de features de Estágio 3 e os dados X/Y (mesmo preparo de amostragem do wrapper).
- Saídas:
  - Lista final: features com frequência ≥ `stability_threshold`.
  - Tabela Parquet: `feature`, `frequency`, `n_selected`, `n_bootstrap`.
  - Gráfico de barras (PNG) com top‑K por frequência (opcional).

Performance

- Reuso de pipeline de Estágio 3; limitar linhas com `selection_max_rows` e, se aplicável, reduzir `n_estimators` para bootstrap.
- Paralelizar por iteração quando possível (cuidado com memória); caso contrário, iterar sequencialmente.

Logs (esperados)

- "Stage 4 stability | { iter, n_bootstrap }"
- "Stage 4 stability done | { kept, threshold }"

Integração

- A saída substitui/filtra o conjunto de Estágio 3 para produção.
- Persistir artefatos para auditoria e reprodutibilidade.

