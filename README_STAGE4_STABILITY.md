Stage 4 — Estabilidade (Bootstrap em Blocos)

Objetivo

- Validar a robustez da seleção multivariada (Estágio 3) medindo, por reamostragem temporal, a frequência com que cada feature é escolhida por um seletor embutido. A lista final mantém apenas as features “estáveis” (frequência ≥ limiar).

Visão Geral

- Mantém a estrutura temporal: cada iteração amostra uma janela contígua (block bootstrap) e reaplica o seletor do Estágio 3 sobre essa janela.
- Conta ocorrência de seleção por feature e calcula frequência em [0..1].
- Aplica um limiar de estabilidade e persiste artefatos (JSON/PNG) para auditoria.
- Backend do wrapper é o mesmo do Estágio 3 (LGBM/XGB GPU/RandomForest), reaproveitando suas configurações.

Como Funciona (algoritmo)

- Parâmetros principais: `n_bootstrap`, `block_size`, `stability_threshold`, `random_state`.
- Para i em 1..`n_bootstrap`:
  - Escolhe início aleatório s ~ U{0, n_rows − block_size}; usa fatia `[s:s+block_size)` de X/Y (ordem preservada).
  - Roda o seletor do Estágio 3 sobre os candidatos do Estágio 2 (VIF/MI) nessa fatia, obtendo uma lista `sel_i`.
  - Incrementa um contador por feature em `sel_i`.
- Frequência por feature: `freq[f] = count[f] / n_bootstrap`.
- Lista estável: `stable = { f | freq[f] ≥ stability_threshold }`.
- Broadcast no frame: `stage4_features`, `stage4_features_count`.

Configuração (config/config.yaml)

- `stage4_enabled`: habilita o estágio de estabilidade.
- `stage4_n_bootstrap`: número de janelas/iterações.
- `stage4_block_size`: tamanho da janela contígua (linhas). É truncado para [10, n_rows].
- `stage4_stability_threshold`: limiar da frequência de seleção (ex.: 0.7).
- `stage4_random_state`: semente do gerador pseudoaleatório.
- `stage4_plot`: salva gráfico de barras com Top‑50 frequências.
- `stage4_bootstrap_method`: `block|tssplit` (ambos suportados; `tssplit` usa folds crescentes como janelas).

Exemplo YAML (equivalente às chaves atuais)

features:
  stage4_enabled: true
  stage4_n_bootstrap: 30
  stage4_block_size: 5000
  stage4_stability_threshold: 0.7
  stage4_plot: true
  stage4_random_state: 42
  stage4_bootstrap_method: block

Entradas e Saídas

- Entrada: matriz X/Y já preparada e lista de candidatos do Estágio 2 (após VIF/MI). O seletor do Estágio 3 é aplicado em cada janela.
- Saídas no DataFrame (broadcast):
  - `stage4_features`: lista em string separada por vírgulas.
  - `stage4_features_count`: número de features estáveis.
- Artefatos por par/alvo:
- `.../artifacts/stage4/<target>/frequencies.json` — `{ n_bootstrap, block_size, method, frequencies }`.
  - `.../artifacts/stage4/<target>/stable.json` — `{ threshold, features }`.
  - `.../artifacts/stage4/<target>/frequencies.png` — Top‑50 frequências com linha do limiar (opcional).

Performance e Boas Práticas

- Custo ~ O(n_bootstrap × custo do Estágio 3). Para acelerar:
  - Reduza `stage3_lgbm_n_estimators` e ative `stage3_lgbm_early_stopping_rounds`.
  - Limite linhas em wrappers com `selection_max_rows` quando aplicável.
  - Prefira GPU (XGBoost `gpu_hist`) quando disponível; fallback automático para LGBM/CPU.
  - Mantenha o conjunto de candidatos enxuto no Estágio 2 (VIF/MI) — ele é a base do Stage 4.
- Memória: block_size grande aumenta consumo; ajuste de acordo com o hardware e o tamanho do dataset.

Logs Esperados

- "Stage 4 stability starting" com `{ n_bootstrap, block_size }`.
- "Stage 4 stability | iteration" com `{ iter, n_bootstrap }`.
- "Stage 4 stability done" com `{ kept, elapsed }`.
- Avisos: "Stage 4 persist failed" / "Stage 4 frequency plot failed" quando necessário.

Comportamento e Edge Cases

- `n_rows < 10` ou `candidates` vazio → estágio é pulado.
- `block_size` é truncado para `[10, n_rows]`; se `n_rows == block_size`, usa janela inteira.
- Falhas pontuais numa iteração do wrapper não abortam o procedimento; a iteração é contada como seleção vazia.
- `always_keep_features` e `always_keep_prefixes` são respeitados (limitados ao conjunto de candidatos do Estágio 2) e unidos ao conjunto estável.

Integração com Estágios 2 e 3

- Recebe candidatos do Estágio 2 (VIF/MI) e usa o seletor do Estágio 3 em cada janela.
- A lista estável pode diferir da seleção única do Estágio 3 — por design, prioriza robustez temporal. Em produção, usar `stage4_features` como lista final.

Referências de Código

- Chamada do estágio: `features/statistical_tests.py:3165`.
- Implementação: `features/statistical_tests.py:3257` (`_stage4_stability`).
