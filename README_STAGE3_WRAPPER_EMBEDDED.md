Stage 3 — Seleção Multivariada (Wrapper/Embutido)

Objetivo

- Selecionar um subconjunto de features com alto valor preditivo marginal em conjunto, usando um modelo com importâncias embutidas e um limiar de seleção.

Abordagem

- `SelectFromModel` com LightGBM (`LGBMRegressor`/`LGBMClassifier`).
- `importance_type=gain` e limiar `median` (ou valor numérico): mantém features com importância ≥ limiar.
- Validação leve com `TimeSeriesSplit` opcional; early‑stopping quando apropriado.

Performance

- GPU quando disponível (`device='gpu'` no LightGBM; fallback gracioso para CPU).
- Amostragem superior limitada por `features.selection_max_rows` para conter custo.
- `float32` em matrizes; `feature_fraction` e `bagging_fraction` para acelerar.

Configuração (YAML sugerido)

selection:
  stage3:
    model: lgbm            # lgbm
    task: auto             # auto|regression|classification
    importance_threshold: median   # median|float
    use_gpu: true
    random_state: 42
    n_estimators: 300
    learning_rate: 0.05
    feature_fraction: 0.8
    bagging_fraction: 0.8
    early_stopping_rounds: 0

Entradas e Saídas

- Entrada: `stage2_features` (lista) e matriz X/Y (linhas amostradas quando configurado).
- Saída: `selected_features` (lista final) e `feature_importances_` por coluna (artefato Parquet/JSON).

Logs (esperados)

- "Stage 3 wrapper fit | { model: 'lgbm', use_gpu: true, rows, cols }"
- "Stage 3 selected features | { kept, threshold }"

Integração

- Recebe um conjunto já desredundado (VIF/MI) do Estágio 2.
- Entrega a lista final ao Estágio 4 (Estabilidade) para validação de robustez.

Referências de Código

- Implementação atual: blocos de wrappers em `features/statistical_tests.py` (LightGBM já usado como backend). A migração para um estágio dedicado poderá isolar a lógica e simplificar artefatos.

