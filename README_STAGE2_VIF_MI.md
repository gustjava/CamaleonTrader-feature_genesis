Stage 2 — Redundância (VIF + MI)

Objetivo

- Remover recursos redundantes depois do Estágio 1 (gating univariado por dCor/Pearson/MI/F‑test). A ideia é reduzir multicolinearidade linear (VIF) e redundância geral (linear e não‑linear) via MI, mantendo um conjunto enxuto de variáveis informativas para wrappers (Estágio 3) e modelos finais.

Visão Geral do Fluxo

1) Amostragem para CPU
- Toma até features.selection_max_rows linhas (head) e converte para pandas/NumPy.
- Motivo: os algoritmos utilizados a seguir (scikit‑learn VIF/MI e clusterização) são maduros/estáveis em CPU, e o custo é mais por coluna do que por linha — uma amostra bem escolhida preserva a estrutura de redundância e acelera muito.

2) VIF Iterativo (colinearidade linear)
- O que é: VIF (Variance Inflation Factor) mede quanto a variância de um coeficiente cresce pela correlação com os demais preditores. VIF alto indica colinearidade.
- Como usamos: calculamos VIF sobre X (features) e removemos iterativamente a coluna com maior VIF acima do limiar (features.vif_threshold). Repetimos até que todos VIFs fiquem abaixo do limiar, reduzindo a multicolinearidade linear.
- Benefícios: estabiliza regressões e modelos lineares, evita “duplicatas lineares” que distorcem importância de features.

3) MI (Informação Mútua) para Redundância Não‑Linear
- O que é: MI mede dependência (linear e não‑linear) entre variáveis (0 = independência).
- Estratégias suportadas:
  - Clusterização por MI (preferida): computa uma matriz de MI (por blocos, controlada por features.mi_chunk_size), normaliza (opcional), constrói uma distância D=1−MI_norm e aplica AgglomerativeClustering. Seleciona 1 representante por cluster (por exemplo, o maior dCor do Estágio 1), reduzindo redundância de forma global e escalável (limitada por features.mi_max_candidates).
  - Redundância par‑a‑par: quando clusterização não está disponível, percorre pares de candidatos e remove o menos informativo (menor dCor) quando MI ≥ features.mi_threshold.
- Benefícios: remove duplicatas informacionais mesmo quando a relação entre features não é linear.

Por que roda em CPU?

- Bibliotecas e algoritmos: a versão robusta e padrão de VIF e de MI (mutual_info_regression) é do scikit‑learn, otimizada para CPU.
- Clusterização (AgglomerativeClustering): CPU por padrão; implementações estáveis estão disponíveis em sklearn.
- Amostragem: a redundância é estrutural (entre colunas) e se preserva bem com subset de linhas, tornando o custo em CPU aceitável. A sobrecarga para traduzir tudo para GPU (e voltar) não se paga neste passo.
- Estabilidade/Diagnóstico: em caso de problemas, stacktrace/validação na CPU é mais simples (ecossistema mais maduro para essas rotas específicas).

Entradas e Saídas

- Entrada: lista de candidatos retida do Estágio 1 (após thresholds/percentis/top‑N; broadcast em `stage1_features`).
- Saída: stage2_features (lista final após VIF + MI) e stage2_features_count (broadcast como scalars no Dask).

Principais Parâmetros

- features.selection_max_rows: linhas usadas no subset CPU (ex.: 100k → considerar reduzir se muito lento).
- features.vif_threshold: limiar de VIF (ex.: 5.0). Quanto menor, mais agressiva a remoção linear.
- features.mi_cluster_enabled: habilita clusterização global por MI (preferido).
- features.mi_cluster_method: atualmente 'agglo'.
- features.mi_cluster_threshold: controla a distância/MI para formação dos clusters (ex.: 0.3).
- features.mi_max_candidates: cap de candidatos para a matriz MI (top por dCor).
- features.mi_chunk_size: tamanho de bloco para computar MI por partes, economizando memória.

Boas Práticas e Diagnóstico

- Se VIF remove “demais”: aumente vif_threshold (ex.: de 5 → 8) ou revise normalização/escala prévia.
- Se MI cluster leva muito tempo: reduza mi_max_candidates (ex.: 800 → 400) e/ou aumente mi_chunk_size para melhor throughput (trade‑off memória/tempo).
- Se pairwise MI fica lento: prefira clusterização quando possível; é mais global e com melhor escalabilidade para N alto.
- Ajuste selection_max_rows: comece menor para diagnosticar e aumente progressivamente.

Complexidade (intuição)

- VIF: custo ~ O(p²) por iteração (p = número de features). Iterativo reduz o conjunto rapidamente.
- MI cluster: custo ~ O(p²) em blocos (controlado por mi_chunk_size e mi_max_candidates), seguido de clusterização Agglo em D (p×p).

Logs Esperados (aprimorados)

- "Stage 2 (VIF+MI) sampling CPU | { max_rows, candidates }"
- "Stage 2 VIF starting | { n_features, n_rows }" → "Stage 2 VIF done | { kept, elapsed }"
- MI: "Stage 2 MI clustering starting | { cand, chunk }" → "Stage 2 MI done | { kept, elapsed, total_elapsed }"
- Ao final, broadcast: stage2_features, stage2_features_count.

Integração com Estágios 1 e 3

- Recebe candidatos do Estágio 1 (dCor) já ordenados ou filtrados.
- Entrega um conjunto enxuto para Estágio 3 (wrappers), onde Lasso/árvores aplica consenso de importância. Menos redundância = wrappers mais estáveis e rápidos.
