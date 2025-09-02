Motor de Estacionarização — O Que Faz e Como Registra Logs

Este documento explica, em termos práticos, o que o StationarizationEngine está fazendo durante o pipeline, como ler seus logs e quais features ele adiciona ao dataset. Também resume as configurações relevantes e comportamento operacional (Dask, memória GPU, caminhos de segurança).

Escopo (Estágio 1: Estacionarização)

O motor prepara as séries para testes estatísticos e seleção aplicando quatro transformações:

## Features Trabalhadas pelo Motor

### **1. Features de Entrada (Colunas Originais com Prefixo 'y_')**

O motor detecta e categoriza automaticamente as seguintes features de entrada:

#### **Features de Preço:**
- `y_close` - Preço de fechamento
- `y_open` - Preço de abertura  
- `y_high` - Preço máximo
- `y_low` - Preço mínimo
- Qualquer coluna que contenha 'close', 'open', 'high', 'low' no nome

#### **Features de Retorno:**
- `y_ret_1m` - Retornos de 1 minuto (padrão)
- `y_returns` - Retornos gerais
- `y_ret_*` - Qualquer coluna que contenha 'ret' ou 'return' no nome

#### **Features de Volume:**
- `y_volume` - Volume de negociação
- `y_tickvol_z_15m` - Volume por tick normalizado (Z-score 15m)
- Qualquer coluna que contenha 'volume' ou 'tick' no nome

#### **Features de Volatilidade:**
- `y_vol_*` - Medidas de volatilidade
- `y_rv_*` - Volatilidade realizada
- `y_volatility_*` - Volatilidade calculada
- Qualquer coluna que contenha 'vol', 'rv', 'volatility' no nome

#### **Features de Spread:**
- `y_spread_*` - Spreads de preços
- `y_spread_rel` - Spread relativo
- Qualquer coluna que contenha 'spread' no nome

#### **Features OFI (Order Flow Imbalance):**
- `y_ofi_*` - Indicadores de desequilíbrio de fluxo de ordens
- Qualquer coluna que contenha 'ofi' no nome

### **2. Features Geradas pelo Motor**

#### **A) Diferenciação Fracionária (FracDiff):**
- `frac_diff_<price_col>` - Série diferenciada fracionariamente da primeira feature de preço
- `frac_diff_<ret_col>` - Série diferenciada fracionariamente da primeira feature de retorno
- `frac_diff_<col>_d` - Ordem d ótima encontrada
- `frac_diff_<col>_stationary` - Flag indicando se a série é estacionária
- `frac_diff_<col>_variance_ratio` - Razão de variância
- `frac_diff_<col>_mean` - Média da série diferenciada
- `frac_diff_<col>_std` - Desvio padrão da série diferenciada
- `frac_diff_<col>_skewness` - Assimetria da série diferenciada
- `frac_diff_<col>_kurtosis` - Curtose da série diferenciada

#### **B) Correlações Rolantes (Proxy):**
- `rolling_corr_<col1>_<col2>_<window>w` - Correlação rolante entre pares de features
- Exemplos típicos:
  - `rolling_corr_y_ret_1m_y_volume_20w` - Retornos vs Volume (janela 20)
  - `rolling_corr_y_ret_1m_y_ofi_50w` - Retornos vs OFI (janela 50)
  - `rolling_corr_y_spread_y_vol_100w` - Spread vs Volatilidade (janela 100)
  - `rolling_corr_y_close_y_volume_200w` - Preço vs Volume (janela 200)

#### **C) Estacionarização Rolante (Z-score):**
- `rolling_stationary_<ret_col>` - Z-score rolante da primeira feature de retorno
- Exemplo: `rolling_stationary_y_ret_1m`

#### **D) Estabilização de Variância (Log):**
- `log_stabilized_<price_col>` - Transformação log da primeira feature de preço
- Exemplo: `log_stabilized_y_close`

#### **E) Features Rolantes Básicas (Fallback):**
- `rolling_mean_<price_col>_20` - Média rolante de 20 períodos
- `rolling_std_<price_col>_20` - Desvio padrão rolante de 20 períodos

### **3. Pares de Features Analisados**

O motor cria automaticamente pares de correlação entre:

1. **Retornos vs Volume** - `y_ret_1m` × `y_volume`
2. **Retornos vs OFI** - `y_ret_1m` × `y_ofi`
3. **Spread vs Volatilidade** - `y_spread` × `y_vol`
4. **Preço vs Volume** - `y_close` × `y_volume`
5. **Retornos vs Spread** - `y_ret_1m` × `y_spread`
6. **Volume vs Spread** - `y_volume` × `y_spread`

### **4. Configurações que Afetam as Features**

- **Janelas Rolantes**: `features.rolling_windows` (padrão: [10, 20, 50, 100, 200])
- **Períodos Mínimos**: `features.rolling_min_periods` (padrão: 1)
- **Valores d FracDiff**: `features.frac_diff_values` (padrão: [0.1, 0.2, 0.3, 0.4, 0.5])
- **Lag Máximo**: `features.frac_diff_max_lag` (padrão: 1000)
- **Threshold**: `features.frac_diff_threshold` (padrão: 1e-5)

### **5. Limitações de Performance**

- **Máximo de 4 pares** de correlação rolante para evitar explosão computacional
- **Máximo de 2 janelas** por par para manter performance
- **Cache de pesos** FracDiff limitado a 32 entradas
- **Threshold de partição** de 4096 linhas para processamento em chunks

O motor é projetado para ser **adaptativo** - ele detecta automaticamente quais features estão disponíveis no dataset e aplica as transformações apropriadas, gerando um conjunto completo de features estacionarizadas para análise estatística posterior.

## Transformações Aplicadas

1) Diferenciação Fracionária (FracDiff)
- Objetivo: alcançar estacionaridade preservando a estrutura de memória longa.
- O que computamos: uma versão fracionariamente diferenciada de algumas séries-chave.
- Seleção atual (para throughput): primeira série de preço (ex: y_close) e primeira série de retorno (ex: y_ret_1m).
- Como é computado: pesos GPU (CuPy) + convolução (direta ou FFT via cuSignal/fallback CPU).
- Colunas de saída: frac_diff_<col> (float32).

2) Correlações Rolantes (Proxy leve)
- Objetivo: rastrear relacionamentos dinâmicos entre pares significativos (retornos vs volume/OFI; spread vs volatilidade; etc.).
- O que computamos: uma métrica de relacionamento rolante simples e determinística por par/janela, limitada a [-1, 1].
- Limites (para evitar explosão): até 4 pares, e até 2 tamanhos de janela de features.rolling_windows.
- Colunas de saída: rolling_corr_<col1>_<col2>_<window>w (float32 proxy para correlação).

3) Estacionarização Rolante (Z-score)
- Objetivo: normalizar/padronizar uma série de retorno-chave sobre uma janela rolante.
- O que computamos: z-score rolante com window=252, min_periods=50.
- Coluna de saída: rolling_stationary_<return_col> (float32).

4) Estabilização de Variância (Log)
- Objetivo: estabilizar variância em séries tipo-preço.
- O que computamos: log(x + shift) onde shift garante positividade (seguro se série ≤ 0).
- Coluna de saída: log_stabilized_<price_col> (float32).

Todos os map_partitions são implementados via funções determinísticas de nível de módulo para que o Dask possa fazer hash delas de forma confiável (sem pulos ocultos).

Passeio pelos Logs — Como Ler as Linhas

Exemplos de linhas e o que significam:

- Starting stationarization pipeline...
  - O motor está entrando no Estágio 1 para o instrumento atual.

- Applying fractional differentiation...
  - Inicia FracDiff nas colunas selecionadas. Internamente computa pesos GPU e convolui com a série em cada partição.
  - Saídas frac_diff_<col>.

- Applying rolling correlations...
  - Detecta colunas disponíveis e as categoriza: preço, volume, retornos, volatilidade, spread, OFI.
  - Registra contagens e nomes de exemplo:
    - Price features: 17 - ['y_bb_lower_20_2', 'y_close', 'y_close_rel_sma_20'] → 17 colunas tipo-preço detectadas; primeiras 3 são mostradas como exemplos.
  - Criados N pares de features para correlações rolantes + lista dos primeiros pares considerados.
  - Computa no máximo 4 colunas de relacionamento rolante (proxy para correlação), 2 janelas cada, para manter runtime limitado.

- Applying rolling stationarization...
  - Computa um z-score rolante na primeira série de retorno encontrada (ex: y_ret_1m).
  - Saídas rolling_stationary_<return_col>.

- Applying variance stabilization...
  - Aplica log(x + shift) à primeira série de preço (ex: y_close).
  - Saídas log_stabilized_<price_col>.

- Stationarization pipeline completed successfully
  - Tudo acima foi completado para o instrumento/conjunto de partições atual. Se qualquer transformação falhar de forma crítica, o motor registra um erro CRÍTICO e para o pipeline.

Funções Determinísticas de Partição Dask

Para evitar problemas de tokenização e pulos silenciosos, o StationarizationEngine usa funções puras de nível de módulo para map_partitions:

- _fracdiff_series_partition(series, d, max_lag, tol) → frac_diff_<col>
- _rolling_corr_simple_partition(pdf, col1, col2, window, min_periods, new_col) → rolling_corr_*
- _rolling_zscore_partition(series, window, min_periods) → rolling_stationary_<col>
- _variance_log_partition(series) → log_stabilized_<col>

Este design garante que o Dask possa fazer hash das funções deterministicamente e agendar trabalho de forma confiável.

Configuração (chaves relevantes)

- features.frac_diff_values: grade de candidatos d; valor padrão último é usado como d_default.
- features.frac_diff_max_lag, features.frac_diff_threshold: controlam pesos FracDiff e poda.
- features.rolling_windows, features.rolling_min_periods: janelas para métricas rolantes.
- Limites de runtime dentro do motor mantêm a execução limitada (ex: máx 4 pares de correlação; 2 janelas).

Relacionado (de outros estágios, mostrado aqui para contexto):
- features.stage1_rolling_enabled: habilita rolling dCor em StatisticalTests (já implementado com segurança).
- features.dcor_batch_size: controla tamanho do lote para logs de progresso de ranking dCor.

Saídas — Resumo das Colunas Adicionadas

- Diferenciação Fracionária:
  - frac_diff_<price_col> (ex: frac_diff_y_close)
  - frac_diff_<ret_col> (ex: frac_diff_y_ret_1m)

- Correlações Rolantes (proxy):
  - rolling_corr_<col1>_<col2>_<w>w (até 4 total)

- Estacionarização Rolante:
  - rolling_stationary_<ret_col>

- Estabilização de Variância:
  - log_stabilized_<price_col>

Os nomes exatos das colunas dependem das colunas detectadas como "primeiro preço" e "primeiros retornos" para o dataset.

Comportamento de Falha e Logging

- Qualquer erro crítico (ex: OOM GPU inesperado não recuperável) dispara _critical_error, que registra:
  - "🚨 ERRO CRÍTICO: <mensagem>"
  - "🛑 Parando pipeline imediatamente devido a erro crítico."
- Problemas não-críticos por partição retornam NaN para a nova coluna para evitar parar toda a execução, e a falha é registrada como WARNING/ERROR.

Notas de Performance

- FracDiff usa CuPy e, quando kernels são longos, FFT via cuSignal (fallback para SciPy se necessário). Pesos são cacheados para minimizar recomputação.
- Computações rolantes usam cuDF .rolling() internamente ou proxies simples para manter uso de memória estável.
- Trabalho é limitado restringindo número de pares e janelas. Estes limites podem ser removidos depois de validar throughput.

Exemplo End-to-End (Logs + Ações)

1) "Applying fractional differentiation..." → escreve frac_diff_y_close, frac_diff_y_ret_1m.
2) "Applying rolling correlations..." → registra categorias e pares; escreve até 4 colunas rolling_corr_*.
3) "Applying rolling stationarization..." → escreve rolling_stationary_y_ret_1m.
4) "Applying variance stabilization..." → escreve log_stabilized_y_close.
5) "Stationarization pipeline completed successfully" → prossegue para StatisticalTests (ADF/dCor).

