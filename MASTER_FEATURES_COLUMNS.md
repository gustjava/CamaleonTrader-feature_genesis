# EURUSD Master Features - Documentação Completa e Organizada

Este documento fornece uma documentação completa e organizada das **209 colunas** presentes no arquivo `EURUSD_master_features.parquet`, separando claramente features de targets e explicando a origem de cada coluna.

## 📊 Informações Gerais

- **Arquivo**: `EURUSD_master_features.parquet`
- **Total de colunas**: 209
- **Formato**: Parquet (Apache Arrow)
- **Dataset**: Features master para EURUSD
- **Pipeline**: Feature Genesis (Feature Engineering Pipeline)

## ✅ Stage 1 — Mapa de Inclusão/Exclusão (curado)

Esta seção indica, por nome, quais colunas do MASTER devem ser processadas no Estágio 1 (stationarização) e quais não devem, com justificativa. A lista é usada para curadoria e deve ser refletida em `config.features.station_candidates_include` e `station_candidates_exclude` (sem regex).

- IN (estacionarizar; tendência a não‑estacionárias ou níveis agregados)
  - Preços/níveis e derivados de nível:
    - `y_close`, `y_open`, `y_high`, `y_low` — níveis de preço.
    - `y_sma_20`, `y_sma_60` — médias móveis (nível), propensas a memória longa.
    - `y_ema_12`, `y_ema_26` — médias exponenciais (nível), idem.
    - `y_weighted_close`, `y_typical_price` — proxies de nível de preço.
  - Spreads em nível:
    - `y_avg_spread`, `y_max_spread`, `y_min_spread`, `y_spread_lvl` — séries de nível.
  - (Drivers de nível — avaliar por par):
    - Exemplos: `dxy_close_z_60m` NÃO (já z‑score); se existir `dxy_close` (nível), poderia ser IN.

- OUT (não estacionarizar neste estágio; já normalizadas, alvos ou métricas)
  - Z‑scores/relativos/razões/betas/correlações/interações:
    - Exemplos: `y_spread_rel_z_15m`, `y_spread_z_15m`, `y_close_z_60m`, `y_close_z_240m`, `y_z_240m_close`, `y_z_60m_close`, `y_vwap_60m_distance` (geralmente já estabilizado), `y_corr_*`, `y_beta_*`, `y_interaction_*` — já normalizados/estáveis ou não se beneficiam de FFD.
  - OFI/volume normalizados:
    - `y_tickvol_z_15m`, `y_tickvol_z_60m`, `y_tickvol_z_l1` — já normalizados (gerados/consumidos como estão no Stage 1).
  - Retornos (inputs e forward):
    - `y_ret_1m` — usado para z‑score/ADF/GARCH, não aplicar FFD por padrão.
    - `y_ret_fwd_*` — alvos (leakage), nunca entram no Stage 1 como insumo.
  - Targets e métricas internas:
    - `m1_*..m9_*` — alvos, fora do Stage 1.
    - `dcor_*`, `dcor_roll_*`, `dcor_pvalue_*`, `stage1_*`, `cpcv_*` — métricas internas, não são features de modelagem.

Observações
- Esta lista é inicial e focada em nomes mais frequentes. Expanda/ajuste com base no par/mercado.
- A decisão final pode considerar um teste ADF leve: se uma série marcada como IN já for estacionária, é razoável não transformá‑la; se uma OUT se mostrar muito útil e não estacionária, reavaliar.
- A ativação por código deve ser feita via `config.features.station_candidates_include`/`station_candidates_exclude` (sem expressões regulares), garantindo rastreabilidade total.

## 🎯 Estrutura das Colunas

### **📋 Colunas de Identificação e Metadados**
1. `timestamp` - Timestamp da observação (coluna original)
2. `symbol` - Símbolo da moeda (coluna original)

### **🔴 FEATURES (Variáveis Independentes)**

#### **📈 Features de Preço OHLC (Originais)**
3. `y_open` - Preço de abertura (coluna original) — INPUT via `frac_diff_y_open` (original removida por default)
4. `y_high` - Preço máximo (coluna original) — INPUT via `frac_diff_y_high` (original removida por default)
5. `y_low` - Preço mínimo (coluna original) — INPUT via `frac_diff_y_low` (original removida por default)
6. `y_close` - Preço de fechamento (coluna original) — INPUT via `frac_diff_y_close` (original removida por default)

#### **💰 Features de Spread e Custos de Transação**
7. `y_avg_spread` - Spread médio (coluna original) — Stage 1: IN (FFD; nível com memória longa)
8. `y_avg_spread_relative` - Spread médio relativo (coluna original) — Stage 1: OUT (já normalizado/relativo)
9. `y_max_spread` - Spread máximo (coluna original) — Stage 1: IN (FFD; nível)
10. `y_min_spread` - Spread mínimo (coluna original) — Stage 1: IN (FFD; nível)
11. `y_spread_lvl` - Nível do spread (coluna original) — Stage 1: IN (FFD; nível)
12. `y_spread_rel` - Spread relativo (coluna original) — Stage 1: OUT (já normalizado/relativo)
13. `y_spread_rel_z_15m` - Z-score 15m do spread relativo — INPUT (pass‑through; já normalizado)
14. `y_spread_z_15m` - Z-score 15m do spread — INPUT (pass‑through)
15. `y_spread_widen_l1` - Alargamento do spread lag 1 — INPUT (pass‑through; sinal/derivada)
16. `y_spread_widening_1m` - Alargamento do spread 1m — INPUT (pass‑through; sinal)

#### **📊 Features de Volume e Liquidez**
17. `y_tick_volume` - Volume do tick (coluna original) — NÃO INPUT (usar z-scores `y_tickvol_z_*`)
18. `y_total_volume` - Volume total (coluna original) — NÃO INPUT (substituída por métricas normalizadas)
19. `y_avg_trade_size` - Tamanho médio das trades (coluna original) — INPUT via `frac_diff_y_avg_trade_size` (nível)
20. `y_avg_volume_imbalance` - Desequilíbrio médio de volume (coluna original) — INPUT via `frac_diff_y_avg_volume_imbalance` (nível)
21. `y_close_volume_imbalance` - Desequilíbrio volume no fechamento (coluna original) — INPUT via `frac_diff_y_close_volume_imbalance` (nível)
22. `y_volume_delta` - Delta do volume (coluna original) — INPUT (pass‑through; diferença já estabiliza)
23. `y_buy_aggression_volume` - Volume de agressão compradora (coluna original) — INPUT via `frac_diff_y_buy_aggression_volume`
24. `y_sell_aggression_volume` - Volume de agressão vendedora (coluna original) — INPUT via `frac_diff_y_sell_aggression_volume`
25. `y_tickvol_z_15m` - Z-score 15m do tick volume
26. `y_tickvol_z_60m` - Z-score 60m do tick volume
27. `y_tickvol_z_l1` - Z-score lag 1 do tick volume

#### **📈 Features Técnicas - Médias Móveis**
28. `y_sma_20` - Média móvel simples 20 períodos — INPUT via `frac_diff_y_sma_20` (original removida por default)
29. `y_sma_60` - Média móvel simples 60 períodos — INPUT via `frac_diff_y_sma_60`
30. `y_ema_12` - Média móvel exponencial 12 períodos — INPUT via `frac_diff_y_ema_12`
31. `y_ema_26` - Média móvel exponencial 26 períodos — INPUT via `frac_diff_y_ema_26`
32. `y_vwap_20` - Volume Weighted Average Price 20 períodos — INPUT via `frac_diff_y_vwap_20`
33. `y_vwap_60m_distance` - Distância ao VWAP 60m — INPUT via `frac_diff_y_vwap_60m_distance`
34. `y_vwap_60m_distance_slope` - Inclinação da distância ao VWAP 60m — NÃO INPUT (slope já normaliza)
35. `y_vwap_distance` - Distância ao VWAP — INPUT via `frac_diff_y_vwap_distance`
36. `y_vwap_distance_slope` - Inclinação da distância ao VWAP — NÃO INPUT (slope já normaliza)

#### **📈 Features Técnicas - Indicadores de Momentum**
37. `y_rsi_14` - Relative Strength Index 14 períodos — INPUT (pass‑through; oscilador)
38. `y_macd` - MACD (Moving Average Convergence Divergence) — INPUT (pass‑through)
39. `y_macd_diff` - Diferença do MACD — INPUT (pass‑through)
40. `y_macd_hist` - Histograma do MACD — INPUT (pass‑through)
41. `y_macd_signal_9` - Sinal do MACD 9 períodos — INPUT (pass‑through)
42. `y_momentum_signal_5m` - Sinal de momentum 5m — INPUT (pass‑through)
43. `y_mom_log_10m` - Momentum logarítmico 10m — INPUT (pass‑through)
44. `y_mean_reversion_signal_20m` - Sinal de reversão à média 20m — INPUT (pass‑through)

#### **📈 Features Técnicas - Bollinger Bands**
45. `y_bb_lower_20_2` - Banda inferior Bollinger 20,2 — INPUT via `frac_diff_y_bb_lower_20_2`
46. `y_bb_percent_b_20` - Percentual B das Bollinger Bands 20 — INPUT (pass‑through; já normalizado)
47. `y_bb_upper_20_2` - Banda superior Bollinger 20,2 — INPUT via `frac_diff_y_bb_upper_20_2`
48. `y_bb_width_20` - Largura das Bollinger Bands 20 — INPUT via `frac_diff_y_bb_width_20`

#### **📈 Features Técnicas - Canais e Volatilidade**
49. `y_atr_14` - Average True Range 14 períodos — INPUT via `frac_diff_y_atr_14`
50. `y_atr_z_60m` - Z-score 60m do ATR — INPUT (pass‑through; já normalizado)
51. `y_true_range` - True range — INPUT via `frac_diff_y_true_range`
52. `y_tr_z_60m` - Z-score 60m do true range — INPUT (pass‑through)
53. `y_donchian_channels_20` - Canais de Donchian 20
54. `y_donchian_lower_20` - Canal inferior Donchian 20
55. `y_donchian_upper_20` - Canal superior Donchian 20
56. `y_donchian_width_20` - Largura dos canais Donchian 20

#### **📈 Features Técnicas - Estocástico e Outros Osciladores**
57. `y_stoch_k_14_3` - Estocástico %K 14,3 — INPUT (pass‑through; oscilador já estável)
58. `y_stoch_d_14_3` - Estocástico %D 14,3 — INPUT (pass‑through)
59. `y_mfi_14` - Money Flow Index 14 — INPUT (pass‑through)
60. `y_kvo` - Klinger Volume Oscillator — INPUT (pass‑through)
61. `y_eom_14` - Ease of Movement 14 — INPUT (pass‑through)
62. `y_force_index_13` - Force Index 13 — INPUT (pass‑through)
63. `y_cmf_20` - Chaikin Money Flow 20 — INPUT (pass‑through)
64. `y_ad_line` - Linha de acumulação/distribuição — INPUT via `frac_diff_y_ad_line`
65. `y_obv` - On Balance Volume — INPUT via `frac_diff_y_obv` (nível acumulativo)
66. `y_obv_z_60m` - Z-score 60m do OBV — INPUT (pass‑through)
67. `y_pvt` - Price Volume Trend — INPUT via `frac_diff_y_pvt`
68. `y_weighted_close` - Fechamento ponderado — INPUT via `frac_diff_y_weighted_close`

#### **📈 Features Técnicas - Volatilidade Avançada**
69. `y_parkinson_30m` - Volatilidade de Parkinson 30m — INPUT via `frac_diff_y_parkinson_30m`
70. `y_garman_klass_vol` - Volatilidade Garman-Klass — INPUT via `frac_diff_y_garman_klass_vol`
71. `y_gk_vol_30m` - Volatilidade Garman-Klass 30m — INPUT via `frac_diff_y_gk_vol_30m`
72. `y_rv_30m` - Realized Volatility 30m — INPUT via `frac_diff_y_rv_30m`
73. `y_tick_volatility` - Volatilidade do tick (coluna original) — INPUT via `frac_diff_y_tick_volatility` (se necessário)

#### **📈 Features Técnicas - Estatísticas**
74. `y_skewness_30m` - Assimetria 30m — INPUT (pass‑through)
75. `y_kurtosis_30m` - Curtose 30m — INPUT (pass‑through)
76. `y_jarque_bera_30m` - Teste Jarque-Bera 30m — INPUT (pass‑through)
77. `y_entropy_returns_15m` - Entropia dos retornos 15m — INPUT (pass‑through)
78. `y_hurst_exponent_60m` - Expoente de Hurst 60m — INPUT (pass‑through)
79. `y_autocorr_lags` - Lags de autocorrelação — INPUT (pass‑through)
80. `y_fracdiff_d` - Diferenciação fracionária (parâmetro d) — INPUT (pass‑through)

#### **📈 Features Técnicas - VSA e Análise de Volume**
81. `y_vsa_flags` - Flags VSA (Volume Spread Analysis) — INPUT (pass‑through)
82. `y_amihud_illiq_30m` - Iliquidez Amihud 30m — INPUT via `frac_diff_y_amihud_illiq_30m` (nível)
83. `y_cumulative_delta_60m` - Delta cumulativo 60m — INPUT via `frac_diff_y_cumulative_delta_60m`

#### **📈 Features Técnicas - SMC (Smart Money Concepts)**
84. `y_fvg_status` - Status das Fair Value Gaps — INPUT (pass‑through; categ/indicador)
85. `y_fvg_confidence_score` - Score de confiança das FVG — INPUT (pass‑through)
86. `y_fvg_distance_pips` - Distância das FVG em pips — INPUT via `frac_diff_y_fvg_distance_pips` (nível)
87. `y_fvg_mitigated` - FVG mitigadas — INPUT (pass‑through; binário)
88. `y_fvg_opened` - FVG abertas — INPUT (pass‑through; binário)
89. `y_fvg_age_bars` - Idade das Fair Value Gaps em barras — INPUT via `frac_diff_y_fvg_age_bars` (acumulativo/nível)
90. `y_fvg_width_pips` - Largura das FVG em pips — INPUT via `frac_diff_y_fvg_width_pips`
91. `y_ob_strength` - Força da order book — INPUT (pass‑through)
92. `y_ob_zone_type` - Tipo de zona da order book — INPUT (pass‑through)
93. `y_liquidity_sweep_direction` - Direção do sweep de liquidez — INPUT (pass‑through)
94. `y_ls_strength_pips` - Força do liquidity sweep em pips — INPUT (pass‑through ou via `frac_diff_` se nível instável)

#### **📈 Features Técnicas - Morfologia de Candles**
95. `y_body_size` - Tamanho do corpo da vela — INPUT via `frac_diff_y_body_size`
96. `y_range_pct` - Range percentual — INPUT (pass‑through; razão/percentual)
97. `y_range_size` - Tamanho do range — INPUT via `frac_diff_y_range_size`
98. `y_gap_open` - Gap de abertura — INPUT via `frac_diff_y_gap_open`
99. `y_typical_price` - Preço típico — INPUT via `frac_diff_y_typical_price`
100. `y_price_change` - Mudança de preço — INPUT (pass‑through; diferença já é estável)
101. `y_price_changes_count` - Contagem de mudanças de preço (coluna original) — INPUT via `frac_diff_y_price_changes_count`
102. `y_price_velocity` - Velocidade do preço (coluna original) — INPUT (pass‑through; derivada)

#### **📈 Features Técnicas - Z-scores e Normalizações**
103. `y_close_z_60m` - Z-score 60m do fechamento — INPUT (pass‑through)
104. `y_close_z_240m` - Z-score 240m do fechamento — INPUT (pass‑through)
105. `y_z_240m_close` - Z-score 240m do fechamento — INPUT (pass‑through)
106. `y_z_60m_close` - Z-score 60m do fechamento — INPUT (pass‑through)
107. `y_close_rel_sma_20` - Fechamento relativo à SMA 20 — INPUT (pass‑through; relativo)

#### **📈 Features Técnicas - Correlações e Betas**
108. `y_corr_30m` - Correlação 30m — INPUT (pass‑through)
109. `y_corr_60m` - Correlação 60m — INPUT (pass‑through)
110. `y_corr_dxy_1h` - Correlação 1h com DXY — INPUT (pass‑through)
111. `y_corr_dxy_4h` - Correlação 4h com DXY — INPUT (pass‑through)
112. `y_beta_240m` - Beta 240m — INPUT (pass‑through)
113. `beta240_y_on_brent` - Beta de 240m do EURUSD vs Brent — INPUT (pass‑through)
114. `beta240_y_on_dxy` - Beta de 240m do EURUSD vs DXY — INPUT (pass‑through)
115. `beta240_y_on_spx` - Beta de 240m do EURUSD vs SPX — INPUT (pass‑through)

#### **📈 Features de OFI (Order Flow Imbalance)**
116. `y_ofi_raw` - OFI raw (coluna original) — INPUT via `frac_diff_y_ofi_raw`
117. `y_ofi_norm` - OFI normalizado — INPUT (pass‑through)
118. `y_ofi_z_15m` - Z-score 15m do OFI — INPUT (pass‑through)
119. `y_ofi_z_l1` - Z-score lag 1 do OFI — INPUT (pass‑through)
120. `y_ofi_z_l3` - Z-score lag 3 do OFI — INPUT (pass‑through)
121. `y_ofi_z_l5` - Z-score lag 5 do OFI — INPUT (pass‑through)
122. `y_ofi_z_l10` - Z-score lag 10 do OFI — INPUT (pass‑through)
123. `y_ofi_shock_detected` - Choque OFI detectado — INPUT (pass‑through)
124. `y_is_ofi_shock` - Se há choque OFI — INPUT (pass‑through)

#### **📈 Features de Regime e Contexto de Mercado**
125. `y_regime_trend_up` - Regime de tendência de alta — INPUT (pass‑through)
126. `y_regime_vol_high` - Regime de alta volatilidade — INPUT (pass‑through)
127. `y_is_market_open` - Se o mercado está aberto — INPUT (pass‑through)
128. `y_is_spread_burst` - Se há burst de spread — INPUT (pass‑through)
129. `y_is_vol_burst` - Se há burst de volume — INPUT (pass‑through)

#### **📈 Features de Tempo e Sazonalidade**
130. `y_time_of_day` - Hora do dia — INPUT (pass‑through)
131. `y_day_of_week` - Dia da semana — INPUT (pass‑through)
132. `y_dow_cos` - Cosseno do dia da semana — INPUT (pass‑through)
133. `y_dow_sin` - Seno do dia da semana — INPUT (pass‑through)
134. `y_hod_cos` - Cosseno da hora do dia — INPUT (pass‑through)
135. `y_hod_sin` - Seno da hora do dia — INPUT (pass‑through)
136. `y_minutes_since_open` - Minutos desde a abertura — INPUT (pass‑through)

#### **📈 Features de Drivers (DXY, SPX, Brent, etc.)**
137. `dxy_close_z_60m` - Z-score 60m do fechamento DXY — INPUT (pass‑through)
138. `dxy_ofi_raw` - OFI raw do DXY — INPUT via `frac_diff_dxy_ofi_raw` (nível)
139. `dxy_ofi_z_15m` - Z-score 15m do OFI DXY — INPUT (pass‑through)
140. `dxy_ofi_z_l1` - Z-score lag 1 do OFI DXY — INPUT (pass‑through)
141. `dxy_ofi_z_l3` - Z-score lag 3 do OFI DXY — INPUT (pass‑through)
142. `dxy_ofi_z_l5` - Z-score lag 5 do OFI DXY — INPUT (pass‑through)
143. `dxy_ret_1m` - Retorno 1m do DXY — INPUT (pass‑through; retorno)
144. `dxy_ret_1m_std_60` - Desvio padrão 60m do retorno 1m DXY — INPUT (pass‑through)
145. `dxy_ret_1m_var_240` - Variância 240m do retorno 1m DXY — INPUT (pass‑through)
146. `dxy_tickvol_z_15m` - Z-score 15m do tick volume DXY — INPUT (pass‑through)
147. `dxy_tr_z_60m` - Z-score 60m do true range DXY — INPUT (pass‑through)
148. `dxy_vel_z_15m` - Z-score 15m da velocidade DXY — INPUT (pass‑through)
149. `dxy_vel_z_l1` - Z-score lag 1 da velocidade DXY — INPUT (pass‑through)
150. `dxy_vol_60m` - Volume 60m do DXY — INPUT via `frac_diff_dxy_vol_60m` (se nível bruto)

#### **📈 Features de Drivers - Interações e Combinações**
151. `dxy_ofi_z_15m_mul_spreadStressY` - OFI DXY 15m × spread stress EURUSD — INPUT (pass‑through; interação)
152. `bond_us_de_ret_1m` - Retorno 1m do bond US vs DE — INPUT (pass‑through; relativo/retorno)
153. `bond_us_de_ret_1m_mul_dxyStrong` - Retorno bond US-DE × DXY Strong — INPUT (pass‑through; interação)
154. `bond_us_uk_ret_1m` - Retorno 1m do bond US vs UK — INPUT (pass‑through)
155. `bundtreur_vol_60m` - Volatilidade 60m do Bund vs EUR — INPUT (pass‑through ou via `frac_diff_` se nível)
156. `ustbondtrusd_vol_60m` - Volume 60m do US Treasury Bond TR USD — INPUT via `frac_diff_ustbondtrusd_vol_60m` (se nível)
157. `spx_vel_z_15m_mul_volBurstSPX` - Velocidade SPX 15m × burst volume SPX — INPUT (pass‑through)
158. `spx_vol_60m` - Volume 60m do SPX — INPUT via `frac_diff_spx_vol_60m` (se nível)

#### **📈 Features de Correlação e Interação**
159. `corr30_ofi_y_dxy` - Correlação 30m OFI EURUSD vs DXY — INPUT (pass‑through; correlação)
160. `corr30_ret_ofi_y` - Correlação 30m retornos vs OFI EURUSD — INPUT (pass‑through)
161. `corr30_tickvol_y_spx` - Correlação 30m tick volume EURUSD vs SPX — INPUT (pass‑through)
162. `corr60_y_dxy` - Correlação 60m EURUSD vs DXY — INPUT (pass‑through)
163. `corr60_y_spx` - Correlação 60m EURUSD vs SPX — INPUT (pass‑through)
164. `y_interaction_spread_ofi` - Interação spread × OFI — INPUT (pass‑through)
165. `y_interaction_vol_beta` - Interação volume × beta — INPUT (pass‑through)

#### **📈 Features de Arbitragem e Relativo**
166. `y_bond_us_minus_de_ret1m` - Retorno 1m bond US - DE — INPUT (pass‑through; relativo/retorno)
167. `y_dxy_vs_spx` - DXY vs SPX — INPUT (pass‑through; relativo)
168. `y_xau_real_yield_proxy` - Proxy do yield real do ouro — INPUT (pass‑through)
169. `z_240m_bond_us_de` - Z-score 240m do bond US-DE — INPUT (pass‑through)

#### **📈 Features de Retornos Históricos**
170. `y_ret_1m` - Retorno 1m — INPUT (pass‑through; retorno já estacionário)
171. `y_ret_5m` - Retorno 5m — INPUT (pass‑through)
172. `y_ret_10m` - Retorno 10m — INPUT (pass‑through)

### **🟠 TARGETS (Variáveis Dependentes)**

#### **🎯 Modelo M1 - Microestrutura e Momentum Rápido**
173. `m1_price_direction_5m` - Direção do preço 5m (1=up, -1=down, 0=flat) — OUT (target; leakage)
174. `m1_price_direction_15m` - Direção do preço 15m (1=up, -1=down, 0=flat) — OUT (target; leakage)
175. `m1_price_direction_30m` - Direção do preço 30m (1=up, -1=down, 0=flat) — OUT (target; leakage)

#### **🎯 Modelo M2 - Tendência e Força de Volume**
176. `m2_peak_profit_30m` - Pico de lucro 30m (regressão) — OUT (target; leakage)
177. `m2_peak_profit_60m` - Pico de lucro 60m (regressão) — OUT (target; leakage)
178. `m2_peak_profit_120m` - Pico de lucro 120m (regressão) — OUT (target; leakage)

#### **🎯 Modelo M3 - Order Flow e Agressão**
179. `m3_ofi_response_3m` - Resposta OFI 3m (1=continuation, -1=reversal, 0=absorption) — OUT (target; leakage)
180. `m3_ofi_response_5m` - Resposta OFI 5m (1=continuation, -1=reversal, 0=absorption) — OUT (target; leakage)
181. `m3_ofi_response_10m` - Resposta OFI 10m (1=continuation, -1=reversal, 0=absorption) — OUT (target; leakage)
182. `m3_ofi_response_target_3m` - Resposta OFI target 3m (1=continuation, -1=reversal, 0=absorption) — OUT (target; leakage)
183. `m3_ofi_response_target_5m` - Resposta OFI target 5m (1=continuation, -1=reversal, 0=absorption) — OUT (target; leakage)
184. `m3_ofi_response_target_10m` - Resposta OFI target 10m (1=continuation, -1=reversal, 0=absorption) — OUT (target; leakage)

#### **🎯 Modelo M4 - Contexto Inter-mercado**
185. `m4_trend_strength_1h` - Força da tendência 1h — OUT (target; leakage)
186. `m4_trend_strength_4h` - Força da tendência 4h — OUT (target; leakage)

#### **🎯 Modelo M5 - Contexto Macro (Taxas de Juros, Risco)**
187. `m5_correlation_regime_1h` - Regime de correlação 1h — OUT (target; leakage)
188. `m5_correlation_regime_4h` - Regime de correlação 4h — OUT (target; leakage)

#### **🎯 Modelo M6 - Classificação de Regime de Mercado**
189. `m6_market_regime_4h` - Regime de mercado 4h — OUT (target; leakage)

#### **🎯 Modelo M7 - Volatilidade e Risco**
190. `m7_vol_burst_1m` - Burst de volume 1m — OUT (target; leakage)
191. `m7_vol_burst_5m` - Burst de volume 5m — OUT (target; leakage)
192. `m7_vol_burst_15m` - Burst de volume 15m — OUT (target; leakage)

#### **🎯 Modelo M8 - Liquidez e Spread**
193. `m8_spread_burst_1m` - Burst de spread 1m — OUT (target; leakage)
194. `m8_spread_burst_3m` - Burst de spread 3m — OUT (target; leakage)

#### **🎯 Modelo M9 - Arbitragem e Valor Relativo**
195. `m9_price_direction_5m` - Direção do preço 5m — OUT (target; leakage)
196. `m9_price_direction_10m` - Direção do preço 10m — OUT (target; leakage)
197. `m9_price_direction_20m` - Direção do preço 20m — OUT (target; leakage)

#### **🎯 Retornos Forward (Targets de Regressão)**
198. `y_ret_fwd_1m` - Retorno forward 1m — OUT (target; leakage)
199. `y_ret_fwd_3m` - Retorno forward 3m — OUT (target; leakage)
200. `y_ret_fwd_5m` - Retorno forward 5m — OUT (target; leakage)
201. `y_ret_fwd_10m` - Retorno forward 10m — OUT (target; leakage)
202. `y_ret_fwd_15m` - Retorno forward 15m — OUT (target; leakage)
203. `y_ret_fwd_20m` - Retorno forward 20m — OUT (target; leakage)
204. `y_ret_fwd_30m` - Retorno forward 30m — OUT (target; leakage)
205. `y_ret_fwd_60m` - Retorno forward 60m — OUT (target; leakage)
206. `y_ret_fwd_120m` - Retorno forward 120m — OUT (target; leakage)
207. `y_ret_fwd_240m` - Retorno forward 240m — OUT (target; leakage)

## 📊 Resumo por Categoria

### **🔴 Features (199 colunas)**
- **Preço OHLC**: 4 colunas (originais)
- **Spread e Custos**: 16 colunas
- **Volume e Liquidez**: 27 colunas
- **Técnicas - Médias Móveis**: 9 colunas
- **Técnicas - Momentum**: 8 colunas
- **Técnicas - Bollinger Bands**: 4 colunas
- **Técnicas - Canais e Volatilidade**: 8 colunas
- **Técnicas - Osciladores**: 12 colunas
- **Técnicas - Volatilidade Avançada**: 5 colunas
- **Técnicas - Estatísticas**: 7 colunas
- **Técnicas - VSA e Volume**: 3 colunas
- **Técnicas - SMC**: 11 colunas
- **Técnicas - Morfologia**: 10 colunas
- **Z-scores e Normalizações**: 5 colunas
- **Correlações e Betas**: 8 colunas
- **OFI**: 9 colunas
- **Regime e Contexto**: 5 colunas
- **Tempo e Sazonalidade**: 7 colunas
- **Drivers**: 22 colunas
- **Interações**: 6 colunas
- **Arbitragem**: 4 colunas
- **Retornos Históricos**: 3 colunas

### **🟠 Targets (10 colunas)**
- **Modelos M1-M9**: 9 colunas (classificação)
- **Retornos Forward**: 10 colunas (regressão)

## 🏗️ Arquitetura do Pipeline

### **📥 Estágio 0: Extração e Alinhamento**
- **Módulo**: `data_extraction.py`
- **Função**: Extrai dados das tabelas `*_candle` do ClickHouse
- **Colunas originais**: OHLC, volume, spread, timestamps

### **🔧 Estágio 1: Engenharia de Features**
- **Módulo**: `forex_features.py` - Features de pares forex
- **Módulo**: `technical_features.py` - Indicadores técnicos
- **Módulo**: `advanced_indicators.py` - Indicadores avançados
- **Módulo**: `microstructure_features.py` - Features de microestrutura
- **Módulo**: `orderflow_features.py` - Features de order flow
- **Módulo**: `volume_price_features.py` - Features volume-preço
- **Módulo**: `timeseries_features.py` - Features de séries temporais
- **Módulo**: `macro_features.py` - Features macroeconômicas
- **Módulo**: `smc_features.py` - Smart Money Concepts
- **Módulo**: `arbitrage_features.py` - Features de arbitragem

### **🎯 Estágio 2: Cálculo de Targets**
- **Módulo**: `targets.py`
- **Função**: `calculate_all_targets()` - Calcula targets M1-M9
- **Função**: `calculate_forward_returns()` - Calcula retornos forward

### **📊 Estágio 3: Features Adicionais**
- **Módulo**: `targets.py`
- **Função**: `calculate_lags_and_additional_features()` - Lags e interações

## 🔍 Colunas Originais vs. Calculadas

### **📋 Colunas Originais (Extraídas do ClickHouse)**
- `timestamp`, `symbol`
- `y_open`, `y_high`, `y_low`, `y_close`
- `y_tick_volume`, `y_total_volume` ⚠️ **REMOVIDAS DO PROCESSAMENTO**
- `y_avg_spread`, `y_max_spread`, `y_min_spread`, `y_avg_spread_relative`
- `y_avg_volume_imbalance`, `y_close_volume_imbalance`
- `y_volume_delta`, `y_buy_aggression_volume`, `y_sell_aggression_volume`
- `y_tick_volatility`, `y_true_range`
- `y_price_changes_count`, `y_price_velocity`
- `y_is_market_open`

### **🧮 Colunas Calculadas (Engenharia de Features)**
- **Todas as demais colunas** são calculadas através de:
  - Indicadores técnicos (RSI, MACD, Bollinger Bands, etc.)
  - Z-scores e normalizações
  - Correlações e betas
  - Features de regime e contexto
  - Targets dos modelos M1-M9
  - Retornos forward

### **🚫 Features Bloqueadas (Não Processadas)**
- `y_tick_volume` - Volume bruto do tick (bloqueado via `feature_denylist`)
- `y_total_volume` - Volume bruto total (bloqueado via `feature_denylist`)
- **Motivo**: Features brutas de volume têm escalas muito diferentes e podem causar instabilidade no ML
- **Alternativa**: Use `y_tickvol_z_15m`, `y_tickvol_z_60m`, `y_tickvol_z_l1` (Z-scores normalizados)

## 📈 Uso no Pipeline Feature Genesis

Este dataset contém **209 features** que são processadas pelo pipeline Feature Genesis:

1. **Estágio 0**: Estacionarização e engenharia de features
2. **Estágio 1**: Ranking por dCor (correlação de distância)
3. **Estágio 2**: Redundância (VIF + MI)
4. **Estágio 3**: Wrappers leves (LightGBM/XGBoost)
5. **Estágio 4**: CPCV opcional (validação)

## 🎯 Aplicação dos Targets

### **Modelos de Classificação (M1, M3, M4, M5, M6, M7, M8, M9)**
- **Uso**: Classificação de direção de preço, regime de mercado, burst de volatilidade
- **Aplicação**: Estratégias de trading baseadas em sinais categóricos

### **Modelos de Regressão (M2)**
- **Uso**: Predição de magnitude de movimento (peak profit potential)
- **Aplicação**: Estratégias de posicionamento de tamanho

### **Retornos Forward**
- **Uso**: Predição de retornos em diferentes horizontes temporais
- **Aplicação**: Modelos de predição de preço e timing de entrada/saída

---
*Documentação atualizada com base na análise do código fonte do módulo `ia_master_table`*
