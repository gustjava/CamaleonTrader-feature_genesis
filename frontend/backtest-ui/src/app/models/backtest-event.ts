export interface BacktestSignalEvent {
  type: 'tick';
  timestamp: string;
  price: number;
  prediction: number;
  signal: 'BUY' | 'SELL' | 'HOLD';
  features: Record<string, number>;
  trade_state?: TradeStateSnapshot | null;
  trade_close?: TradeCloseSnapshot | null;
}

export interface TradeStateSnapshot {
  direction: 'BUY' | 'SELL';
  entry_time: string;
  entry_price: number;
  take_profit: number;
  stop_loss: number;
  prediction: number;
}

export interface TradeCloseSnapshot {
  direction: 'BUY' | 'SELL';
  exit_reason: 'TP' | 'SL' | 'TIMEOUT' | 'EOD' | string;
  return_pct: number;
  pnl: number;
}

export interface BacktestSummaryEvent {
  type: 'summary';
  symbol: string;
  timeframe?: string | null;
  metrics: Record<string, number>;
  trades: TradeSummary[];
}

export interface TradeSummary {
  direction: 'BUY' | 'SELL';
  entry_time: string;
  exit_time: string;
  entry_price: number;
  exit_price: number;
  prediction: number;
  exit_reason: string;
  return_pct: number;
  pnl: number;
  duration_minutes: number;
}

export type BacktestEvent = BacktestSignalEvent | BacktestSummaryEvent;

export interface AvailableModel {
  symbol: string;
  target: string;
  metadataPath: string;
  modelPath: string;
  datasetPath: string;
}
