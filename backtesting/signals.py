"""Signal generation and performance metrics for backtesting."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, List, Optional

import pandas as pd  # type: ignore

from .config import SignalThresholds


@dataclass
class TradeState:
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    take_profit_pct: float
    stop_loss_pct: float
    size: float
    prediction: float
    max_holding: Optional[timedelta]

    def take_profit_price(self) -> float:
        if self.direction == "BUY":
            return self.entry_price * (1 + self.take_profit_pct)
        return self.entry_price * (1 - self.take_profit_pct)

    def stop_loss_price(self) -> float:
        if self.direction == "BUY":
            return self.entry_price * (1 - self.stop_loss_pct)
        return self.entry_price * (1 + self.stop_loss_pct)


@dataclass
class TradeResult:
    direction: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    prediction: float
    exit_reason: str
    return_pct: float
    pnl: float
    duration_minutes: float


class SignalGenerator:
    """Generate trading signals and compute backtest metrics."""

    def __init__(self, thresholds: SignalThresholds):
        self.thresholds = thresholds
        self.current: Optional[TradeState] = None
        self.trade_log: List[TradeResult] = []
        self.cooldown_until: Optional[pd.Timestamp] = None

    def _can_enter(self, timestamp: pd.Timestamp) -> bool:
        if self.cooldown_until is None:
            return True
        return timestamp >= self.cooldown_until

    def _open_trade(self, direction: str, timestamp: pd.Timestamp, price: float, prediction: float):
        self.current = TradeState(
            direction=direction,
            entry_time=timestamp,
            entry_price=float(price),
            take_profit_pct=self.thresholds.take_profit_pct,
            stop_loss_pct=self.thresholds.stop_loss_pct,
            size=self.thresholds.position_size,
            prediction=float(prediction),
            max_holding=(
                timedelta(minutes=self.thresholds.max_holding_minutes)
                if self.thresholds.max_holding_minutes
                else None
            ),
        )

    def _close_trade(self, timestamp: pd.Timestamp, price: float, reason: str):
        if not self.current:
            return None
        trade = self.current
        self.current = None
        if self.thresholds.cooldown_minutes:
            self.cooldown_until = timestamp + timedelta(minutes=self.thresholds.cooldown_minutes)
        return_pct = self._compute_return_pct(trade.direction, trade.entry_price, price)
        pnl = return_pct * trade.size
        duration = (timestamp - trade.entry_time).total_seconds() / 60.0
        result = TradeResult(
            direction=trade.direction,
            entry_time=trade.entry_time,
            exit_time=timestamp,
            entry_price=trade.entry_price,
            exit_price=float(price),
            prediction=trade.prediction,
            exit_reason=reason,
            return_pct=return_pct,
            pnl=pnl,
            duration_minutes=duration,
        )
        self.trade_log.append(result)
        return result

    def _compute_return_pct(self, direction: str, entry_price: float, exit_price: float) -> float:
        if direction == "BUY":
            return (exit_price - entry_price) / entry_price
        return (entry_price - exit_price) / entry_price

    def step(
        self,
        timestamp: pd.Timestamp,
        price: float,
        prediction: float,
        feature_snapshot: Dict[str, float],
    ) -> Dict[str, object]:
        """Advance the generator by one observation."""
        ts = pd.Timestamp(timestamp)
        price = float(price)
        prediction = float(prediction)
        trade_event: Optional[TradeResult] = None
        signal = "HOLD"

        if self.current:
            exit_reason = self._check_exit_conditions(ts, price)
            if exit_reason:
                trade_event = self._close_trade(ts, price, exit_reason)

        if self.current is None and self._can_enter(ts):
            if prediction >= self.thresholds.buy_probability:
                self._open_trade("BUY", ts, price, prediction)
                signal = "BUY"
            elif prediction <= self.thresholds.sell_probability:
                self._open_trade("SELL", ts, price, prediction)
                signal = "SELL"

        trade_state = None
        if self.current:
            trade_state = {
                "direction": self.current.direction,
                "entry_time": self.current.entry_time.isoformat(),
                "entry_price": self.current.entry_price,
                "take_profit": self.current.take_profit_price(),
                "stop_loss": self.current.stop_loss_price(),
                "prediction": self.current.prediction,
            }

        event = {
            "type": "tick",
            "timestamp": ts.isoformat(),
            "price": price,
            "prediction": prediction,
            "signal": signal,
            "features": feature_snapshot,
            "trade_state": trade_state,
        }
        if trade_event:
            event["trade_close"] = {
                "direction": trade_event.direction,
                "exit_reason": trade_event.exit_reason,
                "return_pct": trade_event.return_pct,
                "pnl": trade_event.pnl,
            }
        return event

    def _check_exit_conditions(self, timestamp: pd.Timestamp, price: float) -> Optional[str]:
        if not self.current:
            return None
        trade = self.current
        if trade.direction == "BUY":
            if price >= trade.take_profit_price():
                return "TP"
            if price <= trade.stop_loss_price():
                return "SL"
        else:
            if price <= trade.take_profit_price():
                return "TP"
            if price >= trade.stop_loss_price():
                return "SL"
        if trade.max_holding is not None:
            if timestamp - trade.entry_time >= trade.max_holding:
                return "TIMEOUT"
        return None

    def finalize(self, timestamp: Optional[pd.Timestamp], price: Optional[float]):
        if self.current and timestamp is not None and price is not None:
            self._close_trade(pd.Timestamp(timestamp), float(price), "EOD")
        return self.trade_log

    def performance_metrics(self) -> Dict[str, float]:
        if not self.trade_log:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_return_pct": 0.0,
                "total_pnl": 0.0,
                "max_drawdown_pct": 0.0,
                "sharpe_ratio": 0.0,
                "avg_trade_duration_min": 0.0,
            }
        df = pd.DataFrame([trade.__dict__ for trade in self.trade_log])
        total_return = df["return_pct"].sum()
        total_pnl = df["pnl"].sum()
        win_rate = (df["return_pct"] > 0).mean()
        avg_duration = df["duration_minutes"].mean()
        equity = (1 + df["return_pct"]).cumprod()
        rolling_max = equity.cummax()
        drawdown = equity / rolling_max - 1
        max_drawdown = drawdown.min()
        sharpe = self._compute_sharpe(df["return_pct"].values)
        return {
            "total_trades": int(len(df)),
            "win_rate": float(win_rate),
            "total_return_pct": float(total_return),
            "total_pnl": float(total_pnl),
            "max_drawdown_pct": float(max_drawdown),
            "sharpe_ratio": float(sharpe),
            "avg_trade_duration_min": float(avg_duration),
        }

    def _compute_sharpe(self, returns) -> float:
        if len(returns) < 2:
            return 0.0
        mean = returns.mean()
        std = returns.std(ddof=1)
        if std == 0:
            return 0.0
        # Assume trades roughly correspond to one per day equivalent for sqrt(252)
        annual_factor = 252 ** 0.5
        return float((mean / std) * annual_factor)
