"""
Production Backtesting Engine for FinGraph
Implements realistic trading simulation with transaction costs
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    initial_capital: float = 100000
    position_sizing: str = 'equal_weight'
    max_positions: int = 10

    # Transaction costs
    commission_pct: float = 0.001   # 10 bps
    slippage_pct: float = 0.0005    # 5 bps

    # Risk limits
    max_position_pct: float = 0.10  # 10% per position
    max_leverage: float = 1.0
    stop_loss: float = 0.08         # 8% trailing from entry

    # Rebalancing / exits
    rebalance_frequency: str = 'weekly'  # 'daily' | 'weekly' | 'monthly'
    min_holding_days: int = 2
    max_holding_days: int = 20           # ✱ NEW: force exits
    risk_quantile_max: float = 0.6       # ✱ NEW: filter by risk rank per date (0..1)


class FinGraphBacktester:
    """
    Production-grade backtesting engine with:
    - t+1 execution
    - rank-based selection (cross-sectional)
    - scheduled rebalancing & exits
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.results = {}

    def backtest(
        self,
        predictions: pd.DataFrame,
        prices: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict:
        logger.info("Starting backtest...")

        # --- Align & sanitize
        preds = predictions.copy()
        px = prices.copy()
        preds['date'] = pd.to_datetime(preds['date'])
        px['date'] = pd.to_datetime(px['date'])
        preds = preds.sort_values(['date', 'symbol'])
        px = px.sort_values(['date', 'symbol'])

        if start_date:
            preds = preds[preds['date'] >= pd.to_datetime(start_date)]
            px = px[px['date'] >= pd.to_datetime(start_date)]
        if end_date:
            preds = preds[preds['date'] <= pd.to_datetime(end_date)]
            px = px[px['date'] <= pd.to_datetime(end_date)]

        # --- t+1 execution map
        ### ✱ NEW
        px['next_date'] = px.groupby('symbol')['date'].shift(-1)
        px['next_open'] = px.groupby('symbol')['open'].shift(-1)
        next_open_map = px.set_index(['date', 'symbol'])['next_open']

        # helpful lookups
        close_map = px.set_index(['date', 'symbol'])['close']

        dates = sorted(preds['date'].unique())

        # --- Portfolio state
        portfolio = {
            'cash': float(self.config.initial_capital),
            'positions': {},  # symbol -> dict(shares, entry_price, entry_date, peak_price)
            'values': [],
            'dates': [],
            'trades': []
        }

        # --- Loop over prediction dates
        for d in dates:
            day_preds = preds[preds['date'] == d].copy()
            if day_preds.empty:
                continue

            # Cross-sectional rank logic
            ### ✱ NEW: rank by return_pred; filter by risk quantile
            day_preds['ret_rank'] = day_preds['return_pred'].rank(ascending=False, pct=True)
            day_preds['risk_rank'] = day_preds['risk_pred'].rank(ascending=True, pct=True)  # lower risk = better

            # Select top-k by ret among those with acceptable risk
            day_sel = day_preds[day_preds['risk_rank'] <= self.config.risk_quantile_max]
            day_sel = day_sel.sort_values(['ret_rank', 'risk_rank']).head(self.config.max_positions)
            target_symbols = set(day_sel['symbol'].tolist())

            # --- Rebalance frequency gate
            if not self._is_rebalance_day(d, self.config.rebalance_frequency):
                # still mark-to-market + exit rules like stop-loss / max holding
                self._apply_exits_tplus1(d, next_open_map, close_map, portfolio, allow_rebalance=False)
            else:
                # Full rebalance: sell names that are not in target, buy missing targets
                self._rebalance_tplus1(d, target_symbols, day_sel, next_open_map, close_map, portfolio)

            # Mark portfolio value at **today's close**
            v = self._mark_to_market(d, close_map, portfolio)
            portfolio['values'].append(v)
            portfolio['dates'].append(d)

        # --- Liquidate at the end (t+1 open after last date if available)
        ### ✱ NEW
        if dates:
            self._liquidate_all_tplus1(dates[-1], next_open_map, portfolio)

        # Final marking (last close available)
        if dates:
            v = self._mark_to_market(dates[-1], close_map, portfolio)
            portfolio['values'].append(v)
            portfolio['dates'].append(dates[-1])

        # --- Metrics
        return self._calculate_metrics(portfolio)

    # ---------- Trade/Exit helpers ----------

    def _is_rebalance_day(self, date: pd.Timestamp, freq: str) -> bool:
        ### ✱ NEW
        if freq == 'daily':
            return True
        if freq == 'weekly':
            # Monday rebalance
            return pd.Timestamp(date).weekday() == 0
        if freq == 'monthly':
            # First trading day of month
            return pd.Timestamp(date).day <= 3
        return True

    def _position_value_hint(self, close_map: pd.Series, date: pd.Timestamp, portfolio: Dict) -> float:
        ### ✱ NEW
        value = portfolio['cash']
        for sym, pos in portfolio['positions'].items():
            px = close_map.get((date, sym), np.nan)
            if np.isfinite(px):
                value += pos['shares'] * float(px)
        return float(value)

    def _size_per_position(self, close_map: pd.Series, date: pd.Timestamp, portfolio: Dict) -> float:
        ### ✱ NEW
        equity = self._position_value_hint(close_map, date, portfolio)
        if self.config.position_sizing == 'equal_weight':
            return equity * self.config.max_position_pct
        else:
            return equity / max(1, self.config.max_positions)

    def _rebalance_tplus1(
        self,
        date: pd.Timestamp,
        target_symbols: set,
        day_sel: pd.DataFrame,
        next_open_map: pd.Series,
        close_map: pd.Series,
        portfolio: Dict
    ):
        """Sell names not in target, exit rules, then buy missing target names — all at T+1 open."""
        # 1) Exits for: not in target, stop-loss, max holding
        self._apply_exits_tplus1(date, next_open_map, close_map, portfolio, allow_rebalance=True, target_symbols=target_symbols)

        # 2) Buys for missing target names
        cash_before = portfolio['cash']
        per_pos_val = self._size_per_position(close_map, date, portfolio)

        have = set(portfolio['positions'].keys())
        to_buy = [s for s in target_symbols if s not in have]

        for sym in to_buy:
            px = next_open_map.get((date, sym), np.nan)
            if not np.isfinite(px) or px <= 0:
                continue
            shares = int(per_pos_val // px)
            if shares <= 0:
                continue

            cost = shares * px * (1 + self.config.commission_pct + self.config.slippage_pct)
            if cost > portfolio['cash']:
                continue

            portfolio['cash'] -= cost
            portfolio['positions'][sym] = {
                'shares': shares,
                'entry_price': float(px),
                'entry_date': pd.to_datetime(date) + pd.Timedelta(days=1),
                'peak_price': float(px)
            }
            portfolio['trades'].append({
                'date': pd.to_datetime(date) + pd.Timedelta(days=1),
                'symbol': sym,
                'action': 'BUY',
                'shares': int(shares),
                'price': float(px),
                'cost': float(cost)
            })

    def _apply_exits_tplus1(
        self,
        date: pd.Timestamp,
        next_open_map: pd.Series,
        close_map: pd.Series,
        portfolio: Dict,
        allow_rebalance: bool,
        target_symbols: Optional[set] = None
    ):
        """Exit logic at T+1 open: stop-loss, max holding, and (if allowed) not-in-target sells."""
        symbols = list(portfolio['positions'].keys())
        for sym in symbols:
            pos = portfolio['positions'][sym]
            # Update trailing peak with today's close
            todays_close = close_map.get((date, sym), np.nan)
            if np.isfinite(todays_close):
                pos['peak_price'] = max(pos.get('peak_price', pos['entry_price']), float(todays_close))

            exit_reason = None

            # Max holding days
            held_days = (pd.to_datetime(date) - pd.to_datetime(pos['entry_date'])).days
            if held_days >= self.config.max_holding_days:
                exit_reason = 'MAX_HOLD'
            # Stop loss (from entry or trailing peak; choose stricter)
            elif np.isfinite(todays_close):
                entry_stop = todays_close <= pos['entry_price'] * (1 - self.config.stop_loss)
                trail_stop = todays_close <= pos['peak_price'] * (1 - self.config.stop_loss)
                if entry_stop or trail_stop:
                    exit_reason = 'STOP'

            # Rebalance sell (not in target)
            if allow_rebalance and (target_symbols is not None) and (sym not in target_symbols):
                exit_reason = exit_reason or 'REBAL'

            if exit_reason:
                # Execute at next day's open
                px = next_open_map.get((date, sym), np.nan)
                if np.isfinite(px) and pos['shares'] > 0:
                    proceeds = pos['shares'] * px * (1 - self.config.commission_pct - self.config.slippage_pct)
                    portfolio['cash'] += proceeds
                    portfolio['trades'].append({
                        'date': pd.to_datetime(date) + pd.Timedelta(days=1),
                        'symbol': sym,
                        'action': 'SELL',
                        'shares': int(pos['shares']),
                        'price': float(px),
                        'proceeds': float(proceeds),
                        'reason': exit_reason
                    })
                    del portfolio['positions'][sym]

    def _liquidate_all_tplus1(self, last_date: pd.Timestamp, next_open_map: pd.Series, portfolio: Dict):
        """Liquidate all positions at the first available T+1 open after the last prediction date."""
        ### ✱ NEW
        for sym, pos in list(portfolio['positions'].items()):
            px = next_open_map.get((pd.to_datetime(last_date), sym), np.nan)
            if np.isfinite(px) and pos['shares'] > 0:
                proceeds = pos['shares'] * px * (1 - self.config.commission_pct - self.config.slippage_pct)
                portfolio['cash'] += proceeds
                portfolio['trades'].append({
                    'date': pd.to_datetime(last_date) + pd.Timedelta(days=1),
                    'symbol': sym,
                    'action': 'SELL',
                    'shares': int(pos['shares']),
                    'price': float(px),
                    'proceeds': float(proceeds),
                    'reason': 'EOP'
                })
                del portfolio['positions'][sym]

    # ---------- Valuation & metrics ----------

    def _mark_to_market(self, date: pd.Timestamp, close_map: pd.Series, portfolio: Dict) -> float:
        """Mark portfolio at today's close."""
        total = portfolio['cash']
        for sym, pos in portfolio['positions'].items():
            px = close_map.get((date, sym), np.nan)
            if np.isfinite(px):
                total += pos['shares'] * float(px)
        return float(total)

    def _calculate_metrics(self, portfolio: Dict) -> Dict:
        values = np.array(portfolio['values'], dtype=float)
        dates = pd.to_datetime(portfolio['dates'])

        if len(values) < 2:
            return {
                'error': 'Insufficient data for metrics',
                'num_trades': len(portfolio['trades']),
                'num_sells': len([t for t in portfolio['trades'] if t['action'] == 'SELL']),
                'final_value': float(values[-1]) if len(values) else self.config.initial_capital
            }

        rets = np.diff(values) / (values[:-1] + 1e-12)
        total_return = (values[-1] / values[0]) - 1
        days = max(1, (dates[-1] - dates[0]).days)
        years = days / 365.25
        vol = np.std(rets)
        sharpe = (np.mean(rets) / (vol + 1e-12)) * np.sqrt(252) if len(rets) else 0.0

        cumulative = np.cumprod(1 + rets) if len(rets) else np.array([1.0])
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / (running_max + 1e-12)
        max_drawdown = float(np.min(drawdown)) if len(drawdown) else 0.0

        sells = [t for t in portfolio['trades'] if t['action'] == 'SELL']
        if sells:
            profitable = sum(1 for t in sells if t.get('proceeds', 0) > t.get('cost', float('inf')))
            win_rate = profitable / len(sells)
        else:
            win_rate = 0.0

        metrics = {
            'total_return': float(total_return),
            'annualized_return': float(((1 + total_return) ** (1 / years) - 1) if years > 0 else total_return),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'num_trades': int(len(portfolio['trades'])),
            'num_sells': int(len(sells)),
            'final_value': float(values[-1]),
            'profitable': bool(total_return > 0)
        }

        logger.info("Backtest complete:")
        logger.info(f"  Total Return: {metrics['total_return']:.2%}")
        logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        logger.info(f"  Win Rate: {metrics['win_rate']:.2%}")

        return metrics
