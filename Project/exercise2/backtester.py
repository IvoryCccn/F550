#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 15:51:00 2026

@author: fabriziocoiai
"""
# backtester.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd


@dataclass
class BacktestConfig:
    initial_cash: float = 100_000.0
    transaction_cost_bps: float = 10.0
    trade_size_units: float = 5.0
    allow_short: bool = False


class EventBacktester:
    """
    One-asset backtest that only asks the agent for a decision on event dates
    (here: SEC filing dates), then holds until next event.
    """        
     
    def __init__(self, prices: pd.Series, cfg: Optional[BacktestConfig] = None):
        # Convert DataFrame to Series defensively
        if isinstance(prices, pd.DataFrame):
            if "Close" in prices.columns:
                prices = prices["Close"]
            elif prices.shape[1] == 1:
                prices = prices.iloc[:, 0]
            else:
                raise ValueError(f"prices DataFrame must have Close or be single-column. Got {prices.columns}")

        prices.index = pd.to_datetime(prices.index)
        self.prices = prices.sort_index().astype(float)
        self.cfg = cfg or BacktestConfig()  
        
    def run(
        self,
        *,
        ticker: str,
        agent,
        valuation_inputs_by_event_date: Dict[pd.Timestamp, object],
        
    ) -> pd.DataFrame:
        cash = self.cfg.initial_cash
        pos = 0.0
        tc = self.cfg.transaction_cost_bps / 10_000.0

        rows: List[dict] = []

        last_score = None
        last_conf = None
        last_thesis = None
        last_action = "hold"

        for dt, px in self.prices.items():
            dt = pd.Timestamp(dt)
            px = float(px)

            decision = None

            vin = valuation_inputs_by_event_date.get(dt)
            if vin is not None:
                decision = agent.decide(vin)
                action = decision.action

                if action == "buy":
                    trade_value = self.cfg.trade_size_units * px
                    cost = trade_value * tc
                    if cash >= trade_value + cost:
                        cash -= trade_value + cost
                        pos += self.cfg.trade_size_units

                elif action == "sell":
                    if self.cfg.allow_short:
                        trade_value = self.cfg.trade_size_units * px
                        cost = trade_value * tc
                        cash += trade_value - cost
                        pos -= self.cfg.trade_size_units
                    else:
                        if pos >= self.cfg.trade_size_units:
                            trade_value = self.cfg.trade_size_units * px
                            cost = trade_value * tc
                            cash += trade_value - cost
                            pos -= self.cfg.trade_size_units

                last_score = decision.score
                last_conf = decision.confidence
                last_thesis = decision.thesis
                last_action = decision.action

            pv = cash + pos * px

            rows.append({
                "date": dt,
                "ticker": ticker,
                "price": px,
                "action": last_action,
                "position": pos,
                "cash": cash,
                "portfolio_value": pv,
                "decision_score": last_score,
                "decision_confidence": last_conf,
                "decision_thesis": last_thesis,
                })

        df = pd.DataFrame(rows).set_index("date")
        df["returns"] = df["portfolio_value"].pct_change().fillna(0.0)
        return df    


def compute_sharpe(df: pd.DataFrame, risk_free_rate: float = 0.05, trading_days: int = 252) -> float:
    """
    Compute annualised Sharpe Ratio from a backtest result DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Output of EventBacktester.run() — must contain a 'returns' column.
    risk_free_rate : float
        Annual risk-free rate (default 5%, approx US T-bill rate in 2024).
    trading_days : int
        Number of trading days per year (default 252).
    
    Returns
    -------
    float
        Annualised Sharpe Ratio. Returns NaN if std is zero.
    """
    daily_returns = df["returns"].dropna()
    if daily_returns.std() == 0:
        return float("nan")
    daily_rf = risk_free_rate / trading_days
    excess = daily_returns - daily_rf
    sharpe = (excess.mean() / excess.std()) * (trading_days ** 0.5)
    return float(sharpe)
        