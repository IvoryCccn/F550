#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backtester.py  —  Modified for Exercise 2
Changes vs original:
  - Added compute_sharpe() static method
  - Added print_summary() helper for clean comparison output
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
import numpy as np


@dataclass
class BacktestConfig:
    initial_cash: float = 100_000.0
    transaction_cost_bps: float = 10.0
    trade_size_units: float = 5.0
    allow_short: bool = False


class EventBacktester:
    def __init__(self, prices: pd.Series, cfg=None):
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

    def run(self, *, ticker, agent, valuation_inputs_by_event_date):
        cash = self.cfg.initial_cash
        pos  = 0.0
        tc   = self.cfg.transaction_cost_bps / 10_000.0
        rows = []
        last_score, last_conf, last_thesis, last_action = None, None, None, "hold"

        for dt, px in self.prices.items():
            dt = pd.Timestamp(dt)
            px = float(px)

            vin = valuation_inputs_by_event_date.get(dt)
            if vin is not None:
                decision    = agent.decide(vin)
                last_action = decision.action

                if last_action == "buy":
                    trade_value = self.cfg.trade_size_units * px
                    cost = trade_value * tc
                    if cash >= trade_value + cost:
                        cash -= trade_value + cost
                        pos  += self.cfg.trade_size_units
                elif last_action == "sell":
                    if self.cfg.allow_short:
                        trade_value = self.cfg.trade_size_units * px
                        cash += trade_value * (1 - tc)
                        pos  -= self.cfg.trade_size_units
                    else:
                        if pos >= self.cfg.trade_size_units:
                            trade_value = self.cfg.trade_size_units * px
                            cash += trade_value * (1 - tc)
                            pos  -= self.cfg.trade_size_units

                last_score  = decision.score
                last_conf   = decision.confidence
                last_thesis = decision.thesis

            rows.append({"date": dt, "ticker": ticker, "price": px,
                         "action": last_action, "position": pos, "cash": cash,
                         "portfolio_value": cash + pos * px,
                         "decision_score": last_score,
                         "decision_confidence": last_conf,
                         "decision_thesis": last_thesis})

        df = pd.DataFrame(rows).set_index("date")
        df["returns"] = df["portfolio_value"].pct_change().fillna(0.0)
        return df

    @staticmethod
    def compute_sharpe(df, risk_free_rate_annual=0.05, trading_days_per_year=252):
        """
        Annualised Sharpe Ratio.
        df must contain a 'returns' column.
        risk_free_rate_annual: default 5% (US 2024 approx.)
        """
        rets = df["returns"].dropna()
        if len(rets) < 2:
            return float("nan")
        
        pv = df["portfolio_value"].dropna()
        if pv.max() - pv.min() < 1e-6:
            return float("nan")
        
        rf_daily    = risk_free_rate_annual / trading_days_per_year
        excess_rets = rets - rf_daily
        std = excess_rets.std(ddof=1)
        
        if std < 1e-8 or np.isnan(std):
            return float("nan")
        
        return float(excess_rets.mean() / std * np.sqrt(trading_days_per_year))

    @staticmethod
    def print_summary(label, df, sharpe, decisions=None):
        init_val  = df["portfolio_value"].iloc[0]
        final_val = df["portfolio_value"].iloc[-1]
        total_ret = final_val / init_val - 1
        print("=" * 60)
        print(f"  {label}")
        print("=" * 60)
        print(f"  Period        : {df.index[0].date()} -> {df.index[-1].date()}")
        print(f"  Initial value : ${init_val:,.0f}")
        print(f"  Final value   : ${final_val:,.0f}")
        print(f"  Total return  : {total_ret:+.2%}")
        print(f"  Sharpe Ratio  : {sharpe:.4f}")
        if decisions:
            print()
            print("  Decisions made:")
            for d in decisions:
                print(f"    [{d['date']}]  {d['action'].upper():4s}  conf={d['confidence']:.2f}  score={d['score']:+.2f}")
                print(f"      Thesis: {d['thesis']}")
        print("=" * 60)
        print()
