#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py  —  Exercise 2: NVDA Valuation Agent with / without Sentiment
=======================================================================
Two TTM windows (as required by the exercise):
  Window 1:  05/02/2024 - 10/02/2024  (event date = 2024-02-07)
  Window 2:  05/08/2024 - 10/08/2024  (event date = 2024-08-07)

Sentiment scores from Exercise 1 (NVDA earnings-call NLP pipeline):
  Window 1 uses 2023Q4 earnings call (filed 2024-02-21)
  Window 2 uses 2024Q2 earnings call (filed 2024-08-28)

Runs:
  (A) Agent WITHOUT sentiment -> backtest -> Sharpe
  (B) Agent WITH    sentiment -> backtest -> Sharpe
  Then prints side-by-side comparison.

NOTE: SEC EDGAR XBRL data for NVDA has missing revenue/FCF due to
NVDA's non-standard fiscal year (ends Jan 31). We use yfinance
quarterly financials instead, which is cleaner and more reliable.
"""
from __future__ import annotations

import os
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
load_dotenv()

from llm_backend     import OpenAIBackend
from market_data     import get_price_series
from valuation_agent import ValuationAgent, ValuationInputs, ValuationAgentConfig
from backtester      import EventBacktester, BacktestConfig
from filing_rag      import FilingRAG


# ===========================================================================
# Configuration
# ===========================================================================

TICKER         = "NVDA"
BACKTEST_START = "2024-01-01"
BACKTEST_END   = "2024-12-31"
RISK_FREE_RATE = 0.05   # 5% annualised (US 2024 approximate)

# Two event dates — one trading day inside each required TTM window
EVENT_DATES = [
    pd.Timestamp("2024-02-07"),   # Window 1: 05/02 - 10/02/2024
    pd.Timestamp("2024-08-07"),   # Window 2: 05/08 - 10/08/2024
]

# Sentiment scores sourced from Exercise 1 NLP pipeline (mean aggregation)
# Window 1 -> most recent earnings call before window = 2023Q4 (filed 2024-02-21)
# Window 2 -> most recent earnings call before window = 2024Q2 (filed 2024-08-28)
SENTIMENT_BY_EVENT = {
    pd.Timestamp("2024-02-07"): {
        "lm_score":      0.0748,
        "finbert_score": 0.3265,
        "gpt_score":     6.81,
        "quarter":       "2023Q4",
    },
    pd.Timestamp("2024-08-07"): {
        "lm_score":      0.0316,
        "finbert_score": 0.3650,
        "gpt_score":     6.61,
        "quarter":       "2024Q2",
    },
}


# ===========================================================================
# Quarter table builder — yfinance version
# (replaces build_quarter_table from sec_fundamentals.py)
# Reason: NVDA uses a non-standard fiscal year ending Jan 31, which causes
# the SEC EDGAR XBRL revenue/FCF tags to return NaN via the original code.
# ===========================================================================

def build_nvda_quarter_table_from_yfinance(ticker: str) -> pd.DataFrame:
    """
    Fetch NVDA quarterly financials from yfinance and return a DataFrame
    compatible with the ValuationAgent.compute_metrics() interface.

    Columns: end, filed, revenue, op_income, net_income, ocf, capex, fcf,
             shares_outstanding
    """
    t = yf.Ticker(ticker)

    income   = t.quarterly_income_stmt   # columns = quarter-end dates
    cashflow = t.quarterly_cashflow

    rows = []
    for col in income.columns:
        end = pd.Timestamp(col)

        # --- Income statement items ---
        def get_income(label):
            try:
                return float(income.loc[label, col]) if label in income.index else None
            except Exception:
                return None

        revenue    = get_income("Total Revenue")
        net_income = get_income("Net Income")
        op_income  = get_income("Operating Income")

        # --- Cash flow items ---
        def get_cf(label):
            try:
                return float(cashflow.loc[label, col]) if label in cashflow.index else None
            except Exception:
                return None

        ocf   = get_cf("Operating Cash Flow")
        capex = get_cf("Capital Expenditure")

        # yfinance returns capex as negative; convert to positive for consistency
        if capex is not None:
            capex = abs(capex)

        fcf = (ocf - capex) if (ocf is not None and capex is not None) else None

        # Approximate filing date: SEC requires 10-Q within 40 days of quarter end
        filed = end + pd.Timedelta(days=45)

        rows.append({
            "end":               end,
            "filed":             filed,
            "revenue":           revenue,
            "op_income":         op_income,
            "net_income":        net_income,
            "ocf":               ocf,
            "capex":             capex,
            "fcf":               fcf,
            "shares_outstanding": None,   # filled below
        })

    # Get current shares outstanding from info
    try:
        shares = float(t.info.get("sharesOutstanding") or 0) or None
    except Exception:
        shares = None

    df = pd.DataFrame(rows).sort_values("end").reset_index(drop=True)
    if shares:
        df["shares_outstanding"] = shares

    return df


# ===========================================================================
# Helpers
# ===========================================================================

def align_to_trading_day(prices: pd.Series, dt: pd.Timestamp) -> pd.Timestamp:
    """Snap dt forward to the nearest available trading day."""
    dt  = pd.Timestamp(dt)
    idx = prices.index
    if dt in idx:
        return dt
    later = idx[idx >= dt]
    if len(later) == 0:
        return pd.Timestamp(idx[-1])
    return pd.Timestamp(later[0])


def get_recent_prices(ticker: str, asof: pd.Timestamp, n: int = 5) -> list:
    """
    Return the last n closing prices on or before asof, oldest first.
    Uses a 20-day lookback window to ensure enough trading days are available.
    """
    start = (asof - pd.Timedelta(days=20)).strftime("%Y-%m-%d")
    end   = (asof + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df    = yf.download(ticker, start=start, end=end,
                        auto_adjust=False, progress=False)
    if df.empty:
        return []
    # squeeze() converts single-column DataFrame to Series
    closes = df["Close"].squeeze().dropna().astype(float)
    closes = closes[closes.index <= asof]
    return closes.tail(n).tolist()


def extract_decisions(df: pd.DataFrame) -> list:
    """
    Extract unique agent decisions from the backtest result DataFrame.
    Returns a list of dicts with date, action, confidence, score, thesis.
    """
    decided = df.dropna(subset=["decision_thesis"]).copy()
    out, seen = [], set()
    for dt, row in decided.iterrows():
        thesis = row["decision_thesis"]
        if thesis not in seen:
            seen.add(thesis)
            out.append({
                "date":       dt.date(),
                "action":     row["action"],
                "confidence": row["decision_confidence"] or 0.0,
                "score":      row["decision_score"] or 0.0,
                "thesis":     thesis,
            })
    return out


# ===========================================================================
# Main
# ===========================================================================

def main():
    # -----------------------------------------------------------------------
    # 0. API key check
    # -----------------------------------------------------------------------
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY first.  e.g.  export OPENAI_API_KEY=sk-...")

    print(f"\n{'='*60}")
    print(f"  Exercise 2 -- NVDA Valuation Agent (with / without Sentiment)")
    print(f"{'='*60}\n")

    # -----------------------------------------------------------------------
    # 1. Price data
    # -----------------------------------------------------------------------
    print("Fetching NVDA price data from Yahoo Finance...")
    prices = get_price_series(TICKER, start=BACKTEST_START, end=BACKTEST_END)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index().astype(float)
    prices = prices[~prices.index.duplicated(keep="first")]
    print(f"  {len(prices)} trading days loaded "
          f"({prices.index[0].date()} to {prices.index[-1].date()})\n")

    # -----------------------------------------------------------------------
    # 2. Quarterly fundamentals (yfinance — avoids NVDA EDGAR tag issue)
    # -----------------------------------------------------------------------
    print("Fetching NVDA quarterly fundamentals via yfinance...")
    quarter_table = build_nvda_quarter_table_from_yfinance(TICKER)
    print(f"  {len(quarter_table)} quarterly rows loaded\n")

    # Diagnostic: verify revenue and FCF are populated
    print("  Sample fundamentals (last 6 quarters):")
    print(quarter_table[["end", "filed", "revenue", "net_income", "fcf"]]
          .tail(6).to_string(index=False))
    print()

    # -----------------------------------------------------------------------
    # 3. LLM backend
    # -----------------------------------------------------------------------
    llm = OpenAIBackend(
        chat_model="gpt-4o",
        embed_model="text-embedding-3-small",
        temperature=0.2,
        max_output_tokens=800,
    )

    # -----------------------------------------------------------------------
    # 4. RAG: add NVDA filing text snippets
    # -----------------------------------------------------------------------
    filing_rag = FilingRAG()

    filing_rag.add_document(
        llm,
        doc_id="nvda_q4_fy24_snippet",
        ticker=TICKER,
        filed=pd.Timestamp("2024-02-21"),
        source="10-Q",
        text=(
            "NVIDIA Q4 FY2024 revenue of $22.1 billion, up 265% YoY, above $20B outlook. "
            "Full fiscal 2024 revenue was $60.9 billion, up 126%. "
            "Data Center revenue tripled YoY to $47.5 billion. "
            "Next generation products (Blackwell) expected to be supply constrained. "
            "Management guided Q1 FY2025 revenue of approximately $24 billion. "
            "Gross margins expected to return to mid-70s percent range beyond Q1."
        ),
    )
    filing_rag.add_document(
        llm,
        doc_id="nvda_q2_fy25_snippet",
        ticker=TICKER,
        filed=pd.Timestamp("2024-08-28"),
        source="10-Q",
        text=(
            "NVIDIA Q2 FY2025 revenue of $30 billion, up 122% YoY, above $28B outlook. "
            "Data Center revenue reached $26.3 billion. "
            "Blackwell demand well above supply; shipments expected in billions by year-end. "
            "Gross margins stable in mid-70s percent range. "
            "Sovereign AI revenue expected to reach low double-digit billions for the year. "
            "Automotive and robotics verticals identified as next major growth drivers."
        ),
    )
    print("RAG snippets added.\n")

    # -----------------------------------------------------------------------
    # 5. Build ValuationInputs for each event date
    # -----------------------------------------------------------------------
    print("Event dates and market data:")
    base_inputs = {}

    for event_dt in EVENT_DATES:
        trade_dt = align_to_trading_day(prices, event_dt)
        px_val   = prices.loc[trade_dt]
        price    = float(px_val.iloc[0]) if hasattr(px_val, "iloc") else float(px_val)
        recent   = get_recent_prices(TICKER, trade_dt, n=5)

        # Diagnostic: confirm TTM metrics are now populated
        diag_agent   = ValuationAgent(llm=llm)
        diag_vin     = ValuationInputs(
            asof=trade_dt, ticker=TICKER, price=price,
            market_cap=None, shares_outstanding=None,
            quarter_table=quarter_table,
        )
        diag_metrics = diag_agent.compute_metrics(diag_vin)
        print(f"  Event {event_dt.date()} -> trading day {trade_dt.date()} | price=${price:.2f}")
        print(f"    Last 5 prices : {[f'{p:.2f}' for p in recent]}")
        print(f"    TTM metrics   : "
              f"P/S={diag_metrics.get('ps')}  "
              f"P/E={diag_metrics.get('pe')}  "
              f"P/FCF={diag_metrics.get('p_fcf')}")
        sent = SENTIMENT_BY_EVENT[event_dt]
        print(f"    Sentiment ({sent['quarter']}): "
              f"LM={sent['lm_score']:.4f}  "
              f"FinBERT={sent['finbert_score']:.4f}  "
              f"GPT={sent['gpt_score']:.2f}")

        base_inputs[trade_dt] = {
            "trade_dt":      trade_dt,
            "price":         price,
            "recent_prices": recent,
            "sentiment":     SENTIMENT_BY_EVENT.get(event_dt, {}),
        }
    print()

    # -----------------------------------------------------------------------
    # 6. Run both agent versions and backtest
    # -----------------------------------------------------------------------
    results = {}

    for use_sent, label in [(False, "WITHOUT Sentiment"), (True, "WITH Sentiment")]:
        print(f"Running agent {label}...")

        agent = ValuationAgent(
            llm=llm,
            config=ValuationAgentConfig(
                use_llm=True,
                use_sentiment=use_sent,
                # NVDA trades at high multiples — adjust thresholds for growth stock
                cheap_threshold_ps=10.0,
                expensive_threshold_ps=30.0,
                cheap_threshold_pe=25.0,
                expensive_threshold_pe=60.0,
                cheap_threshold_pfcf=20.0,
                expensive_threshold_pfcf=50.0,
            ),
            filing_rag=filing_rag,
        )

        # Build event dict: map each trading-day timestamp to a ValuationInputs
        event_dict = {}
        for trade_dt, info in base_inputs.items():
            vin = ValuationInputs(
                asof=info["trade_dt"],
                ticker=TICKER,
                price=info["price"],
                market_cap=None,
                shares_outstanding=None,
                quarter_table=quarter_table,
                sentiment_scores=info["sentiment"] if use_sent else {},
                recent_prices=info["recent_prices"],
            )
            event_dict[info["trade_dt"]] = vin

        bt = EventBacktester(
            prices=prices,
            cfg=BacktestConfig(
                initial_cash=100_000.0,
                transaction_cost_bps=10.0,
                trade_size_units=5.0,
                allow_short=True,
            ),
        )

        df_bt  = bt.run(ticker=TICKER, agent=agent,
                        valuation_inputs_by_event_date=event_dict)
        sharpe = EventBacktester.compute_sharpe(df_bt,
                                                risk_free_rate_annual=RISK_FREE_RATE)
        decs   = extract_decisions(df_bt)

        results[label] = {"df": df_bt, "sharpe": sharpe, "decisions": decs}
        EventBacktester.print_summary(label, df_bt, sharpe, decs)

    # -----------------------------------------------------------------------
    # 7. Side-by-side comparison
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  COMPARISON: With vs Without Sentiment")
    print("=" * 60)

    for label, res in results.items():
        df  = res["df"]
        ret = df["portfolio_value"].iloc[-1] / df["portfolio_value"].iloc[0] - 1
        print(f"\n  [{label}]")
        print(f"    Total return : {ret:+.2%}")
        print(f"    Sharpe Ratio : {res['sharpe']:.4f}" if not pd.isna(res['sharpe'])
              else f"    Sharpe Ratio : N/A (no trades executed)")
        print("    Signals      :")
        for d in res["decisions"]:
            print(f"      {d['date']}  {d['action'].upper():4s}  "
                  f"conf={d['confidence']:.2f}  -> {d['thesis'][:80]}...")

    no_sent_sharpe   = results["WITHOUT Sentiment"]["sharpe"]
    with_sent_sharpe = results["WITH Sentiment"]["sharpe"]

    print("\n" + "=" * 60)
    if pd.isna(no_sent_sharpe) or pd.isna(with_sent_sharpe):
        print("  Sharpe comparison not available (one or both runs had no trades).")
    else:
        diff = with_sent_sharpe - no_sent_sharpe
        if diff > 0:
            print(f"  Sentiment IMPROVED Sharpe by {diff:+.4f}")
        elif diff < 0:
            print(f"  Sentiment REDUCED  Sharpe by {diff:+.4f}")
        else:
            print("  Sentiment had no effect on Sharpe Ratio")
    print("=" * 60)
    print("\nDone.")


if __name__ == "__main__":
    main()
