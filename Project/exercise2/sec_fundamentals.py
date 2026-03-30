# sec_fundamentals.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import time

import pandas as pd
import requests


SEC_TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"


@dataclass
class SecConfig:
    user_agent: str
    sleep_seconds: float = 0.2
    timeout: int = 30


def _cik10(cik: str) -> str:
    return str(cik).zfill(10)


def _sec_get_json(url: str, cfg: SecConfig) -> Dict[str, Any]:
    headers = {
        "User-Agent": cfg.user_agent,
        "Accept-Encoding": "gzip, deflate",
    }
    r = requests.get(url, headers=headers, timeout=cfg.timeout)
    r.raise_for_status()
    time.sleep(cfg.sleep_seconds)
    return r.json()


def ticker_to_cik(ticker: str, cfg: SecConfig) -> str:
    data = _sec_get_json(SEC_TICKER_CIK_URL, cfg)
    t = ticker.upper()
    for _, row in data.items():
        if str(row.get("ticker", "")).upper() == t:
            return str(int(row["cik_str"]))
    raise ValueError(f"Ticker {ticker} not found in SEC mapping.")


def fetch_companyfacts(cik: str, cfg: SecConfig) -> Dict[str, Any]:
    url = SEC_COMPANYFACTS_URL.format(cik10=_cik10(cik))
    return _sec_get_json(url, cfg)


def _extract_series(companyfacts: Dict[str, Any], taxonomy: str, tag: str, unit: str) -> pd.DataFrame:
    facts = companyfacts.get("facts", {}).get(taxonomy, {}).get(tag, {})
    vals  = facts.get("units", {}).get(unit, [])

    rows: List[Dict[str, Any]] = []
    for x in vals:
        end = x.get("end")
        val = x.get("val")
        if end is None or val is None:
            continue
        rows.append({
            "end":   pd.Timestamp(end),
            "filed": pd.Timestamp(x["filed"]) if x.get("filed") else pd.NaT,
            "val":   float(val),
            "form":  x.get("form"),
            "fy":    x.get("fy"),
            "fp":    x.get("fp"),
            "frame": x.get("frame"),
        })

    if not rows:
        return pd.DataFrame(columns=["end", "filed", "val", "form", "fy", "fp", "frame"])
    return pd.DataFrame(rows).sort_values(["end", "filed"]).reset_index(drop=True)


def _extract_usd_series(companyfacts: Dict[str, Any], taxonomy: str, tag: str) -> pd.DataFrame:
    return _extract_series(companyfacts, taxonomy, tag, "USD")


def _extract_shares_series(companyfacts: Dict[str, Any], taxonomy: str, tag: str) -> pd.DataFrame:
    return _extract_series(companyfacts, taxonomy, tag, "shares")


def _latest_per_end(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "frame" in df.columns:
        q_frame = df["frame"].astype(str).str.match(r"CY\d{4}Q\d", na=False)
        if q_frame.any():
            single_q   = df[q_frame].sort_values(["end","filed"]).groupby("end", as_index=False).tail(1)
            non_q_ends = set(df["end"]) - set(single_q["end"])
            rest = df[df["end"].isin(non_q_ends)].sort_values(["end","filed"]).groupby("end", as_index=False).tail(1)
            return pd.concat([single_q, rest]).sort_values("end").reset_index(drop=True)
    return df.sort_values(["end", "filed"]).groupby("end", as_index=False).tail(1).reset_index(drop=True)


def _first_nonempty_tag(
    companyfacts: Dict[str, Any],
    taxonomy: str,
    tags: List[str],
    unit: str = "USD",
) -> pd.DataFrame:
    for tag in tags:
        if unit == "USD":
            df = _latest_per_end(_extract_usd_series(companyfacts, taxonomy, tag))
        elif unit == "shares":
            df = _latest_per_end(_extract_shares_series(companyfacts, taxonomy, tag))
        else:
            raise ValueError(f"Unsupported unit: {unit}")
        if not df.empty:
            print(f"Using tag: {tag}")
            return df
    return pd.DataFrame(columns=["end", "filed", "val", "form", "fy", "fp", "frame"])


def _quarter_only(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["end"] = pd.to_datetime(df["end"])

    import re

    if "frame" in df.columns:
        sq_mask  = df["frame"].astype(str).str.match(r"CY\d{4}Q\d$", na=False)
        sq_df    = df[sq_mask].copy()
        ann_mask = df["frame"].astype(str).str.match(r"CY\d{4}$", na=False)
        ann_df   = df[ann_mask].copy()

        q4_rows = []
        for _, arow in ann_df.iterrows():
            m = re.match(r"CY(\d{4})$", str(arow["frame"]))
            if not m:
                continue
            cy   = m.group(1)
            qtrs = sq_df[sq_df["frame"].astype(str).str.startswith(f"CY{cy}Q")]
            if len(qtrs) == 3:
                q4_row = arow.copy()
                q4_row["val"]   = float(arow["val"]) - float(qtrs["val"].sum())
                q4_row["fp"]    = "Q4"
                q4_row["frame"] = f"CY{cy}Q4"
                q4_rows.append(q4_row)

        if q4_rows:
            sq_df = pd.concat([sq_df, pd.DataFrame(q4_rows)], ignore_index=True)

        if not sq_df.empty:
            years_with_q123 = sq_df[sq_df["frame"].astype(str).str.match(r"CY\d{4}Q[123]$", na=False)]
            if not years_with_q123.empty:
                return sq_df.sort_values(["end", "filed"]).reset_index(drop=True)

    q_mask = df["form"].astype(str).str.contains("10-Q", case=False, na=False)
    q_df   = df[q_mask]
    if not q_df.empty:
        return q_df.sort_values(["end", "filed"]).reset_index(drop=True)
    return df.sort_values(["end", "filed"]).reset_index(drop=True)


def _rename(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["end", col, f"{col}_filed", f"{col}_fy", f"{col}_fp", f"{col}_form"])
    out = df[["end", "filed", "val", "fy", "fp", "form"]].copy()
    return out.rename(columns={
        "val":   col,
        "filed": f"{col}_filed",
        "fy":    f"{col}_fy",
        "fp":    f"{col}_fp",
        "form":  f"{col}_form",
    })


def ytd_to_quarterly(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if df.empty:
        return df
    work    = df.copy().sort_values(["fy", "end"]).reset_index(drop=True)
    out_col = f"{value_col}_quarter"
    work[out_col] = work[value_col]

    for fy, grp in work.groupby("fy", dropna=False):
        grp      = grp.sort_values("end")
        prev_ytd = None
        for idx, row in grp.iterrows():
            fp  = str(row.get("fp", "")).upper()
            val = row[value_col]
            if fp in ("Q1", "FY"):
                work.at[idx, out_col] = val
                prev_ytd = val
            elif fp in ("Q2", "Q3"):
                if prev_ytd is not None and pd.notna(val) and pd.notna(prev_ytd):
                    work.at[idx, out_col] = val - prev_ytd
                prev_ytd = val
            else:
                if prev_ytd is not None and pd.notna(val) and pd.notna(prev_ytd):
                    work.at[idx, out_col] = val - prev_ytd
                prev_ytd = val
    return work


def build_quarter_table(ticker: str, cfg: SecConfig) -> pd.DataFrame:
    """
    Build a per-quarter fundamental table.

    IMPORTANT – `filed` column:
      Populated only from revenue_filed / op_income_filed / net_income_filed.
      ocf_filed and capex_filed are intentionally excluded: their end-dates
      come from 10-Q YTD rows whose dates don't align with revenue quarter-end
      dates, which would corrupt the point-in-time TTM window calculation in
      ValuationAgent.compute_metrics.
    """
    cik   = ticker_to_cik(ticker, cfg)
    facts = fetch_companyfacts(cik, cfg)

    rev = _first_nonempty_tag(
        facts, "us-gaap",
        [
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
            "SalesRevenueNet",
            "SalesRevenueGoodsNet",
            "RevenuesNetOfInterestExpense",
        ],
        unit="USD",
    )

    op_inc  = _extract_usd_series(facts, "us-gaap", "OperatingIncomeLoss")
    net_inc = _extract_usd_series(facts, "us-gaap", "NetIncomeLoss")
    ocf     = _extract_usd_series(facts, "us-gaap", "NetCashProvidedByUsedInOperatingActivities")
    capex   = _extract_usd_series(facts, "us-gaap", "PaymentsToAcquireProductiveAssets")
    if capex.empty:
        capex = _extract_usd_series(facts, "us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment")

    shares = _first_nonempty_tag(
        facts, "dei",
        ["EntityCommonStockSharesOutstanding", "EntityCommonStockSharesIssued"],
        unit="shares",
    )

    rev     = _latest_per_end(_quarter_only(rev))
    op_inc  = _latest_per_end(_quarter_only(op_inc))
    net_inc = _latest_per_end(_quarter_only(net_inc))
    shares  = _latest_per_end(shares.sort_values(["end", "filed"]))

    ocf_10q   = _latest_per_end(ocf[ocf["form"].astype(str).str.contains("10-Q", na=False)].copy())
    capex_10q = _latest_per_end(capex[capex["form"].astype(str).str.contains("10-Q", na=False)].copy())

    def _ytd_to_q(df_10q, df_all, col):
        df10q = _latest_per_end(df_10q)
        tmp   = _rename(df10q, col)
        tmp2  = tmp[["end", f"{col}_fy", f"{col}_fp", col]].copy()
        tmp2  = tmp2.rename(columns={f"{col}_fy": "fy", f"{col}_fp": "fp"})
        tmp2  = ytd_to_quarterly(tmp2, col)
        tmp2[col] = tmp2[f"{col}_quarter"].where(tmp2[f"{col}_quarter"].notna(), tmp2[col])
        q_df  = tmp2[["end", col, "fy", "fp"]].copy()

        k_df = df_all[df_all["form"].astype(str).str.contains("10-K", na=False)].copy()
        k_df["end"] = pd.to_datetime(k_df["end"])
        k_df = k_df.sort_values(["end","filed"]).drop_duplicates("end", keep="last")
        q4_rows = []
        df_10q_d = df_10q.copy()
        df_10q_d["end"] = pd.to_datetime(df_10q_d["end"])
        for _, krow in k_df.iterrows():
            fy_end = pd.Timestamp(krow["end"])
            prev_q = df_10q_d[df_10q_d["end"] < fy_end].sort_values("end")
            if prev_q.empty:
                continue
            q3_ytd = float(prev_q.tail(1)["val"].iloc[0])
            q4_rows.append({"end": fy_end, col: float(krow["val"]) - q3_ytd, "fp": "Q4"})

        if q4_rows:
            q_df = pd.concat([q_df, pd.DataFrame(q4_rows)], ignore_index=True)
        return q_df[["end", col]].drop_duplicates("end").sort_values("end").reset_index(drop=True)

    ocf_q   = _ytd_to_q(ocf_10q,   ocf,   "ocf")
    capex_q = _ytd_to_q(capex_10q, capex, "capex")

    rev_r  = _rename(rev,     "revenue")
    op_r   = _rename(op_inc,  "op_income")
    net_r  = _rename(net_inc, "net_income")

    table = rev_r
    for piece in [op_r, net_r]:
        table = table.merge(piece, on="end", how="outer")

    # Shares via merge_asof: DEI end-dates don't align with quarter-end dates
    table = table.sort_values("end").reset_index(drop=True)
    shares_clean = (shares[["end", "val"]]
                    .rename(columns={"val": "shares_outstanding"})
                    .dropna().sort_values("end").reset_index(drop=True))
    table = pd.merge_asof(table, shares_clean, on="end", direction="backward")

    table = table.merge(ocf_q,   on="end", how="outer")
    table = table.merge(capex_q, on="end", how="outer")

    # filed: ONLY from income-statement sources (revenue / op_income / net_income)
    table["filed"] = pd.NaT
    for src in ["revenue_filed", "net_income_filed", "op_income_filed"]:
        if src in table.columns:
            table["filed"] = table["filed"].fillna(table[src])

    for fy_src in ["revenue_fy", "net_income_fy"]:
        if fy_src in table.columns:
            table["fy"] = table[fy_src]; break
    for fp_src in ["revenue_fp", "net_income_fp"]:
        if fp_src in table.columns:
            table["fp"] = table[fp_src]; break

    table["fcf"] = pd.NA
    mask = table["ocf"].notna() & table["capex"].notna()
    table.loc[mask, "fcf"] = table.loc[mask, "ocf"] - table.loc[mask, "capex"]

    keep_cols = [
        "end",
        "revenue_filed", "revenue",
        "op_income_filed", "op_income",
        "net_income_filed", "net_income",
        "ocf", "capex",
        "shares_outstanding",
        "filed", "fcf", "fy", "fp",
    ]
    keep_cols = [c for c in keep_cols if c in table.columns]
    table = table[keep_cols].copy()

    table["end"]   = pd.to_datetime(table["end"])
    table["filed"] = pd.to_datetime(table["filed"], errors="coerce")

    for col in ["revenue", "op_income", "net_income", "ocf", "capex", "fcf", "shares_outstanding"]:
        if col in table.columns:
            table[col] = pd.to_numeric(table[col], errors="coerce")

    table = table.dropna(subset=["revenue", "op_income", "net_income", "ocf", "capex", "fcf"], how="all")
    return table.sort_values("end").reset_index(drop=True)


def ttm_from_quarters(q: pd.DataFrame, asof_end: pd.Timestamp) -> Dict[str, Optional[float]]:
    """Sum the last 4 quarters with end <= asof_end."""
    empty = {
        "ttm_revenue": None, "ttm_op_income": None,
        "ttm_net_income": None, "ttm_fcf": None,
        "shares_outstanding": None,
    }
    if q.empty:
        return empty

    q2 = q.copy()
    q2["end"] = pd.to_datetime(q2["end"])
    q2 = q2[q2["end"] <= pd.Timestamp(asof_end)].sort_values("end")
    if q2.empty:
        return empty

    last4 = q2.tail(4).copy()

    def ttm_sum(col: str) -> Optional[float]:
        if col not in last4.columns:
            return None
        vals = last4[col].dropna()
        return float(vals.sum()) if len(vals) > 0 else None

    shares = None
    if "shares_outstanding" in q2.columns:
        s = q2["shares_outstanding"].dropna()
        if len(s) > 0:
            shares = float(s.iloc[-1])

    return {
        "ttm_revenue":        ttm_sum("revenue"),
        "ttm_op_income":      ttm_sum("op_income"),
        "ttm_net_income":     ttm_sum("net_income"),
        "ttm_fcf":            ttm_sum("fcf"),
        "shares_outstanding": shares,
    }


def build_filing_date_events(q: pd.DataFrame) -> List[pd.Timestamp]:
    if q.empty or "filed" not in q.columns:
        return []
    filed = (
        pd.to_datetime(q["filed"], errors="coerce")
        .dropna().drop_duplicates().sort_values().tolist()
    )
    return [pd.Timestamp(x) for x in filed]
