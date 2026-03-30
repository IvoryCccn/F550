# step5_aggregation.py
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────
SCORED_CSV  = r"C:\Users\cn_55\Desktop\上课文件\2 - F550\F550\Project\exercise1\scored\scored_all.csv"
OUTPUT_DIR  = r"C:\Users\cn_55\Desktop\上课文件\2 - F550\F550\Project\exercise1\aggregated"

# Quarters to process
QUARTERS = ["2023Q4", "2024Q1", "2024Q2", "2024Q3"]

# ── Market cap fetch ──────────────────────────────────────
def fetch_market_caps(tickers: list) -> dict:
    """Fetch market cap for each ticker via yfinance."""
    mcap = {}
    print("Fetching market caps...")
    for tkr in tqdm(tickers, desc="Market cap"):
        try:
            info = yf.Ticker(tkr).info
            cap  = info.get("marketCap") or info.get("enterpriseValue")
            mcap[tkr] = cap if cap and cap > 0 else np.nan
        except Exception:
            mcap[tkr] = np.nan
    return mcap

# ── Normalise GPT score to [-1, +1] ──────────────────────
def normalise_gpt(s):
    """Map 0-10 → -1 to +1."""
    return (s - 5) / 5

# ── Weighted aggregation per group ───────────────────────
def weighted_mean(values, weights):
    """Compute weight-average, fallback to equal-weight if all NaN."""
    mask = ~np.isnan(values) & ~np.isnan(weights) & (weights > 0)
    if mask.sum() == 0:
        return np.nanmean(values)
    return np.average(values[mask], weights=weights[mask])

# ── Main ──────────────────────────────────────────────────
def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load scored sentences
    df = pd.read_csv(SCORED_CSV, encoding="utf-8-sig")
    print(f"Loaded {len(df)} scored sentences")

    # Normalise GPT to same scale as LM and FinBERT
    df["gpt_score_norm"] = normalise_gpt(df["gpt_score"])

    # Keep only management sentences for primary index
    # (analyst sentences used separately for divergence)
    mgmt = df[df["speaker_type"] == "management"].copy()
    mgmt["gpt_score_intensity_wt"] = (
        mgmt["gpt_score_norm"] * mgmt["gpt_intensity"].fillna(2)
    )
    analyst = df[df["speaker_type"] == "analyst"].copy()
    analyst["gpt_score_intensity_wt"] = (
        analyst["gpt_score_norm"] * analyst["gpt_intensity"].fillna(2)
    )

    # Fetch market caps
    all_tickers = df["ticker"].dropna().unique().tolist()
    mcap_map    = fetch_market_caps(all_tickers)
    df["market_cap"]     = df["ticker"].map(mcap_map)
    mgmt["market_cap"]   = mgmt["ticker"].map(mcap_map)
    analyst["market_cap"] = analyst["ticker"].map(mcap_map)

    # ── 1. Company-quarter scores ──────────────────────────────
    # lm/finbert/gpt_norm: equal weight across sentences
    # gpt_score_intensity_wt: intensity-weighted GPT score
    score_cols = ["lm_score", "finbert_score", "gpt_score_norm", "gpt_score_intensity_wt", "lm_uncertainty_score"]
    company_q  = (
        mgmt.groupby(["ticker", "quarter", "sector", "sub_sector", "is_removal"])[score_cols]
        .mean()
        .reset_index()
    )
    company_q["market_cap"] = company_q["ticker"].map(mcap_map)

    company_q.to_csv(output_dir / "company_quarter_scores.csv", index=False)
    print(f"Company-quarter scores: {len(company_q)} rows")

    # ── 2. Sector-quarter index (market-cap weighted) ──────
    sector_records = []

    for sector in ["IT", "Industrials"]:
        for quarter in QUARTERS:
            subset = company_q[
                (company_q["sector"] == sector) &
                (company_q["quarter"] == quarter) &
                (~company_q["is_removal"])   # exclude removal companies
            ].copy()

            if len(subset) == 0:
                continue

            weights = subset["market_cap"].values.astype(float)
            # Fallback: equal weight where market cap missing
            weights = np.where(np.isnan(weights), np.nanmean(weights), weights)

            row = {"sector": sector, "quarter": quarter,
                   "n_companies": len(subset)}

            for col in score_cols:
                vals = subset[col].values.astype(float)
                row[f"{col}_capwt"]  = weighted_mean(vals, weights)
                row[f"{col}_eqwt"]   = np.nanmean(vals)

            sector_records.append(row)

    sector_df = pd.DataFrame(sector_records)
    sector_df.to_csv(output_dir / "sector_quarter_index.csv", index=False)
    print(f"\nSector-quarter index:")
    print(sector_df[["sector","quarter","lm_score_capwt",
                      "finbert_score_capwt","gpt_score_norm_capwt"]].to_string(index=False))

    # ── 3. Sub-sector heatmap data ─────────────────────────
    subsector_records = []

    for sector in ["IT", "Industrials"]:
        sub_q = (
            company_q[
                (company_q["sector"] == sector) &
                (~company_q["is_removal"])
            ]
            .groupby(["sub_sector", "quarter"])
            .agg(
                lm_mean    =("lm_score",           "mean"),
                finbert_mean=("finbert_score",      "mean"),
                gpt_mean   =("gpt_score_norm",      "mean"),
                n_companies=("ticker",              "count"),
            )
            .reset_index()
        )
        sub_q["sector"] = sector
        subsector_records.append(sub_q)

    subsector_df = pd.concat(subsector_records, ignore_index=True)
    subsector_df.to_csv(output_dir / "subsector_heatmap.csv", index=False)
    print(f"\nSub-sector heatmap: {len(subsector_df)} rows")

    # ── 4. Management vs Analyst divergence ───────────────
    # Compute per-ticker-quarter score for each group, then divergence
    mgmt_q = (
        mgmt.groupby(["ticker","quarter","sector"])[score_cols]
        .mean().reset_index()
        .rename(columns={c: f"mgmt_{c}" for c in score_cols})
    )
    analyst_q = (
        analyst.groupby(["ticker","quarter","sector"])[score_cols]
        .mean().reset_index()
        .rename(columns={c: f"analyst_{c}" for c in score_cols})
    )

    divergence = mgmt_q.merge(analyst_q, on=["ticker","quarter","sector"], how="inner")

    # Divergence = management sentiment - analyst sentiment
    # Positive = mgmt more bullish than analysts (potential warning signal)
    for col in ["lm_score","finbert_score","gpt_score_norm"]:
        divergence[f"div_{col}"] = (
            divergence[f"mgmt_{col}"] - divergence[f"analyst_{col}"]
        )

    # Sector-level divergence per quarter
    div_sector = (
        divergence.groupby(["sector","quarter"])
        [[f"div_{c}" for c in ["lm_score","finbert_score","gpt_score_norm"]]]
        .mean().reset_index()
    )
    div_sector.to_csv(output_dir / "divergence_sector.csv", index=False)
    divergence.to_csv(output_dir / "divergence_company.csv", index=False)

    print(f"\nManagement vs Analyst divergence (sector level):")
    print(div_sector.to_string(index=False))

    # ── 5. Uncertainty index ───────────────────────────────
    unc_sector = (
        mgmt[~mgmt["is_removal"]]
        .groupby(["sector","quarter"])["lm_uncertainty_score"]
        .mean().reset_index()
        .rename(columns={"lm_uncertainty_score": "uncertainty_index"})
    )
    unc_sector.to_csv(output_dir / "uncertainty_index.csv", index=False)

    print(f"\nUncertainty index:")
    print(unc_sector.to_string(index=False))

    print(f"\n✅ Step 5 complete → {output_dir}")

if __name__ == "__main__":
    main()