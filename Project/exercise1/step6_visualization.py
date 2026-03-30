# step6_visualization.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import yfinance as yf
from pathlib import Path

# ── Config ────────────────────────────────────────────────
AGG_DIR    = r"C:\Users\cn_55\Desktop\上课文件\2 - F550\F550\Project\exercise1\aggregated"
OUTPUT_DIR = r"C:\Users\cn_55\Desktop\上课文件\2 - F550\F550\Project\exercise1\figures"

QUARTERS      = ["2023Q4", "2024Q1", "2024Q2", "2024Q3"]
QUARTER_DATES = ["2023-11-15", "2024-02-15", "2024-05-15", "2024-08-15"]
QUARTER_LABELS = ["Q4 2023", "Q1 2024", "Q2 2024", "Q3 2024"]

# Macro event annotations (for IT chart)
MACRO_EVENTS = {
    "2024Q1": "NVDA earnings\n+AI capex surge",
    "2024Q2": "AI regulation\ndiscussions",
    "2024Q3": "Fed pivot\nsignals",
}

PALETTE = {
    "IT":          "#2563EB",   # blue
    "Industrials": "#D97706",   # amber
    "mgmt":        "#059669",   # green
    "analyst":     "#DC2626",   # red
    "capwt":       "#1E40AF",   # dark blue
    "eqwt":        "#93C5FD",   # light blue
}

plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         150,
})

# ── Load data ─────────────────────────────────────────────
def load():
    sector   = pd.read_csv(f"{AGG_DIR}/sector_quarter_index.csv")
    subsector= pd.read_csv(f"{AGG_DIR}/subsector_heatmap.csv")
    div      = pd.read_csv(f"{AGG_DIR}/divergence_sector.csv")
    unc      = pd.read_csv(f"{AGG_DIR}/uncertainty_index.csv")
    company  = pd.read_csv(f"{AGG_DIR}/company_quarter_scores.csv")
    return sector, subsector, div, unc, company

# ── ETF price fetch ───────────────────────────────────────
def fetch_etf():
    print("Fetching ETF prices...")
    xlk = yf.download("XLK", start="2023-10-01", end="2024-11-01",
                       auto_adjust=True, progress=False)["Close"]
    xli = yf.download("XLI", start="2023-10-01", end="2024-11-01",
                       auto_adjust=True, progress=False)["Close"]
    # Flatten to Series if needed
    if isinstance(xlk, pd.DataFrame):
        xlk = xlk.iloc[:, 0]
    if isinstance(xli, pd.DataFrame):
        xli = xli.iloc[:, 0]
    return xlk, xli

def etf_at_quarters(etf_series):
    """Sample ETF price at each quarter midpoint."""
    # Handle both Series and DataFrame from yfinance
    if isinstance(etf_series, pd.DataFrame):
        etf_series = etf_series.iloc[:, 0]
    vals = []
    for d in QUARTER_DATES:
        idx = etf_series.index.searchsorted(pd.Timestamp(d))
        idx = min(idx, len(etf_series) - 1)
        vals.append(float(etf_series.iloc[idx]))
    return np.array(vals)

def normalise(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn) if mx > mn else arr * 0

# ═══════════════════════════════════════════════════════════
# FIGURE 1 — Sector sentiment time series (3 methods)
# ═══════════════════════════════════════════════════════════
def fig1_sector_timeseries(sector):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    methods = [
        ("lm_score_capwt",          "LM Lexicon score"),
        ("finbert_score_capwt",     "FinBERT score"),
        ("gpt_score_norm_capwt",    "GPT intensity (normalised)"),
    ]

    for ax, (col, title) in zip(axes, methods):
        for sector_name, color in [("IT", PALETTE["IT"]),
                                    ("Industrials", PALETTE["Industrials"])]:
            sub = sector[sector["sector"] == sector_name].sort_values("quarter")
            ax.plot(QUARTER_LABELS, sub[col].values,
                    marker="o", linewidth=2.2, color=color,
                    label=sector_name)
            ax.fill_between(QUARTER_LABELS, sub[col].values,
                            alpha=0.08, color=color)

        # Macro annotations on first panel only
        if col == "lm_score_capwt":
            it_vals = sector[sector["sector"]=="IT"].sort_values("quarter")[col].values
            for i, (q, label) in enumerate(MACRO_EVENTS.items()):
                qi = QUARTERS.index(q)
                ax.annotate(label, xy=(QUARTER_LABELS[qi], it_vals[qi]),
                            xytext=(0, 18), textcoords="offset points",
                            fontsize=7.5, ha="center", color="#6B7280",
                            arrowprops=dict(arrowstyle="-", color="#D1D5DB", lw=0.8))

        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15)
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Forward-looking sentiment: IT vs Industrials (2023Q4–2024Q3)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig1_sector_timeseries.png",
                bbox_inches="tight", dpi=150)
    plt.close()
    print("fig1 saved")

# ═══════════════════════════════════════════════════════════
# FIGURE 2 — Sentiment vs ETF price (dual-axis)
# ═══════════════════════════════════════════════════════════
def fig2_etf_overlay(sector, xlk, xli):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    pairs = [
        ("IT",          xlk, "XLK", PALETTE["IT"]),
        ("Industrials", xli, "XLI", PALETTE["Industrials"]),
    ]

    for ax, (sec_name, etf, etf_label, color) in zip(axes, pairs):
        sub = sector[sector["sector"] == sec_name].sort_values("quarter")
        sent = sub["finbert_score_capwt"].values
        prices = etf_at_quarters(etf)

        sent_n  = normalise(sent)
        price_n = normalise(prices)

        ax.plot(QUARTER_LABELS, sent_n,  marker="o", color=color,
                linewidth=2.2, label="FinBERT sentiment (normalised)")
        ax.fill_between(QUARTER_LABELS, sent_n, alpha=0.1, color=color)

        ax2 = ax.twinx()
        ax2.plot(QUARTER_LABELS, price_n, marker="s", color="#6B7280",
                 linewidth=2.2, linestyle="--", label=f"{etf_label} price (normalised)")
        ax2.set_ylabel(f"{etf_label} price (normalised)", fontsize=9, color="#6B7280")
        ax2.tick_params(axis="y", labelcolor="#6B7280")
        ax2.spines["top"].set_visible(False)

        ax.set_title(f"{sec_name} — Sentiment vs {etf_label}",
                     fontsize=11, fontweight="bold")
        ax.set_ylabel("Sentiment (normalised)", fontsize=9)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower left")

    fig.suptitle("Sentiment indicator vs sector ETF price (both normalised)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig2_etf_overlay.png",
                bbox_inches="tight", dpi=150)
    plt.close()
    print("fig2 saved")

# ═══════════════════════════════════════════════════════════
# FIGURE 3 — Cap-weighted vs Equal-weighted comparison
# ═══════════════════════════════════════════════════════════
def fig3_capwt_vs_eqwt(sector):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, sec_name in zip(axes, ["IT", "Industrials"]):
        sub = sector[sector["sector"] == sec_name].sort_values("quarter")
        ax.plot(QUARTER_LABELS, sub["finbert_score_capwt"].values,
                marker="o", color=PALETTE["capwt"], linewidth=2.2,
                label="Market-cap weighted")
        ax.plot(QUARTER_LABELS, sub["finbert_score_eqwt"].values,
                marker="s", color=PALETTE["eqwt"], linewidth=2.2,
                linestyle="--", label="Equal weighted")
        ax.fill_between(QUARTER_LABELS,
                        sub["finbert_score_capwt"].values,
                        sub["finbert_score_eqwt"].values,
                        alpha=0.12, color=PALETTE["capwt"])

        ax.set_title(f"{sec_name} — Cap-weighted vs Equal-weighted",
                     fontsize=11, fontweight="bold")
        ax.set_ylabel("FinBERT score", fontsize=9)
        ax.tick_params(axis="x", rotation=15)
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Aggregation method comparison: cap-weighted vs equal-weighted",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig3_capwt_vs_eqwt.png",
                bbox_inches="tight", dpi=150)
    plt.close()
    print("fig3 saved")

# ═══════════════════════════════════════════════════════════
# FIGURE 4 — Management vs Analyst divergence
# ═══════════════════════════════════════════════════════════
def fig4_divergence(div):
    fig, ax = plt.subplots(figsize=(9, 4.5))

    for sec_name, color, style in [
        ("IT",          PALETTE["IT"],          "-"),
        ("Industrials", PALETTE["Industrials"], "-"),
        ("Removal",     "#DC2626",           "--"),
    ]:
        sub = div[div["sector"] == sec_name].sort_values("quarter")
        if len(sub) == 0:
            continue
        ax.plot(QUARTER_LABELS, sub["div_finbert_score"].values,
                marker="o", color=color, linewidth=2.2,
                linestyle=style, label=sec_name)
        ax.fill_between(QUARTER_LABELS, sub["div_finbert_score"].values,
                        alpha=0.08, color=color)

    ax.axhline(0, color="#9CA3AF", linewidth=1, linestyle="--")
    ax.set_ylabel("Divergence (management − analyst sentiment)", fontsize=9)
    ax.set_title("Management vs analyst sentiment divergence\n"
                 "(positive = management more bullish than analysts)",
                 fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig4_divergence.png",
                bbox_inches="tight", dpi=150)
    plt.close()
    print("fig4 saved")

# ═══════════════════════════════════════════════════════════
# FIGURE 5 — Sub-sector heatmap
# ═══════════════════════════════════════════════════════════
def fig5_heatmap(subsector):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, sec_name in zip(axes, ["IT", "Industrials"]):
        sub = subsector[subsector["sector"] == sec_name].copy()

        # Keep top sub-sectors by coverage
        top_subs = (sub.groupby("sub_sector")["n_companies"]
                    .sum().nlargest(8).index.tolist())
        sub = sub[sub["sub_sector"].isin(top_subs)]

        pivot = sub.pivot(index="sub_sector", columns="quarter",
                          values="finbert_mean")[QUARTERS]
        pivot.columns = QUARTER_LABELS

        sns.heatmap(
            pivot, ax=ax, cmap="RdYlGn", center=0,
            vmin=-0.3, vmax=0.6,
            annot=True, fmt=".2f", annot_kws={"size": 9},
            linewidths=0.5, linecolor="#E5E7EB",
            cbar_kws={"shrink": 0.7, "label": "FinBERT score"},
        )
        ax.set_title(f"{sec_name} — Sub-sector sentiment heatmap",
                     fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=15)
        ax.tick_params(axis="y", rotation=0)

    fig.suptitle("Forward-looking sentiment by sub-sector and quarter",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig5_subsector_heatmap.png",
                bbox_inches="tight", dpi=150)
    plt.close()
    print("fig5 saved")

# ═══════════════════════════════════════════════════════════
# FIGURE 6 — Uncertainty index + sentiment overlay
# ═══════════════════════════════════════════════════════════
def fig6_uncertainty(sector, unc):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, sec_name in zip(axes, ["IT", "Industrials"]):
        sent = sector[sector["sector"]==sec_name].sort_values("quarter")
        u    = unc[unc["sector"]==sec_name].sort_values("quarter")

        color = PALETTE[sec_name]
        ax.plot(QUARTER_LABELS, sent["finbert_score_capwt"].values,
                marker="o", color=color, linewidth=2.2, label="Sentiment")

        ax2 = ax.twinx()
        ax2.plot(QUARTER_LABELS, u["uncertainty_index"].values,
                 marker="^", color="#7C3AED", linewidth=2,
                 linestyle="--", label="Uncertainty index")
        ax2.set_ylabel("Uncertainty index (LM)", fontsize=9, color="#7C3AED")
        ax2.tick_params(axis="y", labelcolor="#7C3AED")
        ax2.spines["top"].set_visible(False)

        ax.set_title(f"{sec_name} — Sentiment & uncertainty",
                     fontsize=11, fontweight="bold")
        ax.set_ylabel("FinBERT sentiment", fontsize=9)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1+lines2, labels1+labels2, fontsize=8)

    fig.suptitle("Sentiment vs uncertainty index by sector",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig6_uncertainty.png",
                bbox_inches="tight", dpi=150)
    plt.close()
    print("fig6 saved")

# ═══════════════════════════════════════════════════════════
# FIGURE 7 — Cross-method correlation heatmap
# ═══════════════════════════════════════════════════════════
def fig7_correlation(company):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, sec_name in zip(axes, ["IT", "Industrials"]):
        sub  = company[company["sector"] == sec_name]
        cols = ["lm_score", "finbert_score", "gpt_score_norm"]
        corr = sub[cols].corr()
        labels = ["LM", "FinBERT", "GPT"]
        corr.index   = labels
        corr.columns = labels

        sns.heatmap(corr, ax=ax, cmap="Blues", vmin=0, vmax=1,
                    annot=True, fmt=".3f", annot_kws={"size": 11},
                    linewidths=0.5, square=True,
                    cbar_kws={"shrink": 0.8})
        ax.set_title(f"{sec_name}", fontsize=11, fontweight="bold")

    fig.suptitle("Cross-method correlation: LM vs FinBERT vs GPT",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig7_correlation.png",
                bbox_inches="tight", dpi=150)
    plt.close()
    print("fig7 saved")

# ═══════════════════════════════════════════════════════════
# FIGURE 8 — GPT uniform vs intensity-weighted score
# ═══════════════════════════════════════════════════════════

def fig8_intensity_comparison(sector):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, sec_name in zip(axes, ["IT", "Industrials"]):
        sub   = sector[sector["sector"] == sec_name].sort_values("quarter")
        color = PALETTE[sec_name]

        ax.plot(QUARTER_LABELS, sub["gpt_score_norm_capwt"].values,
                marker="o", color=color, linewidth=2.2,
                label="GPT score (uniform weight)")
        ax.plot(QUARTER_LABELS, sub["gpt_score_intensity_wt_capwt"].values,
                marker="s", color=color, linewidth=2.2,
                linestyle="--", alpha=0.7,
                label="GPT score (intensity weighted)")
        ax.fill_between(QUARTER_LABELS,
                        sub["gpt_score_norm_capwt"].values,
                        sub["gpt_score_intensity_wt_capwt"].values,
                        alpha=0.12, color=color)

        ax.set_title(f"{sec_name} — GPT uniform vs intensity-weighted",
                     fontsize=11, fontweight="bold")
        ax.set_ylabel("GPT score (normalised)", fontsize=9)
        ax.tick_params(axis="x", rotation=15)
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Effect of speaker tone intensity weighting on GPT sentiment score",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig8_intensity_comparison.png",
                bbox_inches="tight", dpi=150)
    plt.close()
    print("fig8 saved")

# ── Main ──────────────────────────────────────────────────
def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    sector, subsector, div, unc, company = load()
    xlk, xli = fetch_etf()

    # Normalise GPT for company-level correlation
    if "gpt_score_norm" not in company.columns:
        company["gpt_score_norm"] = (company["gpt_score"] - 5) / 5

    fig1_sector_timeseries(sector)
    fig2_etf_overlay(sector, xlk, xli)
    fig3_capwt_vs_eqwt(sector)
    fig4_divergence(div)
    fig5_heatmap(subsector)
    fig6_uncertainty(sector, unc)
    fig7_correlation(company)
    fig8_intensity_comparison(sector)

    print(f"\n✅ All 8 figures saved → {OUTPUT_DIR}")

if __name__ == "__main__":
    main()