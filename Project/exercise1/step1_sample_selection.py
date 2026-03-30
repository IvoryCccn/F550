import pandas as pd
from xbbg import blp
from datetime import datetime

# ─────────────────────────────────────────
# 配置
# ─────────────────────────────────────────
SECTORS = {
    "IT":          "S5INFT Index",   # S&P500 IT sector index
    "Industrials": "S5INDU Index",   # S&P500 Industrials sector index
}

# 四个季度快照时间点（覆盖全年成分股变动）
SNAPSHOT_DATES = ["20240101", "20240401", "20240701", "20241001"]

OUTPUT_FILE = "step1_sample.xlsx"


# ─────────────────────────────────────────
# 1. 拉取历史成分股快照
# ─────────────────────────────────────────
def get_constituents(index_ticker: str, dates: list) -> pd.DataFrame:
    """
    对每个快照日期拉取成分股列表，合并去重，
    记录每家公司首次/末次出现的日期快照。
    """
    all_records = []

    for date in dates:
        print(f"  拉取 {index_ticker} @ {date} ...")
        try:
            df = blp.bds(
                index_ticker,
                "INDX_MWEIGHT_HIST",
                END_DT=date,
            )
            if df is None or df.empty:
                print(f"  ⚠️  {date} 无数据，跳过")
                continue

            # Bloomberg 返回列名不固定，统一处理
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]

            # 找到 ticker 列和 weight 列
            ticker_col = [c for c in df.columns if "ticker" in c or "member" in c][0]
            weight_col = [c for c in df.columns if "weight" in c][0]

            df = df[[ticker_col, weight_col]].copy()
            df.columns = ["ticker", "weight"]
            df["snapshot_date"] = date
            all_records.append(df)

        except Exception as e:
            print(f"  ERROR @ {date}: {e}")

    if not all_records:
        return pd.DataFrame()

    combined = pd.concat(all_records, ignore_index=True)
    return combined


# ─────────────────────────────────────────
# 2. 拉取公司元数据（市值、GICS sub-sector）
# ─────────────────────────────────────────
def get_metadata(tickers: list) -> pd.DataFrame:
    """
    批量拉取每只股票的：
    - 公司名称
    - GICS sub-industry（sub-sector 分类用）
    - 市值（用于后续加权聚合）
    """
    print("\n拉取元数据（名称 / GICS / 市值）...")

    fields = [
        "NAME",
        "GICS_SUB_INDUSTRY_NAME",
        "GICS_INDUSTRY_NAME",
        "CUR_MKT_CAP",        # 当前市值（USD mn）
    ]

    try:
        df = blp.bdp(tickers, fields)
        df.index.name = "ticker"
        df.reset_index(inplace=True)
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception as e:
        print(f"  ERROR 拉取元数据: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────
# 3. 识别成分股变动（进出记录）
# ─────────────────────────────────────────
def identify_changes(constituent_df: pd.DataFrame) -> pd.DataFrame:
    """
    对比相邻快照，找出新加入和被剔除的成分股。
    这是处理 survivorship bias 的关键记录。
    """
    dates_sorted = sorted(constituent_df["snapshot_date"].unique())
    changes = []

    for i in range(1, len(dates_sorted)):
        prev_date = dates_sorted[i - 1]
        curr_date = dates_sorted[i]

        prev_set = set(constituent_df[constituent_df["snapshot_date"] == prev_date]["ticker"])
        curr_set = set(constituent_df[constituent_df["snapshot_date"] == curr_date]["ticker"])

        added   = curr_set - prev_set
        removed = prev_set - curr_set

        for t in added:
            changes.append({"ticker": t, "change": "added",   "date": curr_date})
        for t in removed:
            changes.append({"ticker": t, "change": "removed", "date": curr_date})

    return pd.DataFrame(changes)


# ─────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────
def main():
    writer = pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl")
    summary_rows = []

    for sector_name, index_ticker in SECTORS.items():
        print(f"\n{'='*50}")
        print(f"处理板块: {sector_name}  ({index_ticker})")
        print(f"{'='*50}")

        # 1. 成分股快照
        constituent_df = get_constituents(index_ticker, SNAPSHOT_DATES)
        if constituent_df.empty:
            print(f"⚠️  {sector_name} 无成分股数据，跳过")
            continue

        # 2. 去重得到完整样本池
        all_tickers = constituent_df["ticker"].unique().tolist()
        print(f"\n样本池大小（含全年进出）: {len(all_tickers)} 家公司")

        # 3. 元数据
        meta_df = get_metadata(all_tickers)

        # 4. 成分股变动记录
        changes_df = identify_changes(constituent_df)
        print(f"成分股变动事件: {len(changes_df)} 条")

        # 5. 合并：每家公司的最终信息表
        # 取每个 ticker 在各快照中的平均权重（近似年度权重）
        avg_weight = (
            constituent_df.groupby("ticker")["weight"]
            .mean()
            .reset_index()
            .rename(columns={"weight": "avg_weight_2024"})
        )

        final_df = avg_weight.merge(meta_df, on="ticker", how="left")
        final_df["sector"] = sector_name

        # 标记是否全年稳定在成分股中
        snapshot_count = constituent_df.groupby("ticker")["snapshot_date"].count().reset_index()
        snapshot_count.columns = ["ticker", "snapshot_count"]
        final_df = final_df.merge(snapshot_count, on="ticker", how="left")
        final_df["stable_full_year"] = final_df["snapshot_count"] == len(SNAPSHOT_DATES)

        # 写入 Excel
        sheet_name = sector_name[:31]  # Excel sheet name 限制
        final_df.to_excel(writer, sheet_name=sheet_name, index=False)
        changes_df["sector"] = sector_name

        # 汇总统计
        summary_rows.append({
            "sector":           sector_name,
            "total_companies":  len(all_tickers),
            "stable_full_year": final_df["stable_full_year"].sum(),
            "added_during_2024":   len(changes_df[changes_df["change"] == "added"]),
            "removed_during_2024": len(changes_df[changes_df["change"] == "removed"]),
        })

        print(f"✓ {sector_name} 写入完成")

    # 汇总表
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_excel(writer, sheet_name="Summary", index=False)

    writer.close()
    print(f"\n✅ All transcripts collected → {OUTPUT_FILE}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()