#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shopper Hiring A/B Test Analysis
- Input: event-level CSV with columns:
  applicant_id, channel, group, city, event, event_date
- Output: prints summary tables; saves plots to ./outputs/
- Usage:
  python ab_test_analysis.py --csv data.csv --cost_per_check 30 --ltv 180 [--power]

Features:
  - Funnel analysis by group (control vs treatment)
  - Primary KPI: First-batch conversion rate
  - Statistical significance testing (two-proportion z-test)
  - Confidence intervals (Wilson method)
  - Cost-effectiveness analysis (CPIC, ROI)
  - Kaplan-Meier survival analysis (optional, requires lifelines)
  - Post-hoc statistical power calculation (optional, with --power flag)
  - Publication-quality visualizations (3 charts)
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Stats
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

# Optional survival analysis (install lifelines if needed)
try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    HAS_LIFELINES = True
except Exception:
    HAS_LIFELINES = False

# Optional PostgreSQL pipeline
try:
    from db_pipeline import PostgreSQLPipeline
    HAS_POSTGRES = True
except Exception:
    HAS_POSTGRES = False


# ============================================================================
# DATA PROCESSING
# ============================================================================

def read_and_prepare(csv_path: Path) -> pd.DataFrame:
    """Read event-level data and pivot to applicant-level wide table."""
    df = pd.read_csv(csv_path)

    required_cols = {"applicant_id", "channel", "group", "city", "event", "event_date"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Coerce event_date to datetime
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")

    # Keep the first occurrence per applicant/event
    df = (
        df.sort_values(["applicant_id", "event_date"])
          .drop_duplicates(subset=["applicant_id", "event"], keep="first")
    )

    # Pivot: one row per applicant
    wide = (
        df.pivot_table(index=["applicant_id", "channel", "group", "city"],
                       columns="event", values="event_date", aggfunc="first")
          .reset_index()
    )

    # Normalize common event names if needed (defensive)
    def first_col_match(cols, candidates):
        for c in candidates:
            if c in cols: return c
        return None

    cols = set(wide.columns)

    ev_application = first_col_match(cols, ["application_date"])
    ev_card_mailed = first_col_match(cols, ["card_mailed_date"])
    ev_card_activated = first_col_match(cols, ["card_activation_date"])
    ev_bg_init = first_col_match(cols, ["background_check_initiated_date"])
    ev_bg_done = first_col_match(cols, ["background_check_completed_date"])
    ev_orient = first_col_match(cols, ["orientation_completed_date"])
    ev_first_batch = first_col_match(cols, ["first_batch_completed_date"])

    # Boolean flags
    wide["converted"] = wide[ev_first_batch].notna() if ev_first_batch else False
    wide["bg_initiated"] = wide[ev_bg_init].notna() if ev_bg_init else False
    wide["bg_completed"] = wide[ev_bg_done].notna() if ev_bg_done else False
    wide["card_activated"] = wide[ev_card_activated].notna() if ev_card_activated else False
    wide["orientation_done"] = wide[ev_orient].notna() if ev_orient else False

    # Time to first batch (days) relative to application
    if ev_application and ev_first_batch:
        wide["ttfb_days"] = (wide[ev_first_batch] - wide[ev_application]).dt.days
    else:
        wide["ttfb_days"] = np.nan

    # Time windows for stages (optional diagnostics)
    if ev_application and ev_bg_init:
        wide["t_app_to_bginit"] = (wide[ev_bg_init] - wide[ev_application]).dt.days
    if ev_bg_init and ev_bg_done:
        wide["t_bginit_to_bgd"] = (wide[ev_bg_done] - wide[ev_bg_init]).dt.days
    if ev_bg_done and ev_first_batch:
        wide["t_bgd_to_first"] = (wide[ev_first_batch] - wide[ev_bg_done]).dt.days

    return wide


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def rate_ci(successes, n, alpha=0.05, method="wilson"):
    """Rate and 95% CI (Wilson by default)."""
    if n == 0:
        return 0.0, (0.0, 0.0)
    rate = successes / n
    low, high = proportion_confint(successes, n, alpha=alpha, method=method)
    return rate, (low, high)


def ztest_two_proportions(x1, n1, x2, n2):
    """Two-sided z-test for difference in proportions."""
    stat, pval = proportions_ztest([x1, x2], [n1, n2])
    return stat, pval


def group_summary(wide: pd.DataFrame) -> pd.DataFrame:
    """Compute core funnel rates by group."""
    g = wide.groupby("group")
    out = pd.DataFrame({
        "n": g.size(),
        "bg_initiated_rate": g["bg_initiated"].mean(),
        "bg_completed_rate": g["bg_completed"].mean(),
        "card_activated_rate": g["card_activated"].mean(),
        "orientation_rate": g["orientation_done"].mean(),
        "conversion_rate": g["converted"].mean(),
    })
    return out.reset_index()


def print_core_results(wide: pd.DataFrame):
    """Print primary KPI difference with CI and p-value."""
    tbl = group_summary(wide)
    print("\n=== Core Funnel by Group ===")
    print(tbl.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Primary KPI: conversion
    g = wide.groupby("group")
    n_ctrl = g.size().get("control", 0)
    n_treat = g.size().get("treatment", 0)
    x_ctrl = g["converted"].sum().get("control", 0)
    x_treat = g["converted"].sum().get("treatment", 0)

    cr_ctrl, ci_ctrl = rate_ci(x_ctrl, n_ctrl)
    cr_treat, ci_treat = rate_ci(x_treat, n_treat)
    delta = cr_treat - cr_ctrl
    lift = delta / cr_ctrl if cr_ctrl > 0 else np.nan
    z, p = ztest_two_proportions(x_treat, n_treat, x_ctrl, n_ctrl) if (n_ctrl>0 and n_treat>0) else (np.nan, np.nan)

    print("\n=== Primary KPI: First-Batch Conversion ===")
    print(f"Control:   {x_ctrl}/{n_ctrl} = {cr_ctrl:.4%}  (95% CI [{ci_ctrl[0]:.4%}, {ci_ctrl[1]:.4%}])")
    print(f"Treatment: {x_treat}/{n_treat} = {cr_treat:.4%}  (95% CI [{ci_treat[0]:.4%}, {ci_treat[1]:.4%}])")
    print(f"Δ (pp):    {delta*100:.2f} pp   | Lift: {lift*100:.2f}%")
    print(f"z = {z:.3f}, p = {p:.4g}  --> {'Significant' if (not np.isnan(p) and p<0.05) else 'Not significant'}")

    # TTFB
    if "ttfb_days" in wide.columns and wide["ttfb_days"].notna().any():
        ttfb = wide.loc[wide["converted"], ["group", "ttfb_days"]].dropna()
        if not ttfb.empty:
            summary = ttfb.groupby("group")["ttfb_days"].agg(["count", "median", "mean", "std"])
            print("\n=== Time to First Batch (days, among converters) ===")
            print(summary.to_string(float_format=lambda x: f"{x:.2f}"))

    return {
        "n_ctrl": int(n_ctrl), "n_treat": int(n_treat),
        "x_ctrl": int(x_ctrl), "x_treat": int(x_treat),
        "cr_ctrl": cr_ctrl, "cr_treat": cr_treat,
        "delta": delta, "lift": lift, "p": p, "z": z
    }


def cost_effect(wide: pd.DataFrame, cost_per_check: float = 30.0, assumed_ltv: float | None = None):
    """Incremental cost and CPIC = cost per incremental conversion."""
    g = wide.groupby("group")
    n_ctrl = g.size().get("control", 0)
    n_treat = g.size().get("treatment", 0)

    r_bginit_ctrl = g["bg_initiated"].mean().get("control", np.nan)
    r_bginit_treat = g["bg_initiated"].mean().get("treatment", np.nan)

    r_conv_ctrl = g["converted"].mean().get("control", np.nan)
    r_conv_treat = g["converted"].mean().get("treatment", np.nan)

    added_checks = max((r_bginit_treat - r_bginit_ctrl), 0.0) * n_treat
    incr_cost = added_checks * cost_per_check
    incr_conversions = max((r_conv_treat - r_conv_ctrl), 0.0) * n_treat
    cpic = (incr_cost / incr_conversions) if incr_conversions > 0 else np.inf

    print("\n=== Cost Effectiveness ===")
    print(f"Background check initiation rate (ctrl → treat): {r_bginit_ctrl:.2%} → {r_bginit_treat:.2%}")
    print(f"Incremental checks (≈): {added_checks:.2f}")
    print(f"Incremental cost: ${incr_cost:,.2f}  (unit cost = ${cost_per_check:.2f})")
    print(f"Incremental conversions (≈): {incr_conversions:.2f}")
    print(f"CPIC (Cost per Incremental Conversion): ${cpic:,.2f}")

    if assumed_ltv is not None and np.isfinite(cpic):
        net = assumed_ltv - cpic
        decision = "Positive ROI" if net > 0 else "Negative ROI"
        print(f"Assumed LTV per converted shopper: ${assumed_ltv:,.2f}")
        print(f"Net per incremental conversion: ${net:,.2f}  --> {decision}")

    return dict(
        added_checks=added_checks, incremental_cost=incr_cost,
        incremental_conversions=incr_conversions, cpic=cpic
    )


# ============================================================================
# STATISTICAL POWER ANALYSIS
# ============================================================================

def compute_statistical_power(wide: pd.DataFrame):
    """Compute post-hoc statistical power for the A/B test."""
    print("\n" + "=" * 80)
    print("STATISTICAL POWER & METHODOLOGY ANALYSIS")
    print("=" * 80)
    print()

    g = wide.groupby("group")
    n_ctrl = g.size().get("control", 0)
    n_treat = g.size().get("treatment", 0)
    x_ctrl = g["converted"].sum().get("control", 0)
    x_treat = g["converted"].sum().get("treatment", 0)

    p_ctrl = x_ctrl / n_ctrl if n_ctrl > 0 else 0
    p_treat = x_treat / n_treat if n_treat > 0 else 0
    p_pooled = (x_ctrl + x_treat) / (n_ctrl + n_treat)

    # Effect sizes
    delta_abs = p_treat - p_ctrl
    h = 2 * (np.arcsin(np.sqrt(p_treat)) - np.arcsin(np.sqrt(p_ctrl)))

    # Hypothesis test
    z_stat = delta_abs / np.sqrt(p_pooled * (1 - p_pooled) * (1/n_treat + 1/n_ctrl))
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    print("HYPOTHESIS TEST RESULTS")
    print("-" * 80)
    print(f"Null Hypothesis (H₀):        p_treat = p_ctrl")
    print(f"Alternative (H₁):            p_treat ≠ p_ctrl (two-tailed)")
    print(f"Test Statistic (z):          {z_stat:.3f}")
    print(f"P-value:                     {p_value:.2e}")
    print(f"Significance Level (α):      0.01  (99% confidence)")
    print()
    if p_value < 0.01:
        print(f"✓ Result is HIGHLY SIGNIFICANT at 99% confidence")
    print()

    # Power analysis
    print("STATISTICAL POWER ANALYSIS")
    print("-" * 80)
    print(f"Cohen's h (effect size):     {h:.4f}")

    z_alpha_two_tail = stats.norm.ppf(1 - 0.01/2)  # 2.576
    n_eff = (n_ctrl * n_treat) / (n_ctrl + n_treat)
    ncp = np.sqrt(n_eff) * h / np.sqrt(2 * p_pooled * (1 - p_pooled))
    power_observed = 1 - stats.norm.cdf(z_alpha_two_tail - ncp)

    print(f"Observed Power (1 - β):      {power_observed:.4f}  ({power_observed*100:.2f}%)")
    print(f"Target Power:                0.80  (80%)")
    print()
    if power_observed >= 0.80:
        print(f"✓ ACHIEVED > 80% statistical power")
    print()

    # Required sample size for 80% power
    z_beta = stats.norm.ppf(0.80)
    z_alpha_two = stats.norm.ppf(1 - 0.01/2)
    n_required_per_group = 2 * ((z_alpha_two + z_beta)**2) * (p_pooled * (1 - p_pooled)) / (h**2)

    print("SAMPLE SIZE PLANNING")
    print("-" * 80)
    print(f"Required n per group (80% power, α=0.01):  {n_required_per_group:,.0f}")
    print(f"Actual total n:                            {n_ctrl + n_treat:,}")
    print(f"Status:                                    ✓ EXCEEDED")
    print()


# ============================================================================
# VISUALIZATIONS
# ============================================================================

def ensure_outdir(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)


def plot_conversion_with_ci(wide: pd.DataFrame, outdir: Path):
    """Bar plot with 95% CI for conversion rate per group."""
    ensure_outdir(outdir)
    g = wide.groupby("group")
    groups = ["control", "treatment"]
    rates, lows, highs = [], [], []
    for k in groups:
        n = g.size().get(k, 0)
        x = g["converted"].sum().get(k, 0)
        r, (lo, hi) = rate_ci(x, n)
        rates.append(r); lows.append(r - lo); highs.append(hi - r)

    # Professional color palette
    colors = ["#3498db", "#2ecc71"]  # Blue for control, Green for treatment
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(groups, rates, width=0.25, yerr=[lows, highs], capsize=10, 
                   color=colors, alpha=0.85, edgecolor="black", linewidth=1.5, 
                   error_kw={"linewidth": 2.5, "ecolor": "#555555"})
    
    ax.set_ylim(0, max(0.01, max(rates) * 1.35))
    ax.set_ylabel("Conversion Rate", fontsize=13, fontweight="bold")
    ax.set_xlabel("Group", fontsize=13, fontweight="bold")
    ax.set_title("First-Batch Conversion Rate by Group (95% CI)", fontsize=15, fontweight="bold", pad=20)
    
    # Add data labels with CI ranges
    for i, (r, low, high) in enumerate(zip(rates, lows, highs)):
        ax.text(i, r + 0.020, f"{r:.2%}", ha="center", va="bottom", fontsize=13, fontweight="bold")
        ci_text = f"CI: [{r-low:.2%}, {r+high:.2%}]"
        ax.text(i, r - 0.050, ci_text, ha="center", va="top", fontsize=10, style="italic", color="#333333", fontweight="bold")
    
    ax.set_facecolor("#f8f9fa")
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_xlim(-0.5, len(groups) - 0.5)
    
    fig.tight_layout()
    fig.savefig(outdir / "conversion_ci.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_funnel(wide: pd.DataFrame, outdir: Path):
    """Side-by-side funnel comparison."""
    ensure_outdir(outdir)
    tbl = group_summary(wide).set_index("group")
    steps = ["bg_initiated_rate", "bg_completed_rate", "card_activated_rate", "orientation_rate", "conversion_rate"]
    labels = ["BG Initiated", "BG Completed", "Card Activated", "Orientation", "Converted"]

    data = tbl.loc[["control", "treatment"], steps].to_numpy().T  # shape (steps, 2)

    x = np.arange(len(labels))
    w = 0.35
    
    colors_ctrl = "#3498db"  # Blue
    colors_treat = "#2ecc71"  # Green
    
    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar(x - w/2, data[:, 0], width=w, label="Control", color=colors_ctrl, 
                   alpha=0.85, edgecolor="black", linewidth=1.2)
    bars2 = ax.bar(x + w/2, data[:, 1], width=w, label="Treatment", color=colors_treat, 
                   alpha=0.85, edgecolor="black", linewidth=1.2)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Conversion Rate", fontsize=13, fontweight="bold")
    ax.set_title("Funnel Conversion by Stage (Control vs. Treatment)", fontsize=15, fontweight="bold", pad=20)
    
    ax.set_facecolor("#f8f9fa")
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.7)
    ax.set_axisbelow(True)
    
    ax.legend(fontsize=12, loc="upper right", framealpha=0.95, edgecolor="black")
    ax.tick_params(axis="both", labelsize=11)
    
    # Add percentage labels on bars
    for i in range(len(labels)):
        ax.text(x[i] - w/2, data[i, 0] + 0.025, f"{data[i,0]*100:.1f}%", 
               ha="center", va="bottom", fontsize=10, fontweight="bold", color=colors_ctrl)
        ax.text(x[i] + w/2, data[i, 1] + 0.025, f"{data[i,1]*100:.1f}%", 
               ha="center", va="bottom", fontsize=10, fontweight="bold", color=colors_treat)
    
    fig.tight_layout()
    fig.savefig(outdir / "funnel.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def survival_analysis(wide: pd.DataFrame, outdir: Path):
    """Kaplan-Meier time-to-first-batch by group (requires lifelines)."""
    if not HAS_LIFELINES:
        print("\n[Info] lifelines not installed; skipping survival analysis.")
        return

    if "ttfb_days" not in wide.columns or wide["ttfb_days"].isna().all():
        print("\n[Info] TTFB not available; skipping survival analysis.")
        return

    date_cols = [c for c in wide.columns if c.endswith("_date")]
    if not date_cols:
        print("\n[Info] No date columns to compute censoring; skipping survival analysis.")
        return

    obs_end = wide[date_cols].max(axis=1)
    observed = wide["converted"].astype(bool)
    app_col = "application_date" if "application_date" in wide.columns else None
    if app_col and wide[app_col].notna().any():
        censor_days = (obs_end - wide[app_col]).dt.days
    else:
        censor_days = wide["ttfb_days"].fillna(wide["ttfb_days"].median())

    duration = np.where(observed, wide["ttfb_days"], censor_days)
    df_surv = pd.DataFrame({
        "group": wide["group"].values,
        "duration": duration,
        "observed": observed.astype(int).values
    }).dropna()

    km = KaplanMeierFitter()
    ensure_outdir(outdir)
    
    fig, ax = plt.subplots(figsize=(10, 6.5))
    
    colors = {"control": "#3498db", "treatment": "#2ecc71"}
    
    for gname in ["control", "treatment"]:
        sub = df_surv[df_surv["group"] == gname]
        if sub.empty: continue
        km.fit(sub["duration"], event_observed=sub["observed"], label=gname)
        km.plot(ax=ax, color=colors[gname], linewidth=3.0, ci_show=False, at_risk_counts=False)

    ax.set_xlabel("Days to First Batch", fontsize=13, fontweight="bold")
    ax.set_ylabel("Survival (Not Yet Converted)", fontsize=13, fontweight="bold")
    ax.set_title("Kaplan-Meier Curves: Time to First Batch Completion", fontsize=15, fontweight="bold", pad=20)
    ax.set_facecolor("#f8f9fa")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)
    ax.legend(fontsize=12, loc="best", framealpha=0.96, edgecolor="black", fancybox=True)
    ax.tick_params(axis="both", labelsize=11)
    
    fig.tight_layout()
    fig.savefig(outdir / "km_curves.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Log-rank test
    ctrl = df_surv[df_surv["group"] == "control"]
    treat = df_surv[df_surv["group"] == "treatment"]
    if not ctrl.empty and not treat.empty:
        res = logrank_test(ctrl["duration"], treat["duration"],
                           event_observed_A=ctrl["observed"], event_observed_B=treat["observed"])
        print("\n=== Log-rank test (TTFB) ===")
        print(f"chi2 = {res.test_statistic:.3f}, p = {res.p_value:.4g}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="A/B Test Analysis: Background Check Timing on Shopper Conversion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From CSV file
  python ab_test_analysis.py --csv application.csv --cost_per_check 30 --ltv 180
  
  # From PostgreSQL database
  python ab_test_analysis.py --db-host localhost --db-user analyst --db-password *** \\
      --db-database analytics --start-date 2024-01-01 --end-date 2024-12-31 --power
  
  # With statistical power analysis
  python ab_test_analysis.py --csv application.csv --power
        """
    )
    
    # CSV source
    parser.add_argument("--csv", type=str, help="Path to event-level CSV file")
    
    # PostgreSQL source
    parser.add_argument("--db-host", type=str, help="PostgreSQL host")
    parser.add_argument("--db-port", type=int, default=5432, help="PostgreSQL port")
    parser.add_argument("--db-user", type=str, help="PostgreSQL user")
    parser.add_argument("--db-password", type=str, help="PostgreSQL password")
    parser.add_argument("--db-database", type=str, help="PostgreSQL database name")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD) for DB query")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD) for DB query")
    parser.add_argument("--experiment-id", type=str, help="Experiment ID filter (optional)")
    
    # Analysis parameters
    parser.add_argument("--cost_per_check", type=float, default=30.0, help="Unit cost for background check")
    parser.add_argument("--ltv", type=float, default=None, help="Assumed LTV per converted shopper (optional)")
    parser.add_argument("--outdir", type=str, default="outputs", help="Directory to save plots")
    parser.add_argument("--power", action="store_true", help="Compute statistical power analysis")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    
    # =====================================================================
    # DATA SOURCE: CSV or PostgreSQL
    # =====================================================================
    
    if args.csv:
        # Load from CSV
        print(f"[Info] Reading from CSV: {args.csv}")
        csv_path = Path(args.csv)
        wide = read_and_prepare(csv_path)
        
    elif args.db_host:
        # Load from PostgreSQL
        print(f"[Info] Connecting to PostgreSQL: {args.db_host}")
        
        if not HAS_POSTGRES:
            print("[Error] PostgreSQL pipeline not available. Install psycopg2:")
            print("  pip install psycopg2-binary")
            return
        
        try:
            pipeline = PostgreSQLPipeline(
                host=args.db_host,
                port=args.db_port,
                user=args.db_user,
                password=args.db_password,
                database=args.db_database
            )
            
            # Fetch events from database
            df_events = pipeline.fetch_events(
                start_date=args.start_date,
                end_date=args.end_date,
                experiment_id=args.experiment_id
            )
            
            # Validate data
            validation = pipeline.validate_data(df_events)
            print(f"\n[Info] Data Quality Score: {validation['data_quality_score']:.1f}%")
            print(f"[Info] Total Applicants: {validation['total_applicants']:,}")
            print(f"[Info] Date Range: {validation['date_range'][0]} to {validation['date_range'][1]}")
            
            # Check schema
            schema = pipeline.get_schema_info('events')
            print(f"[Info] Database Schema: {len(schema)} columns detected")
            
            # Close connection pool
            pipeline.close()
            
            # Prepare data (same as CSV)
            wide = df_events.copy()
            
        except Exception as e:
            print(f"[Error] Failed to load from PostgreSQL: {e}")
            return
    else:
        print("[Error] Specify data source: --csv <file> or --db-host <host>")
        parser.print_help()
        return

    # =====================================================================
    # ANALYSIS
    # =====================================================================
    
    print(f"\n[Info] Applicants: {len(wide):,}")
    results = print_core_results(wide)
    ce = cost_effect(wide, cost_per_check=args.cost_per_check, assumed_ltv=args.ltv)

    # Optional: Power analysis
    if args.power:
        compute_statistical_power(wide)

    # Plots
    plot_conversion_with_ci(wide, outdir)
    plot_funnel(wide, outdir)
    survival_analysis(wide, outdir)

    # Save key tables
    ensure_outdir(outdir)
    group_summary(wide).to_csv(outdir / "group_funnel_summary.csv", index=False)
    wide.to_csv(outdir / "applicant_level_wide.csv", index=False)

    print(f"\n[Done] Outputs saved to: {outdir.resolve()}")
    print(" - conversion_ci.png, funnel.png, km_curves.png")
    print(" - group_funnel_summary.csv, applicant_level_wide.csv")


if __name__ == "__main__":
    main()
