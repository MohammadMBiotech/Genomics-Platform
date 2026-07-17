"""
Quality Control Module — Publication Quality
────────────────────────────────────────────
Comprehensive QC for SNP genotype data:
  - Missing rate (per marker & per sample)
  - MAF filtering
  - Hardy-Weinberg Equilibrium (HWE) test (vectorized, fast)
  - Sample outlier detection (heterozygosity)
  - Chromosome-wise QC
  - Population-aware HWE (if metadata available)
  - Interactive filtering with preview
  - Comprehensive QC report
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import chisquare

from utils import (
    get_geno_from_session, get_meta_from_session,
    calc_maf, calc_missing_rate, calc_het_obs, calc_het_exp,
    calc_allele_freq,
    download_plotly_html, download_dataframe,
    sort_chromosomes, build_sample_pop_map,
)

st.title("🧹 Quality Control")
st.markdown("---")

geno, marker_info = get_geno_from_session()
if geno is None:
    st.stop()

meta = get_meta_from_session()


# ═══════════════════════════════════════════
# GLOBAL: Metadata configuration (optional)
# ═══════════════════════════════════════════
pop_map = {}
if meta is not None:
    with st.expander("🔧 Metadata Configuration (optional — for population-aware QC)"):
        mc1, mc2 = st.columns(2)
        with mc1:
            sample_col = st.selectbox("Sample ID column",
                                        meta.columns.tolist(),
                                        key="qc_samcol")
        with mc2:
            pop_col = st.selectbox("Population column",
                                     ["None"] + meta.columns.tolist(),
                                     key="qc_popcol")

        if pop_col != "None":
            pop_map = build_sample_pop_map(meta, sample_col, pop_col)
            st.success(f"✅ Population info loaded: "
                        f"{len(set(pop_map.values()))} populations")


# ═══════════════════════════════════════════
# INITIAL BACKUP (for reset)
# ═══════════════════════════════════════════
if "qc_geno_backup" not in st.session_state:
    st.session_state["qc_geno_backup"] = geno.copy()
    st.session_state["qc_marker_info_backup"] = (
        marker_info.copy() if marker_info is not None else None
    )


# ═══════════════════════════════════════════
# RAW DATA SUMMARY
# ═══════════════════════════════════════════
st.subheader("📊 Raw Data Summary")

r1, r2, r3, r4, r5 = st.columns(5)
r1.metric("Samples", f"{geno.shape[0]:,}")
r2.metric("Markers", f"{geno.shape[1]:,}")
missing_pct = geno.isna().sum().sum() / geno.size * 100
r3.metric("Total Missing %", f"{missing_pct:.2f}%")
r4.metric("Monomorphic Markers", f"{int((geno.nunique() <= 1).sum()):,}")

# Chromosome count
if marker_info is not None and "Chrom" in marker_info.columns:
    n_chr = marker_info["Chrom"].nunique()
    r5.metric("Chromosomes", n_chr)
else:
    r5.metric("Chromosomes", "N/A")


# ═══════════════════════════════════════════
# PRE-COMPUTE ALL METRICS
# ═══════════════════════════════════════════
with st.spinner("Computing QC metrics..."):
    miss_marker = calc_missing_rate(geno, axis=0)
    miss_sample = calc_missing_rate(geno, axis=1)
    maf = calc_maf(geno)
    ho = calc_het_obs(geno)
    he = calc_het_exp(geno)


# ═══════════════════════════════════════════
# COMPREHENSIVE DIAGNOSTIC PANEL
# ═══════════════════════════════════════════
st.markdown("---")
st.subheader("📈 Comprehensive Diagnostic Panel")

fig_multi = make_subplots(
    rows=2, cols=3,
    subplot_titles=(
        "MAF Distribution", "Marker Missing Rate", "Sample Missing Rate",
        "He Distribution", "Ho Distribution", "Ho vs He",
    ),
)

fig_multi.add_trace(go.Histogram(x=maf.values, marker_color="steelblue",
                                     name="MAF", showlegend=False), 1, 1)
fig_multi.add_trace(go.Histogram(x=miss_marker.values,
                                     marker_color="orange",
                                     name="Marker miss",
                                     showlegend=False), 1, 2)
fig_multi.add_trace(go.Histogram(x=miss_sample.values,
                                     marker_color="red",
                                     name="Sample miss",
                                     showlegend=False), 1, 3)
fig_multi.add_trace(go.Histogram(x=he.values,
                                     marker_color="purple",
                                     name="He", showlegend=False), 2, 1)
fig_multi.add_trace(go.Histogram(x=ho.dropna().values,
                                     marker_color="green",
                                     name="Ho", showlegend=False), 2, 2)
fig_multi.add_trace(go.Scatter(x=he.values, y=ho.values, mode="markers",
                                   marker=dict(size=3, color="purple",
                                                opacity=0.4),
                                   name="Ho vs He",
                                   showlegend=False), 2, 3)

fig_multi.update_layout(height=650, template="plotly_white",
                          title="Multi-metric QC diagnostics")
st.plotly_chart(fig_multi, use_container_width=True)


# ═══════════════════════════════════════════
# TABS FOR DETAILED ANALYSES
# ═══════════════════════════════════════════
tab_missing, tab_maf, tab_het, tab_hwe, tab_samples, tab_chr = st.tabs([
    "❓ Missing Data",
    "🎯 MAF",
    "🧬 Heterozygosity",
    "⚖️ HWE Test",
    "👤 Sample QC",
    "🎨 Per-Chromosome",
])


# ═══════════════════════════════════════════
# TAB 1 — Missing Data
# ═══════════════════════════════════════════
with tab_missing:
    st.subheader("Missing Rate Analysis")

    mc1, mc2 = st.columns(2)

    with mc1:
        st.markdown("#### Per-Marker Missing Rate")
        fig_miss_m = px.histogram(
            x=miss_marker.values, nbins=50,
            labels={"x": "Missing Rate", "y": "Count"},
        )
        fig_miss_m.add_vline(x=0.1, line_dash="dash", line_color="orange",
                                annotation_text="10%")
        fig_miss_m.add_vline(x=0.2, line_dash="dash", line_color="red",
                                annotation_text="20%")
        fig_miss_m.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig_miss_m, use_container_width=True)

        st.metric("Markers > 10% missing",
                    f"{int((miss_marker > 0.1).sum()):,}")
        st.metric("Markers > 20% missing",
                    f"{int((miss_marker > 0.2).sum()):,}")

    with mc2:
        st.markdown("#### Per-Sample Missing Rate")
        fig_miss_s = px.histogram(
            x=miss_sample.values, nbins=50,
            labels={"x": "Missing Rate", "y": "Count"},
        )
        fig_miss_s.add_vline(x=0.1, line_dash="dash", line_color="orange",
                                annotation_text="10%")
        fig_miss_s.add_vline(x=0.3, line_dash="dash", line_color="red",
                                annotation_text="30%")
        fig_miss_s.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig_miss_s, use_container_width=True)

        st.metric("Samples > 10% missing",
                    f"{int((miss_sample > 0.1).sum()):,}")
        st.metric("Samples > 30% missing",
                    f"{int((miss_sample > 0.3).sum()):,}")

    # Top offenders
    st.markdown("#### Top 20 Samples with Highest Missing Rate")
    top_miss_samples = miss_sample.sort_values(ascending=False).head(20)
    top_miss_df = pd.DataFrame({
        "Sample": top_miss_samples.index,
        "Missing_rate": top_miss_samples.values,
    })
    st.dataframe(top_miss_df.style.format({"Missing_rate": "{:.4f}"}),
                  use_container_width=True)


# ═══════════════════════════════════════════
# TAB 2 — MAF
# ═══════════════════════════════════════════
with tab_maf:
    st.subheader("Minor Allele Frequency (MAF)")

    fig_maf = px.histogram(
        x=maf.values, nbins=50,
        title="MAF Distribution",
        labels={"x": "MAF", "y": "Count"},
    )
    fig_maf.add_vline(x=0.01, line_dash="dot", line_color="orange",
                        annotation_text="Rare (0.01)")
    fig_maf.add_vline(x=0.05, line_dash="dash", line_color="red",
                        annotation_text="Common threshold (0.05)")
    fig_maf.update_layout(template="plotly_white", height=450)
    st.plotly_chart(fig_maf, use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mean MAF", f"{maf.mean():.4f}")
    m2.metric("Median MAF", f"{maf.median():.4f}")
    m3.metric("Markers with MAF < 0.01",
                f"{int((maf < 0.01).sum()):,}")
    m4.metric("Markers with MAF < 0.05",
                f"{int((maf < 0.05).sum()):,}")

    # Cumulative MAF
    st.markdown("#### Cumulative MAF Distribution")
    sorted_maf = np.sort(maf.values)
    cum_pct = np.arange(1, len(sorted_maf) + 1) / len(sorted_maf) * 100

    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(
        x=sorted_maf, y=cum_pct, mode="lines",
        line=dict(color="steelblue", width=2),
        name="Cumulative %",
    ))
    fig_cum.add_vline(x=0.05, line_dash="dash", line_color="red")
    fig_cum.update_layout(
        title="Cumulative distribution of MAF",
        xaxis_title="MAF threshold",
        yaxis_title="% of markers below threshold",
        template="plotly_white", height=400,
    )
    st.plotly_chart(fig_cum, use_container_width=True)


# ═══════════════════════════════════════════
# TAB 3 — Heterozygosity
# ═══════════════════════════════════════════
with tab_het:
    st.subheader("Heterozygosity Analysis")

    het_df = pd.DataFrame({"Ho": ho, "He": he,
                             "Marker": geno.columns}).dropna()

    # Marker-level Fis
    het_df["Fis"] = 1 - (het_df["Ho"] / het_df["He"].replace(0, np.nan))

    # Detect outliers
    max_val = max(het_df["He"].max(), het_df["Ho"].max(), 0.6)
    het_df["Deviation"] = np.abs(het_df["Ho"] - het_df["He"])
    outlier_thresh = het_df["Deviation"].quantile(0.95)
    het_df["Outlier"] = het_df["Deviation"] > outlier_thresh

    fig_het = px.scatter(
        het_df, x="He", y="Ho", color="Outlier",
        opacity=0.6,
        title="Observed vs Expected Heterozygosity per Marker",
        color_discrete_map={True: "red", False: "steelblue"},
        hover_data=["Marker", "Fis"],
    )
    fig_het.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                       line=dict(color="black", dash="dash"))
    fig_het.update_layout(template="plotly_white", height=550)
    st.plotly_chart(fig_het, use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mean Ho", f"{ho.mean():.4f}")
    m2.metric("Mean He", f"{he.mean():.4f}")
    m3.metric("Mean Fis", f"{het_df['Fis'].mean():.4f}")
    m4.metric("Deviation outliers",
                f"{int(het_df['Outlier'].sum()):,} ({het_df['Outlier'].mean()*100:.1f}%)")

    with st.expander("View outlier markers"):
        st.dataframe(het_df[het_df["Outlier"]].head(50).style.format({
            "Ho": "{:.4f}", "He": "{:.4f}", "Fis": "{:.4f}",
            "Deviation": "{:.4f}",
        }), use_container_width=True)


# ═══════════════════════════════════════════
# TAB 4 — HWE Test (Vectorized)
# ═══════════════════════════════════════════
with tab_hwe:
    st.subheader("Hardy-Weinberg Equilibrium (HWE) Test")
    st.write(
        "Chi-square test comparing observed genotype counts to "
        "Hardy-Weinberg expectations. Significant p-values suggest "
        "departure from HWE (genotyping errors, inbreeding, selection, etc.)."
    )

    # VECTORIZED HWE test — much faster
    with st.spinner("Running vectorized HWE tests..."):
        # Count genotypes per marker (vectorized)
        n0 = (geno == 0).sum(axis=0).values.astype(float)
        n1 = (geno == 1).sum(axis=0).values.astype(float)
        n2 = (geno == 2).sum(axis=0).values.astype(float)
        n_total = n0 + n1 + n2

        # Allele frequencies
        with np.errstate(divide="ignore", invalid="ignore"):
            p_arr = (2 * n0 + n1) / (2 * n_total)
        q_arr = 1 - p_arr

        # Expected counts
        exp0 = p_arr ** 2 * n_total
        exp1 = 2 * p_arr * q_arr * n_total
        exp2 = q_arr ** 2 * n_total

        # Chi-square statistic
        with np.errstate(divide="ignore", invalid="ignore"):
            chi2_stat = (
                ((n0 - exp0) ** 2 / np.maximum(exp0, 1e-10)) +
                ((n1 - exp1) ** 2 / np.maximum(exp1, 1e-10)) +
                ((n2 - exp2) ** 2 / np.maximum(exp2, 1e-10))
            )

        # p-value from chi-square distribution (df=1)
        from scipy.stats import chi2 as chi2_dist
        hwe_pvals_arr = 1 - chi2_dist.cdf(chi2_stat, df=1)

        # Handle edge cases: monomorphic markers → NaN
        hwe_pvals_arr = np.where(
            (n_total < 10) | (p_arr <= 0) | (p_arr >= 1),
            np.nan, hwe_pvals_arr,
        )

        hwe_pvals = pd.Series(hwe_pvals_arr, index=geno.columns)

    # Distribution
    neg_log_hwe = -np.log10(np.clip(hwe_pvals.dropna().values, 1e-300, 1))

    fig_hwe = px.histogram(
        x=neg_log_hwe, nbins=60,
        title="HWE Test: -log10(p-value) Distribution",
        labels={"x": "-log10(p)", "y": "Count"},
    )
    fig_hwe.add_vline(x=-np.log10(0.05), line_dash="dot",
                        line_color="orange",
                        annotation_text="p = 0.05")
    fig_hwe.add_vline(x=-np.log10(0.001), line_dash="dash",
                        line_color="red",
                        annotation_text="p = 0.001")
    fig_hwe.add_vline(x=-np.log10(1e-6), line_dash="dot",
                        line_color="darkred",
                        annotation_text="p = 1e-6")
    fig_hwe.update_layout(template="plotly_white", height=450)
    st.plotly_chart(fig_hwe, use_container_width=True)

    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Markers failing p<0.05",
                f"{int((hwe_pvals < 0.05).sum()):,}")
    h2.metric("Markers failing p<0.001",
                f"{int((hwe_pvals < 0.001).sum()):,}")
    h3.metric("Markers failing p<1e-6",
                f"{int((hwe_pvals < 1e-6).sum()):,}")
    h4.metric("Valid HWE tests",
                f"{int(hwe_pvals.notna().sum()):,}")

    # Population-aware HWE (if metadata)
    if pop_map:
        st.markdown("#### Per-Population HWE")
        st.info("Computing HWE per population (may take a moment)...")

        pops = sorted(set(pop_map.values()))
        pop_hwe_results = []

        for pop in pops:
            samples_pop = [s for s in geno.index.astype(str)
                            if pop_map.get(s) == pop]
            if len(samples_pop) < 10:
                continue
            geno_pop = geno.loc[samples_pop]

            # Vectorized per-population HWE
            n0_p = (geno_pop == 0).sum(axis=0).values.astype(float)
            n1_p = (geno_pop == 1).sum(axis=0).values.astype(float)
            n2_p = (geno_pop == 2).sum(axis=0).values.astype(float)
            n_tot_p = n0_p + n1_p + n2_p

            with np.errstate(divide="ignore", invalid="ignore"):
                p_p = (2 * n0_p + n1_p) / (2 * n_tot_p)
            q_p = 1 - p_p

            exp0_p = p_p ** 2 * n_tot_p
            exp1_p = 2 * p_p * q_p * n_tot_p
            exp2_p = q_p ** 2 * n_tot_p

            with np.errstate(divide="ignore", invalid="ignore"):
                chi2_p = (
                    ((n0_p - exp0_p) ** 2 / np.maximum(exp0_p, 1e-10)) +
                    ((n1_p - exp1_p) ** 2 / np.maximum(exp1_p, 1e-10)) +
                    ((n2_p - exp2_p) ** 2 / np.maximum(exp2_p, 1e-10))
                )
            hwe_p = 1 - chi2_dist.cdf(chi2_p, df=1)
            hwe_p = np.where(
                (n_tot_p < 5) | (p_p <= 0) | (p_p >= 1),
                np.nan, hwe_p,
            )

            n_fail = int((hwe_p < 0.001).sum())
            pop_hwe_results.append({
                "Population": pop,
                "N_samples": len(samples_pop),
                "N_markers_tested": int(np.sum(~np.isnan(hwe_p))),
                "N_failing_HWE": n_fail,
                "Mean_-log10(p)": np.nanmean(
                    -np.log10(np.clip(hwe_p, 1e-300, 1))
                ),
            })

        pop_hwe_df = pd.DataFrame(pop_hwe_results)
        st.dataframe(pop_hwe_df.style.format({
            "Mean_-log10(p)": "{:.3f}",
        }), use_container_width=True)


# ═══════════════════════════════════════════
# TAB 5 — Sample QC (Outlier Detection)
# ═══════════════════════════════════════════
with tab_samples:
    st.subheader("👤 Sample-Level QC")

    # Compute per-sample statistics
    sample_stats = pd.DataFrame({
        "Sample": geno.index.astype(str),
        "Missing_rate": miss_sample.values,
        "Het_rate": (geno == 1).sum(axis=1).values /
                     np.maximum(geno.notna().sum(axis=1).values, 1),
        "Hom_ref": (geno == 0).sum(axis=1).values /
                     np.maximum(geno.notna().sum(axis=1).values, 1),
        "Hom_alt": (geno == 2).sum(axis=1).values /
                     np.maximum(geno.notna().sum(axis=1).values, 1),
    })

    # Detect outliers using IQR method
    het_q1 = sample_stats["Het_rate"].quantile(0.25)
    het_q3 = sample_stats["Het_rate"].quantile(0.75)
    het_iqr = het_q3 - het_q1
    het_lo = het_q1 - 3 * het_iqr
    het_hi = het_q3 + 3 * het_iqr

    sample_stats["Het_outlier"] = (
        (sample_stats["Het_rate"] < het_lo) |
        (sample_stats["Het_rate"] > het_hi)
    )
    sample_stats["Missing_outlier"] = sample_stats["Missing_rate"] > 0.3
    sample_stats["Any_outlier"] = (
        sample_stats["Het_outlier"] | sample_stats["Missing_outlier"]
    )

    # Scatter: Missing vs Het
    fig_sample = px.scatter(
        sample_stats, x="Missing_rate", y="Het_rate",
        color="Any_outlier",
        color_discrete_map={True: "red", False: "steelblue"},
        hover_data=["Sample"],
        title="Sample QC: Missing rate vs Heterozygosity",
        labels={"Missing_rate": "Missing rate",
                "Het_rate": "Heterozygosity rate"},
    )
    fig_sample.add_hline(y=het_lo, line_dash="dash", line_color="orange",
                           annotation_text="Het lower bound")
    fig_sample.add_hline(y=het_hi, line_dash="dash", line_color="orange",
                           annotation_text="Het upper bound")
    fig_sample.add_vline(x=0.3, line_dash="dash", line_color="red",
                           annotation_text="Missing > 30%")
    fig_sample.update_traces(marker=dict(size=8, opacity=0.7,
                                           line=dict(width=0.5,
                                                     color="darkslategrey")))
    fig_sample.update_layout(template="plotly_white", height=550)
    st.plotly_chart(fig_sample, use_container_width=True)

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Total samples", len(sample_stats))
    sc2.metric("Missing outliers",
                int(sample_stats["Missing_outlier"].sum()))
    sc3.metric("Het outliers",
                int(sample_stats["Het_outlier"].sum()))
    sc4.metric("Any outlier",
                int(sample_stats["Any_outlier"].sum()))

    # Table of outliers
    if int(sample_stats["Any_outlier"].sum()) > 0:
        st.markdown("#### Outlier Samples")
        st.dataframe(sample_stats[sample_stats["Any_outlier"]].style.format({
            "Missing_rate": "{:.4f}",
            "Het_rate": "{:.4f}",
            "Hom_ref": "{:.4f}",
            "Hom_alt": "{:.4f}",
        }), use_container_width=True)

        download_dataframe(sample_stats, "sample_qc_stats.csv",
                            key="dl_sample_qc")


# ═══════════════════════════════════════════
# TAB 6 — Per-Chromosome QC
# ═══════════════════════════════════════════
with tab_chr:
    st.subheader("🎨 Per-Chromosome QC Statistics")

    if marker_info is None or "Chrom" not in marker_info.columns:
        st.warning("⚠️ Chromosome information not available.")
    else:
        # Merge marker info
        chr_stats_df = pd.DataFrame({
            "Marker": geno.columns.astype(str),
            "MAF": maf.values,
            "Missing_rate": miss_marker.values,
            "Ho": ho.values,
            "He": he.values,
            "HWE_pval": hwe_pvals.values,
        }).merge(marker_info[["Marker", "Chrom"]], on="Marker", how="left")

        # Aggregate per chromosome
        chr_agg = chr_stats_df.groupby("Chrom").agg({
            "Marker": "count",
            "MAF": "mean",
            "Missing_rate": "mean",
            "Ho": "mean",
            "He": "mean",
        }).reset_index().rename(columns={"Marker": "N_markers"})

        # Sort chromosomes
        chr_order = sort_chromosomes(chr_agg["Chrom"].tolist())
        chr_agg["_sort"] = chr_agg["Chrom"].apply(
            lambda x: chr_order.index(x) if x in chr_order else 999)
        chr_agg = chr_agg.sort_values("_sort").drop(columns=["_sort"])

        st.dataframe(chr_agg.style.format({
            "MAF": "{:.4f}", "Missing_rate": "{:.4f}",
            "Ho": "{:.4f}", "He": "{:.4f}",
        }), use_container_width=True)

        # Multi-panel plot
        fig_chr = make_subplots(
            rows=2, cols=2,
            subplot_titles=("N markers per chromosome",
                              "Mean MAF per chromosome",
                              "Mean missing rate per chromosome",
                              "Mean He per chromosome"),
        )

        fig_chr.add_trace(go.Bar(x=chr_agg["Chrom"],
                                     y=chr_agg["N_markers"],
                                     marker_color="steelblue",
                                     showlegend=False), 1, 1)
        fig_chr.add_trace(go.Bar(x=chr_agg["Chrom"],
                                     y=chr_agg["MAF"],
                                     marker_color="green",
                                     showlegend=False), 1, 2)
        fig_chr.add_trace(go.Bar(x=chr_agg["Chrom"],
                                     y=chr_agg["Missing_rate"],
                                     marker_color="orange",
                                     showlegend=False), 2, 1)
        fig_chr.add_trace(go.Bar(x=chr_agg["Chrom"],
                                     y=chr_agg["He"],
                                     marker_color="purple",
                                     showlegend=False), 2, 2)

        fig_chr.update_layout(height=650, template="plotly_white",
                                title="Chromosome-wise QC")
        st.plotly_chart(fig_chr, use_container_width=True)

        download_dataframe(chr_agg, "chromosome_qc.csv", key="dl_chr_qc")


# ═══════════════════════════════════════════
# FILTERING SECTION
# ═══════════════════════════════════════════
st.markdown("---")
st.subheader("🔧 Apply QC Filters")

st.write(
    "Configure thresholds below and preview the impact before applying filters."
)

# Filter configuration
fc1, fc2, fc3, fc4 = st.columns(4)
with fc1:
    max_miss_marker = st.slider(
        "Max marker missing rate", 0.0, 1.0, 0.2, 0.05,
        key="qc_mmm",
        help="Remove markers with missing rate above this threshold",
    )
with fc2:
    max_miss_sample = st.slider(
        "Max sample missing rate", 0.0, 1.0, 0.3, 0.05,
        key="qc_mms",
        help="Remove samples with missing rate above this threshold",
    )
with fc3:
    min_maf = st.slider(
        "Min MAF", 0.0, 0.5, 0.05, 0.01,
        key="qc_maf",
        help="Remove markers with MAF below this threshold",
    )
with fc4:
    hwe_threshold = st.number_input(
        "HWE p threshold", value=0.001, format="%.6f",
        key="qc_hwe",
        help="Remove markers with HWE p-value below this threshold",
    )

# Additional filters
fc5, fc6 = st.columns(2)
with fc5:
    remove_monomorphic = st.checkbox(
        "Remove monomorphic markers", True, key="qc_mono",
        help="Remove markers where all samples have the same genotype",
    )
with fc6:
    remove_sample_outliers = st.checkbox(
        "Remove sample heterozygosity outliers", False, key="qc_sample_out",
        help="Remove samples with abnormally high/low heterozygosity",
    )

# Preview button
if st.button("🔍 Preview Filter Impact", key="qc_preview"):
    with st.spinner("Computing filter impact..."):
        # Marker filters
        keep_markers = (
            (miss_marker <= max_miss_marker) &
            (maf >= min_maf) &
            (hwe_pvals >= hwe_threshold)
        )
        if remove_monomorphic:
            not_mono = geno.nunique() > 1
            keep_markers = keep_markers & not_mono

        # Sample filters
        keep_samples_mask = miss_sample <= max_miss_sample
        if remove_sample_outliers:
            # Compute IQR outliers on het rate
            het_rate = (geno == 1).sum(axis=1) / geno.notna().sum(axis=1)
            q1_ = het_rate.quantile(0.25)
            q3_ = het_rate.quantile(0.75)
            iqr_ = q3_ - q1_
            lo_ = q1_ - 3 * iqr_
            hi_ = q3_ + 3 * iqr_
            no_outlier = (het_rate >= lo_) & (het_rate <= hi_)
            keep_samples_mask = keep_samples_mask & no_outlier

        n_markers_kept = int(keep_markers.sum())
        n_samples_kept = int(keep_samples_mask.sum())

    st.info(
        f"📊 **Preview:** After applying filters, you will have "
        f"**{n_samples_kept:,}** samples × **{n_markers_kept:,}** markers "
        f"(removing {geno.shape[0] - n_samples_kept} samples and "
        f"{geno.shape[1] - n_markers_kept} markers)."
    )

    pc1, pc2, pc3, pc4 = st.columns(4)
    pc1.metric("Removed (missing)",
                int((miss_marker > max_miss_marker).sum()))
    pc2.metric("Removed (MAF)",
                int((maf < min_maf).sum()))
    pc3.metric("Removed (HWE)",
                int((hwe_pvals < hwe_threshold).sum()))
    pc4.metric("Removed (samples)",
                int((~keep_samples_mask).sum()))


# ─── APPLY FILTERS ───
col_apply, col_reset = st.columns(2)

with col_apply:
    if st.button("🚀 Apply Filters", use_container_width=True, key="qc_apply",
                 type="primary"):
        # Apply marker filters
        keep_markers = (
            (miss_marker <= max_miss_marker) &
            (maf >= min_maf) &
            (hwe_pvals >= hwe_threshold)
        )
        if remove_monomorphic:
            not_mono = geno.nunique() > 1
            keep_markers = keep_markers & not_mono

        geno_filtered = geno.loc[:, keep_markers]

        # Apply sample filters
        miss_s_filtered = geno_filtered.isna().mean(axis=1)
        keep_samples_mask = miss_s_filtered <= max_miss_sample

        if remove_sample_outliers:
            het_rate = (geno_filtered == 1).sum(axis=1) / \
                        geno_filtered.notna().sum(axis=1)
            q1_ = het_rate.quantile(0.25)
            q3_ = het_rate.quantile(0.75)
            iqr_ = q3_ - q1_
            lo_ = q1_ - 3 * iqr_
            hi_ = q3_ + 3 * iqr_
            no_outlier = (het_rate >= lo_) & (het_rate <= hi_)
            keep_samples_mask = keep_samples_mask & no_outlier

        geno_filtered = geno_filtered.loc[keep_samples_mask]

        # Update session state
        st.session_state["genotype_matrix"] = geno_filtered
        if marker_info is not None:
            st.session_state["marker_info"] = marker_info[
                marker_info["Marker"].astype(str).isin(
                    geno_filtered.columns.astype(str))
            ].reset_index(drop=True)

        st.success(
            f"✅ **Filters applied!** New dataset: "
            f"**{geno_filtered.shape[0]:,}** samples × "
            f"**{geno_filtered.shape[1]:,}** markers.\n\n"
            f"Removed {geno.shape[0] - geno_filtered.shape[0]} samples "
            f"and {geno.shape[1] - geno_filtered.shape[1]} markers.\n\n"
            f"All other modules will now use the filtered dataset."
        )
        st.balloons()

        # Save filter history
        if "qc_history" not in st.session_state:
            st.session_state["qc_history"] = []
        st.session_state["qc_history"].append({
            "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Max_missing_marker": max_miss_marker,
            "Max_missing_sample": max_miss_sample,
            "Min_MAF": min_maf,
            "HWE_threshold": hwe_threshold,
            "N_samples_after": geno_filtered.shape[0],
            "N_markers_after": geno_filtered.shape[1],
        })

with col_reset:
    if st.button("🔄 Reset to Original Data", use_container_width=True,
                 key="qc_reset"):
        st.session_state["genotype_matrix"] = st.session_state["qc_geno_backup"].copy()
        if st.session_state["qc_marker_info_backup"] is not None:
            st.session_state["marker_info"] = st.session_state["qc_marker_info_backup"].copy()
        st.success("✅ Data restored to original state. Please refresh the page.")
        st.rerun()


# ═══════════════════════════════════════════
# FILTER HISTORY
# ═══════════════════════════════════════════
if "qc_history" in st.session_state and len(st.session_state["qc_history"]) > 0:
    with st.expander("📜 Filter History"):
        history_df = pd.DataFrame(st.session_state["qc_history"])
        st.dataframe(history_df, use_container_width=True)


# ═══════════════════════════════════════════
# QC SUMMARY TABLE (per marker)
# ═══════════════════════════════════════════
st.markdown("---")
st.subheader("📋 Per-Marker QC Summary Table")

qc_summary = pd.DataFrame({
    "Marker": geno.columns,
    "Missing_Rate": miss_marker.values,
    "MAF": maf.values,
    "Ho": ho.values,
    "He": he.values,
    "HWE_pval": hwe_pvals.values,
    "HWE_neg_log10p": -np.log10(np.clip(hwe_pvals.values, 1e-300, 1)),
})

# Add pass/fail flag based on current thresholds
qc_summary["Pass_QC"] = (
    (qc_summary["Missing_Rate"] <= max_miss_marker) &
    (qc_summary["MAF"] >= min_maf) &
    (qc_summary["HWE_pval"] >= hwe_threshold)
)

if marker_info is not None:
    qc_summary = qc_summary.merge(marker_info, on="Marker", how="left")

st.dataframe(qc_summary.head(200).style.format({
    "Missing_Rate": "{:.4f}",
    "MAF": "{:.4f}",
    "Ho": "{:.4f}",
    "He": "{:.4f}",
    "HWE_pval": "{:.3e}",
    "HWE_neg_log10p": "{:.2f}",
}), use_container_width=True)

download_dataframe(qc_summary, "qc_summary_full.csv", key="dl_qc")


# ═══════════════════════════════════════════
# QC REPORT
# ═══════════════════════════════════════════
st.markdown("---")
st.subheader("📄 Generate QC Report")

if st.button("📊 Generate comprehensive QC report", key="qc_report"):
    report = f"""
# Quality Control Report

**Dataset:** {geno.shape[0]:,} samples × {geno.shape[1]:,} markers
**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Missing Data
- Overall missing rate: {missing_pct:.2f}%
- Markers with >10% missing: {int((miss_marker > 0.1).sum()):,}
- Markers with >20% missing: {int((miss_marker > 0.2).sum()):,}
- Samples with >10% missing: {int((miss_sample > 0.1).sum()):,}
- Samples with >30% missing: {int((miss_sample > 0.3).sum()):,}

## MAF
- Mean MAF: {maf.mean():.4f}
- Median MAF: {maf.median():.4f}
- Rare markers (MAF < 0.01): {int((maf < 0.01).sum()):,}
- Uncommon markers (MAF < 0.05): {int((maf < 0.05).sum()):,}

## Heterozygosity
- Mean Ho: {ho.mean():.4f}
- Mean He: {he.mean():.4f}
- Mean Fis: {(1 - ho/he.replace(0, np.nan)).mean():.4f}

## Hardy-Weinberg Equilibrium
- Markers with p<0.05: {int((hwe_pvals < 0.05).sum()):,}
- Markers with p<0.001: {int((hwe_pvals < 0.001).sum()):,}
- Markers with p<1e-6: {int((hwe_pvals < 1e-6).sum()):,}

## Current Filter Settings
- Max marker missing rate: {max_miss_marker}
- Max sample missing rate: {max_miss_sample}
- Min MAF: {min_maf}
- HWE p threshold: {hwe_threshold}

## Interpretation & Recommendations
"""

    # Add recommendations
    if missing_pct > 20:
        report += "- ⚠️ **High missing data (>20%).** Consider imputation or stricter filtering.\n"
    if int((maf < 0.05).sum()) > geno.shape[1] * 0.3:
        report += "- ⚠️ **Many rare variants (>30% MAF<0.05).** Apply MAF filter for population analyses.\n"
    if int((hwe_pvals < 0.001).sum()) > geno.shape[1] * 0.05:
        report += "- ⚠️ **Many HWE failures.** Check for genotyping errors or population stratification.\n"
    if ho.mean() < he.mean() * 0.5:
        report += "- ⚠️ **Low observed heterozygosity.** Suggests inbreeding or population structure.\n"

    st.markdown(report)

    st.download_button(
        "📥 Download QC Report (Markdown)",
        report, "qc_report.md", "text/markdown",
        key="dl_qc_report",
    )