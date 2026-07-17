"""
Interactive Reports — Publication Quality
──────────────────────────────────────────
Comprehensive analysis summary with:
  - Executive dashboard with data quality grading
  - Auto-generated insights and recommendations
  - Genome-wide statistical summary
  - Per-population comparison (radar, heatmap, tables)
  - Chromosome-level analysis
  - Sample-level QC report
  - Publication-ready HTML report
  - Multiple export formats (MD, HTML, Excel, CSV)
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

from utils import (
    get_geno_from_session, get_meta_from_session,
    calc_allele_freq, calc_maf, calc_het_obs, calc_het_exp, calc_pic,
    calc_fis, calc_missing_rate, calc_shannon_diversity,
    calc_effective_alleles,
    build_sample_pop_map, sort_chromosomes,
    download_plotly_html, download_dataframe, download_excel,
)

st.title("📑 Interactive Reports")
st.markdown("---")

geno, marker_info = get_geno_from_session()
if geno is None:
    st.stop()

meta = get_meta_from_session()


# ═══════════════════════════════════════════
# METADATA CONFIGURATION
# ═══════════════════════════════════════════
pop_map = {}
sample_col = None
pop_col = None

if meta is not None:
    with st.expander("🔧 Metadata Configuration", expanded=True):
        mc1, mc2 = st.columns(2)
        with mc1:
            sample_col = st.selectbox("Sample ID column",
                                        meta.columns.tolist(),
                                        key="rep_samcol")
        with mc2:
            pop_col_opt = st.selectbox("Population column",
                                         ["None"] + meta.columns.tolist(),
                                         key="rep_popcol")
            pop_col = None if pop_col_opt == "None" else pop_col_opt

        if pop_col:
            pop_map = build_sample_pop_map(meta, sample_col, pop_col)


# ═══════════════════════════════════════════
# PRE-COMPUTE ALL STATISTICS
# ═══════════════════════════════════════════
with st.spinner("Computing all statistics..."):
    maf = calc_maf(geno)
    ho = calc_het_obs(geno)
    he = calc_het_exp(geno)
    pic = calc_pic(geno)
    fis = calc_fis(geno)
    shannon = calc_shannon_diversity(geno)
    ne = calc_effective_alleles(geno)
    miss_marker = calc_missing_rate(geno, axis=0)
    miss_sample = calc_missing_rate(geno, axis=1)


# ═══════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════
tab_exec, tab_summary, tab_diag, tab_pop, tab_chr, tab_sample, tab_export = st.tabs([
    "🎯 Executive Summary",
    "📊 Statistical Summary",
    "📈 Diagnostics",
    "🌍 Populations",
    "🎨 Chromosomes",
    "👤 Samples",
    "💾 Export Reports",
])


# ═══════════════════════════════════════════
# TAB 1 — EXECUTIVE SUMMARY
# ═══════════════════════════════════════════
with tab_exec:
    st.subheader("🎯 Executive Summary")

    # Header metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📊 Samples", f"{geno.shape[0]:,}")
    c2.metric("🧬 Markers", f"{geno.shape[1]:,}")
    missing_pct = geno.isna().sum().sum() / geno.size * 100
    c3.metric("❓ Missing", f"{missing_pct:.2f}%")

    if meta is not None:
        c4.metric("📋 Metadata rows", f"{meta.shape[0]:,}")
    else:
        c4.metric("📋 Metadata", "Not loaded")

    # Additional key metrics
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Mean MAF", f"{maf.mean():.4f}")
    c6.metric("Mean He", f"{he.mean():.4f}")
    c7.metric("Mean PIC", f"{pic.mean():.4f}")
    c8.metric("Mean Fis", f"{fis.mean():.4f}")

    st.markdown("---")

    # ─── DATA QUALITY DASHBOARD ───
    st.markdown("### 🎨 Data Quality Dashboard")

    quality_items = []

    # Sample size
    if geno.shape[0] >= 100:
        quality_items.append(("Sample size", "✅ Excellent", "success",
                                f"{geno.shape[0]:,} samples"))
    elif geno.shape[0] >= 50:
        quality_items.append(("Sample size", "✅ Good", "success",
                                f"{geno.shape[0]:,} samples"))
    elif geno.shape[0] >= 20:
        quality_items.append(("Sample size", "⚡ Moderate", "warning",
                                f"{geno.shape[0]:,} — some analyses limited"))
    else:
        quality_items.append(("Sample size", "❌ Small", "error",
                                f"{geno.shape[0]:,} — many analyses limited"))

    # Marker count
    if geno.shape[1] >= 5000:
        quality_items.append(("Marker count", "✅ Excellent", "success",
                                f"{geno.shape[1]:,} markers"))
    elif geno.shape[1] >= 500:
        quality_items.append(("Marker count", "✅ Good", "success",
                                f"{geno.shape[1]:,} markers"))
    elif geno.shape[1] >= 100:
        quality_items.append(("Marker count", "⚡ Moderate", "warning",
                                f"{geno.shape[1]:,} markers"))
    else:
        quality_items.append(("Marker count", "❌ Small", "error",
                                f"{geno.shape[1]:,} markers"))

    # Missing data
    if missing_pct < 5:
        quality_items.append(("Missing data", "✅ Low", "success",
                                f"{missing_pct:.2f}%"))
    elif missing_pct < 15:
        quality_items.append(("Missing data", "⚡ Moderate", "warning",
                                f"{missing_pct:.2f}%"))
    else:
        quality_items.append(("Missing data", "❌ High", "error",
                                f"{missing_pct:.2f}% — QC filtering needed"))

    # MAF
    low_maf_pct = (maf < 0.05).mean() * 100
    if low_maf_pct < 10:
        quality_items.append(("Low-MAF markers", "✅ Few", "success",
                                f"{low_maf_pct:.1f}% < 0.05"))
    elif low_maf_pct < 30:
        quality_items.append(("Low-MAF markers", "⚡ Moderate", "warning",
                                f"{low_maf_pct:.1f}% < 0.05"))
    else:
        quality_items.append(("Low-MAF markers", "⚠️ Many", "warning",
                                f"{low_maf_pct:.1f}% < 0.05 — apply MAF filter"))

    # Genetic diversity
    if he.mean() > 0.35:
        quality_items.append(("Genetic diversity", "✅ High", "success",
                                f"Mean He = {he.mean():.3f}"))
    elif he.mean() > 0.2:
        quality_items.append(("Genetic diversity", "⚡ Moderate", "warning",
                                f"Mean He = {he.mean():.3f}"))
    else:
        quality_items.append(("Genetic diversity", "⚠️ Low", "warning",
                                f"Mean He = {he.mean():.3f}"))

    # Fis
    if abs(fis.mean()) < 0.05:
        quality_items.append(("HWE conformity", "✅ Near HWE", "success",
                                f"Mean Fis = {fis.mean():.3f}"))
    elif fis.mean() > 0.1:
        quality_items.append(("HWE conformity", "⚠️ Inbreeding", "warning",
                                f"Mean Fis = {fis.mean():.3f}"))
    elif fis.mean() < -0.1:
        quality_items.append(("HWE conformity", "⚠️ Het excess", "warning",
                                f"Mean Fis = {fis.mean():.3f}"))
    else:
        quality_items.append(("HWE conformity", "⚡ Slight deviation",
                                "warning", f"Mean Fis = {fis.mean():.3f}"))

    # Marker info
    if marker_info is not None and "Chrom" in marker_info.columns:
        n_chr = marker_info["Chrom"].nunique()
        quality_items.append(("Marker positions", "✅ Available", "success",
                                f"{n_chr} chromosomes"))
    else:
        quality_items.append(("Marker positions", "⚠️ Missing", "warning",
                                "LD/Manhattan analyses limited"))

    # Metadata
    if meta is not None:
        n_pops = len(set(pop_map.values())) if pop_map else 0
        if n_pops > 0:
            quality_items.append(("Population info", "✅ Available",
                                    "success", f"{n_pops} populations"))
        else:
            quality_items.append(("Metadata", "⚡ Loaded",
                                    "warning", "No population column selected"))
    else:
        quality_items.append(("Metadata", "⚠️ Not loaded", "warning",
                                "Population analyses unavailable"))

    # Display as styled table
    quality_df = pd.DataFrame(quality_items,
                                columns=["Item", "Status", "level", "Details"])

    for _, row in quality_df.iterrows():
        if row["level"] == "success":
            st.success(f"**{row['Item']}** — {row['Status']} — {row['Details']}")
        elif row["level"] == "warning":
            st.warning(f"**{row['Item']}** — {row['Status']} — {row['Details']}")
        elif row["level"] == "error":
            st.error(f"**{row['Item']}** — {row['Status']} — {row['Details']}")

    # ─── AUTO-GENERATED INSIGHTS ───
    st.markdown("---")
    st.markdown("### 💡 Auto-Generated Insights")

    insights = []

    # Diversity
    if he.mean() > 0.35:
        insights.append("🌿 **High genetic diversity detected** — This population is genetically variable, suitable for association and diversity studies.")
    elif he.mean() < 0.15:
        insights.append("⚠️ **Low genetic diversity** — May indicate inbreeding, bottleneck, or population isolation.")

    # HWE
    if fis.mean() > 0.1:
        insights.append("🔴 **Significant inbreeding detected** (Fis > 0.1) — Consider population substructure or non-random mating.")
    elif fis.mean() < -0.1:
        insights.append("🔵 **Heterozygote excess** (Fis < -0.1) — May indicate hybrid origin, balancing selection, or genotyping issues.")

    # MAF
    if low_maf_pct > 30:
        insights.append(f"⚠️ **{low_maf_pct:.0f}% of markers are rare (MAF < 0.05)** — Consider applying a MAF filter.")

    # Missing
    if missing_pct > 15:
        insights.append(f"⚠️ **High missing data ({missing_pct:.1f}%)** — Apply QC filters or imputation before analyses.")

    # PIC
    if pic.mean() > 0.5:
        insights.append("✨ **Highly informative markers** (mean PIC > 0.5) — Excellent for genetic differentiation studies.")
    elif pic.mean() < 0.25:
        insights.append("ℹ️ **Markers have low informativeness** (mean PIC < 0.25).")

    # Sample count
    if geno.shape[0] < 30:
        insights.append(f"⚠️ **Small sample size (n={geno.shape[0]})** — STRUCTURE, GWAS, and phylogenetic analyses may be underpowered.")

    # Marker count
    if geno.shape[1] < 500:
        insights.append(f"⚠️ **Few markers ({geno.shape[1]})** — LD, selection, and fine-scale structure analyses may be limited.")

    # Multiple populations
    if pop_map:
        n_pops = len(set(pop_map.values()))
        pop_sizes = pd.Series(list(pop_map.values())).value_counts()
        if pop_sizes.min() < 5:
            small_pops = pop_sizes[pop_sizes < 5].index.tolist()
            insights.append(f"⚠️ **Populations with < 5 samples:** {', '.join(small_pops[:5])}. Consider excluding or combining.")

    if not insights:
        insights.append("✅ **Data looks great!** All key quality metrics are within acceptable ranges.")

    for insight in insights:
        st.markdown(f"- {insight}")

    # ─── RECOMMENDED NEXT STEPS ───
    st.markdown("---")
    st.markdown("### 🚀 Recommended Next Steps")

    steps = []
    if missing_pct > 10 or low_maf_pct > 20:
        steps.append("1. Go to **🧹 Quality Control** and apply appropriate filters.")

    if pop_map:
        steps.append("2. Explore **🧩 Population Structure** to identify genetic clusters.")
        steps.append("3. Compute **🌍 Geographic Genetics** (Fst, DEST) between populations.")
        steps.append("4. Run **🧬 Genetic Diversity** for per-population statistics.")
    else:
        steps.append("2. Upload metadata to enable population-based analyses.")

    if geno.shape[0] >= 20:
        steps.append("5. Build phylogenetic tree via **🌳 Phylogenetics**.")
        steps.append("6. Compute kinship matrix via **👥 Kinship & Relatedness**.")

    if geno.shape[1] >= 100:
        steps.append("7. Analyze LD patterns via **🔗 Linkage Disequilibrium**.")

    if pop_map:
        steps.append("8. Detect selection signatures via **🤖 Machine Learning → Selection Detection**.")

    steps.append("9. Export all results via **💾 Export Results**.")

    for step in steps:
        st.markdown(f"- {step}")


# ═══════════════════════════════════════════
# TAB 2 — STATISTICAL SUMMARY
# ═══════════════════════════════════════════
with tab_summary:
    st.subheader("📊 Comprehensive Statistical Summary")

    summary = pd.DataFrame({
        "Statistic": ["MAF", "PIC", "Ho", "He", "Fis",
                       "Shannon (I)", "Ne (alleles)",
                       "Marker missing rate", "Sample missing rate"],
        "Mean": [maf.mean(), pic.mean(), ho.mean(), he.mean(),
                  fis.mean(), shannon.mean(), ne.mean(),
                  miss_marker.mean(), miss_sample.mean()],
        "Median": [maf.median(), pic.median(), ho.median(), he.median(),
                    fis.median(), shannon.median(), ne.median(),
                    miss_marker.median(), miss_sample.median()],
        "Std": [maf.std(), pic.std(), ho.std(), he.std(),
                 fis.std(), shannon.std(), ne.std(),
                 miss_marker.std(), miss_sample.std()],
        "Min": [maf.min(), pic.min(), ho.min(), he.min(),
                 fis.min(), shannon.min(), ne.min(),
                 miss_marker.min(), miss_sample.min()],
        "Max": [maf.max(), pic.max(), ho.max(), he.max(),
                 fis.max(), shannon.max(), ne.max(),
                 miss_marker.max(), miss_sample.max()],
    })

    st.dataframe(summary.style.format({
        "Mean": "{:.4f}", "Median": "{:.4f}", "Std": "{:.4f}",
        "Min": "{:.4f}", "Max": "{:.4f}"
    }), use_container_width=True)

    download_dataframe(summary, "statistical_summary.csv",
                        key="dl_summary")

    # Genotype composition
    st.markdown("### Genotype Composition")

    all_vals = geno.values.flatten()
    all_vals = all_vals[~np.isnan(all_vals)]
    n_0 = int((all_vals == 0).sum())
    n_1 = int((all_vals == 1).sum())
    n_2 = int((all_vals == 2).sum())
    total = n_0 + n_1 + n_2

    gc1, gc2, gc3 = st.columns(3)
    gc1.metric("Homozygous reference (0)",
                f"{n_0:,}", f"{n_0/total*100:.1f}%")
    gc2.metric("Heterozygous (1)",
                f"{n_1:,}", f"{n_1/total*100:.1f}%")
    gc3.metric("Homozygous alternate (2)",
                f"{n_2:,}", f"{n_2/total*100:.1f}%")

    fig_geno = px.pie(
        values=[n_0, n_1, n_2],
        names=["Hom Ref (0)", "Het (1)", "Hom Alt (2)"],
        title="Genotype dosage distribution",
        color_discrete_sequence=["#4CAF50", "#FFC107", "#F44336"],
    )
    fig_geno.update_layout(template="plotly_white", height=400)
    st.plotly_chart(fig_geno, use_container_width=True)


# ═══════════════════════════════════════════
# TAB 3 — DIAGNOSTICS
# ═══════════════════════════════════════════
with tab_diag:
    st.subheader("📈 Comprehensive Diagnostic Panel")

    fig_multi = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            "MAF distribution", "Marker missing rate",
            "Sample missing rate",
            "Ho vs He", "PIC distribution", "Fis distribution",
        ),
    )

    fig_multi.add_trace(go.Histogram(x=maf.values, marker_color="steelblue",
                                         showlegend=False), 1, 1)
    fig_multi.add_trace(go.Histogram(x=miss_marker.values,
                                         marker_color="orange",
                                         showlegend=False), 1, 2)
    fig_multi.add_trace(go.Histogram(x=miss_sample.values,
                                         marker_color="red",
                                         showlegend=False), 1, 3)
    fig_multi.add_trace(go.Scatter(x=he.values, y=ho.values, mode="markers",
                                       marker=dict(size=3, color="purple",
                                                    opacity=0.5),
                                       showlegend=False), 2, 1)
    fig_multi.add_trace(go.Histogram(x=pic.values, marker_color="green",
                                         showlegend=False), 2, 2)
    fig_multi.add_trace(go.Histogram(x=fis.dropna().values,
                                         marker_color="teal",
                                         showlegend=False), 2, 3)

    fig_multi.update_layout(height=650, template="plotly_white",
                              title="Comprehensive QC & Diversity Diagnostics")
    st.plotly_chart(fig_multi, use_container_width=True)

    download_plotly_html(fig_multi, "diagnostic_panel.html",
                          key="dl_diag_html")


# ═══════════════════════════════════════════
# TAB 4 — POPULATIONS
# ═══════════════════════════════════════════
with tab_pop:
    st.subheader("🌍 Per-Population Summary")

    if not pop_map:
        st.warning("Metadata configuration required.")
    else:
        rows = []
        for pop in sorted(set(pop_map.values())):
            samples = [s for s in geno.index.astype(str)
                        if pop_map.get(s) == pop]
            if len(samples) < 2:
                continue
            sub = geno.loc[samples]

            maf_p = calc_maf(sub)
            ho_p = calc_het_obs(sub)
            he_p = calc_het_exp(sub)
            pic_p = calc_pic(sub)
            fis_p = calc_fis(sub)
            shannon_p = calc_shannon_diversity(sub)

            rows.append({
                "Population": pop,
                "N_samples": len(samples),
                "N_polymorphic": int((maf_p > 0).sum()),
                "Mean_MAF": maf_p.mean(),
                "Mean_PIC": pic_p.mean(),
                "Mean_Ho": ho_p.mean(),
                "Mean_He": he_p.mean(),
                "Mean_Fis": fis_p.mean(),
                "Mean_Shannon": shannon_p.mean(),
            })

        pop_summary = pd.DataFrame(rows)
        st.dataframe(pop_summary.style.format({
            c: "{:.4f}" for c in pop_summary.columns
            if pop_summary[c].dtype in [np.float64, np.float32]
        }), use_container_width=True)

        # Grouped bar chart
        st.markdown("#### Diversity Metrics by Population")
        fig_pop = px.bar(
            pop_summary.melt(
                id_vars=["Population", "N_samples"],
                value_vars=["Mean_MAF", "Mean_PIC", "Mean_Ho", "Mean_He"]),
            x="Population", y="value", color="variable",
            barmode="group",
            title="Genetic diversity by population",
        )
        fig_pop.update_layout(template="plotly_white", height=500,
                                xaxis_tickangle=45)
        st.plotly_chart(fig_pop, use_container_width=True)

        # Radar chart
        if len(pop_summary) <= 15:
            st.markdown("#### Multi-Metric Radar Comparison")
            radar_metrics = ["Mean_MAF", "Mean_PIC", "Mean_Ho",
                              "Mean_He", "Mean_Shannon"]

            selected_pops = st.multiselect(
                "Select populations to compare",
                pop_summary["Population"].tolist(),
                default=pop_summary["Population"].head(5).tolist(),
                key="rep_radar_pops",
            )

            if selected_pops:
                fig_radar = go.Figure()
                for pop in selected_pops:
                    row = pop_summary[
                        pop_summary["Population"] == pop
                    ].iloc[0]
                    fig_radar.add_trace(go.Scatterpolar(
                        r=[row[m] for m in radar_metrics],
                        theta=radar_metrics,
                        fill="toself",
                        name=str(pop),
                    ))
                fig_radar.update_layout(
                    template="plotly_white", height=600,
                    title="Multi-metric population comparison",
                )
                st.plotly_chart(fig_radar, use_container_width=True)

        # Population heatmap
        st.markdown("#### Population × Metric Heatmap (z-scored)")
        heat_metrics = ["Mean_MAF", "Mean_PIC", "Mean_Ho",
                          "Mean_He", "Mean_Fis", "Mean_Shannon"]
        heat_data = pop_summary.set_index("Population")[heat_metrics]
        heat_z = (heat_data - heat_data.mean()) / heat_data.std()

        fig_heat = px.imshow(
            heat_z.values,
            x=heat_metrics, y=heat_data.index.tolist(),
            text_auto=".2f", color_continuous_scale="RdBu_r",
            aspect="auto",
            title="Population diversity heatmap (z-scored)",
        )
        fig_heat.update_layout(template="plotly_white", height=500)
        st.plotly_chart(fig_heat, use_container_width=True)

        download_dataframe(pop_summary, "per_population_summary.csv",
                            key="dl_pop_summary")


# ═══════════════════════════════════════════
# TAB 5 — CHROMOSOMES
# ═══════════════════════════════════════════
with tab_chr:
    st.subheader("🎨 Chromosome-Level Summary")

    if marker_info is None or "Chrom" not in marker_info.columns:
        st.warning("⚠️ Chromosome information not available.")
    else:
        stats_by_chrom = pd.DataFrame({
            "Marker": geno.columns.astype(str),
            "MAF": maf.values,
            "PIC": pic.values,
            "Ho": ho.values,
            "He": he.values,
            "Fis": fis.values,
            "Shannon": shannon.values,
        }).merge(marker_info[["Marker", "Chrom"]],
                  on="Marker", how="left")

        chr_summary = stats_by_chrom.groupby("Chrom").agg({
            "Marker": "count",
            "MAF": "mean",
            "PIC": "mean",
            "Ho": "mean",
            "He": "mean",
            "Fis": "mean",
            "Shannon": "mean",
        }).reset_index().rename(columns={"Marker": "N_markers"})

        # Sort chromosomes
        chr_sorted = sort_chromosomes(chr_summary["Chrom"].tolist())
        chr_summary["_sort"] = chr_summary["Chrom"].apply(
            lambda x: chr_sorted.index(x) if x in chr_sorted else 999)
        chr_summary = chr_summary.sort_values("_sort").drop(columns=["_sort"])

        st.dataframe(chr_summary.style.format({
            c: "{:.4f}" for c in chr_summary.columns
            if chr_summary[c].dtype in [np.float64, np.float32]
        }), use_container_width=True)

        # Multi-panel
        fig_chr_multi = make_subplots(
            rows=2, cols=2,
            subplot_titles=("N markers per chromosome",
                              "Mean MAF per chromosome",
                              "Mean PIC per chromosome",
                              "Mean He per chromosome"),
        )
        fig_chr_multi.add_trace(go.Bar(x=chr_summary["Chrom"],
                                          y=chr_summary["N_markers"],
                                          marker_color="steelblue",
                                          showlegend=False,
                                          text=chr_summary["N_markers"],
                                          textposition="outside"), 1, 1)
        fig_chr_multi.add_trace(go.Bar(x=chr_summary["Chrom"],
                                          y=chr_summary["MAF"],
                                          marker_color="green",
                                          showlegend=False), 1, 2)
        fig_chr_multi.add_trace(go.Bar(x=chr_summary["Chrom"],
                                          y=chr_summary["PIC"],
                                          marker_color="orange",
                                          showlegend=False), 2, 1)
        fig_chr_multi.add_trace(go.Bar(x=chr_summary["Chrom"],
                                          y=chr_summary["He"],
                                          marker_color="purple",
                                          showlegend=False), 2, 2)

        fig_chr_multi.update_layout(height=700, template="plotly_white",
                                        title="Chromosome-wise diversity")
        st.plotly_chart(fig_chr_multi, use_container_width=True)

        # Z-scored heatmap
        st.markdown("#### Chromosome × Metric Heatmap (z-scored)")
        heat_chr_metrics = ["MAF", "PIC", "Ho", "He", "Fis", "Shannon"]
        heat_chr = chr_summary.set_index("Chrom")[heat_chr_metrics]
        heat_chr_z = (heat_chr - heat_chr.mean()) / heat_chr.std()

        fig_heat_chr = px.imshow(
            heat_chr_z.T.values,
            x=chr_summary["Chrom"].tolist(),
            y=heat_chr_metrics,
            text_auto=".2f", color_continuous_scale="RdBu_r",
            aspect="auto",
            title="Chromosome diversity heatmap (z-scored)",
        )
        fig_heat_chr.update_layout(template="plotly_white", height=500)
        st.plotly_chart(fig_heat_chr, use_container_width=True)

        download_dataframe(chr_summary, "chromosome_summary.csv",
                            key="dl_chr_summary")


# ═══════════════════════════════════════════
# TAB 6 — SAMPLES
# ═══════════════════════════════════════════
with tab_sample:
    st.subheader("👤 Sample-Level QC Report")

    # Compute per-sample stats
    sample_stats = pd.DataFrame({
        "Sample": geno.index.astype(str),
        "Missing_rate": miss_sample.values,
        "N_valid_markers": geno.notna().sum(axis=1).values,
        "Het_rate": ((geno == 1).sum(axis=1).values /
                       np.maximum(geno.notna().sum(axis=1).values, 1)),
        "Hom_ref_rate": ((geno == 0).sum(axis=1).values /
                           np.maximum(geno.notna().sum(axis=1).values, 1)),
        "Hom_alt_rate": ((geno == 2).sum(axis=1).values /
                           np.maximum(geno.notna().sum(axis=1).values, 1)),
    })

    if pop_map:
        sample_stats["Population"] = sample_stats["Sample"].map(pop_map)

    # Add QC flags
    sample_stats["QC_flag"] = "✅ OK"
    sample_stats.loc[sample_stats["Missing_rate"] > 0.3,
                       "QC_flag"] = "⚠️ High missing"

    # Het outliers (IQR)
    q1_h = sample_stats["Het_rate"].quantile(0.25)
    q3_h = sample_stats["Het_rate"].quantile(0.75)
    iqr_h = q3_h - q1_h
    lo_h = q1_h - 3 * iqr_h
    hi_h = q3_h + 3 * iqr_h
    het_outlier_mask = (sample_stats["Het_rate"] < lo_h) | \
                       (sample_stats["Het_rate"] > hi_h)
    sample_stats.loc[het_outlier_mask, "QC_flag"] = "⚠️ Het outlier"

    n_ok = int((sample_stats["QC_flag"] == "✅ OK").sum())
    n_flagged = len(sample_stats) - n_ok

    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Total samples", len(sample_stats))
    sc2.metric("Passing QC", n_ok)
    sc3.metric("Flagged", n_flagged)

    st.dataframe(sample_stats.style.format({
        "Missing_rate": "{:.4f}",
        "Het_rate": "{:.4f}",
        "Hom_ref_rate": "{:.4f}",
        "Hom_alt_rate": "{:.4f}",
    }), use_container_width=True)

    # Scatter plot
    color_col = "Population" if "Population" in sample_stats.columns else "QC_flag"
    fig_samples = px.scatter(
        sample_stats, x="Missing_rate", y="Het_rate",
        color=color_col, hover_data=["Sample", "QC_flag"],
        title="Sample QC: Missing vs Heterozygosity",
    )
    fig_samples.add_hline(y=lo_h, line_dash="dash", line_color="orange")
    fig_samples.add_hline(y=hi_h, line_dash="dash", line_color="orange")
    fig_samples.add_vline(x=0.3, line_dash="dash", line_color="red")
    fig_samples.update_traces(marker=dict(size=8, opacity=0.7,
                                             line=dict(width=0.5,
                                                        color="darkslategrey")))
    fig_samples.update_layout(template="plotly_white", height=550)
    st.plotly_chart(fig_samples, use_container_width=True)

    download_dataframe(sample_stats, "sample_qc_report.csv",
                        key="dl_sample_qc")


# ═══════════════════════════════════════════
# TAB 7 — EXPORT REPORTS
# ═══════════════════════════════════════════
with tab_export:
    st.subheader("💾 Export Comprehensive Reports")

    # ─── Markdown report ───
    st.markdown("### 📝 Markdown Report")

    interpretation_diversity = (
        'high' if he.mean() > 0.3
        else 'moderate' if he.mean() > 0.15
        else 'low'
    )
    interpretation_fis = (
        'excess heterozygotes' if fis.mean() < -0.05
        else 'inbreeding tendency' if fis.mean() > 0.05
        else 'Hardy-Weinberg near equilibrium'
    )

    report_md = f"""
# Population Genomics Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 📊 Dataset Overview

| Metric | Value |
|--------|-------|
| Samples | {geno.shape[0]:,} |
| Markers | {geno.shape[1]:,} |
| Overall missing rate | {missing_pct:.2f}% |
| Metadata | {'Loaded' if meta is not None else 'Not loaded'} |
| Populations | {len(set(pop_map.values())) if pop_map else 'N/A'} |
| Chromosomes | {marker_info['Chrom'].nunique() if marker_info is not None and 'Chrom' in marker_info.columns else 'N/A'} |

## 🧬 Genome-Wide Statistics

| Statistic | Mean | Median | Std |
|-----------|------|--------|-----|
| MAF | {maf.mean():.4f} | {maf.median():.4f} | {maf.std():.4f} |
| PIC | {pic.mean():.4f} | {pic.median():.4f} | {pic.std():.4f} |
| Ho | {ho.mean():.4f} | {ho.median():.4f} | {ho.std():.4f} |
| He | {he.mean():.4f} | {he.median():.4f} | {he.std():.4f} |
| Fis | {fis.mean():.4f} | {fis.median():.4f} | {fis.std():.4f} |
| Shannon I | {shannon.mean():.4f} | {shannon.median():.4f} | {shannon.std():.4f} |

## 🎯 Quality Control Summary

- Markers with MAF < 0.05: **{int((maf < 0.05).sum()):,}** ({(maf < 0.05).mean()*100:.1f}%)
- Markers with MAF < 0.01: **{int((maf < 0.01).sum()):,}** ({(maf < 0.01).mean()*100:.1f}%)
- Markers with missing > 20%: **{int((miss_marker > 0.2).sum()):,}** ({(miss_marker > 0.2).mean()*100:.1f}%)
- Samples with missing > 30%: **{int((miss_sample > 0.3).sum()):,}** ({(miss_sample > 0.3).mean()*100:.1f}%)
- Monomorphic markers: **{int((geno.nunique() <= 1).sum()):,}**

## 🎨 Interpretation

- Mean expected heterozygosity of **{he.mean():.4f}** suggests **{interpretation_diversity}** genetic diversity.
- Mean Fis of **{fis.mean():.4f}** indicates **{interpretation_fis}**.
- Mean PIC of **{pic.mean():.4f}** indicates {'highly' if pic.mean() > 0.5 else 'moderately' if pic.mean() > 0.25 else 'poorly'} informative markers.

## 💡 Recommendations

"""

    if missing_pct > 10:
        report_md += "- ⚠️ **Apply QC filtering** — missing data > 10%\n"
    if (maf < 0.05).mean() > 0.2:
        report_md += "- ⚠️ **Apply MAF filter** — >20% markers have MAF < 0.05\n"
    if abs(fis.mean()) > 0.1:
        report_md += f"- ⚠️ **Investigate HWE deviation** — Fis = {fis.mean():.3f}\n"
    if geno.shape[0] < 30:
        report_md += "- ⚠️ **Small sample size** — consider expanding cohort\n"

    if not any(x in report_md for x in ["Apply QC", "Apply MAF",
                                          "Investigate HWE", "Small sample"]):
        report_md += "- ✅ **Data quality is good** — all metrics within acceptable ranges\n"

    # Add population section if metadata
    if pop_map:
        n_pops = len(set(pop_map.values()))
        pop_sizes = pd.Series(list(pop_map.values())).value_counts()
        report_md += f"""
## 🌍 Population Analysis

- Number of populations: **{n_pops}**
- Largest population: **{pop_sizes.index[0]}** (n={pop_sizes.iloc[0]})
- Smallest population: **{pop_sizes.index[-1]}** (n={pop_sizes.iloc[-1]})
- Mean samples per population: **{pop_sizes.mean():.1f}**
"""

    report_md += """
---

*Report generated by Interactive Population Genomics Platform.*
"""

    st.markdown(report_md)

    ec1, ec2, ec3 = st.columns(3)

    with ec1:
        st.download_button(
            "📥 Download Markdown",
            report_md,
            "population_genomics_report.md",
            "text/markdown",
            key="dl_report_md",
        )

    # ─── HTML report ───
    with ec2:
        html_report = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Population Genomics Report</title>
<style>
body {{ font-family: Arial, sans-serif; max-width: 1000px;
        margin: 30px auto; padding: 20px;
        background: #f8f9fa; color: #333; line-height: 1.6; }}
h1 {{ color: #1E88E5; border-bottom: 3px solid #1E88E5;
     padding-bottom: 10px; }}
h2 {{ color: #43A047; border-left: 5px solid #43A047;
     padding-left: 12px; margin-top: 30px; }}
h3 {{ color: #FB8C00; margin-top: 20px; }}
table {{ border-collapse: collapse; width: 100%; margin: 15px 0;
         background: white; }}
th, td {{ border: 1px solid #ddd; padding: 10px 12px; text-align: left; }}
th {{ background: linear-gradient(90deg, #1E88E5, #43A047);
     color: white; font-weight: bold; }}
tr:nth-child(even) {{ background: #f2f2f2; }}
.metric-box {{ background: white; padding: 15px 20px;
              border-left: 5px solid #1E88E5; margin: 10px 0;
              border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.warning {{ border-left-color: #FB8C00; }}
.success {{ border-left-color: #43A047; }}
.error {{ border-left-color: #E53935; }}
code {{ background: #eee; padding: 2px 6px; border-radius: 3px;
       font-family: monospace; }}
.footer {{ text-align: center; color: #888; margin-top: 40px;
          padding: 15px; border-top: 1px solid #ddd; }}
</style>
</head>
<body>
<h1>🧬 Population Genomics Analysis Report</h1>
<p><em>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>

<h2>📊 Dataset Overview</h2>
<div class="metric-box success">
<strong>Samples:</strong> {geno.shape[0]:,} |
<strong>Markers:</strong> {geno.shape[1]:,} |
<strong>Missing rate:</strong> {missing_pct:.2f}%
</div>

<h2>🧬 Genome-wide Statistics</h2>
<table>
<tr><th>Statistic</th><th>Mean</th><th>Median</th><th>Std</th><th>Min</th><th>Max</th></tr>
<tr><td>MAF</td><td>{maf.mean():.4f}</td><td>{maf.median():.4f}</td><td>{maf.std():.4f}</td><td>{maf.min():.4f}</td><td>{maf.max():.4f}</td></tr>
<tr><td>PIC</td><td>{pic.mean():.4f}</td><td>{pic.median():.4f}</td><td>{pic.std():.4f}</td><td>{pic.min():.4f}</td><td>{pic.max():.4f}</td></tr>
<tr><td>Ho</td><td>{ho.mean():.4f}</td><td>{ho.median():.4f}</td><td>{ho.std():.4f}</td><td>{ho.min():.4f}</td><td>{ho.max():.4f}</td></tr>
<tr><td>He</td><td>{he.mean():.4f}</td><td>{he.median():.4f}</td><td>{he.std():.4f}</td><td>{he.min():.4f}</td><td>{he.max():.4f}</td></tr>
<tr><td>Fis</td><td>{fis.mean():.4f}</td><td>{fis.median():.4f}</td><td>{fis.std():.4f}</td><td>{fis.min():.4f}</td><td>{fis.max():.4f}</td></tr>
<tr><td>Shannon I</td><td>{shannon.mean():.4f}</td><td>{shannon.median():.4f}</td><td>{shannon.std():.4f}</td><td>{shannon.min():.4f}</td><td>{shannon.max():.4f}</td></tr>
</table>

<h2>🎯 Quality Control</h2>
<table>
<tr><th>Metric</th><th>Value</th><th>Percentage</th></tr>
<tr><td>MAF &lt; 0.05</td><td>{int((maf < 0.05).sum()):,}</td><td>{(maf < 0.05).mean()*100:.1f}%</td></tr>
<tr><td>MAF &lt; 0.01</td><td>{int((maf < 0.01).sum()):,}</td><td>{(maf < 0.01).mean()*100:.1f}%</td></tr>
<tr><td>Missing marker &gt; 20%</td><td>{int((miss_marker > 0.2).sum()):,}</td><td>{(miss_marker > 0.2).mean()*100:.1f}%</td></tr>
<tr><td>Missing sample &gt; 30%</td><td>{int((miss_sample > 0.3).sum()):,}</td><td>{(miss_sample > 0.3).mean()*100:.1f}%</td></tr>
<tr><td>Monomorphic markers</td><td>{int((geno.nunique() <= 1).sum()):,}</td><td>{(geno.nunique() <= 1).mean()*100:.1f}%</td></tr>
</table>

<h2>💡 Interpretation</h2>
<div class="metric-box">
<p><strong>Genetic diversity (He = {he.mean():.4f}):</strong> {interpretation_diversity}</p>
<p><strong>HWE (Fis = {fis.mean():.4f}):</strong> {interpretation_fis}</p>
<p><strong>Marker informativeness (PIC = {pic.mean():.4f}):</strong>
{'highly informative' if pic.mean() > 0.5 else 'moderately informative' if pic.mean() > 0.25 else 'less informative'}</p>
</div>

<div class="footer">
🧬 Interactive Population Genomics Platform
</div>

</body>
</html>
"""

        st.download_button(
            "📥 Download HTML",
            html_report,
            "population_genomics_report.html",
            "text/html",
            key="dl_report_html",
        )

    # ─── Excel report ───
    with ec3:
        excel_sheets = {
            "Statistical_Summary": summary,
            "Per_Marker_Stats": pd.DataFrame({
                "Marker": geno.columns,
                "MAF": maf.values,
                "PIC": pic.values,
                "Ho": ho.values,
                "He": he.values,
                "Fis": fis.values,
                "Missing_rate": miss_marker.values,
            }),
        }

        if pop_map:
            pop_summary_local = pd.DataFrame([
                {"Population": pop,
                  "N_samples": len([s for s in geno.index.astype(str)
                                     if pop_map.get(s) == pop])}
                for pop in sorted(set(pop_map.values()))
            ])
            excel_sheets["Per_Population_Stats"] = pop_summary_local

        download_excel(excel_sheets,
                        "genomics_report.xlsx",
                        label="📥 Download Excel",
                        key="dl_report_xlsx")

    st.info(
        "💡 **Tip:** Use the **💾 Export Results** page to download raw data "
        "in genotype formats (Numeric, HapMap, VCF, PED) or a full analysis bundle."
    )