"""
SNP Statistics — Publication Quality
────────────────────────────────────
Comprehensive per-marker and per-population statistics:
  - Allele frequencies (p, q, MAF)
  - Diversity indices (PIC, Ho, He, Fis, Shannon, Ne)
  - Per-chromosome breakdown with heatmap
  - Per-population statistics with comparison
  - Interactive scatter plots
  - Marker filtering & search
  - Correlation matrix of metrics
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import (
    get_geno_from_session, get_meta_from_session,
    calc_allele_freq, calc_maf, calc_pic,
    calc_het_obs, calc_het_exp, calc_fis,
    calc_shannon_diversity, calc_effective_alleles,
    download_plotly_html, download_dataframe,
    sort_chromosomes, build_sample_pop_map,
)

st.title("🧬 SNP Statistics")
st.markdown("---")

geno, marker_info = get_geno_from_session()
if geno is None:
    st.stop()

meta = get_meta_from_session()


# ═══════════════════════════════════════════
# GLOBAL: Metadata configuration
# ═══════════════════════════════════════════
pop_map = {}
sample_col = None
pop_col = None

if meta is not None:
    with st.expander("🔧 Metadata Configuration (for per-population analyses)",
                     expanded=True):
        mc1, mc2 = st.columns(2)
        with mc1:
            sample_col = st.selectbox("Sample ID column",
                                        meta.columns.tolist(),
                                        key="snp_samcol")
        with mc2:
            pop_col_opt = st.selectbox("Population column",
                                        ["None"] + meta.columns.tolist(),
                                        key="snp_popcol")
            pop_col = None if pop_col_opt == "None" else pop_col_opt

        if pop_col:
            pop_map = build_sample_pop_map(meta, sample_col, pop_col)
            st.success(f"✅ Loaded {len(set(pop_map.values()))} populations")


# ═══════════════════════════════════════════
# PRE-COMPUTE ALL STATISTICS
# ═══════════════════════════════════════════
with st.spinner("Computing SNP statistics..."):
    p, q = calc_allele_freq(geno)
    maf = calc_maf(geno)
    pic = calc_pic(geno)
    ho = calc_het_obs(geno)
    he = calc_het_exp(geno)
    fis = calc_fis(geno)
    shannon = calc_shannon_diversity(geno)
    ne = calc_effective_alleles(geno)

    stats_df = pd.DataFrame({
        "Marker": geno.columns,
        "Freq_Ref_p": p.values,
        "Freq_Alt_q": q.values,
        "MAF": maf.values,
        "PIC": pic.values,
        "Ho": ho.values,
        "He": he.values,
        "Fis": fis.values,
        "Shannon_I": shannon.values,
        "Ne_alleles": ne.values,
    })

    if marker_info is not None:
        stats_df = stats_df.merge(marker_info, on="Marker", how="left")


# ═══════════════════════════════════════════
# GENOME-WIDE SUMMARY
# ═══════════════════════════════════════════
st.subheader("📈 Genome-wide Summary")

sc1, sc2, sc3, sc4, sc5, sc6 = st.columns(6)
sc1.metric("Mean MAF", f"{maf.mean():.4f}")
sc2.metric("Mean PIC", f"{pic.mean():.4f}")
sc3.metric("Mean Ho", f"{ho.mean():.4f}")
sc4.metric("Mean He", f"{he.mean():.4f}")
sc5.metric("Mean Fis", f"{fis.mean():.4f}")
sc6.metric("Mean Shannon I", f"{shannon.mean():.4f}")

# Additional counts
n_polymorphic = int((maf > 0).sum())
n_high_maf = int((maf >= 0.05).sum())
n_high_pic = int((pic >= 0.375).sum())  # PIC > 0.375 = "highly informative"

ss1, ss2, ss3, ss4 = st.columns(4)
ss1.metric("Total markers", f"{len(stats_df):,}")
ss2.metric("Polymorphic (MAF>0)", f"{n_polymorphic:,}")
ss3.metric("Common (MAF≥0.05)", f"{n_high_maf:,}")
ss4.metric("Highly informative (PIC≥0.375)", f"{n_high_pic:,}")


# ═══════════════════════════════════════════
# MULTI-PANEL DIAGNOSTIC
# ═══════════════════════════════════════════
st.markdown("---")
st.subheader("📊 Multi-Metric Diagnostic Panel")

fig_multi = make_subplots(
    rows=2, cols=3,
    subplot_titles=(
        "MAF distribution", "PIC distribution",
        "Observed heterozygosity (Ho)",
        "Expected heterozygosity (He)",
        "Fis distribution", "Ho vs He",
    ),
)

fig_multi.add_trace(go.Histogram(x=maf.values, marker_color="steelblue",
                                     showlegend=False), 1, 1)
fig_multi.add_trace(go.Histogram(x=pic.values, marker_color="green",
                                     showlegend=False), 1, 2)
fig_multi.add_trace(go.Histogram(x=ho.dropna().values,
                                     marker_color="purple",
                                     showlegend=False), 1, 3)
fig_multi.add_trace(go.Histogram(x=he.values, marker_color="orange",
                                     showlegend=False), 2, 1)
fig_multi.add_trace(go.Histogram(x=fis.dropna().values,
                                     marker_color="red",
                                     showlegend=False), 2, 2)
fig_multi.add_trace(go.Scatter(x=he.values, y=ho.values, mode="markers",
                                   marker=dict(size=3, color="teal",
                                                opacity=0.5),
                                   showlegend=False), 2, 3)

fig_multi.update_layout(height=650, template="plotly_white",
                          title="Comprehensive SNP statistics overview")
st.plotly_chart(fig_multi, use_container_width=True)


# ═══════════════════════════════════════════
# TABS FOR DETAILED VIEWS
# ═══════════════════════════════════════════
tab_table, tab_dist, tab_scatter, tab_chr, tab_pop = st.tabs([
    "📋 Statistics Table",
    "📊 Distributions",
    "🔀 Interactive Scatter",
    "🎨 Per-Chromosome",
    "🌍 Per-Population",
])


# ═══════════════════════════════════════════
# TAB 1 — Statistics Table (with filtering)
# ═══════════════════════════════════════════
with tab_table:
    st.subheader("📋 Per-Marker Statistics Table")

    # Filtering
    with st.expander("🔍 Filter markers", expanded=False):
        fc1, fc2 = st.columns(2)
        with fc1:
            maf_range = st.slider("MAF range", 0.0, 0.5,
                                     (0.0, 0.5), 0.01, key="tbl_maf")
            pic_range = st.slider("PIC range", 0.0, 0.5,
                                     (0.0, 0.5), 0.01, key="tbl_pic")
        with fc2:
            he_range = st.slider("He range", 0.0, 0.6,
                                    (0.0, 0.6), 0.01, key="tbl_he")
            search_marker = st.text_input(
                "Search marker name (partial match)", key="tbl_search")

        # Column selector
        available_cols = stats_df.columns.tolist()
        default_cols = ["Marker", "MAF", "PIC", "Ho", "He", "Fis"]
        if "Chrom" in available_cols:
            default_cols = ["Marker", "Chrom", "Pos"] + default_cols[1:]

        show_cols = st.multiselect(
            "Columns to display",
            available_cols,
            default=[c for c in default_cols if c in available_cols],
            key="tbl_cols",
        )

    # Apply filters
    filtered_df = stats_df[
        (stats_df["MAF"] >= maf_range[0]) &
        (stats_df["MAF"] <= maf_range[1]) &
        (stats_df["PIC"] >= pic_range[0]) &
        (stats_df["PIC"] <= pic_range[1]) &
        (stats_df["He"] >= he_range[0]) &
        (stats_df["He"] <= he_range[1])
    ]

    if search_marker:
        filtered_df = filtered_df[
            filtered_df["Marker"].astype(str).str.contains(
                search_marker, case=False, na=False)
        ]

    st.info(f"Showing {len(filtered_df):,} of {len(stats_df):,} markers "
             f"after filtering")

    if show_cols:
        display_df = filtered_df[show_cols].head(500)
    else:
        display_df = filtered_df.head(500)

    # Format numeric columns
    format_dict = {c: "{:.4f}" for c in display_df.columns
                    if display_df[c].dtype in [np.float64, np.float32]}
    st.dataframe(display_df.style.format(format_dict),
                  use_container_width=True)

    dc1, dc2 = st.columns(2)
    with dc1:
        download_dataframe(filtered_df, "snp_statistics_filtered.csv",
                            key="dl_snp_filt")
    with dc2:
        download_dataframe(stats_df, "snp_statistics_full.csv",
                            key="dl_snp_full")


# ═══════════════════════════════════════════
# TAB 2 — Distributions with Statistics
# ═══════════════════════════════════════════
with tab_dist:
    st.subheader("📊 Distribution Analysis")

    metric_choices = ["MAF", "PIC", "Ho", "He", "Fis",
                       "Shannon_I", "Ne_alleles",
                       "Freq_Ref_p", "Freq_Alt_q"]

    dc1, dc2 = st.columns(2)
    with dc1:
        dist_metric = st.selectbox(
            "Select statistic to visualize",
            metric_choices,
            key="snp_dist_metric",
        )
    with dc2:
        n_bins = st.slider("Number of bins", 20, 100, 50, 5,
                             key="snp_dist_bins")

    values = stats_df[dist_metric].dropna()

    # Distribution plot with box
    fig_dist = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=(f"Distribution of {dist_metric}",
                          "Box plot"),
        vertical_spacing=0.1,
    )

    fig_dist.add_trace(
        go.Histogram(x=values, nbinsx=n_bins,
                       marker_color="steelblue",
                       showlegend=False), 1, 1,
    )
    fig_dist.add_trace(
        go.Box(x=values, marker_color="steelblue",
                showlegend=False,
                boxmean="sd"), 2, 1,
    )

    # Add mean and median lines
    mean_val = values.mean()
    median_val = values.median()
    fig_dist.add_vline(x=mean_val, line_dash="dash", line_color="red",
                         annotation_text=f"Mean = {mean_val:.4f}",
                         row=1, col=1)
    fig_dist.add_vline(x=median_val, line_dash="dot", line_color="green",
                         annotation_text=f"Median = {median_val:.4f}",
                         row=1, col=1)

    fig_dist.update_layout(template="plotly_white", height=600)
    st.plotly_chart(fig_dist, use_container_width=True)

    # Descriptive statistics
    st.markdown(f"#### Descriptive Statistics for {dist_metric}")
    desc_stats = pd.DataFrame({
        "Statistic": ["N", "Mean", "Std", "Min",
                        "Q1 (25%)", "Median", "Q3 (75%)", "Max",
                        "Skewness", "Kurtosis"],
        "Value": [
            f"{len(values):,}",
            f"{values.mean():.6f}",
            f"{values.std():.6f}",
            f"{values.min():.6f}",
            f"{values.quantile(0.25):.6f}",
            f"{values.median():.6f}",
            f"{values.quantile(0.75):.6f}",
            f"{values.max():.6f}",
            f"{values.skew():.4f}",
            f"{values.kurtosis():.4f}",
        ],
    })
    st.dataframe(desc_stats, use_container_width=True)

    # Top / Bottom markers
    st.markdown("#### Top 10 Highest & Lowest Markers")
    tc1, tc2 = st.columns(2)
    with tc1:
        st.markdown(f"**Top 10 highest {dist_metric}**")
        top10 = stats_df.nlargest(10, dist_metric)[
            ["Marker", dist_metric]
        ]
        st.dataframe(top10.style.format({dist_metric: "{:.4f}"}),
                      use_container_width=True)
    with tc2:
        st.markdown(f"**Top 10 lowest {dist_metric}**")
        bot10 = stats_df.nsmallest(10, dist_metric)[
            ["Marker", dist_metric]
        ]
        st.dataframe(bot10.style.format({dist_metric: "{:.4f}"}),
                      use_container_width=True)


# ═══════════════════════════════════════════
# TAB 3 — Interactive Scatter Plots
# ═══════════════════════════════════════════
with tab_scatter:
    st.subheader("🔀 Interactive Scatter Analysis")

    numeric_cols = ["MAF", "PIC", "Ho", "He", "Fis",
                     "Shannon_I", "Ne_alleles",
                     "Freq_Ref_p", "Freq_Alt_q"]

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        x_metric = st.selectbox("X-axis", numeric_cols, index=0,
                                  key="scatter_x")
    with sc2:
        y_metric = st.selectbox("Y-axis", numeric_cols, index=3,
                                  key="scatter_y")
    with sc3:
        color_by = st.selectbox("Color by",
                                  ["None"] + numeric_cols,
                                  index=numeric_cols.index("PIC") + 1
                                  if "PIC" in numeric_cols else 0,
                                  key="scatter_color")

    scatter_df = stats_df.dropna(subset=[x_metric, y_metric])

    hover_data = ["Marker"]
    if "Chrom" in scatter_df.columns:
        hover_data.append("Chrom")

    if color_by == "None":
        fig_scatter = px.scatter(
            scatter_df, x=x_metric, y=y_metric,
            hover_data=hover_data, opacity=0.6,
        )
    else:
        fig_scatter = px.scatter(
            scatter_df, x=x_metric, y=y_metric,
            color=color_by,
            hover_data=hover_data, opacity=0.6,
            color_continuous_scale="Viridis",
        )

    # Add diagonal for He vs Ho
    if (x_metric == "He" and y_metric == "Ho") or \
        (x_metric == "Ho" and y_metric == "He"):
        max_val = max(scatter_df[x_metric].max(),
                       scatter_df[y_metric].max())
        fig_scatter.add_shape(type="line",
                                x0=0, y0=0, x1=max_val, y1=max_val,
                                line=dict(color="red", dash="dash", width=2))

    fig_scatter.update_traces(marker=dict(size=6,
                                             line=dict(width=0.3,
                                                        color="darkslategrey")))
    fig_scatter.update_layout(
        title=f"{y_metric} vs {x_metric}",
        template="plotly_white", height=600,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Correlation
    corr_val = scatter_df[[x_metric, y_metric]].corr().iloc[0, 1]
    st.metric(f"Pearson correlation ({x_metric} vs {y_metric})",
                f"{corr_val:.4f}")

    # Correlation matrix
    st.markdown("#### Correlation Matrix of All Metrics")
    corr_mat = stats_df[numeric_cols].corr()

    fig_corr = px.imshow(
        corr_mat, text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="Correlation matrix of SNP statistics",
        aspect="auto",
        zmin=-1, zmax=1,
    )
    fig_corr.update_layout(template="plotly_white", height=600)
    st.plotly_chart(fig_corr, use_container_width=True)


# ═══════════════════════════════════════════
# TAB 4 — Per-Chromosome Analysis
# ═══════════════════════════════════════════
with tab_chr:
    st.subheader("🎨 Per-Chromosome Statistics")

    if marker_info is None or "Chrom" not in stats_df.columns:
        st.warning("⚠️ Chromosome information not available.")
    else:
        # Aggregate per chromosome
        per_chrom = stats_df.groupby("Chrom").agg({
            "Marker": "count",
            "MAF": ["mean", "median", "std"],
            "PIC": ["mean", "median"],
            "Ho": "mean",
            "He": "mean",
            "Fis": "mean",
            "Shannon_I": "mean",
        }).reset_index()

        # Flatten column names
        per_chrom.columns = [
            "_".join(col).rstrip("_") if isinstance(col, tuple) else col
            for col in per_chrom.columns
        ]
        per_chrom = per_chrom.rename(columns={"Marker_count": "N_markers"})

        # Sort chromosomes
        chr_sorted = sort_chromosomes(per_chrom["Chrom"].tolist())
        per_chrom["_sort"] = per_chrom["Chrom"].apply(
            lambda x: chr_sorted.index(x) if x in chr_sorted else 999)
        per_chrom = per_chrom.sort_values("_sort").drop(columns=["_sort"])

        # Display table
        st.dataframe(per_chrom.style.format({
            c: "{:.4f}" for c in per_chrom.columns
            if per_chrom[c].dtype in [np.float64, np.float32]
        }), use_container_width=True)

        download_dataframe(per_chrom, "per_chromosome_stats.csv",
                            key="dl_chr_stats")

        # Multi-panel plot
        fig_chr_multi = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "N markers per chromosome",
                "Mean MAF per chromosome",
                "Mean PIC per chromosome",
                "Mean He per chromosome",
            ),
        )
        fig_chr_multi.add_trace(go.Bar(
            x=per_chrom["Chrom"], y=per_chrom["N_markers"],
            marker_color="steelblue", showlegend=False,
            text=per_chrom["N_markers"], textposition="outside",
        ), 1, 1)
        fig_chr_multi.add_trace(go.Bar(
            x=per_chrom["Chrom"], y=per_chrom["MAF_mean"],
            marker_color="green", showlegend=False,
        ), 1, 2)
        fig_chr_multi.add_trace(go.Bar(
            x=per_chrom["Chrom"], y=per_chrom["PIC_mean"],
            marker_color="orange", showlegend=False,
        ), 2, 1)
        fig_chr_multi.add_trace(go.Bar(
            x=per_chrom["Chrom"], y=per_chrom["He_mean"],
            marker_color="purple", showlegend=False,
        ), 2, 2)

        fig_chr_multi.update_layout(height=700, template="plotly_white",
                                       title="Chromosome-wise statistics")
        st.plotly_chart(fig_chr_multi, use_container_width=True)

        # Heatmap of stats × chromosome
        st.markdown("#### Heatmap: Chromosomes × Metrics (z-scored)")
        heat_metrics = ["MAF_mean", "PIC_mean", "Ho_mean",
                          "He_mean", "Fis_mean", "Shannon_I_mean"]
        heat_data = per_chrom[heat_metrics].copy()
        # Z-score standardize columns
        heat_z = (heat_data - heat_data.mean()) / heat_data.std()

        fig_heat_chr = px.imshow(
            heat_z.T.values,
            x=per_chrom["Chrom"].tolist(),
            y=heat_metrics,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            aspect="auto",
            title="Chromosome-wise metrics (z-scored)",
        )
        fig_heat_chr.update_layout(template="plotly_white", height=500)
        st.plotly_chart(fig_heat_chr, use_container_width=True)


# ═══════════════════════════════════════════
# TAB 5 — Per-Population Analysis
# ═══════════════════════════════════════════
with tab_pop:
    st.subheader("🌍 Per-Population Statistics")

    if not pop_map:
        st.warning(
            "⚠️ Please configure metadata (Sample ID + Population columns) "
            "at the top of this page to enable per-population analysis."
        )
    else:
        with st.spinner("Computing per-population statistics..."):
            pop_stats = []
            for pop in sorted(set(pop_map.values())):
                samples_in_pop = [s for s in geno.index.astype(str)
                                    if pop_map.get(s) == pop]
                if len(samples_in_pop) < 2:
                    continue

                sub = geno.loc[samples_in_pop]
                maf_pop = calc_maf(sub)
                ho_pop = calc_het_obs(sub)
                he_pop = calc_het_exp(sub)
                fis_pop = calc_fis(sub)
                pic_pop = calc_pic(sub)
                shannon_pop = calc_shannon_diversity(sub)
                ne_pop = calc_effective_alleles(sub)

                pop_stats.append({
                    "Population": pop,
                    "N_samples": len(samples_in_pop),
                    "N_polymorphic": int((maf_pop > 0).sum()),
                    "Mean_MAF": maf_pop.mean(),
                    "Mean_PIC": pic_pop.mean(),
                    "Mean_Ho": ho_pop.mean(),
                    "Mean_He": he_pop.mean(),
                    "Mean_Fis": fis_pop.mean(),
                    "Mean_Shannon": shannon_pop.mean(),
                    "Mean_Ne": ne_pop.mean(),
                })

        pop_stats_df = pd.DataFrame(pop_stats)
        st.dataframe(pop_stats_df.style.format({
            c: "{:.4f}" for c in pop_stats_df.columns
            if pop_stats_df[c].dtype in [np.float64, np.float32]
        }), use_container_width=True)

        download_dataframe(pop_stats_df, "per_population_stats.csv",
                            key="dl_pop_stats")

        # Bar plots
        st.markdown("#### Population Comparison — Bar Charts")

        pop_metric = st.selectbox(
            "Metric to compare",
            ["Mean_MAF", "Mean_PIC", "Mean_Ho", "Mean_He",
             "Mean_Fis", "Mean_Shannon", "Mean_Ne",
             "N_polymorphic", "N_samples"],
            key="pop_metric",
        )

        fig_pop_bar = px.bar(
            pop_stats_df.sort_values(pop_metric, ascending=False),
            x="Population", y=pop_metric,
            color=pop_metric,
            color_continuous_scale="Viridis",
            title=f"{pop_metric} across populations",
            text=pop_stats_df[pop_metric].round(4),
        )
        fig_pop_bar.update_traces(textposition="outside")
        fig_pop_bar.update_layout(template="plotly_white", height=500,
                                     xaxis_tickangle=45)
        st.plotly_chart(fig_pop_bar, use_container_width=True)

        # Grouped bar
        st.markdown("#### Multi-Metric Grouped Comparison")
        melted = pop_stats_df.melt(
            id_vars="Population",
            value_vars=["Mean_MAF", "Mean_PIC", "Mean_Ho", "Mean_He"],
            var_name="Metric", value_name="Value",
        )
        fig_grouped = px.bar(
            melted, x="Population", y="Value", color="Metric",
            barmode="group",
            title="Diversity indices by population",
        )
        fig_grouped.update_layout(template="plotly_white", height=500,
                                     xaxis_tickangle=45)
        st.plotly_chart(fig_grouped, use_container_width=True)

        # Radar chart
        if len(pop_stats_df) <= 15:
            st.markdown("#### Multi-Metric Radar Comparison")
            radar_metrics = ["Mean_MAF", "Mean_PIC", "Mean_Ho",
                              "Mean_He", "Mean_Shannon", "Mean_Ne"]

            selected_pops_radar = st.multiselect(
                "Select populations to compare",
                pop_stats_df["Population"].tolist(),
                default=pop_stats_df["Population"].head(5).tolist(),
                key="radar_pops",
            )

            if selected_pops_radar:
                fig_radar = go.Figure()
                for pop in selected_pops_radar:
                    row = pop_stats_df[
                        pop_stats_df["Population"] == pop
                    ].iloc[0]
                    fig_radar.add_trace(go.Scatterpolar(
                        r=[row[m] for m in radar_metrics],
                        theta=radar_metrics,
                        fill="toself",
                        name=str(pop),
                    ))
                fig_radar.update_layout(
                    template="plotly_white",
                    height=600,
                    title="Multi-metric population comparison",
                )
                st.plotly_chart(fig_radar, use_container_width=True)

        # Heatmap of populations × metrics
        st.markdown("#### Heatmap: Populations × Metrics (z-scored)")
        heat_metrics_pop = ["Mean_MAF", "Mean_PIC", "Mean_Ho",
                              "Mean_He", "Mean_Fis", "Mean_Shannon",
                              "Mean_Ne"]
        heat_data_pop = pop_stats_df.set_index("Population")[heat_metrics_pop]
        heat_z_pop = ((heat_data_pop - heat_data_pop.mean()) /
                       heat_data_pop.std())

        fig_heat_pop = px.imshow(
            heat_z_pop.values,
            x=heat_metrics_pop,
            y=heat_data_pop.index.tolist(),
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            aspect="auto",
            title="Population diversity indices (z-scored)",
        )
        fig_heat_pop.update_layout(template="plotly_white", height=500)
        st.plotly_chart(fig_heat_pop, use_container_width=True)