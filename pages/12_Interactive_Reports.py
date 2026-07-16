"""Interactive Reports — Comprehensive summary of all analyses."""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import (
    get_geno_from_session, get_meta_from_session,
    calc_allele_freq, calc_maf, calc_het_obs, calc_het_exp, calc_pic,
    calc_fis, calc_missing_rate,
)

st.title("📑 Interactive Reports")
st.markdown("---")

geno, marker_info = get_geno_from_session()
if geno is None:
    st.stop()

meta = get_meta_from_session()

# =========================================================
# Header summary
# =========================================================
st.header("📊 Dataset Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Samples", geno.shape[0])
c2.metric("Markers", geno.shape[1])
c3.metric("Overall missing", f"{geno.isna().sum().sum() / geno.size * 100:.2f}%")
if meta is not None:
    c4.metric("Metadata rows", meta.shape[0])
else:
    c4.metric("Metadata", "Not loaded")

# =========================================================
# SNP Statistics Summary
# =========================================================
st.header("🧬 SNP Statistics Summary")

with st.spinner("Computing summary statistics..."):
    maf = calc_maf(geno)
    ho = calc_het_obs(geno)
    he = calc_het_exp(geno)
    pic = calc_pic(geno)
    fis = calc_fis(geno)
    miss_marker = calc_missing_rate(geno, axis=0)
    miss_sample = calc_missing_rate(geno, axis=1)

summary = pd.DataFrame({
    "Statistic": ["MAF", "Ho", "He", "PIC", "Fis",
                  "Marker missing rate", "Sample missing rate"],
    "Mean": [maf.mean(), ho.mean(), he.mean(), pic.mean(),
             fis.mean(), miss_marker.mean(), miss_sample.mean()],
    "Median": [maf.median(), ho.median(), he.median(), pic.median(),
               fis.median(), miss_marker.median(), miss_sample.median()],
    "Std": [maf.std(), ho.std(), he.std(), pic.std(),
            fis.std(), miss_marker.std(), miss_sample.std()],
    "Min": [maf.min(), ho.min(), he.min(), pic.min(),
            fis.min(), miss_marker.min(), miss_sample.min()],
    "Max": [maf.max(), ho.max(), he.max(), pic.max(),
            fis.max(), miss_marker.max(), miss_sample.max()],
})
st.dataframe(summary.style.format({
    "Mean": "{:.4f}", "Median": "{:.4f}", "Std": "{:.4f}",
    "Min": "{:.4f}", "Max": "{:.4f}"
}), use_container_width=True)

# =========================================================
# Multi-panel diagnostic figure
# =========================================================
st.header("📈 Comprehensive Diagnostic Panel")

fig_multi = make_subplots(
    rows=2, cols=3,
    subplot_titles=(
        "MAF distribution", "Marker missing rate",
        "Sample missing rate", "Ho vs He", "PIC distribution",
        "Fis distribution",
    ),
)

fig_multi.add_trace(go.Histogram(x=maf.values, marker_color="steelblue",
                                    name="MAF", showlegend=False), 1, 1)
fig_multi.add_trace(go.Histogram(x=miss_marker.values, marker_color="orange",
                                    name="Marker miss", showlegend=False), 1, 2)
fig_multi.add_trace(go.Histogram(x=miss_sample.values, marker_color="red",
                                    name="Sample miss", showlegend=False), 1, 3)
fig_multi.add_trace(go.Scatter(x=he.values, y=ho.values, mode="markers",
                                  marker=dict(size=4, color="purple", opacity=0.5),
                                  name="Ho vs He", showlegend=False), 2, 1)
fig_multi.add_trace(go.Histogram(x=pic.values, marker_color="green",
                                    name="PIC", showlegend=False), 2, 2)
fig_multi.add_trace(go.Histogram(x=fis.values, marker_color="teal",
                                    name="Fis", showlegend=False), 2, 3)

fig_multi.update_layout(height=650, template="plotly_white",
                          title="Comprehensive QC & Diversity Diagnostics")
st.plotly_chart(fig_multi, use_container_width=True)

# =========================================================
# Per-population summary (if metadata)
# =========================================================
if meta is not None:
    st.header("🌍 Per-Population Summary")

    pop_col_r = st.selectbox("Population column", meta.columns.tolist(),
                              key="rep_popcol")
    sam_col_r = st.selectbox("Sample ID column", meta.columns.tolist(),
                              key="rep_samcol")

    pop_map = dict(zip(meta[sam_col_r].astype(str),
                        meta[pop_col_r].astype(str)))

    rows = []
    for pop in sorted(set(pop_map.values())):
        samples = [s for s in geno.index if pop_map.get(str(s)) == pop]
        if len(samples) < 2:
            continue
        sub = geno.loc[samples]
        rows.append({
            "Population": pop,
            "N_samples": len(samples),
            "Mean_MAF": calc_maf(sub).mean(),
            "Mean_Ho": calc_het_obs(sub).mean(),
            "Mean_He": calc_het_exp(sub).mean(),
            "N_polymorphic": int((calc_maf(sub) > 0).sum()),
        })

    pop_summary = pd.DataFrame(rows)
    st.dataframe(pop_summary.style.format({
        "Mean_MAF": "{:.4f}", "Mean_Ho": "{:.4f}", "Mean_He": "{:.4f}"
    }), use_container_width=True)

    # Grouped bar
    if len(pop_summary) > 0:
        fig_pop = px.bar(
            pop_summary.melt(id_vars=["Population", "N_samples"],
                              value_vars=["Mean_MAF", "Mean_Ho", "Mean_He"]),
            x="Population", y="value", color="variable", barmode="group",
            title="Genetic diversity by population",
        )
        fig_pop.update_layout(template="plotly_white", height=500)
        st.plotly_chart(fig_pop, use_container_width=True)

# =========================================================
# Chromosome-level summary
# =========================================================
if marker_info is not None and "Chrom" in marker_info.columns:
    st.header("🧬 Chromosome-Level Summary")

    stats_by_chrom = pd.DataFrame({
        "Marker": geno.columns,
        "MAF": maf.values,
        "PIC": pic.values,
        "Ho": ho.values,
        "He": he.values,
    }).merge(marker_info[["Marker", "Chrom"]], on="Marker", how="left")

    chr_summary = stats_by_chrom.groupby("Chrom").agg({
        "Marker": "count", "MAF": "mean", "PIC": "mean",
        "Ho": "mean", "He": "mean",
    }).reset_index().rename(columns={"Marker": "N_markers"})

    st.dataframe(chr_summary.style.format({
        "MAF": "{:.4f}", "PIC": "{:.4f}",
        "Ho": "{:.4f}", "He": "{:.4f}"
    }), use_container_width=True)

    fig_chr = px.bar(
        chr_summary.melt(id_vars=["Chrom", "N_markers"],
                          value_vars=["MAF", "PIC", "Ho", "He"]),
        x="Chrom", y="value", color="variable", barmode="group",
        title="Diversity by chromosome",
    )
    fig_chr.update_layout(template="plotly_white", height=500)
    st.plotly_chart(fig_chr, use_container_width=True)

# =========================================================
# Generate final markdown report
# =========================================================
st.header("📝 Auto-Generated Text Report")

report_md = f"""
# Population Genomics Analysis Report

## Dataset
- **Samples:** {geno.shape[0]:,}
- **Markers:** {geno.shape[1]:,}
- **Overall missing rate:** {geno.isna().sum().sum() / geno.size * 100:.2f}%

## Genome-wide statistics
- Mean MAF: {maf.mean():.4f} (SD: {maf.std():.4f})
- Mean observed heterozygosity (Ho): {ho.mean():.4f}
- Mean expected heterozygosity (He): {he.mean():.4f}
- Mean PIC: {pic.mean():.4f}
- Mean Fis: {fis.mean():.4f}

## QC recommendations
- Markers with MAF < 0.05: {int((maf < 0.05).sum()):,} ({(maf < 0.05).mean()*100:.1f}%)
- Markers with missing > 20%: {int((miss_marker > 0.2).sum()):,}
- Samples with missing > 30%: {int((miss_sample > 0.3).sum()):,}

## Interpretation
- Mean expected heterozygosity of {he.mean():.4f} suggests
  {'high' if he.mean() > 0.3 else 'moderate' if he.mean() > 0.15 else 'low'}
  genetic diversity.
- Mean Fis of {fis.mean():.4f} indicates
  {'excess heterozygotes' if fis.mean() < -0.05 else 'inbreeding tendency' if fis.mean() > 0.05 else 'Hardy-Weinberg near equilibrium'}.
"""

st.markdown(report_md)

st.download_button(
    "📥 Download report (Markdown)",
    report_md, "population_genomics_report.md", "text/markdown",
    key="dl_report_md",
)