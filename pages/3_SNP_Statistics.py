"""SNP Statistics — Allele frequencies, PIC, Nei's diversity indices."""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from utils import (
    get_geno_from_session, get_meta_from_session,
    calc_allele_freq, calc_maf, calc_pic,
    calc_het_obs, calc_het_exp, calc_fis,
    download_plotly_html, download_dataframe,
)

st.title("🧬 SNP Statistics")
st.markdown("---")

geno, marker_info = get_geno_from_session()
if geno is None:
    st.stop()

meta = get_meta_from_session()

st.subheader("📊 Per-Marker Statistics")

# Compute all statistics
p, q = calc_allele_freq(geno)
maf = calc_maf(geno)
pic = calc_pic(geno)
ho = calc_het_obs(geno)
he = calc_het_exp(geno)
fis = calc_fis(geno)

stats_df = pd.DataFrame({
    "Marker": geno.columns,
    "Freq_Ref (p)": p.values,
    "Freq_Alt (q)": q.values,
    "MAF": maf.values,
    "PIC": pic.values,
    "Ho": ho.values,
    "He": he.values,
    "Fis": fis.values,
})

# Merge with marker_info if available
if marker_info is not None:
    stats_df = stats_df.merge(marker_info, on="Marker", how="left")

st.dataframe(stats_df.head(100), use_container_width=True)
download_dataframe(stats_df, "snp_statistics.csv", key="dl_snp_stats")

# ── Summary metrics ──
st.subheader("📈 Genome-wide Summary")
sc1, sc2, sc3, sc4, sc5 = st.columns(5)
sc1.metric("Mean MAF", f"{maf.mean():.4f}")
sc2.metric("Mean PIC", f"{pic.mean():.4f}")
sc3.metric("Mean Ho", f"{ho.mean():.4f}")
sc4.metric("Mean He", f"{he.mean():.4f}")
sc5.metric("Mean Fis", f"{fis.mean():.4f}")

# ── Distributions ──
st.subheader("Distribution Plots")

dist_metric = st.selectbox(
    "Select statistic to visualize",
    ["MAF", "PIC", "Ho", "He", "Fis", "Freq_Ref (p)"],
    key="snp_dist_metric"
)

fig_dist = px.histogram(
    stats_df, x=dist_metric, nbins=50,
    title=f"Distribution of {dist_metric} across markers",
    template="plotly_white",
)
fig_dist.update_layout(height=450)
st.plotly_chart(fig_dist, use_container_width=True)

# ── Per-chromosome averages ──
if marker_info is not None and "Chrom" in marker_info.columns:
    st.subheader("Per-Chromosome Averages")
    per_chrom = stats_df.groupby("Chrom").agg({
        "MAF": "mean", "PIC": "mean", "Ho": "mean",
        "He": "mean", "Fis": "mean"
    }).reset_index()
    st.dataframe(per_chrom, use_container_width=True)

    fig_chrom = px.bar(
        per_chrom.melt(id_vars="Chrom",
                       value_vars=["MAF", "PIC", "Ho", "He"]),
        x="Chrom", y="value", color="variable", barmode="group",
        title="Diversity indices by chromosome"
    )
    fig_chrom.update_layout(template="plotly_white", height=500)
    st.plotly_chart(fig_chrom, use_container_width=True)

# ── Per-population statistics (if metadata available) ──
if meta is not None:
    st.subheader("🌍 Per-Population Statistics")

    pop_col = st.selectbox("Select population/group column",
                            meta.columns.tolist(), key="snp_popcol")
    sample_col = st.selectbox("Sample ID column in metadata",
                               meta.columns.tolist(), key="snp_samcol")

    # Build sample→population map
    pop_map = dict(zip(meta[sample_col].astype(str),
                        meta[pop_col].astype(str)))

    # Compute per-population stats
    pop_stats = []
    for pop in sorted(set(pop_map.values())):
        samples_in_pop = [s for s in geno.index if pop_map.get(str(s)) == pop]
        if len(samples_in_pop) < 2:
            continue
        sub = geno.loc[samples_in_pop]
        maf_pop = calc_maf(sub)
        ho_pop = calc_het_obs(sub)
        he_pop = calc_het_exp(sub)
        pop_stats.append({
            "Population": pop,
            "N_samples": len(samples_in_pop),
            "Mean_MAF": maf_pop.mean(),
            "Mean_Ho": ho_pop.mean(),
            "Mean_He": he_pop.mean(),
            "N_polymorphic": int((maf_pop > 0).sum()),
        })

    pop_stats_df = pd.DataFrame(pop_stats)
    st.dataframe(pop_stats_df, use_container_width=True)
    download_dataframe(pop_stats_df, "population_stats.csv", key="dl_pop_stats")

    fig_pop = px.bar(
        pop_stats_df.melt(id_vars="Population",
                           value_vars=["Mean_MAF", "Mean_Ho", "Mean_He"]),
        x="Population", y="value", color="variable", barmode="group",
        title="Diversity indices by population"
    )
    fig_pop.update_layout(template="plotly_white", height=500)
    st.plotly_chart(fig_pop, use_container_width=True)