"""Genetic Diversity — He, Ho, Fis, Na, Ne, Shannon index, AMOVA."""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from utils import (
    get_geno_from_session, get_meta_from_session,
    calc_allele_freq, calc_maf, calc_het_obs, calc_het_exp, calc_fis,
    download_dataframe, download_plotly_html,
)

st.title("🌿 Genetic Diversity")
st.markdown("---")

geno, marker_info = get_geno_from_session()
if geno is None:
    st.stop()

meta = get_meta_from_session()


def compute_diversity_indices(sub_geno):
    """Compute a suite of genetic diversity indices for a subset."""
    p, q = calc_allele_freq(sub_geno)
    maf = calc_maf(sub_geno)
    ho = calc_het_obs(sub_geno)
    he = calc_het_exp(sub_geno)
    fis = calc_fis(sub_geno)

    # Number of alleles (Na): 2 if polymorphic, else 1
    na = ((p > 0) & (p < 1)).astype(int) + 1

    # Effective number of alleles: Ne = 1 / (p² + q²)
    ne = 1.0 / (p**2 + q**2).replace(0, np.nan)

    # Shannon index: I = -[p*ln(p) + q*ln(q)]
    p_safe = p.replace(0, np.nan)
    q_safe = q.replace(0, np.nan)
    shannon = -(p_safe * np.log(p_safe) + q_safe * np.log(q_safe))

    return {
        "N_samples": sub_geno.shape[0],
        "N_markers": sub_geno.shape[1],
        "N_polymorphic": int((maf > 0).sum()),
        "Mean_MAF": float(maf.mean()),
        "Mean_Ho": float(ho.mean()),
        "Mean_He": float(he.mean()),
        "Mean_Fis": float(fis.mean()),
        "Mean_Na": float(na.mean()),
        "Mean_Ne": float(ne.mean()),
        "Shannon_I": float(shannon.mean()),
    }


# ── Global diversity ──
st.subheader("🌍 Global Diversity Indices")

global_stats = compute_diversity_indices(geno)
global_df = pd.DataFrame([global_stats])
st.dataframe(global_df.T.rename(columns={0: "Value"}), use_container_width=True)

# ── Per-population diversity ──
if meta is not None:
    st.subheader("🧬 Per-Population Diversity Analysis")

    pop_col = st.selectbox("Population column", meta.columns.tolist(),
                            key="div_popcol")
    sample_col = st.selectbox("Sample ID column", meta.columns.tolist(),
                               key="div_samcol")

    pop_map = dict(zip(meta[sample_col].astype(str),
                        meta[pop_col].astype(str)))

    diversity_rows = []
    for pop in sorted(set(pop_map.values())):
        samples_in_pop = [s for s in geno.index
                          if pop_map.get(str(s)) == pop]
        if len(samples_in_pop) < 2:
            continue
        sub = geno.loc[samples_in_pop]
        stats = compute_diversity_indices(sub)
        stats["Population"] = pop
        diversity_rows.append(stats)

    if diversity_rows:
        div_df = pd.DataFrame(diversity_rows)
        div_df = div_df[["Population"] + [c for c in div_df.columns
                                            if c != "Population"]]
        st.dataframe(div_df, use_container_width=True)
        download_dataframe(div_df, "diversity_by_population.csv",
                            key="dl_div_pop")

        # Bar plot of key indices
        st.subheader("📊 Diversity Comparison Across Populations")

        div_metric = st.selectbox(
            "Metric to display",
            ["Mean_He", "Mean_Ho", "Mean_MAF", "Mean_Fis",
             "Mean_Ne", "Shannon_I", "N_polymorphic"],
            key="div_metric",
        )
        fig_div = px.bar(
            div_df.sort_values(div_metric, ascending=False),
            x="Population", y=div_metric,
            color=div_metric, color_continuous_scale="Viridis",
            title=f"{div_metric} by Population"
        )
        fig_div.update_layout(template="plotly_white", height=500)
        st.plotly_chart(fig_div, use_container_width=True)

        # Multi-metric radar chart
        st.subheader("🕸️ Multi-Metric Radar Comparison")
        radar_metrics = ["Mean_He", "Mean_Ho", "Mean_MAF",
                         "Mean_Ne", "Shannon_I"]

        selected_pops = st.multiselect(
            "Select populations to compare",
            div_df["Population"].tolist(),
            default=div_df["Population"].head(5).tolist(),
            key="div_radar_pops"
        )

        if selected_pops:
            fig_radar = go.Figure()
            for pop in selected_pops:
                row = div_df[div_df["Population"] == pop].iloc[0]
                fig_radar.add_trace(go.Scatterpolar(
                    r=[row[m] for m in radar_metrics],
                    theta=radar_metrics,
                    fill='toself',
                    name=pop,
                ))
            fig_radar.update_layout(
                template="plotly_white", height=600,
                title="Multi-metric diversity radar",
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    # ── Simplified AMOVA (F-statistics decomposition) ──
    st.markdown("---")
    st.subheader("📊 AMOVA-like Decomposition (F-statistics)")

    with st.spinner("Computing F-statistics..."):
        # Overall
        p_total, q_total = calc_allele_freq(geno)
        Ht = 2 * p_total * q_total

        # Within populations (mean He across pops)
        he_by_pop = []
        for pop in set(pop_map.values()):
            samples = [s for s in geno.index if pop_map.get(str(s)) == pop]
            if len(samples) < 2:
                continue
            sub = geno.loc[samples]
            p_sub, q_sub = calc_allele_freq(sub)
            he_by_pop.append(2 * p_sub * q_sub)

        if he_by_pop:
            Hs = pd.concat(he_by_pop, axis=1).mean(axis=1)

            Fst = (Ht - Hs) / Ht.replace(0, np.nan)
            Fst_mean = float(Fst.mean())

            # Approximate Ho across pops
            ho_by_pop = []
            for pop in set(pop_map.values()):
                samples = [s for s in geno.index
                           if pop_map.get(str(s)) == pop]
                if len(samples) < 2:
                    continue
                sub = geno.loc[samples]
                ho_by_pop.append(calc_het_obs(sub))
            Hi = pd.concat(ho_by_pop, axis=1).mean(axis=1)
            Fis_mean = float((1 - Hi / Hs.replace(0, np.nan)).mean())
            Fit_mean = float((1 - Hi / Ht.replace(0, np.nan)).mean())

            fstat_df = pd.DataFrame({
                "Statistic": ["Fis (within pop)",
                              "Fst (among pops)",
                              "Fit (total)"],
                "Value": [Fis_mean, Fst_mean, Fit_mean],
                "Interpretation": [
                    "Inbreeding within populations",
                    "Genetic differentiation among populations",
                    "Overall inbreeding",
                ],
            })
            st.table(fstat_df)
            download_dataframe(fstat_df, "fstatistics.csv",
                                key="dl_fstat")

            fig_fst = px.bar(fstat_df, x="Statistic", y="Value",
                              color="Statistic",
                              title="Global F-statistics")
            fig_fst.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig_fst, use_container_width=True)
else:
    st.info("Upload metadata with population/group column for per-population analysis.")