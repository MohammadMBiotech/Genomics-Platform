"""
Genetic Diversity — Publication Quality
────────────────────────────────────────
Comprehensive population genetic diversity analysis:
  - Global & per-population diversity indices (He, Ho, Fis, Na, Ne, Shannon, PIC)
  - Allelic richness (rarefaction)
  - Private alleles per population
  - Nei's genetic distance
  - Multi-metric radar comparison
  - Bootstrap confidence intervals
  - Hierarchical AMOVA (regions → populations → individuals)
  - F-statistics decomposition (Fis, Fst, Fit)
  - Population ranking & interpretation
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import (
    get_geno_from_session, get_meta_from_session,
    calc_allele_freq, calc_maf, calc_het_obs, calc_het_exp, calc_fis,
    calc_pic, calc_shannon_diversity, calc_effective_alleles,
    download_dataframe, download_plotly_html,
    build_sample_pop_map, get_samples_by_population,
)

st.title("🌿 Genetic Diversity")
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
region_col = None

if meta is not None:
    with st.expander("🔧 Metadata Configuration", expanded=True):
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            sample_col = st.selectbox("Sample ID column",
                                        meta.columns.tolist(),
                                        key="div_samcol")
        with mc2:
            pop_col_opt = st.selectbox("Population column",
                                         ["None"] + meta.columns.tolist(),
                                         key="div_popcol")
            pop_col = None if pop_col_opt == "None" else pop_col_opt
        with mc3:
            region_col_opt = st.selectbox(
                "Region/Cluster column (for hierarchical AMOVA)",
                ["None"] + meta.columns.tolist(),
                key="div_regcol",
                help="Optional: higher-level grouping (e.g., continents, geographic regions)"
            )
            region_col = None if region_col_opt == "None" else region_col_opt

        if pop_col:
            pop_map = build_sample_pop_map(meta, sample_col, pop_col)
            st.success(
                f"✅ Loaded {len(set(pop_map.values()))} populations"
                + (f" and {meta[region_col].nunique()} regions" if region_col else "")
            )


# ═══════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════
def compute_diversity_indices(sub_geno):
    """Compute comprehensive genetic diversity indices for a subset."""
    p, q = calc_allele_freq(sub_geno)
    maf = calc_maf(sub_geno)
    ho = calc_het_obs(sub_geno)
    he = calc_het_exp(sub_geno)
    fis = calc_fis(sub_geno)
    pic = calc_pic(sub_geno)
    shannon = calc_shannon_diversity(sub_geno)
    ne = calc_effective_alleles(sub_geno)

    # Number of alleles per marker
    na = ((p > 0) & (p < 1)).astype(int) + 1

    return {
        "N_samples": sub_geno.shape[0],
        "N_markers": sub_geno.shape[1],
        "N_polymorphic": int((maf > 0).sum()),
        "% Polymorphic": float((maf > 0).sum() / len(maf) * 100),
        "Mean_MAF": float(maf.mean()),
        "Mean_PIC": float(pic.mean()),
        "Mean_Ho": float(ho.mean()),
        "Mean_He": float(he.mean()),
        "Mean_Fis": float(fis.mean()),
        "Mean_Na": float(na.mean()),
        "Mean_Ne": float(ne.mean()),
        "Shannon_I": float(shannon.mean()),
    }


def bootstrap_ci(sub_geno, statistic_fn, n_bootstrap=100, seed=42):
    """Compute 95% confidence interval via bootstrap resampling of markers."""
    rng = np.random.RandomState(seed)
    n_markers = sub_geno.shape[1]
    bootstrap_values = []

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n_markers, n_markers)
        boot_sub = sub_geno.iloc[:, idx]
        try:
            val = statistic_fn(boot_sub)
            bootstrap_values.append(val)
        except Exception:
            pass

    if len(bootstrap_values) == 0:
        return np.nan, np.nan

    return np.percentile(bootstrap_values, [2.5, 97.5])


def compute_private_alleles(pop_geno_dict):
    """
    Find private alleles per population.
    An allele is "private" if MAF > 0 in exactly one population.
    """
    pops = list(pop_geno_dict.keys())
    private_counts = {pop: 0 for pop in pops}

    for marker in list(pop_geno_dict.values())[0].columns:
        pops_with_alt = []
        for pop in pops:
            if marker in pop_geno_dict[pop].columns:
                pop_data = pop_geno_dict[pop][marker].dropna()
                if len(pop_data) > 0:
                    # Has alt allele if any 1 or 2
                    if (pop_data > 0).any():
                        pops_with_alt.append(pop)

        if len(pops_with_alt) == 1:
            private_counts[pops_with_alt[0]] += 1

    return private_counts


def allelic_richness_rarefaction(sub_geno, min_n=None):
    """
    Compute allelic richness using rarefaction to a common sample size.
    """
    if min_n is None:
        min_n = sub_geno.shape[0]
    if min_n > sub_geno.shape[0]:
        min_n = sub_geno.shape[0]

    n = sub_geno.shape[0]
    if n < 2 or min_n < 2:
        return np.nan

    ar_values = []
    for marker in sub_geno.columns:
        col = sub_geno[marker].dropna()
        if len(col) < 2:
            continue

        # Count alleles at this marker
        n0 = (col == 0).sum()
        n1 = (col == 1).sum()
        n2 = (col == 2).sum()
        total_alleles = 2 * (n0 + n1 + n2)
        n_ref = 2 * n0 + n1
        n_alt = 2 * n2 + n1

        if total_alleles < 2 * min_n:
            continue

        # Probability that at least one ref/alt allele in 2*min_n draws
        try:
            from scipy.special import comb
            total = comb(total_alleles, 2 * min_n)
            if total == 0:
                continue
            p_no_ref = comb(total_alleles - n_ref, 2 * min_n) / total if n_ref > 0 else 1
            p_no_alt = comb(total_alleles - n_alt, 2 * min_n) / total if n_alt > 0 else 1
            expected_alleles = (1 - p_no_ref) + (1 - p_no_alt)
            ar_values.append(expected_alleles)
        except (OverflowError, ValueError):
            # Fallback: use observed
            ar_values.append(((n_ref > 0) + (n_alt > 0)))

    return np.mean(ar_values) if ar_values else np.nan


def calc_nei_distance(geno1, geno2):
    """Nei's standard genetic distance between two populations."""
    p1, q1 = calc_allele_freq(geno1)
    p2, q2 = calc_allele_freq(geno2)

    # Sum of products
    Jxy = (p1 * p2 + q1 * q2).sum()
    Jx = (p1 ** 2 + q1 ** 2).sum()
    Jy = (p2 ** 2 + q2 ** 2).sum()

    if Jx * Jy <= 0:
        return np.nan

    # Nei's genetic identity
    I = Jxy / np.sqrt(Jx * Jy)
    # Nei's genetic distance
    D = -np.log(I) if I > 0 else np.inf
    return float(D)


# ═══════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════
tab_global, tab_pop, tab_advanced, tab_amova, tab_distance = st.tabs([
    "🌍 Global Diversity",
    "🌿 Per-Population",
    "🔬 Advanced Metrics",
    "🧬 AMOVA & F-stats",
    "📏 Genetic Distances",
])


# ═══════════════════════════════════════════
# TAB 1 — Global Diversity
# ═══════════════════════════════════════════
with tab_global:
    st.subheader("🌍 Global Diversity Indices")
    st.write("Diversity indices computed across all samples combined.")

    with st.spinner("Computing global diversity..."):
        global_stats = compute_diversity_indices(geno)

    # Display as metrics
    gc1, gc2, gc3, gc4 = st.columns(4)
    gc1.metric("Samples", global_stats["N_samples"])
    gc2.metric("Markers", global_stats["N_markers"])
    gc3.metric("Polymorphic", f"{global_stats['N_polymorphic']:,}")
    gc4.metric("% Polymorphic",
                f"{global_stats['% Polymorphic']:.1f}%")

    gc5, gc6, gc7, gc8 = st.columns(4)
    gc5.metric("Mean MAF", f"{global_stats['Mean_MAF']:.4f}")
    gc6.metric("Mean He", f"{global_stats['Mean_He']:.4f}")
    gc7.metric("Mean Ho", f"{global_stats['Mean_Ho']:.4f}")
    gc8.metric("Mean Fis", f"{global_stats['Mean_Fis']:.4f}")

    gc9, gc10, gc11, gc12 = st.columns(4)
    gc9.metric("Mean PIC", f"{global_stats['Mean_PIC']:.4f}")
    gc10.metric("Mean Ne (alleles)",
                 f"{global_stats['Mean_Ne']:.4f}")
    gc11.metric("Mean Na (alleles)",
                 f"{global_stats['Mean_Na']:.4f}")
    gc12.metric("Shannon I",
                 f"{global_stats['Shannon_I']:.4f}")

    # Interpretation
    st.markdown("### 💡 Interpretation")

    interpretations = []
    he_val = global_stats["Mean_He"]
    fis_val = global_stats["Mean_Fis"]
    pic_val = global_stats["Mean_PIC"]

    if he_val > 0.35:
        interpretations.append(f"✅ **High genetic diversity** (He = {he_val:.3f})")
    elif he_val > 0.2:
        interpretations.append(f"⚡ **Moderate genetic diversity** (He = {he_val:.3f})")
    else:
        interpretations.append(f"⚠️ **Low genetic diversity** (He = {he_val:.3f})")

    if fis_val > 0.1:
        interpretations.append(f"⚠️ **Inbreeding detected** (Fis = {fis_val:.3f})")
    elif fis_val < -0.05:
        interpretations.append(f"⚠️ **Heterozygote excess** (Fis = {fis_val:.3f})")
    else:
        interpretations.append(f"✅ **Near Hardy-Weinberg equilibrium** (Fis = {fis_val:.3f})")

    if pic_val > 0.5:
        interpretations.append(f"✅ **Highly informative markers** (PIC = {pic_val:.3f})")
    elif pic_val > 0.25:
        interpretations.append(f"⚡ **Reasonably informative markers** (PIC = {pic_val:.3f})")
    else:
        interpretations.append(f"⚠️ **Low marker informativeness** (PIC = {pic_val:.3f})")

    for interp in interpretations:
        st.write(interp)

    # Bootstrap confidence intervals (optional)
    st.markdown("---")
    st.markdown("### 📊 Bootstrap Confidence Intervals (optional)")
    if st.button("Compute 95% CI for global He, Ho, Fis (100 bootstrap)",
                 key="global_ci"):
        with st.spinner("Bootstrapping..."):
            he_ci = bootstrap_ci(geno,
                                   lambda g: calc_het_exp(g).mean(),
                                   n_bootstrap=100)
            ho_ci = bootstrap_ci(geno,
                                   lambda g: calc_het_obs(g).mean(),
                                   n_bootstrap=100)
            fis_ci = bootstrap_ci(geno,
                                    lambda g: calc_fis(g).mean(),
                                    n_bootstrap=100)

        ci_df = pd.DataFrame({
            "Statistic": ["He", "Ho", "Fis"],
            "Estimate": [global_stats["Mean_He"], global_stats["Mean_Ho"],
                          global_stats["Mean_Fis"]],
            "CI Lower (2.5%)": [he_ci[0], ho_ci[0], fis_ci[0]],
            "CI Upper (97.5%)": [he_ci[1], ho_ci[1], fis_ci[1]],
        })
        st.dataframe(ci_df.style.format({
            "Estimate": "{:.4f}", "CI Lower (2.5%)": "{:.4f}",
            "CI Upper (97.5%)": "{:.4f}",
        }), use_container_width=True)


# ═══════════════════════════════════════════
# TAB 2 — Per-Population Diversity
# ═══════════════════════════════════════════
with tab_pop:
    st.subheader("🌿 Per-Population Diversity Analysis")

    if not pop_map:
        st.warning(
            "⚠️ Please configure metadata (Sample ID + Population columns) "
            "at the top of this page."
        )
    else:
        with st.spinner("Computing per-population diversity..."):
            diversity_rows = []
            pop_geno_dict = {}

            for pop in sorted(set(pop_map.values())):
                samples_in_pop = get_samples_by_population(
                    pop_map, geno.index, pop)
                if len(samples_in_pop) < 2:
                    continue

                sub = geno.loc[samples_in_pop]
                pop_geno_dict[pop] = sub
                stats = compute_diversity_indices(sub)
                stats["Population"] = pop
                diversity_rows.append(stats)

        if not diversity_rows:
            st.warning("Insufficient samples per population.")
            st.stop()

        div_df = pd.DataFrame(diversity_rows)
        div_df = div_df[["Population"] + [c for c in div_df.columns
                                            if c != "Population"]]

        # Display formatted table
        st.dataframe(div_df.style.format({
            c: "{:.4f}" for c in div_df.columns
            if div_df[c].dtype in [np.float64, np.float32]
        }), use_container_width=True)

        download_dataframe(div_df, "diversity_by_population.csv",
                            key="dl_div_pop")

        # Bar plot
        st.subheader("📊 Diversity Comparison Across Populations")

        div_metric = st.selectbox(
            "Metric to display",
            ["Mean_He", "Mean_Ho", "Mean_MAF", "Mean_PIC",
             "Mean_Fis", "Mean_Ne", "Mean_Na", "Shannon_I",
             "N_polymorphic", "% Polymorphic"],
            key="div_metric",
        )

        fig_div = px.bar(
            div_df.sort_values(div_metric, ascending=False),
            x="Population", y=div_metric,
            color=div_metric, color_continuous_scale="Viridis",
            title=f"{div_metric} by Population",
            text=div_df[div_metric].round(4),
        )
        fig_div.update_traces(textposition="outside")
        fig_div.update_layout(template="plotly_white", height=500,
                                xaxis_tickangle=45)
        st.plotly_chart(fig_div, use_container_width=True)

        # Grouped bar
        st.markdown("#### Multi-Metric Grouped Comparison")
        melt_metrics = st.multiselect(
            "Select metrics to compare",
            ["Mean_He", "Mean_Ho", "Mean_MAF", "Mean_PIC",
             "Mean_Ne", "Shannon_I"],
            default=["Mean_He", "Mean_Ho", "Mean_PIC"],
            key="melt_metrics",
        )

        if melt_metrics:
            melted = div_df.melt(id_vars="Population",
                                    value_vars=melt_metrics,
                                    var_name="Metric",
                                    value_name="Value")
            fig_grp = px.bar(
                melted, x="Population", y="Value", color="Metric",
                barmode="group",
                title="Multi-metric population comparison",
            )
            fig_grp.update_layout(template="plotly_white", height=500,
                                    xaxis_tickangle=45)
            st.plotly_chart(fig_grp, use_container_width=True)

        # Radar chart
        st.markdown("#### 🕸️ Multi-Metric Radar Comparison")
        radar_metrics = ["Mean_He", "Mean_Ho", "Mean_MAF",
                          "Mean_PIC", "Mean_Ne", "Shannon_I"]

        selected_pops = st.multiselect(
            "Select populations to compare in radar",
            div_df["Population"].tolist(),
            default=div_df["Population"].head(5).tolist(),
            key="div_radar_pops",
        )

        if selected_pops:
            fig_radar = go.Figure()
            for pop in selected_pops:
                row = div_df[div_df["Population"] == pop].iloc[0]
                fig_radar.add_trace(go.Scatterpolar(
                    r=[row[m] for m in radar_metrics],
                    theta=radar_metrics,
                    fill="toself",
                    name=str(pop),
                ))
            fig_radar.update_layout(
                template="plotly_white", height=600,
                title="Multi-metric diversity radar",
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # Heatmap (z-scored)
        st.markdown("#### 🎨 Population × Metric Heatmap (z-scored)")
        heat_metrics = ["Mean_He", "Mean_Ho", "Mean_MAF", "Mean_PIC",
                          "Mean_Fis", "Mean_Ne", "Shannon_I"]
        heat_data = div_df.set_index("Population")[heat_metrics]
        heat_z = (heat_data - heat_data.mean()) / heat_data.std()

        fig_heat = px.imshow(
            heat_z.values,
            x=heat_metrics, y=heat_data.index.tolist(),
            text_auto=".2f", color_continuous_scale="RdBu_r",
            aspect="auto", title="Population diversity (z-scored)",
        )
        fig_heat.update_layout(template="plotly_white", height=500)
        st.plotly_chart(fig_heat, use_container_width=True)

        # Population ranking
        st.markdown("#### 🏆 Population Ranking")
        rank_metric = st.selectbox(
            "Rank populations by",
            ["Mean_He", "Mean_PIC", "Shannon_I", "Mean_Ne",
             "N_polymorphic"],
            key="rank_metric",
        )

        ranked = div_df.sort_values(rank_metric, ascending=False)[
            ["Population", rank_metric, "N_samples", "N_polymorphic"]
        ].reset_index(drop=True)
        ranked.insert(0, "Rank", range(1, len(ranked) + 1))
        st.dataframe(ranked.style.format({
            rank_metric: "{:.4f}",
        }), use_container_width=True)


# ═══════════════════════════════════════════
# TAB 3 — Advanced Metrics (Rarefaction, Private Alleles)
# ═══════════════════════════════════════════
with tab_advanced:
    st.subheader("🔬 Advanced Diversity Metrics")

    if not pop_map:
        st.warning("Metadata configuration required.")
    else:
        # Build pop_geno_dict if not already
        if "pop_geno_dict" not in dir():
            pop_geno_dict = {}
            for pop in sorted(set(pop_map.values())):
                samples_pop = get_samples_by_population(pop_map,
                                                          geno.index, pop)
                if len(samples_pop) >= 2:
                    pop_geno_dict[pop] = geno.loc[samples_pop]

        # ─── Private Alleles ───
        st.markdown("### 🎯 Private Alleles per Population")
        st.write(
            "Alleles found in **only one** population — indicators of "
            "unique genetic material or geographic isolation."
        )

        if st.button("🚀 Compute private alleles",
                     key="private_alleles_run"):
            with st.spinner("Computing..."):
                private_counts = compute_private_alleles(pop_geno_dict)

            private_df = pd.DataFrame([
                {"Population": pop, "Private_alleles": count,
                  "% of markers":
                   count / geno.shape[1] * 100}
                for pop, count in private_counts.items()
            ]).sort_values("Private_alleles", ascending=False)

            st.dataframe(private_df.style.format({
                "% of markers": "{:.2f}",
            }), use_container_width=True)

            fig_priv = px.bar(
                private_df, x="Population", y="Private_alleles",
                color="Private_alleles", color_continuous_scale="Viridis",
                title="Private alleles per population",
                text="Private_alleles",
            )
            fig_priv.update_traces(textposition="outside")
            fig_priv.update_layout(template="plotly_white", height=500,
                                     xaxis_tickangle=45)
            st.plotly_chart(fig_priv, use_container_width=True)

            download_dataframe(private_df, "private_alleles.csv",
                                key="dl_private")

        # ─── Allelic Richness (Rarefaction) ───
        st.markdown("---")
        st.markdown("### 📊 Allelic Richness (Rarefaction)")
        st.write(
            "Number of alleles expected in samples of standardized size. "
            "Allows fair comparison of populations with different N."
        )

        min_sample_size = min(
            len(get_samples_by_population(pop_map, geno.index, p))
            for p in set(pop_map.values())
            if len(get_samples_by_population(pop_map, geno.index, p)) >= 2
        ) if pop_map else 2

        rf_n = st.slider(
            "Rarefy to sample size",
            2, min(50, min_sample_size), min(min_sample_size, 10),
            key="rf_n",
            help="Standardized N for rarefaction. Should be ≤ smallest population size."
        )

        if st.button("🚀 Compute allelic richness (rarefaction)",
                     key="rf_run"):
            with st.spinner("Computing..."):
                ar_results = []
                for pop, pop_geno in pop_geno_dict.items():
                    ar = allelic_richness_rarefaction(pop_geno, min_n=rf_n)
                    ar_results.append({
                        "Population": pop,
                        "N_samples": pop_geno.shape[0],
                        "Allelic_richness (rarefied)": ar,
                    })

            ar_df = pd.DataFrame(ar_results).sort_values(
                "Allelic_richness (rarefied)", ascending=False)

            st.dataframe(ar_df.style.format({
                "Allelic_richness (rarefied)": "{:.4f}",
            }), use_container_width=True)

            fig_ar = px.bar(
                ar_df, x="Population",
                y="Allelic_richness (rarefied)",
                color="Allelic_richness (rarefied)",
                color_continuous_scale="Viridis",
                title=f"Allelic richness rarefied to N={rf_n}",
            )
            fig_ar.update_layout(template="plotly_white", height=500,
                                    xaxis_tickangle=45)
            st.plotly_chart(fig_ar, use_container_width=True)

            download_dataframe(ar_df, "allelic_richness.csv",
                                key="dl_ar")


# ═══════════════════════════════════════════
# TAB 4 — AMOVA & F-Statistics
# ═══════════════════════════════════════════
with tab_amova:
    st.subheader("🧬 AMOVA & F-Statistics Decomposition")
    st.write(
        "Decompose genetic variance into hierarchical components using "
        "F-statistics (Fis, Fst, Fit) and AMOVA."
    )

    if not pop_map:
        st.warning("Metadata configuration required.")
    else:
        if st.button("🚀 Compute F-statistics",
                     use_container_width=True, key="fstats_run"):
            with st.spinner("Computing F-statistics..."):
                p_total, q_total = calc_allele_freq(geno)
                Ht = 2 * p_total * q_total

                # Weighted Hs (heterozygosity within populations)
                he_by_pop_list = []
                ho_by_pop_list = []
                weights = []
                for pop in set(pop_map.values()):
                    samples = get_samples_by_population(pop_map,
                                                          geno.index, pop)
                    if len(samples) < 2:
                        continue
                    sub = geno.loc[samples]
                    p_sub, q_sub = calc_allele_freq(sub)
                    he_by_pop_list.append(2 * p_sub * q_sub *
                                              len(samples))
                    ho_by_pop_list.append(calc_het_obs(sub) *
                                              len(samples))
                    weights.append(len(samples))

                total_weight = sum(weights)
                Hs = (pd.concat(he_by_pop_list, axis=1).sum(axis=1) /
                       total_weight)
                Hi = (pd.concat(ho_by_pop_list, axis=1).sum(axis=1) /
                       total_weight)

                # F-statistics
                with np.errstate(divide="ignore", invalid="ignore"):
                    Fst = (Ht - Hs) / Ht.replace(0, np.nan)
                    Fis = 1 - Hi / Hs.replace(0, np.nan)
                    Fit = 1 - Hi / Ht.replace(0, np.nan)

                Fst_mean = float(Fst.mean())
                Fis_mean = float(Fis.mean())
                Fit_mean = float(Fit.mean())

            # Display F-statistics
            fs1, fs2, fs3 = st.columns(3)
            fs1.metric("Fis (within pop)", f"{Fis_mean:.4f}",
                        help="Inbreeding within populations")
            fs2.metric("Fst (among pops)", f"{Fst_mean:.4f}",
                        help="Genetic differentiation among populations")
            fs3.metric("Fit (total)", f"{Fit_mean:.4f}",
                        help="Overall inbreeding coefficient")

            # Interpretation
            st.markdown("### 💡 Interpretation")

            fst_interp = ""
            if Fst_mean < 0.05:
                fst_interp = "✅ **Little differentiation** among populations"
            elif Fst_mean < 0.15:
                fst_interp = "⚡ **Moderate differentiation**"
            elif Fst_mean < 0.25:
                fst_interp = "⚠️ **Great differentiation**"
            else:
                fst_interp = "🔴 **Very great differentiation**"

            fis_interp = ""
            if Fis_mean > 0.05:
                fis_interp = "⚠️ **Inbreeding present** within populations"
            elif Fis_mean < -0.05:
                fis_interp = "⚠️ **Heterozygote excess** within populations"
            else:
                fis_interp = "✅ **Near HWE** within populations"

            st.write(f"- **Fst** = {Fst_mean:.4f} — {fst_interp}")
            st.write(f"- **Fis** = {Fis_mean:.4f} — {fis_interp}")
            st.write(f"- **Fit** = {Fit_mean:.4f}")

            fstat_df = pd.DataFrame({
                "Statistic": ["Fis (within pop)", "Fst (among pops)",
                                "Fit (total)"],
                "Value": [Fis_mean, Fst_mean, Fit_mean],
                "Interpretation": [
                    "Inbreeding within populations",
                    "Genetic differentiation among populations",
                    "Overall inbreeding",
                ],
            })
            st.table(fstat_df)

            download_dataframe(fstat_df, "fstatistics.csv", key="dl_fstat")

            fig_fst = px.bar(
                fstat_df, x="Statistic", y="Value",
                color="Statistic",
                title="Global F-statistics",
                text=fstat_df["Value"].round(4),
            )
            fig_fst.update_traces(textposition="outside")
            fig_fst.update_layout(template="plotly_white", height=450)
            st.plotly_chart(fig_fst, use_container_width=True)

            # F-statistic per marker
            st.markdown("### 📊 Per-Marker F-statistics Distribution")
            per_marker_df = pd.DataFrame({
                "Marker": geno.columns,
                "Fis": Fis.values,
                "Fst": Fst.values,
                "Fit": Fit.values,
            })

            fig_dist_fst = make_subplots(rows=1, cols=3,
                                             subplot_titles=("Fis", "Fst", "Fit"))
            fig_dist_fst.add_trace(go.Histogram(x=per_marker_df["Fis"],
                                                     marker_color="steelblue",
                                                     showlegend=False), 1, 1)
            fig_dist_fst.add_trace(go.Histogram(x=per_marker_df["Fst"],
                                                     marker_color="orange",
                                                     showlegend=False), 1, 2)
            fig_dist_fst.add_trace(go.Histogram(x=per_marker_df["Fit"],
                                                     marker_color="green",
                                                     showlegend=False), 1, 3)
            fig_dist_fst.update_layout(template="plotly_white", height=400,
                                          title="Per-marker F-statistics")
            st.plotly_chart(fig_dist_fst, use_container_width=True)

            download_dataframe(per_marker_df, "per_marker_fstats.csv",
                                key="dl_pm_fstat")

        # ─── AMOVA-like variance partitioning ───
        st.markdown("---")
        st.markdown("### 📊 AMOVA-like Variance Partitioning")

        if st.button("🚀 Run AMOVA", use_container_width=True,
                     key="amova_run"):
            with st.spinner("Running AMOVA..."):
                # Impute
                geno_imp = geno.fillna(geno.mean())

                samples = [str(s) for s in geno_imp.index]
                samples = [s for s in samples if s in pop_map]
                g_sub = geno_imp.loc[samples]
                groups = np.array([pop_map[s] for s in samples])
                unique_groups = np.unique(groups)

                # Total SS
                centroid = g_sub.mean(axis=0)
                SS_total = ((g_sub - centroid) ** 2).sum().sum()

                # Within-group SS
                SS_within = 0
                for grp in unique_groups:
                    sub = g_sub[groups == grp]
                    if len(sub) < 2:
                        continue
                    grp_centroid = sub.mean(axis=0)
                    SS_within += ((sub - grp_centroid) ** 2).sum().sum()

                SS_among = SS_total - SS_within

                n_total = len(samples)
                n_groups = len(unique_groups)
                df_among = n_groups - 1
                df_within = n_total - n_groups

                MS_among = SS_among / df_among if df_among > 0 else 0
                MS_within = SS_within / df_within if df_within > 0 else 0

                n_bar = n_total / n_groups
                sigma2_within = MS_within
                sigma2_among = max(0, (MS_among - MS_within) / n_bar)
                sigma2_total = sigma2_among + sigma2_within

                phi_st = (sigma2_among / sigma2_total
                            if sigma2_total > 0 else 0)

            amova_df = pd.DataFrame({
                "Source of variation": [
                    "Among populations", "Within populations", "Total"
                ],
                "df": [df_among, df_within, df_among + df_within],
                "SS": [SS_among, SS_within, SS_total],
                "MS": [MS_among, MS_within, np.nan],
                "Est. Var.": [sigma2_among, sigma2_within, sigma2_total],
                "% of variation": [
                    sigma2_among / sigma2_total * 100 if sigma2_total > 0 else 0,
                    sigma2_within / sigma2_total * 100 if sigma2_total > 0 else 0,
                    100,
                ],
            })

            st.subheader("AMOVA Table")
            st.dataframe(amova_df.style.format({
                "SS": "{:.2f}", "MS": "{:.2f}",
                "Est. Var.": "{:.4f}", "% of variation": "{:.2f}",
            }), use_container_width=True)

            st.metric("Φst", f"{phi_st:.4f}")

            # Pie chart
            fig_amova = px.pie(
                amova_df.iloc[:2], values="% of variation",
                names="Source of variation",
                title="Partitioning of genetic variance",
                color_discrete_sequence=["#FF6B6B", "#4ECDC4"],
            )
            fig_amova.update_layout(template="plotly_white", height=450)
            st.plotly_chart(fig_amova, use_container_width=True)

            download_dataframe(amova_df, "amova_results.csv",
                                key="dl_amova")


# ═══════════════════════════════════════════
# TAB 5 — Nei's Genetic Distance
# ═══════════════════════════════════════════
with tab_distance:
    st.subheader("📏 Nei's Genetic Distance Between Populations")
    st.write(
        "Standard measure of genetic differentiation between populations. "
        "**D = -ln(I)** where I is Nei's genetic identity."
    )

    if not pop_map:
        st.warning("Metadata configuration required.")
    else:
        if st.button("🚀 Compute Nei's distance matrix",
                     use_container_width=True, key="nei_run"):
            with st.spinner("Computing pairwise Nei's distances..."):
                pops = sorted(set(pop_map.values()))
                n_pops = len(pops)
                nei_mat = np.full((n_pops, n_pops), np.nan)

                pop_genos = {}
                for pop in pops:
                    samples = get_samples_by_population(pop_map,
                                                          geno.index, pop)
                    if len(samples) >= 2:
                        pop_genos[pop] = geno.loc[samples]

                valid_pops = list(pop_genos.keys())

                for i, pop_i in enumerate(valid_pops):
                    nei_mat[i, i] = 0.0
                    for j, pop_j in enumerate(valid_pops):
                        if j <= i:
                            continue
                        d = calc_nei_distance(pop_genos[pop_i],
                                                 pop_genos[pop_j])
                        i_idx = pops.index(pop_i)
                        j_idx = pops.index(pop_j)
                        nei_mat[i_idx, j_idx] = d
                        nei_mat[j_idx, i_idx] = d

            nei_df = pd.DataFrame(nei_mat, index=pops, columns=pops)

            st.subheader("Nei's Genetic Distance Matrix")
            st.dataframe(nei_df.style.format("{:.4f}"),
                          use_container_width=True)

            # Heatmap
            fig_nei = px.imshow(
                nei_mat, x=pops, y=pops,
                text_auto=".3f",
                color_continuous_scale="YlOrRd",
                title="Nei's genetic distance",
                aspect="auto",
            )
            fig_nei.update_layout(template="plotly_white", height=600)
            st.plotly_chart(fig_nei, use_container_width=True)

            # Summary
            upper = nei_mat[np.triu_indices(n_pops, k=1)]
            upper = upper[~np.isnan(upper)]

            m1, m2, m3 = st.columns(3)
            m1.metric("Mean Nei's D", f"{upper.mean():.4f}")
            m2.metric("Max Nei's D", f"{upper.max():.4f}")
            m3.metric("Min Nei's D", f"{upper.min():.4f}")

            download_dataframe(nei_df.reset_index(),
                                "nei_distance.csv", key="dl_nei")
            download_plotly_html(fig_nei, "nei_distance.html",
                                  key="dl_nei_html")

            # Top most/least distant pairs
            st.markdown("#### 🔝 Most & Least Distant Population Pairs")

            triu_i, triu_j = np.triu_indices(n_pops, k=1)
            pair_df = pd.DataFrame({
                "Pop_1": [pops[i] for i in triu_i],
                "Pop_2": [pops[j] for j in triu_j],
                "Nei_distance": nei_mat[triu_i, triu_j],
            }).dropna().sort_values("Nei_distance", ascending=False)

            pc1, pc2 = st.columns(2)
            with pc1:
                st.markdown("**Top 10 most distant**")
                st.dataframe(pair_df.head(10).style.format({
                    "Nei_distance": "{:.4f}",
                }), use_container_width=True)
            with pc2:
                st.markdown("**Top 10 least distant**")
                st.dataframe(pair_df.tail(10).sort_values(
                    "Nei_distance").style.format({
                    "Nei_distance": "{:.4f}",
                }), use_container_width=True)