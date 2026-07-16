"""Geographic Genetics — Fst, AMOVA, Isolation by Distance (IBD), Mantel, IBE."""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

from utils import (
    get_geno_from_session, get_meta_from_session,
    impute_missing, calc_allele_freq, calc_het_obs, calc_het_exp,
    download_plotly_html, download_dataframe,
)

st.title("🌍 Geographic Genetics")
st.markdown("---")

geno, marker_info = get_geno_from_session()
if geno is None:
    st.stop()

meta = get_meta_from_session()
if meta is None:
    st.warning(
        "⚠️ Metadata is required. Please upload a metadata file with "
        "population, geographic coordinates, and/or environmental variables."
    )
    st.stop()


# ═══════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════
def haversine(lat1, lon1, lat2, lon2):
    """Geographic distance in km between two lat/lon points."""
    R = 6371.0
    lat1r = np.radians(lat1)
    lat2r = np.radians(lat2)
    dlat = lat2r - lat1r
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def pairwise_fst(geno1, geno2):
    """Compute pairwise Fst between two populations using Weir-Cockerham approach."""
    p1, _ = calc_allele_freq(geno1)
    p2, _ = calc_allele_freq(geno2)

    pbar = (p1 + p2) / 2
    Ht = 2 * pbar * (1 - pbar)
    Hs = (2 * p1 * (1 - p1) + 2 * p2 * (1 - p2)) / 2

    fst = (Ht - Hs) / Ht.replace(0, np.nan)
    return float(fst.mean())


def pairwise_gst(geno1, geno2):
    """Nei's Gst between two populations."""
    return pairwise_fst(geno1, geno2)


def mantel_test(D1, D2, n_perm=999):
    """
    Mantel test: correlation between two distance matrices with permutation p-value.
    """
    n = D1.shape[0]
    iu = np.triu_indices(n, k=1)
    d1 = D1[iu]
    d2 = D2[iu]
    mask = ~(np.isnan(d1) | np.isnan(d2))
    d1, d2 = d1[mask], d2[mask]

    if len(d1) < 3:
        return np.nan, np.nan

    r_obs, _ = pearsonr(d1, d2)

    # Permutation
    rng = np.random.RandomState(42)
    perms = []
    for _ in range(n_perm):
        perm = rng.permutation(n)
        D1_perm = D1[np.ix_(perm, perm)]
        d1p = D1_perm[iu][mask]
        rp, _ = pearsonr(d1p, d2)
        perms.append(rp)

    perms = np.array(perms)
    p_val = (np.sum(np.abs(perms) >= abs(r_obs)) + 1) / (n_perm + 1)
    return r_obs, p_val


# ═══════════════════════════════════════════
# Tabs
# ═══════════════════════════════════════════
tab_fst, tab_amova, tab_ibd, tab_mantel, tab_ibe, tab_map = st.tabs([
    "📊 Fst / Gst",
    "🧬 AMOVA",
    "📏 Isolation by Distance",
    "🔗 Mantel Test",
    "🌡️ Isolation by Environment",
    "🗺️ Geographic Map",
])

# =========================================================
# TAB 1 — Fst / Gst pairwise
# =========================================================
with tab_fst:
    st.subheader("Pairwise Fst Between Populations")

    pop_col = st.selectbox("Population column",
                            meta.columns.tolist(), key="fst_popcol")
    sam_col = st.selectbox("Sample ID column",
                            meta.columns.tolist(), key="fst_samcol")

    pop_map = dict(zip(meta[sam_col].astype(str),
                        meta[pop_col].astype(str)))

    if st.button("🚀 Compute pairwise Fst matrix", key="fst_run"):
        pops = sorted(set(pop_map.values()))
        n_pops = len(pops)

        with st.spinner("Computing Fst..."):
            Fst_mat = np.full((n_pops, n_pops), np.nan)

            for i in range(n_pops):
                Fst_mat[i, i] = 0.0
                for j in range(i + 1, n_pops):
                    samples_i = [s for s in geno.index
                                    if pop_map.get(str(s)) == pops[i]]
                    samples_j = [s for s in geno.index
                                    if pop_map.get(str(s)) == pops[j]]
                    if len(samples_i) < 2 or len(samples_j) < 2:
                        continue
                    g1 = geno.loc[samples_i]
                    g2 = geno.loc[samples_j]
                    fst = pairwise_fst(g1, g2)
                    Fst_mat[i, j] = Fst_mat[j, i] = fst

        Fst_df = pd.DataFrame(Fst_mat, index=pops, columns=pops)

        st.subheader("Fst Matrix")
        st.dataframe(Fst_df.style.format("{:.4f}"), use_container_width=True)

        # Heatmap
        fig_fst = px.imshow(
            Fst_mat, x=pops, y=pops, text_auto=".3f",
            color_continuous_scale="Reds",
            title="Pairwise Fst",
        )
        fig_fst.update_layout(template="plotly_white", height=600)
        st.plotly_chart(fig_fst, use_container_width=True)

        # Save for other tabs
        st.session_state["fst_matrix"] = Fst_df
        st.session_state["fst_pops"] = pops

        # Summary
        upper = Fst_mat[np.triu_indices(n_pops, k=1)]
        upper = upper[~np.isnan(upper)]
        m1, m2, m3 = st.columns(3)
        m1.metric("Mean Fst", f"{upper.mean():.4f}")
        m2.metric("Max Fst", f"{upper.max():.4f}")
        m3.metric("Min Fst", f"{upper.min():.4f}")

        download_dataframe(Fst_df.reset_index(),
                            "pairwise_fst.csv", key="dl_fst")
        download_plotly_html(fig_fst, "fst_heatmap.html",
                              key="dl_fst_html")

# =========================================================
# TAB 2 — AMOVA
# =========================================================
with tab_amova:
    st.subheader("Analysis of Molecular Variance (AMOVA)")
    st.write(
        "Partitions genetic variance into components: among populations, "
        "among individuals within populations, and within individuals."
    )

    pop_col_a = st.selectbox("Population column",
                              meta.columns.tolist(), key="amova_popcol")
    sam_col_a = st.selectbox("Sample ID column",
                              meta.columns.tolist(), key="amova_samcol")

    if st.button("🚀 Run AMOVA", key="amova_run"):
        pop_map_a = dict(zip(meta[sam_col_a].astype(str),
                              meta[pop_col_a].astype(str)))

        with st.spinner("Computing AMOVA..."):
            geno_imp = impute_missing(geno, "mean")

            # Sum of squares approach
            samples = geno.index.astype(str).tolist()
            samples = [s for s in samples if str(s) in pop_map_a]
            g_sub = geno_imp.loc[samples]
            groups = np.array([pop_map_a[str(s)] for s in samples])
            unique_groups = np.unique(groups)

            # Total SS (sum of squared distances from centroid)
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

            # Degrees of freedom
            n_total = len(samples)
            n_groups = len(unique_groups)
            df_among = n_groups - 1
            df_within = n_total - n_groups

            MS_among = SS_among / df_among if df_among > 0 else 0
            MS_within = SS_within / df_within if df_within > 0 else 0

            # Variance components
            n_bar = n_total / n_groups  # average group size (approx.)
            sigma2_within = MS_within
            sigma2_among = max(0, (MS_among - MS_within) / n_bar)
            sigma2_total = sigma2_among + sigma2_within

            phi_st = sigma2_among / sigma2_total if sigma2_total > 0 else 0

            # F-statistic
            F_stat = MS_among / MS_within if MS_within > 0 else np.nan

            # Permutation p-value
            rng = np.random.RandomState(42)
            n_perm = 500
            F_perm = []
            for _ in range(n_perm):
                perm_groups = rng.permutation(groups)
                ssw_p = 0
                for grp in unique_groups:
                    sub = g_sub[perm_groups == grp]
                    if len(sub) < 2:
                        continue
                    gc = sub.mean(axis=0)
                    ssw_p += ((sub - gc) ** 2).sum().sum()
                ssa_p = SS_total - ssw_p
                msa_p = ssa_p / df_among if df_among > 0 else 0
                msw_p = ssw_p / df_within if df_within > 0 else 1
                F_perm.append(msa_p / msw_p if msw_p > 0 else 0)

            F_perm = np.array(F_perm)
            p_val = (np.sum(F_perm >= F_stat) + 1) / (n_perm + 1)

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

        m1, m2, m3 = st.columns(3)
        m1.metric("Φst", f"{phi_st:.4f}")
        m2.metric("F-statistic", f"{F_stat:.4f}")
        m3.metric("Permutation p", f"{p_val:.4f}")

        # Pie chart of variance components
        fig_var = px.pie(
            amova_df.iloc[:2], values="% of variation",
            names="Source of variation",
            title="Partitioning of genetic variance",
        )
        fig_var.update_layout(template="plotly_white", height=450)
        st.plotly_chart(fig_var, use_container_width=True)

        download_dataframe(amova_df, "amova_results.csv",
                            key="dl_amova")

# =========================================================
# TAB 3 — Isolation by Distance
# =========================================================
with tab_ibd:
    st.subheader("Isolation by Distance (IBD)")
    st.write(
        "Tests correlation between genetic distance and geographic distance "
        "between populations. Requires lat/lon in metadata."
    )

    lat_col = st.selectbox("Latitude column",
                            meta.columns.tolist(), key="ibd_lat")
    lon_col = st.selectbox("Longitude column",
                            meta.columns.tolist(), key="ibd_lon")
    pop_col_ibd = st.selectbox("Population column",
                                 meta.columns.tolist(), key="ibd_pop")
    sam_col_ibd = st.selectbox("Sample ID column",
                                 meta.columns.tolist(), key="ibd_sam")

    if st.button("🚀 Run IBD test", key="ibd_run"):
        pop_map_ibd = dict(zip(meta[sam_col_ibd].astype(str),
                                 meta[pop_col_ibd].astype(str)))

        # Get one lat/lon per population
        pop_coords = (meta.groupby(pop_col_ibd)[[lat_col, lon_col]]
                       .mean().reset_index())

        # Compute pairwise Fst matrix
        pops = sorted(set(pop_map_ibd.values()))
        n_pops = len(pops)
        Fst_mat = np.full((n_pops, n_pops), np.nan)

        with st.spinner("Computing Fst and geographic distances..."):
            for i in range(n_pops):
                Fst_mat[i, i] = 0.0
                for j in range(i + 1, n_pops):
                    si = [s for s in geno.index
                            if pop_map_ibd.get(str(s)) == pops[i]]
                    sj = [s for s in geno.index
                            if pop_map_ibd.get(str(s)) == pops[j]]
                    if len(si) < 2 or len(sj) < 2:
                        continue
                    fst = pairwise_fst(geno.loc[si], geno.loc[sj])
                    Fst_mat[i, j] = Fst_mat[j, i] = fst

            # Geographic distances
            Geo_mat = np.zeros((n_pops, n_pops))
            pop_lat = pop_coords.set_index(pop_col_ibd)[lat_col].to_dict()
            pop_lon = pop_coords.set_index(pop_col_ibd)[lon_col].to_dict()

            for i in range(n_pops):
                for j in range(i + 1, n_pops):
                    d = haversine(pop_lat[pops[i]], pop_lon[pops[i]],
                                    pop_lat[pops[j]], pop_lon[pops[j]])
                    Geo_mat[i, j] = Geo_mat[j, i] = d

            # Linearize Fst: Fst / (1 - Fst)
            Fst_lin = Fst_mat / (1 - Fst_mat)

        # Extract upper triangle
        iu = np.triu_indices(n_pops, k=1)
        geo_flat = Geo_mat[iu]
        fst_flat = Fst_lin[iu]

        mask = ~np.isnan(fst_flat) & (geo_flat > 0)
        geo_flat = geo_flat[mask]
        fst_flat = fst_flat[mask]

        if len(geo_flat) < 3:
            st.warning("Too few valid pairs.")
        else:
            r, p_val = pearsonr(geo_flat, fst_flat)

            fig_ibd = px.scatter(
                x=geo_flat, y=fst_flat,
                labels={"x": "Geographic distance (km)",
                        "y": "Fst / (1 - Fst)"},
                title=f"Isolation by Distance (r = {r:.3f}, p = {p_val:.3g})",
                trendline="ols",
            )
            fig_ibd.update_layout(template="plotly_white", height=550)
            st.plotly_chart(fig_ibd, use_container_width=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Pearson r", f"{r:.4f}")
            m2.metric("p-value", f"{p_val:.4g}")
            m3.metric("N pairs", f"{len(geo_flat)}")

            # Mantel test
            with st.spinner("Running Mantel test..."):
                r_mantel, p_mantel = mantel_test(Fst_lin, Geo_mat,
                                                    n_perm=999)
            st.info(f"**Mantel test:** r = {r_mantel:.4f}, "
                     f"p = {p_mantel:.4g} (999 permutations)")

            result_df = pd.DataFrame({
                "Geographic_km": geo_flat,
                "Fst_linear": fst_flat,
            })
            download_dataframe(result_df, "ibd_results.csv",
                                key="dl_ibd")

# =========================================================
# TAB 4 — Generic Mantel test
# =========================================================
with tab_mantel:
    st.subheader("Mantel Test")
    st.write(
        "Test correlation between any two distance matrices via permutation. "
        "Choose the variables to compare from your metadata."
    )

    sam_m = st.selectbox("Sample ID column", meta.columns.tolist(),
                          key="man_sam")

    mc1, mc2 = st.columns(2)
    with mc1:
        var1_cols = st.multiselect(
            "Variables for matrix 1", meta.columns.tolist(),
            key="man_v1",
        )
    with mc2:
        var2_cols = st.multiselect(
            "Variables for matrix 2", meta.columns.tolist(),
            key="man_v2",
        )

    n_perm_m = st.slider("Permutations", 99, 9999, 999, 100,
                          key="man_np")

    if st.button("🚀 Run Mantel test", key="man_run"):
        if not var1_cols or not var2_cols:
            st.warning("Select at least one variable in each set.")
            st.stop()

        # Build distance matrices
        samples_use = [s for s in geno.index.astype(str)
                        if s in meta[sam_m].astype(str).values]

        meta_use = meta[meta[sam_m].astype(str).isin(samples_use)]
        meta_use = meta_use.set_index(sam_m)

        try:
            V1 = meta_use.loc[samples_use, var1_cols].apply(
                pd.to_numeric, errors="coerce").fillna(0).values
            V2 = meta_use.loc[samples_use, var2_cols].apply(
                pd.to_numeric, errors="coerce").fillna(0).values
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

        D1 = squareform(pdist(V1))
        D2 = squareform(pdist(V2))

        r_man, p_man = mantel_test(D1, D2, n_perm=n_perm_m)

        m1, m2 = st.columns(2)
        m1.metric("Mantel r", f"{r_man:.4f}")
        m2.metric("p-value", f"{p_man:.4g}")

        # Scatter
        iu = np.triu_indices(D1.shape[0], k=1)
        fig_m = px.scatter(
            x=D1[iu], y=D2[iu],
            labels={"x": "Distance (matrix 1)", "y": "Distance (matrix 2)"},
            title=f"Mantel scatter (r = {r_man:.3f}, p = {p_man:.3g})",
            trendline="ols",
        )
        fig_m.update_layout(template="plotly_white", height=500)
        st.plotly_chart(fig_m, use_container_width=True)

# =========================================================
# TAB 5 — Isolation by Environment
# =========================================================
with tab_ibe:
    st.subheader("Isolation by Environment (IBE)")
    st.write(
        "Correlates genetic distance with environmental distance "
        "(e.g., climate variables). Requires environmental columns in metadata."
    )

    env_cols = st.multiselect(
        "Select environmental variables",
        meta.select_dtypes(include=[np.number]).columns.tolist(),
        key="ibe_env",
    )
    pop_col_ibe = st.selectbox("Population column",
                                meta.columns.tolist(), key="ibe_pop")
    sam_col_ibe = st.selectbox("Sample ID column",
                                meta.columns.tolist(), key="ibe_sam")

    if st.button("🚀 Run IBE analysis", key="ibe_run"):
        if not env_cols:
            st.warning("Select at least one environmental variable.")
            st.stop()

        pop_map_ibe = dict(zip(meta[sam_col_ibe].astype(str),
                                 meta[pop_col_ibe].astype(str)))

        # Environmental centroids per population
        env_by_pop = (meta.groupby(pop_col_ibe)[env_cols]
                       .mean().reset_index())

        pops = sorted(set(pop_map_ibe.values()))
        n_pops = len(pops)
        env_lookup = env_by_pop.set_index(pop_col_ibe)

        # Environmental distance
        env_mat = np.array([env_lookup.loc[p, env_cols].values
                              for p in pops if p in env_lookup.index])
        pops_valid = [p for p in pops if p in env_lookup.index]

        env_scaled = StandardScaler().fit_transform(env_mat)
        D_env = squareform(pdist(env_scaled))

        # Fst matrix
        Fst_mat_ibe = np.full((len(pops_valid), len(pops_valid)), np.nan)
        with st.spinner("Computing Fst..."):
            for i in range(len(pops_valid)):
                Fst_mat_ibe[i, i] = 0.0
                for j in range(i + 1, len(pops_valid)):
                    si = [s for s in geno.index
                            if pop_map_ibe.get(str(s)) == pops_valid[i]]
                    sj = [s for s in geno.index
                            if pop_map_ibe.get(str(s)) == pops_valid[j]]
                    if len(si) < 2 or len(sj) < 2:
                        continue
                    fst = pairwise_fst(geno.loc[si], geno.loc[sj])
                    Fst_mat_ibe[i, j] = Fst_mat_ibe[j, i] = fst

        iu = np.triu_indices(len(pops_valid), k=1)
        env_flat = D_env[iu]
        fst_flat = Fst_mat_ibe[iu]
        mask = ~np.isnan(fst_flat)

        env_flat = env_flat[mask]
        fst_flat = fst_flat[mask]

        r_ibe, p_ibe = pearsonr(env_flat, fst_flat)

        fig_ibe = px.scatter(
            x=env_flat, y=fst_flat,
            labels={"x": "Environmental distance",
                    "y": "Genetic distance (Fst)"},
            title=f"IBE (r = {r_ibe:.3f}, p = {p_ibe:.3g})",
            trendline="ols",
        )
        fig_ibe.update_layout(template="plotly_white", height=550)
        st.plotly_chart(fig_ibe, use_container_width=True)

        # Mantel
        r_m_ibe, p_m_ibe = mantel_test(Fst_mat_ibe, D_env, n_perm=999)
        m1, m2 = st.columns(2)
        m1.metric("Mantel r", f"{r_m_ibe:.4f}")
        m2.metric("Mantel p", f"{p_m_ibe:.4g}")

# =========================================================
# TAB 6 — Geographic map
# =========================================================
with tab_map:
    st.subheader("Sample Locations Map")

    lat_c = st.selectbox("Latitude column", meta.columns.tolist(),
                          key="map_lat")
    lon_c = st.selectbox("Longitude column", meta.columns.tolist(),
                          key="map_lon")
    color_c = st.selectbox("Color by", ["None"] + meta.columns.tolist(),
                             key="map_color")

    try:
        meta_map = meta.copy()
        meta_map[lat_c] = pd.to_numeric(meta_map[lat_c], errors="coerce")
        meta_map[lon_c] = pd.to_numeric(meta_map[lon_c], errors="coerce")
        meta_map = meta_map.dropna(subset=[lat_c, lon_c])

        fig_map = px.scatter_geo(
            meta_map, lat=lat_c, lon=lon_c,
            color=color_c if color_c != "None" else None,
            hover_data=meta_map.columns.tolist(),
            title="Sample geographic locations",
            projection="natural earth",
        )
        fig_map.update_traces(marker=dict(size=8, opacity=0.8))
        fig_map.update_layout(template="plotly_white", height=700)
        st.plotly_chart(fig_map, use_container_width=True)

        download_plotly_html(fig_map, "geographic_map.html",
                              key="dl_map")
    except Exception as e:
        st.error(f"Error building map: {e}")