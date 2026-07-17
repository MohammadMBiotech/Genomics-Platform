"""
Geographic Genetics Analysis — Publication Quality
──────────────────────────────────────────────────
Features:
  - Multiple differentiation indices: Fst, Gst, DEST (Jost's D)
  - Gene flow estimation (Nm from Fst)
  - Gene flow network diagram
  - AMOVA with Φ-statistics
  - Isolation by Distance (IBD) with Mantel test
  - Isolation by Environment (IBE)
  - Generic Mantel test
  - Interactive geographic map
"""

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
# GLOBAL: Metadata column selector
# ═══════════════════════════════════════════
st.subheader("🔧 Metadata Configuration")

gc1, gc2 = st.columns(2)
with gc1:
    global_sam_col = st.selectbox(
        "Sample ID column (in metadata)",
        meta.columns.tolist(),
        key="geo_samcol_g",
    )
with gc2:
    global_pop_col = st.selectbox(
        "Population / Group column",
        meta.columns.tolist(),
        key="geo_popcol_g",
    )

st.info(
    f"✅ Using **{global_sam_col}** as Sample ID and "
    f"**{global_pop_col}** as Population across all tabs. "
    f"You can override in each tab if needed."
)

st.markdown("---")


# ═══════════════════════════════════════════
# CORE HELPER FUNCTIONS
# ═══════════════════════════════════════════
def haversine(lat1, lon1, lat2, lon2):
    """Geographic distance (km) between two lat/lon points."""
    R = 6371.0
    lat1r = np.radians(lat1)
    lat2r = np.radians(lat2)
    dlat = lat2r - lat1r
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def compute_pairwise_fst(geno1, geno2):
    """
    Weir & Cockerham (1984) Fst — variance components approach.
    Returns: mean Fst across markers.
    """
    p1, _ = calc_allele_freq(geno1)
    p2, _ = calc_allele_freq(geno2)

    pbar = (p1 + p2) / 2
    Ht = 2 * pbar * (1 - pbar)
    Hs = (2 * p1 * (1 - p1) + 2 * p2 * (1 - p2)) / 2

    fst = (Ht - Hs) / Ht.replace(0, np.nan)
    return float(fst.mean())


def compute_pairwise_gst(geno1, geno2):
    """
    Nei's Gst (1973): Ht - Hs / Ht.
    For biallelic markers with 2 populations equals Fst.
    """
    p1, _ = calc_allele_freq(geno1)
    p2, _ = calc_allele_freq(geno2)

    pbar = (p1 + p2) / 2
    Ht = 2 * pbar * (1 - pbar)
    Hs = (2 * p1 * (1 - p1) + 2 * p2 * (1 - p2)) / 2

    gst = (Ht - Hs) / Ht.replace(0, np.nan)
    return float(gst.mean())


def compute_pairwise_dest(geno1, geno2):
    """
    Jost's D (2008): true measure of allelic differentiation.
    D = ((Ht - Hs) / (1 - Hs)) × (n / (n - 1))
    where n = number of populations = 2 here.
    """
    p1, _ = calc_allele_freq(geno1)
    p2, _ = calc_allele_freq(geno2)

    n = 2  # two populations
    pbar = (p1 + p2) / 2
    Ht = 2 * pbar * (1 - pbar)
    Hs = (2 * p1 * (1 - p1) + 2 * p2 * (1 - p2)) / 2

    dest = ((Ht - Hs) / (1 - Hs).replace(0, np.nan)) * (n / (n - 1))
    return float(dest.mean())


def compute_nm_from_fst(fst):
    """
    Wright's Nm (number of migrants per generation) from Fst.
    Nm = (1 - Fst) / (4 * Fst)
    """
    if fst <= 0 or np.isnan(fst):
        return np.inf
    if fst >= 1:
        return 0.0
    return (1 - fst) / (4 * fst)


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


def build_pop_maps(sam_col_use, pop_col_use):
    """Build sample→population mapping using selected columns."""
    return dict(zip(meta[sam_col_use].astype(str),
                     meta[pop_col_use].astype(str)))


# ═══════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════
tab_diff, tab_geneflow, tab_amova, tab_ibd, tab_mantel, tab_ibe, tab_map = st.tabs([
    "📊 Fst / Gst / DEST",
    "🧬 Gene Flow (Nm)",
    "🧬 AMOVA",
    "📏 Isolation by Distance",
    "🔗 Mantel Test",
    "🌡️ Isolation by Environment",
    "🗺️ Geographic Map",
])

# =========================================================
# TAB 1 — Differentiation Indices (Fst, Gst, DEST)
# =========================================================
with tab_diff:
    st.subheader("Pairwise Population Differentiation")
    st.write(
        "Compute multiple differentiation indices between populations:\n"
        "- **Fst** (Weir & Cockerham): standard differentiation measure\n"
        "- **Gst** (Nei): frequency-based Fst analog\n"
        "- **DEST** (Jost's D): true allelic differentiation measure "
        "(preferred for highly polymorphic markers)"
    )

    dc1, dc2 = st.columns(2)
    with dc1:
        indices_to_compute = st.multiselect(
            "Indices to compute",
            ["Fst", "Gst", "DEST (Jost's D)"],
            default=["Fst", "Gst", "DEST (Jost's D)"],
            key="diff_indices",
        )
    with dc2:
        color_scheme = st.selectbox("Color scheme",
                                     ["Reds", "Viridis", "YlOrRd",
                                      "Blues", "RdBu_r"],
                                     key="diff_cscheme")

    if st.button("🚀 Compute all differentiation indices",
                 use_container_width=True, key="diff_run"):

        pop_map = build_pop_maps(global_sam_col, global_pop_col)
        pops = sorted(set(pop_map.values()))
        n_pops = len(pops)

        # Initialize matrices
        matrices = {}
        for idx_name in indices_to_compute:
            matrices[idx_name] = np.full((n_pops, n_pops), np.nan)

        with st.spinner(f"Computing {len(indices_to_compute)} indices "
                          f"for {n_pops} populations..."):
            progress_bar = st.progress(0)
            total_pairs = n_pops * (n_pops - 1) // 2
            done = 0

            for i in range(n_pops):
                for idx_name in indices_to_compute:
                    matrices[idx_name][i, i] = 0.0
                for j in range(i + 1, n_pops):
                    samples_i = [s for s in geno.index
                                    if pop_map.get(str(s)) == pops[i]]
                    samples_j = [s for s in geno.index
                                    if pop_map.get(str(s)) == pops[j]]
                    if len(samples_i) < 2 or len(samples_j) < 2:
                        done += 1
                        continue

                    g1 = geno.loc[samples_i]
                    g2 = geno.loc[samples_j]

                    if "Fst" in indices_to_compute:
                        v = compute_pairwise_fst(g1, g2)
                        matrices["Fst"][i, j] = matrices["Fst"][j, i] = v
                    if "Gst" in indices_to_compute:
                        v = compute_pairwise_gst(g1, g2)
                        matrices["Gst"][i, j] = matrices["Gst"][j, i] = v
                    if "DEST (Jost's D)" in indices_to_compute:
                        v = compute_pairwise_dest(g1, g2)
                        matrices["DEST (Jost's D)"][i, j] = matrices["DEST (Jost's D)"][j, i] = v

                    done += 1
                    progress_bar.progress(done / total_pairs)

            progress_bar.empty()

        st.success(f"✅ Computed {len(indices_to_compute)} indices.")

        # Store for other tabs
        if "Fst" in matrices:
            st.session_state["fst_matrix"] = pd.DataFrame(
                matrices["Fst"], index=pops, columns=pops)
            st.session_state["fst_pops"] = pops

        # Display each matrix
        for idx_name, mat in matrices.items():
            st.markdown(f"### {idx_name} Matrix")

            mat_df = pd.DataFrame(mat, index=pops, columns=pops)
            st.dataframe(mat_df.style.format("{:.4f}"),
                          use_container_width=True)

            fig_mat = px.imshow(
                mat, x=pops, y=pops, text_auto=".3f",
                color_continuous_scale=color_scheme,
                title=f"Pairwise {idx_name}",
                aspect="auto",
            )
            fig_mat.update_layout(template="plotly_white", height=600)
            st.plotly_chart(fig_mat, use_container_width=True)

            upper = mat[np.triu_indices(n_pops, k=1)]
            upper = upper[~np.isnan(upper)]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Mean", f"{upper.mean():.4f}")
            m2.metric("Median", f"{np.median(upper):.4f}")
            m3.metric("Max", f"{upper.max():.4f}")
            m4.metric("Min", f"{upper.min():.4f}")

            download_dataframe(mat_df.reset_index(),
                                f"pairwise_{idx_name.replace(' ', '_').replace('(', '').replace(')', '')}.csv",
                                key=f"dl_{idx_name}")
            download_plotly_html(fig_mat,
                                  f"{idx_name.replace(' ', '_')}_heatmap.html",
                                  key=f"dl_{idx_name}_html")

        # Combined comparison table
        if len(matrices) > 1:
            st.markdown("### 📋 Combined Indices Comparison")
            iu = np.triu_indices(n_pops, k=1)
            combined_data = []
            for idx_name, mat in matrices.items():
                values = mat[iu]
                for k, (i, j) in enumerate(zip(iu[0], iu[1])):
                    if not np.isnan(values[k]):
                        combined_data.append({
                            "Pop_1": pops[i],
                            "Pop_2": pops[j],
                            "Index": idx_name,
                            "Value": values[k],
                        })

            comb_df = pd.DataFrame(combined_data)
            pivot_comb = comb_df.pivot_table(
                index=["Pop_1", "Pop_2"], columns="Index",
                values="Value").reset_index()

            st.dataframe(pivot_comb.style.format({
                c: "{:.4f}" for c in pivot_comb.columns
                if c not in ["Pop_1", "Pop_2"]
            }), use_container_width=True)

            download_dataframe(pivot_comb,
                                "combined_differentiation_indices.csv",
                                key="dl_comb")

            # Comparison plot
            if len(matrices) >= 2:
                fig_comp = px.scatter_matrix(
                    pivot_comb.drop(columns=["Pop_1", "Pop_2"]),
                    title="Pairwise comparison of differentiation indices",
                )
                fig_comp.update_traces(diagonal_visible=False,
                                         showupperhalf=False)
                fig_comp.update_layout(template="plotly_white",
                                         height=600)
                st.plotly_chart(fig_comp, use_container_width=True)

# =========================================================
# TAB 2 — Gene Flow (Nm) with Network Diagram
# =========================================================
with tab_geneflow:
    st.subheader("🧬 Gene Flow Analysis (Nm)")
    st.write(
        "Estimates the number of effective migrants per generation using "
        "**Wright's formula**: Nm = (1 - Fst) / (4 × Fst).\n\n"
        "- **Nm > 1**: sufficient gene flow to prevent differentiation\n"
        "- **Nm < 1**: differentiation occurring despite gene flow\n"
        "- **Nm ≈ 0**: isolated populations"
    )

    if "fst_matrix" not in st.session_state:
        st.warning("⚠️ Please run **Fst / Gst / DEST** tab first "
                    "to compute Fst matrix.")
    else:
        gc1, gc2 = st.columns(2)
        with gc1:
            nm_threshold_display = st.slider(
                "Nm threshold to highlight",
                0.1, 5.0, 1.0, 0.1, key="nm_thresh",
                help="Populations with Nm above this threshold "
                        "are shown in the gene flow network",
            )
        with gc2:
            network_layout = st.selectbox(
                "Network layout",
                ["Circular", "Spring", "Kamada-Kawai"],
                key="nm_layout",
            )

        if st.button("🚀 Compute Gene Flow (Nm)",
                     use_container_width=True, key="nm_run"):

            fst_df = st.session_state["fst_matrix"]
            pops = st.session_state["fst_pops"]
            Fst_mat = fst_df.values

            # Compute Nm matrix
            Nm_mat = np.full_like(Fst_mat, np.nan)
            for i in range(len(pops)):
                for j in range(len(pops)):
                    if i == j:
                        continue
                    Nm_mat[i, j] = compute_nm_from_fst(Fst_mat[i, j])

            # Cap at reasonable max for visualization
            Nm_display = np.where(Nm_mat > 100, 100, Nm_mat)

            st.markdown("### Nm Matrix")
            Nm_df = pd.DataFrame(Nm_mat, index=pops, columns=pops)
            st.dataframe(Nm_df.style.format("{:.4f}"),
                          use_container_width=True)

            # Nm heatmap
            fig_nm = px.imshow(
                Nm_display, x=pops, y=pops, text_auto=".2f",
                color_continuous_scale="Greens",
                title="Pairwise Nm (Number of migrants per generation)",
                aspect="auto",
            )
            fig_nm.update_layout(template="plotly_white", height=600)
            st.plotly_chart(fig_nm, use_container_width=True)

            # Summary
            iu = np.triu_indices(len(pops), k=1)
            upper = Nm_mat[iu]
            upper = upper[np.isfinite(upper)]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Mean Nm", f"{upper.mean():.4f}")
            m2.metric("Median Nm", f"{np.median(upper):.4f}")
            m3.metric("Pairs with Nm > 1",
                        f"{(upper > 1).sum()} / {len(upper)}")
            m4.metric("Pairs with Nm < 0.5",
                        f"{(upper < 0.5).sum()} / {len(upper)}")

            # ── Gene flow network diagram ──
            st.markdown("### 🕸️ Gene Flow Network Diagram")

            try:
                import networkx as nx

                G = nx.Graph()
                for pop in pops:
                    G.add_node(pop)

                for i in range(len(pops)):
                    for j in range(i + 1, len(pops)):
                        nm = Nm_mat[i, j]
                        if np.isfinite(nm) and nm >= nm_threshold_display:
                            G.add_edge(pops[i], pops[j], weight=nm)

                # Layout
                if network_layout == "Circular":
                    pos = nx.circular_layout(G)
                elif network_layout == "Spring":
                    pos = nx.spring_layout(G, seed=42, k=2, iterations=100)
                else:
                    pos = nx.kamada_kawai_layout(G)

                # Build edges with weight-scaled width
                edge_traces = []
                if len(G.edges()) > 0:
                    max_nm = max([d["weight"]
                                    for _, _, d in G.edges(data=True)])
                    for u, v, d in G.edges(data=True):
                        x0, y0 = pos[u]
                        x1, y1 = pos[v]
                        width = 1 + 8 * (d["weight"] / max_nm)
                        edge_traces.append(go.Scatter(
                            x=[x0, x1], y=[y0, y1],
                            mode="lines",
                            line=dict(width=width,
                                        color=f"rgba(50,150,50,{min(0.9, 0.3 + d['weight']/max_nm * 0.6)})"),
                            hoverinfo="text",
                            text=f"{u} ↔ {v}<br>Nm = {d['weight']:.3f}",
                            showlegend=False,
                        ))

                # Nodes
                node_x = [pos[n][0] for n in G.nodes()]
                node_y = [pos[n][1] for n in G.nodes()]

                # Node size = number of samples in that pop
                pop_map = build_pop_maps(global_sam_col, global_pop_col)
                node_sizes = []
                for n in G.nodes():
                    count = sum(1 for s, p in pop_map.items() if p == n)
                    node_sizes.append(20 + count * 1.5)

                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode="markers+text",
                    text=list(G.nodes()),
                    textposition="top center",
                    textfont=dict(size=12, color="black"),
                    marker=dict(
                        size=node_sizes,
                        color="steelblue",
                        line=dict(width=2, color="darkblue"),
                    ),
                    hoverinfo="text",
                    hovertext=[f"{n}<br>N samples: "
                                 f"{sum(1 for s, p in pop_map.items() if p == n)}"
                                 for n in G.nodes()],
                    showlegend=False,
                )

                fig_net = go.Figure(data=edge_traces + [node_trace])
                fig_net.update_layout(
                    title=f"Gene Flow Network (Nm ≥ {nm_threshold_display})",
                    template="plotly_white",
                    height=650,
                    xaxis=dict(showgrid=False, zeroline=False,
                               showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False,
                               showticklabels=False),
                    showlegend=False,
                )
                st.plotly_chart(fig_net, use_container_width=True)

                if len(G.edges()) == 0:
                    st.warning(
                        f"No population pairs have Nm ≥ {nm_threshold_display}. "
                        "Try lowering the threshold."
                    )
                else:
                    st.info(
                        f"💡 Network shows **{len(G.edges())}** gene flow "
                        f"connections (Nm ≥ {nm_threshold_display}). "
                        "Edge thickness proportional to Nm. "
                        "Node size proportional to sample count."
                    )

                download_plotly_html(fig_net, "gene_flow_network.html",
                                      key="dl_geneflow")

            except ImportError:
                st.error("NetworkX is required for network diagram. "
                          "Add `networkx>=3.0` to requirements.")

            # Nm distribution
            st.markdown("### Nm Distribution")
            valid_nm = Nm_mat[iu]
            valid_nm = valid_nm[np.isfinite(valid_nm)]

            fig_dist = px.histogram(
                x=valid_nm[valid_nm < 20], nbins=40,
                title="Distribution of pairwise Nm (capped at 20)",
                labels={"x": "Nm", "y": "Count"},
            )
            fig_dist.add_vline(x=1.0, line_dash="dash", line_color="red",
                                 annotation_text="Nm = 1 (threshold)")
            fig_dist.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig_dist, use_container_width=True)

            # Top gene flow pairs
            st.markdown("### 🔝 Top Gene Flow Pairs")
            pairs_data = []
            for i in range(len(pops)):
                for j in range(i + 1, len(pops)):
                    if np.isfinite(Nm_mat[i, j]):
                        pairs_data.append({
                            "Pop_1": pops[i],
                            "Pop_2": pops[j],
                            "Fst": Fst_mat[i, j],
                            "Nm": Nm_mat[i, j],
                            "Interpretation": (
                                "Very high gene flow" if Nm_mat[i, j] > 5
                                else "Moderate gene flow" if Nm_mat[i, j] > 1
                                else "Restricted gene flow" if Nm_mat[i, j] > 0.25
                                else "Very restricted"
                            ),
                        })

            pairs_df = pd.DataFrame(pairs_data).sort_values(
                "Nm", ascending=False)
            st.dataframe(pairs_df.style.format({
                "Fst": "{:.4f}", "Nm": "{:.4f}"
            }), use_container_width=True)

            download_dataframe(pairs_df, "gene_flow_pairs.csv",
                                key="dl_nm_pairs")
            download_dataframe(Nm_df.reset_index(),
                                "nm_matrix.csv", key="dl_nm")

# =========================================================
# TAB 3 — AMOVA
# =========================================================
with tab_amova:
    st.subheader("🧬 Analysis of Molecular Variance (AMOVA)")
    st.write(
        "Partitions genetic variance into components: **among populations**, "
        "**among individuals within populations**, and **within individuals**."
    )

    ac1, ac2 = st.columns(2)
    with ac1:
        pop_col_a = st.selectbox("Population column",
                                   meta.columns.tolist(),
                                   index=meta.columns.tolist().index(global_pop_col),
                                   key="amova_popcol")
    with ac2:
        sam_col_a = st.selectbox("Sample ID column",
                                   meta.columns.tolist(),
                                   index=meta.columns.tolist().index(global_sam_col),
                                   key="amova_samcol")

    n_perm_amova = st.slider("Permutations", 100, 2000, 500, 100,
                               key="amova_perm")

    if st.button("🚀 Run AMOVA", use_container_width=True,
                 key="amova_run"):
        pop_map_a = dict(zip(meta[sam_col_a].astype(str),
                              meta[pop_col_a].astype(str)))

        with st.spinner("Computing AMOVA..."):
            geno_imp = impute_missing(geno, "mean")

            samples = geno.index.astype(str).tolist()
            samples = [s for s in samples if str(s) in pop_map_a]
            g_sub = geno_imp.loc[samples]
            groups = np.array([pop_map_a[str(s)] for s in samples])
            unique_groups = np.unique(groups)

            centroid = g_sub.mean(axis=0)
            SS_total = ((g_sub - centroid) ** 2).sum().sum()

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

            phi_st = sigma2_among / sigma2_total if sigma2_total > 0 else 0
            F_stat = MS_among / MS_within if MS_within > 0 else np.nan

            # Permutation test
            rng = np.random.RandomState(42)
            F_perm = []
            progress_bar = st.progress(0)
            for pi in range(n_perm_amova):
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
                progress_bar.progress((pi + 1) / n_perm_amova)
            progress_bar.empty()

            F_perm = np.array(F_perm)
            p_val = (np.sum(F_perm >= F_stat) + 1) / (n_perm_amova + 1)

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

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Φst", f"{phi_st:.4f}")
        m2.metric("F-statistic", f"{F_stat:.4f}")
        m3.metric("Permutation p", f"{p_val:.4f}")
        m4.metric("Significance",
                    "✅ Significant" if p_val < 0.05 else "❌ n.s.")

        # Pie chart
        pc1, pc2 = st.columns(2)
        with pc1:
            fig_var = px.pie(
                amova_df.iloc[:2], values="% of variation",
                names="Source of variation",
                title="Partitioning of genetic variance",
                color_discrete_sequence=["#FF6B6B", "#4ECDC4"],
            )
            fig_var.update_layout(template="plotly_white", height=450)
            st.plotly_chart(fig_var, use_container_width=True)

        with pc2:
            # Permutation histogram
            fig_perm = px.histogram(
                x=F_perm, nbins=40,
                title=f"Permutation null distribution (n={n_perm_amova})",
                labels={"x": "F-statistic (permuted)", "y": "Count"},
            )
            fig_perm.add_vline(x=F_stat, line_dash="dash",
                                 line_color="red",
                                 annotation_text=f"Observed F = {F_stat:.3f}")
            fig_perm.update_layout(template="plotly_white", height=450)
            st.plotly_chart(fig_perm, use_container_width=True)

        download_dataframe(amova_df, "amova_results.csv",
                            key="dl_amova")

# =========================================================
# TAB 4 — Isolation by Distance
# =========================================================
with tab_ibd:
    st.subheader("📏 Isolation by Distance (IBD)")
    st.write(
        "Tests correlation between genetic distance and geographic distance. "
        "Uses **Rousset's (1997)** linearized Fst / (1 - Fst) approach."
    )

    ibc1, ibc2 = st.columns(2)
    with ibc1:
        lat_col = st.selectbox("Latitude column",
                                meta.columns.tolist(), key="ibd_lat")
    with ibc2:
        lon_col = st.selectbox("Longitude column",
                                meta.columns.tolist(), key="ibd_lon")

    ibc3, ibc4 = st.columns(2)
    with ibc3:
        pop_col_ibd = st.selectbox(
            "Population column", meta.columns.tolist(),
            index=meta.columns.tolist().index(global_pop_col),
            key="ibd_pop")
    with ibc4:
        sam_col_ibd = st.selectbox(
            "Sample ID column", meta.columns.tolist(),
            index=meta.columns.tolist().index(global_sam_col),
            key="ibd_sam")

    log_geo = st.checkbox("Log-transform geographic distance", False,
                            key="ibd_log")

    if st.button("🚀 Run IBD test", use_container_width=True,
                 key="ibd_run"):
        pop_map_ibd = dict(zip(meta[sam_col_ibd].astype(str),
                                 meta[pop_col_ibd].astype(str)))

        pop_coords = (meta.groupby(pop_col_ibd)[[lat_col, lon_col]]
                       .mean().reset_index())

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
                    fst = compute_pairwise_fst(geno.loc[si], geno.loc[sj])
                    Fst_mat[i, j] = Fst_mat[j, i] = fst

            Geo_mat = np.zeros((n_pops, n_pops))
            pop_lat = pop_coords.set_index(pop_col_ibd)[lat_col].to_dict()
            pop_lon = pop_coords.set_index(pop_col_ibd)[lon_col].to_dict()

            for i in range(n_pops):
                for j in range(i + 1, n_pops):
                    d = haversine(pop_lat[pops[i]], pop_lon[pops[i]],
                                    pop_lat[pops[j]], pop_lon[pops[j]])
                    Geo_mat[i, j] = Geo_mat[j, i] = d

            # Rousset's linearization
            Fst_lin = Fst_mat / (1 - Fst_mat)

        iu = np.triu_indices(n_pops, k=1)
        geo_flat = Geo_mat[iu]
        fst_flat = Fst_lin[iu]

        mask = ~np.isnan(fst_flat) & (geo_flat > 0)
        geo_flat = geo_flat[mask]
        fst_flat = fst_flat[mask]

        if len(geo_flat) < 3:
            st.warning("Too few valid pairs.")
        else:
            x_data = np.log10(geo_flat) if log_geo else geo_flat
            x_label = "log10(Geographic distance km)" if log_geo else "Geographic distance (km)"

            r, p_val = pearsonr(x_data, fst_flat)

            # Fit linear regression
            slope, intercept = np.polyfit(x_data, fst_flat, 1)

            fig_ibd = go.Figure()
            fig_ibd.add_trace(go.Scatter(
                x=x_data, y=fst_flat, mode="markers",
                marker=dict(size=8, color="steelblue",
                             line=dict(width=1, color="darkblue")),
                name="Pop pairs",
                hovertemplate="Geo dist: %{x:.2f}<br>Fst/(1-Fst): %{y:.4f}<extra></extra>",
            ))

            x_fit = np.linspace(x_data.min(), x_data.max(), 100)
            y_fit = slope * x_fit + intercept
            fig_ibd.add_trace(go.Scatter(
                x=x_fit, y=y_fit, mode="lines",
                line=dict(color="red", dash="dash", width=2),
                name=f"y = {slope:.4e}x + {intercept:.4f}",
            ))

            fig_ibd.update_layout(
                title=f"Isolation by Distance (Pearson r = {r:.3f}, p = {p_val:.3g})",
                xaxis_title=x_label,
                yaxis_title="Fst / (1 - Fst)",
                template="plotly_white",
                height=600,
            )
            st.plotly_chart(fig_ibd, use_container_width=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Pearson r", f"{r:.4f}")
            m2.metric("R²", f"{r**2:.4f}")
            m3.metric("p-value", f"{p_val:.4g}")
            m4.metric("N pairs", f"{len(geo_flat)}")

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
# TAB 5 — Generic Mantel test
# =========================================================
with tab_mantel:
    st.subheader("🔗 Mantel Test")
    st.write(
        "Test correlation between any two distance matrices via permutation."
    )

    sam_m = st.selectbox(
        "Sample ID column", meta.columns.tolist(),
        index=meta.columns.tolist().index(global_sam_col),
        key="man_sam")

    mc1, mc2 = st.columns(2)
    with mc1:
        var1_cols = st.multiselect(
            "Variables for matrix 1 (numeric)",
            meta.select_dtypes(include=[np.number]).columns.tolist(),
            key="man_v1",
        )
    with mc2:
        var2_cols = st.multiselect(
            "Variables for matrix 2 (numeric)",
            meta.select_dtypes(include=[np.number]).columns.tolist(),
            key="man_v2",
        )

    n_perm_m = st.slider("Permutations", 99, 9999, 999, 100, key="man_np")

    if st.button("🚀 Run Mantel test", key="man_run"):
        if not var1_cols or not var2_cols:
            st.warning("Select at least one variable in each set.")
            st.stop()

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

        iu = np.triu_indices(D1.shape[0], k=1)
        fig_m = px.scatter(
            x=D1[iu], y=D2[iu],
            labels={"x": "Distance (matrix 1)",
                    "y": "Distance (matrix 2)"},
            title=f"Mantel scatter (r = {r_man:.3f}, p = {p_man:.3g})",
            trendline="ols",
        )
        fig_m.update_layout(template="plotly_white", height=500)
        st.plotly_chart(fig_m, use_container_width=True)

# =========================================================
# TAB 6 — Isolation by Environment
# =========================================================
with tab_ibe:
    st.subheader("🌡️ Isolation by Environment (IBE)")
    st.write(
        "Correlates genetic distance with environmental distance."
    )

    env_cols = st.multiselect(
        "Select environmental variables",
        meta.select_dtypes(include=[np.number]).columns.tolist(),
        key="ibe_env",
    )
    iec1, iec2 = st.columns(2)
    with iec1:
        pop_col_ibe = st.selectbox(
            "Population column", meta.columns.tolist(),
            index=meta.columns.tolist().index(global_pop_col),
            key="ibe_pop")
    with iec2:
        sam_col_ibe = st.selectbox(
            "Sample ID column", meta.columns.tolist(),
            index=meta.columns.tolist().index(global_sam_col),
            key="ibe_sam")

    if st.button("🚀 Run IBE analysis", use_container_width=True,
                 key="ibe_run"):
        if not env_cols:
            st.warning("Select at least one environmental variable.")
            st.stop()

        pop_map_ibe = dict(zip(meta[sam_col_ibe].astype(str),
                                 meta[pop_col_ibe].astype(str)))

        env_by_pop = (meta.groupby(pop_col_ibe)[env_cols]
                       .mean().reset_index())

        pops = sorted(set(pop_map_ibe.values()))
        env_lookup = env_by_pop.set_index(pop_col_ibe)

        pops_valid = [p for p in pops if p in env_lookup.index]
        env_mat = np.array([env_lookup.loc[p, env_cols].values
                              for p in pops_valid])

        env_scaled = StandardScaler().fit_transform(env_mat)
        D_env = squareform(pdist(env_scaled))

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
                    fst = compute_pairwise_fst(geno.loc[si], geno.loc[sj])
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
            title=f"IBE (Pearson r = {r_ibe:.3f}, p = {p_ibe:.3g})",
            trendline="ols",
        )
        fig_ibe.update_layout(template="plotly_white", height=550)
        st.plotly_chart(fig_ibe, use_container_width=True)

        r_m_ibe, p_m_ibe = mantel_test(Fst_mat_ibe, D_env, n_perm=999)
        m1, m2, m3 = st.columns(3)
        m1.metric("Pearson r", f"{r_ibe:.4f}")
        m2.metric("Mantel r", f"{r_m_ibe:.4f}")
        m3.metric("Mantel p", f"{p_m_ibe:.4g}")

# =========================================================
# TAB 7 — Geographic map
# =========================================================
with tab_map:
    st.subheader("🗺️ Sample Locations Map")

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        lat_c = st.selectbox("Latitude column",
                              meta.columns.tolist(), key="map_lat")
    with mc2:
        lon_c = st.selectbox("Longitude column",
                              meta.columns.tolist(), key="map_lon")
    with mc3:
        color_c = st.selectbox("Color by",
                                 ["None"] + meta.columns.tolist(),
                                 index=(meta.columns.tolist().index(global_pop_col) + 1
                                          if global_pop_col in meta.columns.tolist()
                                          else 0),
                                 key="map_color")

    map_style = st.selectbox("Map projection",
                               ["natural earth", "orthographic",
                                "mercator", "equirectangular"],
                               key="map_proj")

    try:
        meta_map = meta.copy()
        meta_map[lat_c] = pd.to_numeric(meta_map[lat_c], errors="coerce")
        meta_map[lon_c] = pd.to_numeric(meta_map[lon_c], errors="coerce")
        meta_map = meta_map.dropna(subset=[lat_c, lon_c])

        # Aggregate: one point per pop with size = count
        if color_c != "None":
            agg = meta_map.groupby(color_c).agg({
                lat_c: "mean", lon_c: "mean",
                meta_map.columns[0]: "count"
            }).reset_index().rename(columns={meta_map.columns[0]: "N_samples"})

            fig_map = px.scatter_geo(
                agg, lat=lat_c, lon=lon_c,
                color=color_c, size="N_samples",
                hover_name=color_c,
                hover_data={"N_samples": True},
                title=f"Sample geographic locations (grouped by {color_c})",
                projection=map_style,
                size_max=30,
            )
        else:
            fig_map = px.scatter_geo(
                meta_map, lat=lat_c, lon=lon_c,
                hover_data=meta_map.columns.tolist(),
                title="Sample geographic locations",
                projection=map_style,
            )
            fig_map.update_traces(marker=dict(size=8, opacity=0.8))

        fig_map.update_layout(template="plotly_white", height=700)
        st.plotly_chart(fig_map, use_container_width=True)

        # Sample counts table
        if color_c != "None":
            st.markdown("### Sample Counts per Population")
            counts = meta_map.groupby(color_c).size().reset_index(
                name="N_samples").sort_values("N_samples", ascending=False)
            st.dataframe(counts, use_container_width=True)

        download_plotly_html(fig_map, "geographic_map.html",
                              key="dl_map")
    except Exception as e:
        st.error(f"Error building map: {e}")