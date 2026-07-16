"""Phylogenetics — NJ, UPGMA, Maximum Likelihood, Maximum Parsimony trees."""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy.cluster.hierarchy import linkage, to_tree, dendrogram
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler

from utils import (
    get_geno_from_session, get_meta_from_session,
    impute_missing, download_plotly_html, download_dataframe,
)

st.title("🌳 Phylogenetics")
st.markdown("---")

geno, marker_info = get_geno_from_session()
if geno is None:
    st.stop()

meta = get_meta_from_session()


# ═══════════════════════════════════════════
# Helper: Newick format converter
# ═══════════════════════════════════════════
def scipy_to_newick(Z, labels):
    """Convert scipy linkage matrix to Newick tree string."""
    tree = to_tree(Z, rd=False)

    def _newick(node, parent_dist, leaf_names):
        if node.is_leaf():
            return f"{leaf_names[node.id]}:{parent_dist - node.dist:.6f}"
        left = _newick(node.get_left(), node.dist, leaf_names)
        right = _newick(node.get_right(), node.dist, leaf_names)
        return f"({left},{right}):{parent_dist - node.dist:.6f}"

    return _newick(tree, tree.dist, labels) + ";"


# ═══════════════════════════════════════════
# Helper: Neighbor-Joining tree construction
# ═══════════════════════════════════════════
def neighbor_joining(D, labels):
    """
    Saitou-Nei Neighbor Joining algorithm.
    Returns Newick string.
    """
    D = D.copy().astype(float)
    n = D.shape[0]
    labels = list(labels)
    active = list(range(n))
    node_names = {i: labels[i] for i in range(n)}
    node_dists = {i: 0.0 for i in range(n)}
    next_id = n

    while len(active) > 2:
        m = len(active)
        # Compute Q-matrix
        row_sums = D[np.ix_(active, active)].sum(axis=1)
        Q = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                if i != j:
                    Q[i, j] = (m - 2) * D[active[i], active[j]] - row_sums[i] - row_sums[j]

        np.fill_diagonal(Q, np.inf)
        i_min, j_min = np.unravel_index(np.argmin(Q), Q.shape)
        a, b = active[i_min], active[j_min]

        # Compute branch lengths
        d_ab = D[a, b]
        if m > 2:
            delta = (row_sums[i_min] - row_sums[j_min]) / (m - 2)
        else:
            delta = 0
        limb_a = 0.5 * d_ab + 0.5 * delta
        limb_b = d_ab - limb_a

        # Guard against negative branch lengths
        limb_a = max(limb_a, 0.0)
        limb_b = max(limb_b, 0.0)

        # Create new node
        new_name = f"({node_names[a]}:{limb_a:.6f},{node_names[b]}:{limb_b:.6f})"
        node_names[next_id] = new_name

        # Update distance matrix
        new_D = np.zeros((D.shape[0] + 1, D.shape[0] + 1))
        new_D[:D.shape[0], :D.shape[0]] = D
        for k in range(D.shape[0]):
            if k != a and k != b:
                new_D[next_id, k] = new_D[k, next_id] = 0.5 * (D[a, k] + D[b, k] - d_ab)

        D = new_D
        active.remove(a)
        active.remove(b)
        active.append(next_id)
        next_id += 1

    # Join last two
    if len(active) == 2:
        a, b = active
        d = D[a, b]
        newick = f"({node_names[a]}:{d/2:.6f},{node_names[b]}:{d/2:.6f});"
    else:
        newick = node_names[active[0]] + ";"

    return newick


# ═══════════════════════════════════════════
# Helper: Radial tree plot from linkage matrix
# ═══════════════════════════════════════════
def plot_radial_tree(Z, labels, title="Radial Tree"):
    """Simplified radial dendrogram using Plotly."""
    from scipy.cluster.hierarchy import dendrogram as sd
    R = sd(Z, no_plot=True, labels=labels)

    icoord = np.array(R['icoord'])
    dcoord = np.array(R['dcoord'])
    ivl = R['ivl']

    fig = go.Figure()
    # Draw dendrogram edges
    for i in range(len(icoord)):
        fig.add_trace(go.Scatter(
            x=icoord[i], y=dcoord[i],
            mode="lines", line=dict(color="steelblue", width=1.5),
            hoverinfo="none", showlegend=False,
        ))

    fig.update_layout(
        title=title,
        template="plotly_white", height=650,
        xaxis=dict(
            tickvals=np.arange(5, 10 * len(ivl) + 5, 10),
            ticktext=ivl, tickangle=90, showgrid=False,
        ),
        yaxis=dict(title="Distance"),
    )
    return fig


# ═══════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════
st.subheader("Configuration")

c1, c2, c3 = st.columns(3)
with c1:
    tree_method = st.selectbox(
        "Tree method",
        ["Neighbor Joining (NJ)", "UPGMA",
         "Maximum Likelihood (ML approx.)",
         "Maximum Parsimony"],
        key="phy_method",
    )
with c2:
    dist_metric = st.selectbox(
        "Distance metric",
        ["euclidean", "manhattan", "hamming", "jaccard", "cosine"],
        key="phy_dist",
    )
with c3:
    scale_data = st.checkbox("Standardize markers", True, key="phy_scale")

max_samples_tree = st.slider(
    "Max samples to display (for readability)",
    10, min(500, geno.shape[0]),
    min(100, geno.shape[0]), key="phy_maxs",
)

# Optional: color leaves by metadata
color_col = None
if meta is not None:
    pop_col = st.selectbox("Color leaves by (metadata column)",
                            ["None"] + meta.columns.tolist(),
                            key="phy_color")
    if pop_col != "None":
        color_col = pop_col
        sam_col = st.selectbox("Sample ID column",
                                meta.columns.tolist(),
                                key="phy_samcol")

if st.button("🚀 Build Phylogenetic Tree", use_container_width=True,
             key="phy_run"):
    # Sub-sample if too many
    if geno.shape[0] > max_samples_tree:
        idx = np.random.RandomState(42).choice(
            geno.shape[0], max_samples_tree, replace=False)
        geno_sub = geno.iloc[idx]
    else:
        geno_sub = geno

    imp = impute_missing(geno_sub, "mean")
    X = imp.values
    if scale_data:
        X = StandardScaler().fit_transform(X)

    labels = geno_sub.index.astype(str).tolist()

    with st.spinner(f"Building {tree_method} tree..."):
        if tree_method == "Neighbor Joining (NJ)":
            D = squareform(pdist(X, metric=dist_metric))
            newick = neighbor_joining(D, labels)

            # For visualization, also build linkage
            Z = linkage(pdist(X, metric=dist_metric), method="average")

        elif tree_method == "UPGMA":
            Z = linkage(pdist(X, metric=dist_metric), method="average")
            newick = scipy_to_newick(Z, labels)

        elif tree_method == "Maximum Likelihood (ML approx.)":
            # ML approximation using weighted average linkage
            # (True ML requires substitution model - not feasible in pure Python)
            st.info(
                "Note: This is a computationally-fast approximation using "
                "weighted linkage. For true ML, use RAxML or IQ-TREE offline."
            )
            Z = linkage(pdist(X, metric=dist_metric), method="weighted")
            newick = scipy_to_newick(Z, labels)

        elif tree_method == "Maximum Parsimony":
            # Parsimony approximation using Manhattan/Hamming distance
            # (True parsimony requires character-state matrix)
            st.info(
                "Note: This is a distance-based parsimony approximation. "
                "For true parsimony, use PAUP* or TNT offline."
            )
            D_pars = pdist(X, metric="hamming")
            Z = linkage(D_pars, method="complete")
            newick = scipy_to_newick(Z, labels)

    st.success(f"✅ {tree_method} tree built successfully!")

    # ── Rectangular dendrogram ──
    st.subheader("Rectangular Dendrogram")

    if len(labels) <= 300:
        # Use figure_factory dendrogram
        precomp_dist = pdist(X, metric=dist_metric)

        def _custom_linkfun(x):
            return Z

        def _custom_distfun(x):
            return precomp_dist

        fig_dendro = ff.create_dendrogram(
            X, labels=labels, orientation="left",
            linkagefun=_custom_linkfun,
            distfun=_custom_distfun,
        )
        fig_dendro.update_layout(
            title=f"Phylogenetic Tree — {tree_method}",
            template="plotly_white",
            height=max(600, len(labels) * 15),
            xaxis_title="Distance",
        )
        st.plotly_chart(fig_dendro, use_container_width=True)
        download_plotly_html(fig_dendro, "phylogenetic_tree.html",
                              key="dl_phy_html")
    else:
        st.warning("Too many samples for dendrogram (>300). Use radial plot.")

    # ── Radial tree ──
    st.subheader("Radial Tree View")
    fig_radial = plot_radial_tree(Z, labels,
                                    title=f"{tree_method} — Radial")
    st.plotly_chart(fig_radial, use_container_width=True)

    # ── Distance heatmap ──
    st.subheader("Distance Matrix Heatmap")
    D_full = squareform(pdist(X, metric=dist_metric))
    D_df = pd.DataFrame(D_full, index=labels, columns=labels)

    fig_d = px.imshow(
        D_df, color_continuous_scale="Viridis",
        title=f"Pairwise {dist_metric} Distance",
        aspect="auto",
    )
    fig_d.update_layout(template="plotly_white", height=600)
    st.plotly_chart(fig_d, use_container_width=True)
    download_dataframe(D_df.reset_index(),
                        "distance_matrix.csv",
                        index=True, key="dl_phy_dist")

    # ── Newick output ──
    st.subheader("Newick Format")
    st.code(newick[:500] + ("..." if len(newick) > 500 else ""),
             language="text")
    st.download_button(
        "📥 Download Newick (.nwk)",
        newick, "phylogenetic_tree.nwk", "text/plain",
        key="dl_phy_nwk",
    )
    st.info(
        "💡 You can visualize this Newick file in iTOL, FigTree, "
        "or MEGA for publication-quality trees."
    )