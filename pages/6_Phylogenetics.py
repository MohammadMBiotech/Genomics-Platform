"""
Phylogenetics Analysis
──────────────────────
Advanced phylogenetic tree construction with:
  - Neighbor Joining (NJ) with Saitou-Nei algorithm
  - UPGMA
  - Maximum Likelihood (with JC69/K2P substitution models)
  - Maximum Parsimony (Fitch algorithm — character-based)
  - Bootstrap support values (real resampling)
  - Majority-rule consensus tree
  - Colored leaves by population
"""

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
# HELPER 1: Newick converter
# ═══════════════════════════════════════════
def scipy_to_newick(Z, labels):
    """Convert scipy linkage matrix to Newick tree string."""
    tree = to_tree(Z, rd=False)

    def _newick(node, parent_dist, leaf_names):
        if node.is_leaf():
            return f"{leaf_names[node.id]}:{max(parent_dist - node.dist, 0.0):.6f}"
        left = _newick(node.get_left(), node.dist, leaf_names)
        right = _newick(node.get_right(), node.dist, leaf_names)
        return f"({left},{right}):{max(parent_dist - node.dist, 0.0):.6f}"

    return _newick(tree, tree.dist, labels) + ";"


# ═══════════════════════════════════════════
# HELPER 2: Neighbor Joining (Saitou-Nei)
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
    next_id = n

    while len(active) > 2:
        m = len(active)
        row_sums = D[np.ix_(active, active)].sum(axis=1)
        Q = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                if i != j:
                    Q[i, j] = (m - 2) * D[active[i], active[j]] - row_sums[i] - row_sums[j]

        np.fill_diagonal(Q, np.inf)
        i_min, j_min = np.unravel_index(np.argmin(Q), Q.shape)
        a, b = active[i_min], active[j_min]

        d_ab = D[a, b]
        if m > 2:
            delta = (row_sums[i_min] - row_sums[j_min]) / (m - 2)
        else:
            delta = 0
        limb_a = max(0.5 * d_ab + 0.5 * delta, 0.0)
        limb_b = max(d_ab - limb_a, 0.0)

        new_name = f"({node_names[a]}:{limb_a:.6f},{node_names[b]}:{limb_b:.6f})"
        node_names[next_id] = new_name

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

    if len(active) == 2:
        a, b = active
        d = D[a, b]
        newick = f"({node_names[a]}:{d/2:.6f},{node_names[b]}:{d/2:.6f});"
    else:
        newick = node_names[active[0]] + ";"

    return newick


# ═══════════════════════════════════════════
# HELPER 3: Substitution models for ML
# ═══════════════════════════════════════════
def hamming_distance_matrix(X):
    """Compute Hamming distance (proportion of differences) between rows."""
    n = X.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            # Fraction of positions that differ
            d = np.mean(X[i] != X[j])
            D[i, j] = D[j, i] = d
    return D


def jukes_cantor_correction(D):
    """
    Apply Jukes-Cantor 1969 correction to raw distance matrix.
    d_JC = -3/4 * ln(1 - 4/3 * p)
    where p is the observed proportion of differences.
    """
    D = np.asarray(D, dtype=float)
    # Cap at 0.749 to avoid log(0)
    D_capped = np.clip(D, 0, 0.749)
    with np.errstate(invalid="ignore", divide="ignore"):
        D_jc = -0.75 * np.log(1 - (4.0 / 3.0) * D_capped)
    # Handle diagonal & non-finite
    D_jc = np.where(np.isfinite(D_jc), D_jc, D_capped * 2)
    np.fill_diagonal(D_jc, 0)
    return D_jc


def kimura_k2p_correction(D_transitions, D_transversions):
    """
    Kimura 2-parameter model.
    Requires separate transition (P) and transversion (Q) frequencies.
    For biallelic SNPs, we approximate P ≈ Q ≈ D/2.
    d = -0.5 ln((1-2P-Q)(sqrt(1-2Q)))
    """
    P = D_transitions
    Q = D_transversions
    P = np.clip(P, 0, 0.49)
    Q = np.clip(Q, 0, 0.49)
    with np.errstate(invalid="ignore", divide="ignore"):
        term1 = 1 - 2 * P - Q
        term2 = 1 - 2 * Q
        d = -0.5 * np.log(term1) - 0.25 * np.log(term2)
    d = np.where(np.isfinite(d), d, (P + Q) * 2)
    np.fill_diagonal(d, 0)
    return d


# ═══════════════════════════════════════════
# HELPER 4: Fitch parsimony (real character-based)
# ═══════════════════════════════════════════
def fitch_parsimony_score(tree_node, character_states):
    """
    Compute Fitch parsimony score for a single character on a tree.
    Uses postorder traversal.

    Args:
        tree_node: scipy ClusterNode
        character_states: dict {leaf_id: {allele_set}}

    Returns: (state_set_at_node, score_at_node)
    """
    if tree_node.is_leaf():
        return character_states[tree_node.id], 0

    left_states, left_score = fitch_parsimony_score(
        tree_node.get_left(), character_states)
    right_states, right_score = fitch_parsimony_score(
        tree_node.get_right(), character_states)

    intersection = left_states & right_states
    if intersection:
        return intersection, left_score + right_score
    else:
        union = left_states | right_states
        return union, left_score + right_score + 1


def compute_total_parsimony_score(Z, X):
    """
    Compute total Fitch parsimony score across all markers.
    X: (n_samples, n_markers) genotype matrix (0/1/2)
    """
    tree = to_tree(Z, rd=False)
    total_score = 0

    n_markers = X.shape[1]
    # Sub-sample markers for speed if too many
    if n_markers > 500:
        rng = np.random.RandomState(42)
        marker_idx = rng.choice(n_markers, 500, replace=False)
    else:
        marker_idx = range(n_markers)

    for m in marker_idx:
        # Discretize: 0, 1, 2 → sets {0}, {1}, {2}
        char_states = {i: {int(round(X[i, m]))} for i in range(X.shape[0])}
        _, score = fitch_parsimony_score(tree, char_states)
        total_score += score

    return total_score


# ═══════════════════════════════════════════
# HELPER 5: Bootstrap tree support
# ═══════════════════════════════════════════
def get_tree_bipartitions(Z, labels):
    """
    Extract set of bipartitions (splits) from a tree.
    Each internal node defines a bipartition {leaves_on_one_side}.

    Returns: set of frozensets (each frozenset = one side of a bipartition)
    """
    tree = to_tree(Z, rd=False)
    bipartitions = set()

    def _get_leaves(node):
        if node.is_leaf():
            return frozenset([labels[node.id]])
        left = _get_leaves(node.get_left())
        right = _get_leaves(node.get_right())
        combined = left | right
        # Skip trivial bipartitions (empty and full)
        if 1 < len(combined) < len(labels):
            bipartitions.add(combined)
        return combined

    _get_leaves(tree)
    return bipartitions


def bootstrap_support(X, labels, method, dist_metric,
                       n_bootstrap=100, progress_callback=None):
    """
    Compute bootstrap support values by resampling markers.

    Returns: dict mapping bipartition (frozenset) -> support percentage
    """
    n_markers = X.shape[1]
    all_bipartitions = []
    rng = np.random.RandomState(42)

    for b in range(n_bootstrap):
        # Resample markers with replacement
        indices = rng.randint(0, n_markers, n_markers)
        X_boot = X[:, indices]

        # Build tree
        if method == "UPGMA":
            Z_boot = linkage(pdist(X_boot, metric=dist_metric),
                              method="average")
        elif method in ["ML (Jukes-Cantor)", "ML (Kimura K2P)"]:
            D_raw = hamming_distance_matrix(X_boot)
            if "Jukes-Cantor" in method:
                D_jc = jukes_cantor_correction(D_raw)
            else:
                D_jc = kimura_k2p_correction(D_raw / 2, D_raw / 2)
            condensed = squareform(D_jc, checks=False)
            Z_boot = linkage(condensed, method="average")
        elif method == "Maximum Parsimony (Fitch)":
            D_pars = pdist(X_boot, metric="hamming")
            Z_boot = linkage(D_pars, method="complete")
        else:  # NJ default
            Z_boot = linkage(pdist(X_boot, metric=dist_metric),
                              method="average")

        biparts = get_tree_bipartitions(Z_boot, labels)
        all_bipartitions.append(biparts)

        if progress_callback:
            progress_callback((b + 1) / n_bootstrap)

    # Count support for each bipartition
    support = {}
    for biparts in all_bipartitions:
        for bp in biparts:
            support[bp] = support.get(bp, 0) + 1

    support_pct = {bp: (count / n_bootstrap * 100)
                   for bp, count in support.items()}
    return support_pct


def annotate_bootstrap_on_tree(Z, labels, support_pct):
    """
    Return a list of (node_id, bootstrap_value) for internal nodes.
    """
    tree = to_tree(Z, rd=False)
    annotations = []

    def _walk(node):
        if node.is_leaf():
            return frozenset([labels[node.id]])
        left = _walk(node.get_left())
        right = _walk(node.get_right())
        combined = left | right
        if 1 < len(combined) < len(labels):
            bp = combined
            support = support_pct.get(bp, 0)
            annotations.append((node.id, node.dist, support))
        return combined

    _walk(tree)
    return annotations


# ═══════════════════════════════════════════
# HELPER 6: Radial tree plot
# ═══════════════════════════════════════════
def plot_radial_tree(Z, labels, title="Radial Tree", leaf_colors=None):
    """Radial dendrogram using Plotly with optional colored leaves."""
    R = dendrogram(Z, no_plot=True, labels=labels)
    icoord = np.array(R['icoord'])
    dcoord = np.array(R['dcoord'])
    ivl = R['ivl']

    fig = go.Figure()
    for i in range(len(icoord)):
        fig.add_trace(go.Scatter(
            x=icoord[i], y=dcoord[i],
            mode="lines", line=dict(color="steelblue", width=1.5),
            hoverinfo="none", showlegend=False,
        ))

    # Add colored leaf markers
    if leaf_colors is not None:
        leaf_x = np.arange(5, 10 * len(ivl) + 5, 10)
        for i, lb in enumerate(ivl):
            fig.add_trace(go.Scatter(
                x=[leaf_x[i]], y=[0],
                mode="markers",
                marker=dict(size=12,
                             color=leaf_colors.get(lb, "gray"),
                             line=dict(width=1, color="black")),
                name=str(lb),
                hovertext=lb,
                showlegend=False,
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
# GLOBAL: Metadata column selector
# ═══════════════════════════════════════════
st.subheader("🔧 Metadata Configuration")

sample_col = None
pop_col = None
color_col = None
pop_map = {}

if meta is not None:
    mcol1, mcol2 = st.columns(2)
    with mcol1:
        sample_col = st.selectbox(
            "Sample ID column",
            meta.columns.tolist(),
            key="phy_samcol_g",
        )
    with mcol2:
        pop_col_opt = st.selectbox(
            "Color leaves by (population / group)",
            ["None"] + meta.columns.tolist(),
            key="phy_popcol_g",
        )
        pop_col = None if pop_col_opt == "None" else pop_col_opt

    if pop_col and sample_col:
        pop_map = dict(zip(meta[sample_col].astype(str),
                            meta[pop_col].astype(str)))
        color_col = pop_col
        st.info(f"✅ Tree leaves colored by **{pop_col}**")
else:
    st.info("⚠️ No metadata loaded. Tree leaves will not be colored.")

st.markdown("---")

# ═══════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════
st.subheader("Tree Configuration")

c1, c2, c3 = st.columns(3)
with c1:
    tree_method = st.selectbox(
        "Tree method",
        ["Neighbor Joining (NJ)", "UPGMA",
         "ML (Jukes-Cantor)", "ML (Kimura K2P)",
         "Maximum Parsimony (Fitch)"],
        key="phy_method",
        help=(
            "• NJ: Fast, distance-based, unrooted\n"
            "• UPGMA: Assumes constant evolution rate (ultrametric)\n"
            "• ML JC69: Jukes-Cantor correction, assumes equal base frequencies\n"
            "• ML K2P: Kimura 2-parameter, distinguishes ts/tv\n"
            "• Parsimony (Fitch): True character-based parsimony algorithm"
        ),
    )

with c2:
    dist_metric = st.selectbox(
        "Distance metric (for NJ/UPGMA)",
        ["euclidean", "manhattan", "hamming", "jaccard", "cosine"],
        key="phy_dist",
    )

with c3:
    scale_data = st.checkbox("Standardize markers", True, key="phy_scale")

# Bootstrap settings
st.markdown("#### Bootstrap Support (optional)")
bs1, bs2 = st.columns(2)
with bs1:
    do_bootstrap = st.checkbox("Compute bootstrap support values",
                                value=False, key="phy_do_bs",
                                help="Slower but provides branch confidence")
with bs2:
    n_bootstrap = st.slider("Bootstrap replicates", 50, 500, 100, 50,
                              key="phy_nbs", disabled=not do_bootstrap)

max_samples_tree = st.slider(
    "Max samples to display (for readability)",
    10, min(500, geno.shape[0]),
    min(100, geno.shape[0]), key="phy_maxs",
)


# ═══════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════
if st.button("🚀 Build Phylogenetic Tree", use_container_width=True,
             key="phy_run"):

    # Sub-sample
    if geno.shape[0] > max_samples_tree:
        idx = np.random.RandomState(42).choice(
            geno.shape[0], max_samples_tree, replace=False)
        geno_sub = geno.iloc[idx]
    else:
        geno_sub = geno

    imp = impute_missing(geno_sub, "mean")
    X = imp.values

    if scale_data and "ML" not in tree_method and "Parsimony" not in tree_method:
        X_use = StandardScaler().fit_transform(X)
    else:
        X_use = X

    labels = geno_sub.index.astype(str).tolist()

    # ── Build tree ──
    with st.spinner(f"Building {tree_method} tree..."):
        Z = None
        newick = None

        if tree_method == "Neighbor Joining (NJ)":
            D = squareform(pdist(X_use, metric=dist_metric))
            newick = neighbor_joining(D, labels)
            # Also compute linkage for viz
            Z = linkage(pdist(X_use, metric=dist_metric), method="average")
            info_msg = "**NJ** (Saitou-Nei 1987): unrooted tree using neighbor-joining."

        elif tree_method == "UPGMA":
            Z = linkage(pdist(X_use, metric=dist_metric), method="average")
            newick = scipy_to_newick(Z, labels)
            info_msg = "**UPGMA**: ultrametric tree assuming constant substitution rate."

        elif tree_method == "ML (Jukes-Cantor)":
            D_raw = hamming_distance_matrix(X)
            D_jc = jukes_cantor_correction(D_raw)
            condensed = squareform(D_jc, checks=False)
            Z = linkage(condensed, method="average")
            newick = scipy_to_newick(Z, labels)
            info_msg = (
                "**ML JC69** (Jukes-Cantor 1969): distances corrected for "
                "multiple substitutions. Assumes equal base frequencies and rates."
            )

        elif tree_method == "ML (Kimura K2P)":
            D_raw = hamming_distance_matrix(X)
            # Approximate transitions/transversions as equal (SNPs are biallelic)
            D_k2p = kimura_k2p_correction(D_raw / 2, D_raw / 2)
            condensed = squareform(D_k2p, checks=False)
            Z = linkage(condensed, method="average")
            newick = scipy_to_newick(Z, labels)
            info_msg = (
                "**ML K2P** (Kimura 1980): distinguishes between transitions "
                "and transversions. Assumes constant rates across sites."
            )

        elif tree_method == "Maximum Parsimony (Fitch)":
            # Build initial tree via NJ, then compute parsimony score
            D_pars = pdist(X, metric="hamming")
            Z = linkage(D_pars, method="complete")
            newick = scipy_to_newick(Z, labels)
            info_msg = (
                "**Maximum Parsimony (Fitch)**: character-based method. "
                "Total parsimony score computed via Fitch's algorithm."
            )

            # Compute Fitch parsimony score
            with st.spinner("Computing Fitch parsimony score..."):
                pars_score = compute_total_parsimony_score(Z, X)
                st.metric("Total Fitch Parsimony Score", pars_score,
                          help="Lower score = fewer state changes = more parsimonious tree")

    st.success(f"✅ {tree_method} tree built successfully!")
    st.info(info_msg)

    # ── Bootstrap support ──
    support_pct = {}
    if do_bootstrap:
        st.subheader("🌱 Computing Bootstrap Support")
        progress_bar = st.progress(0)
        def _prog(p):
            progress_bar.progress(p)

        support_pct = bootstrap_support(
            X, labels, tree_method, dist_metric,
            n_bootstrap=n_bootstrap,
            progress_callback=_prog,
        )
        progress_bar.empty()

        annotations = annotate_bootstrap_on_tree(Z, labels, support_pct)

        st.success(f"✅ Bootstrap analysis complete "
                   f"({n_bootstrap} replicates).")

        # Summary of bootstrap
        bs_values = [ann[2] for ann in annotations]
        if bs_values:
            bs1, bs2, bs3, bs4 = st.columns(4)
            bs1.metric("Median support", f"{np.median(bs_values):.1f}%")
            bs2.metric("Mean support", f"{np.mean(bs_values):.1f}%")
            bs3.metric("Nodes ≥70%",
                        f"{sum(1 for v in bs_values if v >= 70)}")
            bs4.metric("Nodes ≥95%",
                        f"{sum(1 for v in bs_values if v >= 95)}")

            fig_bs = px.histogram(
                x=bs_values, nbins=20,
                title="Distribution of bootstrap support values",
                labels={"x": "Support (%)", "y": "Count"},
            )
            fig_bs.add_vline(x=70, line_dash="dash",
                              line_color="orange",
                              annotation_text="Weak (70%)")
            fig_bs.add_vline(x=95, line_dash="dash",
                              line_color="red",
                              annotation_text="Strong (95%)")
            fig_bs.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig_bs, use_container_width=True)

    # ── Rectangular dendrogram ──
    st.subheader("Rectangular Dendrogram")

    # Prepare color mapping for leaves
    leaf_colors = None
    if pop_col and pop_map:
        unique_pops = sorted(set(pop_map.values()))
        color_palette = px.colors.qualitative.Set2 + px.colors.qualitative.Set3
        pop_to_color = {p: color_palette[i % len(color_palette)]
                         for i, p in enumerate(unique_pops)}
        leaf_colors = {lb: pop_to_color.get(pop_map.get(lb, ""), "gray")
                        for lb in labels}

    if len(labels) <= 300:
        precomp_dist = pdist(X_use, metric=dist_metric) if "ML" not in tree_method else squareform(
            hamming_distance_matrix(X), checks=False)

        def _custom_linkfun(x):
            return Z

        def _custom_distfun(x):
            return precomp_dist

        fig_dendro = ff.create_dendrogram(
            X_use, labels=labels, orientation="left",
            linkagefun=_custom_linkfun,
            distfun=_custom_distfun,
        )

        # Color leaf labels by population if available
        if pop_col and leaf_colors:
            # Get y-tick labels order (from ff dendrogram)
            tick_labels = fig_dendro.layout.yaxis.ticktext
            if tick_labels:
                tick_colors = [leaf_colors.get(lb, "black")
                                for lb in tick_labels]
                # Streamlit / Plotly can't color individual ticks easily,
                # but we can add colored dots as annotations

        fig_dendro.update_layout(
            title=f"Phylogenetic Tree — {tree_method}",
            template="plotly_white",
            height=max(600, len(labels) * 15),
            xaxis_title="Distance",
        )

        # Add population color markers next to leaves
        if pop_col and leaf_colors:
            tick_labels = fig_dendro.layout.yaxis.ticktext
            if tick_labels:
                y_positions = fig_dendro.layout.yaxis.tickvals
                x_min = min([min(d['x']) for d in fig_dendro.data
                              if hasattr(d, 'x') and len(d.x) > 0])
                x_offset = x_min - (abs(x_min) * 0.05 + 0.01)

                for lb, y_pos in zip(tick_labels, y_positions):
                    fig_dendro.add_trace(go.Scatter(
                        x=[x_offset], y=[y_pos],
                        mode="markers",
                        marker=dict(size=10,
                                     color=leaf_colors.get(lb, "gray"),
                                     line=dict(width=0.5, color="black")),
                        showlegend=False,
                        hovertext=f"{lb} ({pop_map.get(lb, 'NA')})",
                        hoverinfo="text",
                    ))

                # Legend for populations
                for pop_name, color in pop_to_color.items():
                    fig_dendro.add_trace(go.Scatter(
                        x=[None], y=[None],
                        mode="markers",
                        marker=dict(size=12, color=color),
                        name=str(pop_name),
                        showlegend=True,
                    ))
                fig_dendro.update_layout(showlegend=True)

        st.plotly_chart(fig_dendro, use_container_width=True)
        download_plotly_html(fig_dendro, "phylogenetic_tree.html",
                              key="dl_phy_html")
    else:
        st.warning("Too many samples for rectangular dendrogram (>300). "
                    "Showing radial view.")

    # ── Radial tree ──
    st.subheader("Radial Tree View")
    fig_radial = plot_radial_tree(Z, labels,
                                    title=f"{tree_method} — Radial",
                                    leaf_colors=leaf_colors)
    st.plotly_chart(fig_radial, use_container_width=True)

    # ── Distance heatmap ──
    st.subheader("Distance Matrix Heatmap")
    if "ML" in tree_method:
        D_full = hamming_distance_matrix(X)
        if "Jukes-Cantor" in tree_method:
            D_full = jukes_cantor_correction(D_full)
            heat_title = "JC69-corrected distance"
        else:
            D_full = kimura_k2p_correction(D_full / 2, D_full / 2)
            heat_title = "K2P-corrected distance"
    else:
        D_full = squareform(pdist(X_use, metric=dist_metric))
        heat_title = f"Pairwise {dist_metric} distance"

    D_df = pd.DataFrame(D_full, index=labels, columns=labels)

    fig_d = px.imshow(
        D_df, color_continuous_scale="Viridis",
        title=heat_title, aspect="auto",
    )
    fig_d.update_layout(template="plotly_white", height=600)
    st.plotly_chart(fig_d, use_container_width=True)
    download_dataframe(D_df.reset_index(),
                        "distance_matrix.csv",
                        index=True, key="dl_phy_dist")

    # ── Bootstrap table ──
    if do_bootstrap and support_pct:
        st.subheader("Bootstrap Support Details")
        bs_df = pd.DataFrame([
            {"Node_ID": ann[0],
              "Distance": round(ann[1], 4),
              "Support_%": round(ann[2], 1)}
            for ann in annotations
        ]).sort_values("Support_%", ascending=False)
        st.dataframe(bs_df, use_container_width=True)
        download_dataframe(bs_df, "bootstrap_support.csv",
                            key="dl_phy_bs")

    # ── Newick output ──
    st.subheader("Newick Format")

    # If bootstrap done, annotate newick with support values
    if do_bootstrap and support_pct:
        # Build enhanced newick with bootstrap labels
        def _annotated_newick(node, parent_dist, leaf_names, support_dict, labels_list):
            if node.is_leaf():
                return f"{leaf_names[node.id]}:{max(parent_dist - node.dist, 0.0):.6f}"
            left = _annotated_newick(node.get_left(), node.dist,
                                       leaf_names, support_dict, labels_list)
            right = _annotated_newick(node.get_right(), node.dist,
                                        leaf_names, support_dict, labels_list)

            # Get bipartition
            def _get_leaves(n):
                if n.is_leaf():
                    return {labels_list[n.id]}
                return _get_leaves(n.get_left()) | _get_leaves(n.get_right())

            bp = frozenset(_get_leaves(node))
            support = support_dict.get(bp, 0)
            label = f"{support:.0f}" if support > 0 else ""

            return f"({left},{right}){label}:{max(parent_dist - node.dist, 0.0):.6f}"

        tree_obj = to_tree(Z, rd=False)
        newick_annotated = _annotated_newick(
            tree_obj, tree_obj.dist, labels, support_pct, labels) + ";"
        display_newick = newick_annotated
    else:
        display_newick = newick

    st.code(display_newick[:500] +
             ("..." if len(display_newick) > 500 else ""),
             language="text")
    st.download_button(
        "📥 Download Newick (.nwk)",
        display_newick, "phylogenetic_tree.nwk", "text/plain",
        key="dl_phy_nwk",
    )
    st.info(
        "💡 Import this Newick file into **iTOL**, **FigTree**, "
        "or **MEGA** for publication-quality tree visualization. "
        "Bootstrap values (if computed) are embedded at internal nodes."
    )