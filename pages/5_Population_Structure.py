"""
Population Structure Analysis
─────────────────────────────
Features:
  - PCA (2D, 3D, Biplot, Scree)
  - PCoA (multi-metric)
  - STRUCTURE-like with CLUMPP alignment + Evanno's ΔK
  - fastStructure (Bayesian GMM)
  - True labels in legends
  - Consensus Q-matrix across replicates
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform

from utils import (
    get_geno_from_session, get_meta_from_session,
    impute_missing, download_plotly_html, download_dataframe,
)

st.title("🧩 Population Structure")
st.markdown("---")

geno, marker_info = get_geno_from_session()
if geno is None:
    st.stop()

meta = get_meta_from_session()


# ═══════════════════════════════════════════
# GLOBAL: Metadata column selector (shared across tabs)
# ═══════════════════════════════════════════
st.subheader("🔧 Metadata Configuration")

sample_col = None
pop_col = None
color_col = None

if meta is not None:
    mcol1, mcol2 = st.columns(2)
    with mcol1:
        sample_col = st.selectbox(
            "Sample ID column (in metadata)",
            meta.columns.tolist(),
            key="ps_samcol",
        )
    with mcol2:
        pop_col_opt = st.selectbox(
            "Population / Group column",
            ["None"] + meta.columns.tolist(),
            key="ps_popcol",
        )
        pop_col = None if pop_col_opt == "None" else pop_col_opt

    if pop_col:
        color_col = pop_col
        st.info(f"✅ Using **{pop_col}** as population labels across all tabs.")

    # Build a sample -> population map for later use
    if sample_col and pop_col:
        pop_map = dict(zip(meta[sample_col].astype(str),
                            meta[pop_col].astype(str)))
    else:
        pop_map = {}
else:
    st.warning("⚠️ No metadata loaded. Analyses will use default sample IDs.")
    pop_map = {}


def _attach_metadata(df, sample_column="Sample"):
    """Attach population labels from metadata."""
    df = df.copy()
    df[sample_column] = df[sample_column].astype(str)
    if pop_col and sample_col and meta is not None:
        m_sub = meta[[sample_col, pop_col]].drop_duplicates()
        m_sub[sample_col] = m_sub[sample_col].astype(str)
        df = df.merge(m_sub, left_on=sample_column,
                      right_on=sample_col, how="left")
    return df


st.markdown("---")

tab_pca, tab_pcoa, tab_struct, tab_fast = st.tabs([
    "📉 PCA",
    "📊 PCoA",
    "🧬 STRUCTURE (Admixture)",
    "⚡ fastStructure",
])


# =========================================================
# TAB 1 — PCA
# =========================================================
with tab_pca:
    st.subheader("Principal Component Analysis")

    imp_method = st.selectbox("Missing value imputation",
                               ["mean", "median", "zero"],
                               key="pca_imp")
    scale_data = st.checkbox("Standardize markers", True, key="pca_scale")

    geno_imp = impute_missing(geno, method=imp_method)
    X = geno_imp.values

    if scale_data:
        X = StandardScaler().fit_transform(X)

    max_comp = min(20, X.shape[0], X.shape[1])
    n_components = st.slider("Number of components", 2, max_comp,
                              min(10, max_comp), key="pca_ncomp")

    with st.spinner("Running PCA..."):
        pca = PCA(n_components=n_components)
        scores = pca.fit_transform(X)

    var_exp = pca.explained_variance_ratio_ * 100

    scores_df = pd.DataFrame(
        scores, index=geno.index,
        columns=[f"PC{i+1}" for i in range(n_components)],
    ).reset_index().rename(columns={"index": "Sample"})

    # Attach real labels
    scores_df = _attach_metadata(scores_df)

    # ── Scree plot ──
    st.markdown("#### Scree Plot")
    var_df = pd.DataFrame({
        "Component": [f"PC{i+1}" for i in range(n_components)],
        "Variance (%)": var_exp,
        "Cumulative (%)": np.cumsum(var_exp),
    })
    fig_scree = go.Figure()
    fig_scree.add_trace(go.Bar(x=var_df["Component"],
                                 y=var_df["Variance (%)"],
                                 name="Individual",
                                 marker_color="steelblue"))
    fig_scree.add_trace(go.Scatter(x=var_df["Component"],
                                     y=var_df["Cumulative (%)"],
                                     mode="lines+markers",
                                     name="Cumulative",
                                     yaxis="y2",
                                     line=dict(color="red", width=2)))
    fig_scree.update_layout(
        title="PCA Scree Plot",
        yaxis=dict(title="Variance (%)"),
        yaxis2=dict(title="Cumulative (%)", overlaying="y", side="right"),
        template="plotly_white", height=450,
    )
    st.plotly_chart(fig_scree, use_container_width=True)

    # ── 2D scatter ──
    st.markdown("#### 2D PCA Plot")
    p2c1, p2c2 = st.columns(2)
    with p2c1:
        x_pc = st.selectbox("X axis", var_df["Component"], index=0, key="pca_x")
    with p2c2:
        y_pc = st.selectbox("Y axis", var_df["Component"], index=1, key="pca_y")

    show_labels = st.checkbox("Show sample labels", False, key="pca_lbl")

    x_idx = int(x_pc.replace("PC", "")) - 1
    y_idx = int(y_pc.replace("PC", "")) - 1

    fig_2d = px.scatter(
        scores_df, x=x_pc, y=y_pc, color=color_col,
        hover_data=["Sample"],
        text="Sample" if show_labels else None,
        labels={
            x_pc: f"{x_pc} ({var_exp[x_idx]:.1f}%)",
            y_pc: f"{y_pc} ({var_exp[y_idx]:.1f}%)",
        },
        title="PCA - 2D",
    )
    if show_labels:
        fig_2d.update_traces(textposition="top center")
    fig_2d.update_traces(marker=dict(size=9,
                                       line=dict(width=0.5, color="darkslategrey")))
    fig_2d.update_layout(template="plotly_white", height=650)
    st.plotly_chart(fig_2d, use_container_width=True)

    # ── 3D scatter ──
    if n_components >= 3:
        st.markdown("#### 3D PCA Plot")
        fig_3d = px.scatter_3d(
            scores_df, x="PC1", y="PC2", z="PC3",
            color=color_col, hover_data=["Sample"],
            title="PCA - 3D",
            labels={
                "PC1": f"PC1 ({var_exp[0]:.1f}%)",
                "PC2": f"PC2 ({var_exp[1]:.1f}%)",
                "PC3": f"PC3 ({var_exp[2]:.1f}%)",
            },
        )
        fig_3d.update_traces(marker=dict(size=5))
        fig_3d.update_layout(template="plotly_white", height=700)
        st.plotly_chart(fig_3d, use_container_width=True)

    # ── Loadings & Biplot ──
    st.markdown("#### Loadings & Biplot")
    loadings = pd.DataFrame(
        pca.components_.T,
        index=geno.columns,
        columns=[f"PC{i+1}" for i in range(n_components)],
    ).reset_index().rename(columns={"index": "Marker"})

    st.dataframe(loadings.head(50), use_container_width=True)
    download_dataframe(loadings, "pca_loadings.csv", key="dl_pca_load")

    top_n_load = st.slider("Top N markers in biplot", 5, 50, 20, key="pca_bp_n")
    loadings["magnitude"] = np.sqrt(loadings[x_pc]**2 + loadings[y_pc]**2)
    top_markers = loadings.nlargest(top_n_load, "magnitude")

    scale_factor = 3 * (scores_df[[x_pc, y_pc]].abs().max().max() /
                         top_markers[[x_pc, y_pc]].abs().max().max())

    fig_bp = go.Figure()
    if color_col and color_col in scores_df.columns:
        for cat in scores_df[color_col].dropna().unique():
            sub = scores_df[scores_df[color_col] == cat]
            fig_bp.add_trace(go.Scatter(
                x=sub[x_pc], y=sub[y_pc],
                mode="markers", name=str(cat),
                marker=dict(size=8, opacity=0.7),
            ))
    else:
        fig_bp.add_trace(go.Scatter(
            x=scores_df[x_pc], y=scores_df[y_pc],
            mode="markers", name="Samples",
            marker=dict(size=8, opacity=0.7, color="steelblue"),
        ))

    for _, row in top_markers.iterrows():
        fig_bp.add_annotation(
            x=row[x_pc] * scale_factor, y=row[y_pc] * scale_factor,
            ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1,
            arrowwidth=1.5, arrowcolor="red",
        )
        fig_bp.add_annotation(
            x=row[x_pc] * scale_factor * 1.1,
            y=row[y_pc] * scale_factor * 1.1,
            text=str(row["Marker"])[:8], showarrow=False,
            font=dict(size=9, color="red"),
        )

    fig_bp.update_layout(
        title="PCA Biplot", template="plotly_white",
        xaxis_title=f"{x_pc} ({var_exp[x_idx]:.1f}%)",
        yaxis_title=f"{y_pc} ({var_exp[y_idx]:.1f}%)",
        height=700,
    )
    st.plotly_chart(fig_bp, use_container_width=True)

    download_dataframe(scores_df, "pca_scores.csv", key="dl_pca_scores")


# =========================================================
# TAB 2 — PCoA
# =========================================================
with tab_pcoa:
    st.subheader("Principal Coordinates Analysis (PCoA)")
    st.write(
        "PCoA on a distance matrix. Suitable for non-linear distances "
        "(e.g., Jaccard, IBS)."
    )

    dist_metric = st.selectbox(
        "Distance metric",
        ["euclidean", "manhattan", "hamming", "jaccard", "cosine"],
        key="pcoa_dist",
    )
    n_comp_pcoa = st.slider("Components", 2, 10, 3, key="pcoa_ncomp")

    if st.button("🚀 Run PCoA", key="pcoa_run"):
        with st.spinner("Computing distance matrix..."):
            imp = impute_missing(geno, "mean")
            D = squareform(pdist(imp.values, metric=dist_metric))

            n = D.shape[0]
            J = np.eye(n) - np.ones((n, n)) / n
            B = -0.5 * J @ (D**2) @ J

            eigvals, eigvecs = np.linalg.eigh(B)
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]

            pos_idx = eigvals > 1e-9
            coords = eigvecs[:, pos_idx] * np.sqrt(eigvals[pos_idx])
            coords = coords[:, :n_comp_pcoa]

            var_exp_pcoa = eigvals[pos_idx][:n_comp_pcoa] / eigvals[pos_idx].sum() * 100

        pcoa_df = pd.DataFrame(
            coords, index=geno.index,
            columns=[f"Axis{i+1}" for i in range(coords.shape[1])],
        ).reset_index().rename(columns={"index": "Sample"})

        pcoa_df = _attach_metadata(pcoa_df)

        st.markdown("#### PCoA Variance Explained")
        var_pcoa_df = pd.DataFrame({
            "Axis": [f"Axis{i+1}" for i in range(len(var_exp_pcoa))],
            "Variance (%)": var_exp_pcoa,
        })
        fig_var = px.bar(var_pcoa_df, x="Axis", y="Variance (%)",
                          title="PCoA Variance Explained")
        fig_var.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig_var, use_container_width=True)

        st.markdown("#### PCoA 2D Plot")
        fig_pcoa = px.scatter(
            pcoa_df, x="Axis1", y="Axis2", color=color_col,
            hover_data=["Sample"],
            labels={
                "Axis1": f"Axis1 ({var_exp_pcoa[0]:.1f}%)",
                "Axis2": f"Axis2 ({var_exp_pcoa[1]:.1f}%)",
            },
            title="PCoA - 2D",
        )
        fig_pcoa.update_traces(marker=dict(size=9,
                                             line=dict(width=0.5,
                                                       color="darkslategrey")))
        fig_pcoa.update_layout(template="plotly_white", height=650)
        st.plotly_chart(fig_pcoa, use_container_width=True)

        if coords.shape[1] >= 3:
            fig_pcoa_3d = px.scatter_3d(
                pcoa_df, x="Axis1", y="Axis2", z="Axis3",
                color=color_col, hover_data=["Sample"],
                title="PCoA - 3D",
            )
            fig_pcoa_3d.update_traces(marker=dict(size=5))
            fig_pcoa_3d.update_layout(template="plotly_white", height=700)
            st.plotly_chart(fig_pcoa_3d, use_container_width=True)

        download_dataframe(pcoa_df, "pcoa_coordinates.csv", key="dl_pcoa")


# =========================================================
# TAB 3 — STRUCTURE with CLUMPP alignment + Evanno's ΔK
# =========================================================
def align_q_matrices(Q_list):
    """
    CLUMPP-like alignment of multiple Q-matrices using Hungarian algorithm.
    Uses the first Q-matrix as reference and permutes columns of others
    to maximize match.

    Returns list of aligned Q-matrices.
    """
    if len(Q_list) <= 1:
        return Q_list

    reference = Q_list[0]
    aligned = [reference]

    for Q in Q_list[1:]:
        # Compute cost matrix: negative correlation
        K = Q.shape[1]
        cost = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                corr = np.corrcoef(reference[:, i], Q[:, j])[0, 1]
                if np.isnan(corr):
                    corr = 0
                cost[i, j] = -corr

        # Solve assignment
        row_ind, col_ind = linear_sum_assignment(cost)
        Q_aligned = Q[:, col_ind]
        aligned.append(Q_aligned)

    return aligned


def consensus_q_matrix(Q_list):
    """Compute average Q-matrix across aligned replicates."""
    if len(Q_list) == 0:
        return None
    aligned = align_q_matrices(Q_list)
    return np.mean(aligned, axis=0)


def compute_evanno_delta_k(k_values, ln_probs):
    """
    Evanno's ΔK method.
    ΔK = |L''(K)| / std(L(K))
    where L''(K) = L(K+1) - 2L(K) + L(K-1)
    """
    k_values = np.array(k_values)
    ln_probs = np.array(ln_probs)  # mean L(K) per K
    n = len(k_values)

    if n < 3:
        return np.array([]), np.array([])

    delta_k = []
    valid_k = []
    for i in range(1, n - 1):
        L_prev = ln_probs[i - 1]
        L_curr = ln_probs[i]
        L_next = ln_probs[i + 1]
        second_deriv = abs(L_next - 2 * L_curr + L_prev)
        delta_k.append(second_deriv)
        valid_k.append(k_values[i])

    return np.array(valid_k), np.array(delta_k)


with tab_struct:
    st.subheader("🧬 STRUCTURE-like Admixture Analysis")
    st.write(
        "Uses Gaussian Mixture Models (GMM) as a computational proxy for "
        "STRUCTURE. Implements **CLUMPP-like alignment** across replicates "
        "and **Evanno's ΔK method** for optimal K selection."
    )

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        k_min = st.number_input("K min", 2, 20, 2, key="struct_kmin")
    with sc2:
        k_max = st.number_input("K max", 2, 20, 10, key="struct_kmax")
    with sc3:
        n_reps = st.slider("Repetitions per K", 1, 20, 10, key="struct_reps")

    sc4, sc5 = st.columns(2)
    with sc4:
        n_pcs_struct_slider = st.slider(
            "PCA reduction (# PCs)", 5, 50, 20, key="struct_npcs",
            help="Reduces dimensionality before GMM for speed."
        )
    with sc5:
        cov_type = st.selectbox(
            "GMM covariance type",
            ["full", "diag", "tied", "spherical"],
            key="struct_cov",
        )

    imp_geno = impute_missing(geno, "mean")
    X_struct = StandardScaler().fit_transform(imp_geno.values)

    n_pcs_struct = min(n_pcs_struct_slider, X_struct.shape[0],
                        X_struct.shape[1])
    pca_pre = PCA(n_components=n_pcs_struct)
    X_reduced = pca_pre.fit_transform(X_struct)

    if st.button("🚀 Run STRUCTURE-like analysis", key="struct_run",
                  use_container_width=True):
        all_k_stats = []
        consensus_Q = {}
        best_Q_per_k = {}
        all_replicate_Qs = {}

        with st.spinner(f"Running GMM for K={k_min}..{k_max} with "
                          f"{n_reps} reps each..."):
            progress_bar = st.progress(0)
            total_iters = (int(k_max) - int(k_min) + 1) * int(n_reps)
            done = 0

            for k in range(int(k_min), int(k_max) + 1):
                bics, aics, log_liks = [], [], []
                Q_reps = []
                models = []

                for rep in range(int(n_reps)):
                    gmm = GaussianMixture(
                        n_components=k,
                        covariance_type=cov_type,
                        random_state=rep,
                        max_iter=500,
                        reg_covar=1e-4,
                        n_init=1,
                    )
                    gmm.fit(X_reduced)
                    bics.append(gmm.bic(X_reduced))
                    aics.append(gmm.aic(X_reduced))
                    log_liks.append(gmm.score(X_reduced) * X_reduced.shape[0])
                    Q_reps.append(gmm.predict_proba(X_reduced))
                    models.append(gmm)

                    done += 1
                    progress_bar.progress(done / total_iters)

                # CLUMPP alignment + consensus
                Q_consensus = consensus_q_matrix(Q_reps)
                consensus_Q[k] = Q_consensus
                best_Q_per_k[k] = Q_reps[int(np.argmin(bics))]
                all_replicate_Qs[k] = align_q_matrices(Q_reps)

                all_k_stats.append({
                    "K": k,
                    "BIC_min": min(bics),
                    "BIC_mean": np.mean(bics),
                    "BIC_std": np.std(bics),
                    "AIC_mean": np.mean(aics),
                    "LnL_mean": np.mean(log_liks),
                    "LnL_std": np.std(log_liks),
                })

            progress_bar.empty()

        stats_df = pd.DataFrame(all_k_stats)

        # ── Evanno's ΔK calculation ──
        valid_k, delta_k = compute_evanno_delta_k(
            stats_df["K"].values, stats_df["LnL_mean"].values,
        )
        best_evanno_k = int(valid_k[np.argmax(delta_k)]) if len(delta_k) > 0 else int(stats_df.loc[stats_df["BIC_min"].idxmin(), "K"])
        best_bic_k = int(stats_df.loc[stats_df["BIC_min"].idxmin(), "K"])

        # ── Display K selection metrics ──
        st.markdown("### 🎯 Optimal K Selection")

        oc1, oc2, oc3 = st.columns(3)
        oc1.metric("Best K (BIC)", best_bic_k)
        oc2.metric("Best K (Evanno's ΔK)", best_evanno_k)
        oc3.metric("Best K (AIC)",
                    int(stats_df.loc[stats_df["AIC_mean"].idxmin(), "K"]))

        # BIC / AIC / LnL plot
        fig_metrics = go.Figure()
        fig_metrics.add_trace(go.Scatter(
            x=stats_df["K"], y=stats_df["BIC_min"],
            mode="lines+markers", name="BIC (min)",
            line=dict(color="steelblue"),
            error_y=dict(type="data", array=stats_df["BIC_std"]),
        ))
        fig_metrics.add_trace(go.Scatter(
            x=stats_df["K"], y=stats_df["AIC_mean"],
            mode="lines+markers", name="AIC (mean)",
            line=dict(color="orange", dash="dot"),
        ))
        fig_metrics.add_vline(x=best_bic_k, line_dash="dash",
                                line_color="red",
                                annotation_text=f"BIC K={best_bic_k}")
        fig_metrics.update_layout(
            title="Model Selection: BIC / AIC vs K",
            xaxis_title="K", yaxis_title="Score",
            template="plotly_white", height=450,
        )
        st.plotly_chart(fig_metrics, use_container_width=True)

        # Evanno ΔK plot
        if len(delta_k) > 0:
            fig_evanno = go.Figure()
            fig_evanno.add_trace(go.Scatter(
                x=valid_k, y=delta_k, mode="lines+markers",
                name="ΔK", line=dict(color="purple", width=3),
                marker=dict(size=10),
            ))
            fig_evanno.add_vline(x=best_evanno_k, line_dash="dash",
                                    line_color="red",
                                    annotation_text=f"Best K = {best_evanno_k}")
            fig_evanno.update_layout(
                title="Evanno's ΔK — Optimal K Detection",
                xaxis_title="K", yaxis_title="ΔK",
                template="plotly_white", height=450,
            )
            st.plotly_chart(fig_evanno, use_container_width=True)

        st.dataframe(stats_df.style.format({
            "BIC_min": "{:.2f}", "BIC_mean": "{:.2f}", "BIC_std": "{:.2f}",
            "AIC_mean": "{:.2f}", "LnL_mean": "{:.2f}", "LnL_std": "{:.2f}",
        }), use_container_width=True)

        download_dataframe(stats_df, "structure_model_selection.csv",
                            key="dl_struct_bic")

        # ── Store results in session state for the Q-matrix section ──
        st.session_state["struct_stats_df"] = stats_df
        st.session_state["struct_consensus_Q"] = consensus_Q
        st.session_state["struct_best_bic_k"] = best_bic_k
        st.session_state["struct_best_evanno_k"] = best_evanno_k

    # ── Q-matrix display (works even without re-running) ──
    if "struct_consensus_Q" in st.session_state:
        st.markdown("---")
        st.markdown("### 📊 Q-matrix (Consensus Ancestry Proportions)")

        consensus_Q = st.session_state["struct_consensus_Q"]
        best_bic_k = st.session_state["struct_best_bic_k"]
        best_evanno_k = st.session_state["struct_best_evanno_k"]

        available_ks = sorted(consensus_Q.keys())
        display_k = st.select_slider(
            "Select K for Q-matrix display",
            options=available_ks,
            value=best_evanno_k if best_evanno_k in available_ks else best_bic_k,
            key="struct_k_disp",
        )

        Q_cons = consensus_Q[display_k]
        Q_df = pd.DataFrame(
            Q_cons, index=geno.index,
            columns=[f"K{i+1}" for i in range(display_k)],
        ).reset_index().rename(columns={"index": "Sample"})

        # Attach real population labels
        Q_df = _attach_metadata(Q_df)

        # Sort samples: by population first, then by dominant cluster
        cluster_cols = [c for c in Q_df.columns if c.startswith("K")]
        Q_df["DominantK"] = Q_df[cluster_cols].idxmax(axis=1)

        if pop_col and pop_col in Q_df.columns:
            Q_df = Q_df.sort_values([pop_col, "DominantK"])
        else:
            Q_df = Q_df.sort_values("DominantK")

        # Display
        st.dataframe(Q_df.head(50), use_container_width=True)
        download_dataframe(Q_df, f"Q_consensus_K{display_k}.csv",
                            key="dl_struct_q")

        # ── Distruct-style admixture plot ──
        st.markdown("#### 🎨 Admixture Plot (Distruct-style)")

        # Melt for stacked bar
        melt_ids = ["Sample"] + ([pop_col] if pop_col in Q_df.columns else [])
        plot_data = Q_df.melt(
            id_vars=melt_ids,
            value_vars=cluster_cols,
            var_name="Ancestry",
            value_name="Proportion",
        )

        fig_admix = px.bar(
            plot_data, x="Sample", y="Proportion", color="Ancestry",
            title=f"Consensus Admixture Bar Plot (K = {display_k}, "
                    f"averaged over {n_reps} replicates)",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_admix.update_layout(
            template="plotly_white",
            height=550, barmode="stack",
            xaxis={"categoryorder": "array",
                    "categoryarray": Q_df["Sample"].tolist(),
                    "tickangle": 90},
            yaxis=dict(range=[0, 1]),
        )
        st.plotly_chart(fig_admix, use_container_width=True)

        # ── Population-labeled admixture bar with dividers ──
        if pop_col and pop_col in Q_df.columns:
            st.markdown("#### 🌍 Admixture Plot with Population Labels")

            # Add colored population separator
            fig_pop = go.Figure()

            colors_k = px.colors.qualitative.Set2
            for i, col in enumerate(cluster_cols):
                fig_pop.add_trace(go.Bar(
                    x=Q_df["Sample"],
                    y=Q_df[col],
                    name=col,
                    marker_color=colors_k[i % len(colors_k)],
                ))

            # Add population separators
            pop_groups = Q_df.groupby(pop_col, sort=False).size()
            cumulative = 0
            annotations = []
            for pop_name, count in pop_groups.items():
                mid = cumulative + count / 2 - 0.5
                annotations.append(dict(
                    x=mid, y=-0.08, xref="x", yref="paper",
                    text=f"<b>{pop_name}</b>", showarrow=False,
                    font=dict(size=11, color="black"),
                    textangle=-45,
                ))
                cumulative += count
                if cumulative < len(Q_df):
                    fig_pop.add_vline(
                        x=cumulative - 0.5, line_dash="solid",
                        line_color="black", line_width=1.5,
                    )

            fig_pop.update_layout(
                title=f"Population-labeled Admixture (K = {display_k})",
                barmode="stack",
                template="plotly_white",
                height=600,
                yaxis=dict(range=[0, 1], title="Ancestry proportion"),
                xaxis=dict(tickangle=90, title=""),
                annotations=annotations,
                margin=dict(b=150),
            )
            st.plotly_chart(fig_pop, use_container_width=True)

            download_plotly_html(fig_pop,
                                  f"admixture_labeled_K{display_k}.html",
                                  key="dl_admix_html")

        # ── Cluster vs Population confusion matrix ──
        if pop_col and pop_col in Q_df.columns:
            st.markdown("#### 🔀 Cluster vs Population Correspondence")

            Q_df["AssignedCluster"] = Q_df[cluster_cols].idxmax(axis=1)
            conf = pd.crosstab(Q_df[pop_col], Q_df["AssignedCluster"])

            fig_conf = px.imshow(
                conf.values,
                x=conf.columns.tolist(),
                y=conf.index.tolist(),
                text_auto=True,
                color_continuous_scale="Blues",
                title="Population × Inferred Cluster",
                labels=dict(x="Inferred Cluster",
                            y="True Population",
                            color="Count"),
            )
            fig_conf.update_layout(template="plotly_white", height=500)
            st.plotly_chart(fig_conf, use_container_width=True)


# =========================================================
# TAB 4 — fastStructure-like (Bayesian GMM)
# =========================================================
with tab_fast:
    st.subheader("⚡ fastStructure-like Analysis")
    st.write(
        "Uses **Bayesian Gaussian Mixture Model** as a fast alternative "
        "to fastStructure. Automatically determines K via variational "
        "evidence — no need for repetitions."
    )

    fc1, fc2 = st.columns(2)
    with fc1:
        max_k_fast = st.slider("Maximum K to explore", 2, 20, 8,
                                 key="fast_maxk")
    with fc2:
        weight_prior = st.slider("Weight concentration prior", 0.01,
                                   10.0, 0.1, key="fast_wp",
                                   help="Lower = fewer effective components")

    if st.button("🚀 Run fastStructure-like", key="fast_run",
                  use_container_width=True):
        imp_geno = impute_missing(geno, "mean")
        X_fast = StandardScaler().fit_transform(imp_geno.values)
        pca_pre2 = PCA(n_components=min(20, X_fast.shape[1]))
        X_red2 = pca_pre2.fit_transform(X_fast)

        with st.spinner("Running Bayesian GMM..."):
            bgmm = BayesianGaussianMixture(
                n_components=max_k_fast,
                covariance_type="full",
                weight_concentration_prior=weight_prior,
                random_state=42,
                max_iter=500,
                reg_covar=1e-4,
            )
            bgmm.fit(X_red2)

        weights = bgmm.weights_
        active_k = int((weights > 0.01).sum())

        st.success(f"✅ Bayesian GMM detected **{active_k}** active clusters "
                    f"(out of {max_k_fast} allowed).")

        # Component weights
        w_df = pd.DataFrame({
            "Cluster": [f"K{i+1}" for i in range(len(weights))],
            "Weight": weights,
        }).sort_values("Weight", ascending=False)

        fig_w = px.bar(w_df, x="Cluster", y="Weight",
                        title="Cluster weights (>0.01 = active)",
                        color="Weight", color_continuous_scale="Viridis")
        fig_w.add_hline(y=0.01, line_dash="dash", line_color="red")
        fig_w.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig_w, use_container_width=True)

        # Q-matrix
        Q_fast = bgmm.predict_proba(X_red2)
        Q_fast_df = pd.DataFrame(
            Q_fast, index=geno.index,
            columns=[f"K{i+1}" for i in range(Q_fast.shape[1])],
        ).reset_index().rename(columns={"index": "Sample"})

        # Attach real labels
        Q_fast_df = _attach_metadata(Q_fast_df)

        active_cols = [f"K{i+1}"
                        for i in range(len(weights))
                        if weights[i] > 0.01]

        # Sort by dominant cluster / population
        Q_fast_df["DominantK"] = Q_fast_df[active_cols].idxmax(axis=1)
        if pop_col and pop_col in Q_fast_df.columns:
            Q_fast_df = Q_fast_df.sort_values([pop_col, "DominantK"])

        st.dataframe(Q_fast_df[["Sample"] + active_cols].head(50),
                      use_container_width=True)
        download_dataframe(Q_fast_df, "faststructure_Q.csv",
                            key="dl_fast_q")

        # Admixture plot
        st.markdown("#### 🎨 fastStructure Admixture Plot")
        melt_ids_f = ["Sample"] + ([pop_col] if pop_col in Q_fast_df.columns else [])
        plot_fast = Q_fast_df.melt(
            id_vars=melt_ids_f, value_vars=active_cols,
            var_name="Ancestry", value_name="Proportion",
        )
        fig_fast_admix = px.bar(
            plot_fast, x="Sample", y="Proportion", color="Ancestry",
            title=f"fastStructure Admixture (Active K = {active_k})",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_fast_admix.update_layout(
            template="plotly_white", height=550, barmode="stack",
            xaxis={"categoryorder": "array",
                    "categoryarray": Q_fast_df["Sample"].tolist(),
                    "tickangle": 90},
            yaxis=dict(range=[0, 1]),
        )
        st.plotly_chart(fig_fast_admix, use_container_width=True)

        # Population-labeled version
        if pop_col and pop_col in Q_fast_df.columns:
            st.markdown("#### 🌍 Population-labeled Admixture")
            fig_pop_f = go.Figure()
            colors_k = px.colors.qualitative.Set2
            for i, col in enumerate(active_cols):
                fig_pop_f.add_trace(go.Bar(
                    x=Q_fast_df["Sample"], y=Q_fast_df[col], name=col,
                    marker_color=colors_k[i % len(colors_k)],
                ))

            pop_groups = Q_fast_df.groupby(pop_col, sort=False).size()
            cumulative = 0
            annotations_f = []
            for pop_name, count in pop_groups.items():
                mid = cumulative + count / 2 - 0.5
                annotations_f.append(dict(
                    x=mid, y=-0.08, xref="x", yref="paper",
                    text=f"<b>{pop_name}</b>", showarrow=False,
                    font=dict(size=11, color="black"),
                    textangle=-45,
                ))
                cumulative += count
                if cumulative < len(Q_fast_df):
                    fig_pop_f.add_vline(
                        x=cumulative - 0.5, line_dash="solid",
                        line_color="black", line_width=1.5,
                    )

            fig_pop_f.update_layout(
                title=f"fastStructure Population-labeled (K={active_k})",
                barmode="stack",
                template="plotly_white",
                height=600,
                yaxis=dict(range=[0, 1]),
                xaxis=dict(tickangle=90),
                annotations=annotations_f,
                margin=dict(b=150),
            )
            st.plotly_chart(fig_pop_f, use_container_width=True)
            download_plotly_html(fig_pop_f,
                                  "faststructure_labeled.html",
                                  key="dl_fast_html")