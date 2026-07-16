"""Population Structure — PCA (2D/3D/Biplot), PCoA, STRUCTURE-like admixture."""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

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

    # Preprocess
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
        scores,
        index=geno.index,
        columns=[f"PC{i+1}" for i in range(n_components)],
    ).reset_index().rename(columns={"index": "Sample"})

    # Merge metadata for coloring
    color_col = None
    if meta is not None:
        pop_col = st.selectbox("Color by (metadata column)",
                                ["None"] + meta.columns.tolist(),
                                key="pca_color")
        sample_col = st.selectbox("Sample ID column in metadata",
                                   meta.columns.tolist(),
                                   key="pca_samcol")
        if pop_col != "None":
            meta_sub = meta[[sample_col, pop_col]].drop_duplicates()
            meta_sub[sample_col] = meta_sub[sample_col].astype(str)
            scores_df["Sample"] = scores_df["Sample"].astype(str)
            scores_df = scores_df.merge(meta_sub,
                                          left_on="Sample",
                                          right_on=sample_col,
                                          how="left")
            color_col = pop_col

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
        yaxis2=dict(title="Cumulative (%)", overlaying="y",
                    side="right"),
        template="plotly_white", height=450,
    )
    st.plotly_chart(fig_scree, use_container_width=True)

    # ── 2D scatter ──
    st.markdown("#### 2D PCA Plot")
    p2c1, p2c2 = st.columns(2)
    with p2c1:
        x_pc = st.selectbox("X axis", var_df["Component"], index=0,
                             key="pca_x")
    with p2c2:
        y_pc = st.selectbox("Y axis", var_df["Component"], index=1,
                             key="pca_y")

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
    download_dataframe(loadings, "pca_loadings.csv",
                        key="dl_pca_load")

    # Biplot (top N loading markers)
    top_n_load = st.slider("Top N markers in biplot", 5, 50, 20,
                             key="pca_bp_n")

    loadings["magnitude"] = np.sqrt(loadings[x_pc]**2 + loadings[y_pc]**2)
    top_markers = loadings.nlargest(top_n_load, "magnitude")

    scale_factor = 3 * (scores_df[[x_pc, y_pc]].abs().max().max() /
                         top_markers[[x_pc, y_pc]].abs().max().max())

    fig_bp = go.Figure()
    # Sample points
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

    # Loading arrows
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

    # Download scores
    download_dataframe(scores_df, "pca_scores.csv", key="dl_pca_scores")

# =========================================================
# TAB 2 — PCoA (Principal Coordinates Analysis)
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
            from scipy.spatial.distance import pdist, squareform

            imp = impute_missing(geno, "mean")
            D = squareform(pdist(imp.values, metric=dist_metric))

            # Classical MDS / PCoA
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
            coords,
            index=geno.index,
            columns=[f"Axis{i+1}" for i in range(coords.shape[1])],
        ).reset_index().rename(columns={"index": "Sample"})

        # Metadata merge
        color_col2 = None
        if meta is not None:
            pop2 = st.session_state.get("pca_color", None)
            sam2 = st.session_state.get("pca_samcol", None)
            if pop2 and pop2 != "None" and sam2:
                meta_sub = meta[[sam2, pop2]].drop_duplicates()
                meta_sub[sam2] = meta_sub[sam2].astype(str)
                pcoa_df["Sample"] = pcoa_df["Sample"].astype(str)
                pcoa_df = pcoa_df.merge(meta_sub,
                                          left_on="Sample",
                                          right_on=sam2,
                                          how="left")
                color_col2 = pop2

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
            pcoa_df, x="Axis1", y="Axis2", color=color_col2,
            hover_data=["Sample"],
            labels={
                "Axis1": f"Axis1 ({var_exp_pcoa[0]:.1f}%)",
                "Axis2": f"Axis2 ({var_exp_pcoa[1]:.1f}%)",
            },
            title="PCoA - 2D",
        )
        fig_pcoa.update_layout(template="plotly_white", height=650)
        st.plotly_chart(fig_pcoa, use_container_width=True)

        if coords.shape[1] >= 3:
            fig_pcoa_3d = px.scatter_3d(
                pcoa_df, x="Axis1", y="Axis2", z="Axis3",
                color=color_col2, hover_data=["Sample"],
                title="PCoA - 3D",
            )
            fig_pcoa_3d.update_layout(template="plotly_white", height=700)
            st.plotly_chart(fig_pcoa_3d, use_container_width=True)

        download_dataframe(pcoa_df, "pcoa_coordinates.csv",
                            key="dl_pcoa")

# =========================================================
# TAB 3 — STRUCTURE-like (via GMM/K-means admixture)
# =========================================================
with tab_struct:
    st.subheader("STRUCTURE-like Admixture Analysis")
    st.write(
        "This module uses **Gaussian Mixture Models (GMM)** as a proxy for "
        "STRUCTURE analysis. It produces Q-matrix estimates and generates "
        "admixture bar plots similar to CLUMPP/Distruct output."
    )

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        k_min = st.number_input("K min", 2, 20, 2, key="struct_kmin")
    with sc2:
        k_max = st.number_input("K max", 2, 20, 10, key="struct_kmax")
    with sc3:
        n_reps = st.slider("Repetitions per K", 1, 20, 10,
                             key="struct_reps")

    imp_geno = impute_missing(geno, "mean")
    X_struct = StandardScaler().fit_transform(imp_geno.values)

    # Reduce dimensionality first
    n_pcs_struct = min(20, X_struct.shape[0], X_struct.shape[1])
    pca_pre = PCA(n_components=n_pcs_struct)
    X_reduced = pca_pre.fit_transform(X_struct)

    if st.button("🚀 Run STRUCTURE-like analysis", key="struct_run"):
        best_bic = []
        best_models = {}

        with st.spinner(f"Running GMM for K={k_min}..{k_max}..."):
            for k in range(int(k_min), int(k_max) + 1):
                bics = []
                models = []
                for rep in range(int(n_reps)):
                    gmm = GaussianMixture(
                        n_components=k,
                        covariance_type="full",
                        random_state=rep,
                        max_iter=200,
                        reg_covar=1e-4,
                    )
                    gmm.fit(X_reduced)
                    bics.append(gmm.bic(X_reduced))
                    models.append(gmm)
                best_idx = int(np.argmin(bics))
                best_bic.append({
                    "K": k,
                    "BIC_min": min(bics),
                    "BIC_mean": np.mean(bics),
                    "BIC_std": np.std(bics),
                })
                best_models[k] = models[best_idx]

        bic_df = pd.DataFrame(best_bic)
        st.markdown("#### BIC vs K (lower is better)")
        fig_bic = go.Figure()
        fig_bic.add_trace(go.Scatter(
            x=bic_df["K"], y=bic_df["BIC_min"],
            mode="lines+markers", name="Best BIC",
            error_y=dict(type="data", array=bic_df["BIC_std"]),
        ))
        best_k = int(bic_df.loc[bic_df["BIC_min"].idxmin(), "K"])
        fig_bic.add_vline(x=best_k, line_dash="dash",
                           line_color="red",
                           annotation_text=f"Best K = {best_k}")
        fig_bic.update_layout(template="plotly_white", height=450,
                                xaxis_title="K",
                                yaxis_title="BIC")
        st.plotly_chart(fig_bic, use_container_width=True)

        st.dataframe(bic_df, use_container_width=True)
        download_dataframe(bic_df, "structure_bic.csv",
                            key="dl_struct_bic")

        # Display Q-matrix for user-selected K
        st.markdown("---")
        st.markdown("#### Q-matrix (Ancestry proportions)")
        display_k = st.slider("Select K for Q-matrix display",
                                int(k_min), int(k_max), best_k,
                                key="struct_k_disp")

        gmm_sel = best_models[display_k]
        Q_matrix = gmm_sel.predict_proba(X_reduced)
        Q_df = pd.DataFrame(
            Q_matrix,
            index=geno.index,
            columns=[f"Cluster_{i+1}" for i in range(display_k)],
        ).reset_index().rename(columns={"index": "Sample"})

        # Merge population info
        pop_col_s = None
        if meta is not None:
            sam_col = st.session_state.get("pca_samcol", None)
            pop_col_s = st.session_state.get("pca_color", None)
            if sam_col and pop_col_s and pop_col_s != "None":
                meta_sub = meta[[sam_col, pop_col_s]].drop_duplicates()
                meta_sub[sam_col] = meta_sub[sam_col].astype(str)
                Q_df["Sample"] = Q_df["Sample"].astype(str)
                Q_df = Q_df.merge(meta_sub, left_on="Sample",
                                    right_on=sam_col, how="left")
                Q_df = Q_df.sort_values(pop_col_s)

        st.dataframe(Q_df.head(50), use_container_width=True)
        download_dataframe(Q_df,
                            f"Q_matrix_K{display_k}.csv",
                            key="dl_struct_q")

        # Distruct-style stacked bar plot
        st.markdown("#### Admixture Plot (Distruct-style)")
        cluster_cols = [c for c in Q_df.columns
                        if c.startswith("Cluster_")]
        plot_data = Q_df.melt(id_vars=["Sample"] +
                                  ([pop_col_s] if pop_col_s in Q_df.columns else []),
                                value_vars=cluster_cols,
                                var_name="Cluster",
                                value_name="Proportion")

        fig_admix = px.bar(
            plot_data, x="Sample", y="Proportion", color="Cluster",
            title=f"Admixture bar plot (K = {display_k})",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_admix.update_layout(
            template="plotly_white",
            height=500, barmode="stack",
            xaxis={"categoryorder": "array",
                   "categoryarray": Q_df["Sample"].tolist(),
                   "tickangle": 90},
        )
        st.plotly_chart(fig_admix, use_container_width=True)
        download_plotly_html(fig_admix,
                              f"admixture_K{display_k}.html",
                              key="dl_admix_html")

# =========================================================
# TAB 4 — fastStructure-like (variational Bayesian approx.)
# =========================================================
with tab_fast:
    st.subheader("fastStructure-like Analysis")
    st.write(
        "This module uses **Bayesian Gaussian Mixture Model** as a "
        "fast alternative to fastStructure. It runs much faster than "
        "the STRUCTURE tab and automatically selects K via evidence."
    )

    fc1, fc2 = st.columns(2)
    with fc1:
        max_k_fast = st.slider("Maximum K to explore", 2, 20, 8,
                                 key="fast_maxk")
    with fc2:
        weight_prior = st.slider("Weight concentration prior", 0.01,
                                   10.0, 0.1, key="fast_wp")

    if st.button("🚀 Run fastStructure-like", key="fast_run"):
        from sklearn.mixture import BayesianGaussianMixture

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
            "Cluster": [f"C{i+1}" for i in range(len(weights))],
            "Weight": weights,
        })
        fig_w = px.bar(w_df, x="Cluster", y="Weight",
                        title="Cluster weights (>0.01 = active)")
        fig_w.add_hline(y=0.01, line_dash="dash",
                         line_color="red")
        fig_w.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig_w, use_container_width=True)

        # Q-matrix
        Q_fast = bgmm.predict_proba(X_red2)
        Q_fast_df = pd.DataFrame(
            Q_fast, index=geno.index,
            columns=[f"Cluster_{i+1}" for i in range(Q_fast.shape[1])],
        ).reset_index().rename(columns={"index": "Sample"})

        # Filter to active clusters only
        active_cols = [f"Cluster_{i+1}"
                        for i in range(len(weights))
                        if weights[i] > 0.01]

        st.dataframe(Q_fast_df[["Sample"] + active_cols].head(50),
                      use_container_width=True)
        download_dataframe(Q_fast_df, "faststructure_Q.csv",
                            key="dl_fast_q")

        # Admixture plot with active clusters
        plot_fast = Q_fast_df.melt(
            id_vars="Sample", value_vars=active_cols,
            var_name="Cluster", value_name="Proportion",
        )
        fig_fast_admix = px.bar(
            plot_fast, x="Sample", y="Proportion", color="Cluster",
            title=f"fastStructure Admixture (Active K = {active_k})",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_fast_admix.update_layout(
            template="plotly_white", height=500,
            barmode="stack",
            xaxis=dict(tickangle=90),
        )
        st.plotly_chart(fig_fast_admix, use_container_width=True)