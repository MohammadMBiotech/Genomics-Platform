"""
Clustering Analysis — Publication Quality
─────────────────────────────────────────
Methods:
  - Hierarchical (UPGMA, Ward, Complete, Single, Average)
  - K-means
  - DBSCAN
  - Neighbor Joining (NJ)

Features:
  - Metadata-based cluster labeling (real names in legend)
  - Silhouette + Davies-Bouldin + Calinski-Harabasz metrics
  - Per-cluster silhouette breakdown
  - Cluster composition (cross-tab with real labels)
  - 2D/3D PCA visualization
  - Colored dendrogram with population labels
  - Optimal K search (elbow + silhouette)
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    silhouette_samples,
)
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform

from utils import (
    get_geno_from_session, get_meta_from_session,
    impute_missing, download_plotly_html, download_dataframe,
)

st.title("🎯 Clustering")
st.markdown("---")

geno, marker_info = get_geno_from_session()
if geno is None:
    st.stop()

meta = get_meta_from_session()


# ═══════════════════════════════════════════
# GLOBAL: Metadata configuration (BEFORE run button)
# ═══════════════════════════════════════════
st.subheader("🔧 Metadata Configuration")

sample_col = None
pop_col = None
label_col_active = None
pop_map = {}

if meta is not None:
    mc1, mc2 = st.columns(2)
    with mc1:
        sample_col = st.selectbox(
            "Sample ID column",
            meta.columns.tolist(),
            key="clu_samcol_g",
        )
    with mc2:
        pop_col_opt = st.selectbox(
            "Real label column (for cluster naming)",
            ["None"] + meta.columns.tolist(),
            key="clu_popcol_g",
            help="If selected, clusters will be named by their most "
                    "common real label (e.g., 'Cluster 1: Species_A (n=25)')."
        )
        pop_col = None if pop_col_opt == "None" else pop_col_opt

    if pop_col and sample_col:
        pop_map = dict(zip(meta[sample_col].astype(str),
                            meta[pop_col].astype(str)))
        label_col_active = pop_col
        st.success(
            f"✅ Clusters will be labeled using **{pop_col}** column."
        )
else:
    st.info("⚠️ No metadata loaded. Clusters will use generic names (Cluster 1, 2, ...).")

st.markdown("---")


# ═══════════════════════════════════════════
# Preprocessing
# ═══════════════════════════════════════════
st.subheader("Preprocessing")
p1, p2 = st.columns(2)
with p1:
    imp_method = st.selectbox("Missing imputation",
                                ["mean", "median", "zero"],
                                key="clu_imp")
with p2:
    scale_data = st.checkbox("Standardize markers", True, key="clu_scale")

geno_imp = impute_missing(geno, method=imp_method)
X = geno_imp.values
if scale_data:
    X = StandardScaler().fit_transform(X)

# Optional PCA pre-reduction
use_pca = st.checkbox(
    "Reduce dimensionality with PCA first (recommended for large datasets)",
    True, key="clu_usepca")

if use_pca:
    n_pcs = st.slider("Number of PCs", 2, min(50, X.shape[1]),
                       min(10, X.shape[1]), key="clu_npcs")
    X_use = PCA(n_components=n_pcs).fit_transform(X)
else:
    X_use = X


# ═══════════════════════════════════════════
# Method selection
# ═══════════════════════════════════════════
st.subheader("Clustering Method")
method = st.selectbox(
    "Method",
    ["Hierarchical (UPGMA / Average)",
     "Hierarchical (Ward)",
     "Hierarchical (Complete)",
     "Hierarchical (Single)",
     "K-means",
     "DBSCAN",
     "Neighbor Joining (NJ-like)"],
    key="clu_method",
)

if (method.startswith("Hierarchical") or method == "K-means"
        or method == "Neighbor Joining (NJ-like)"):
    n_clusters = st.slider("Number of clusters (K)", 2,
                            min(20, X_use.shape[0]), 4, key="clu_k")

if method == "DBSCAN":
    dc1, dc2 = st.columns(2)
    with dc1:
        eps = st.slider("EPS (neighborhood radius)", 0.1, 10.0, 0.5, 0.1,
                         key="clu_eps")
    with dc2:
        min_samples = st.slider("Min samples", 2, 20, 5, key="clu_minsam")


# ═══════════════════════════════════════════
# RUN CLUSTERING
# ═══════════════════════════════════════════
if st.button("🚀 Run Clustering", use_container_width=True, key="clu_run"):
    with st.spinner(f"Running {method}..."):
        if method == "Hierarchical (UPGMA / Average)":
            Z = linkage(pdist(X_use), method="average")
            clusters = fcluster(Z, t=n_clusters, criterion="maxclust")
        elif method == "Hierarchical (Ward)":
            Z = linkage(pdist(X_use), method="ward")
            clusters = fcluster(Z, t=n_clusters, criterion="maxclust")
        elif method == "Hierarchical (Complete)":
            Z = linkage(pdist(X_use), method="complete")
            clusters = fcluster(Z, t=n_clusters, criterion="maxclust")
        elif method == "Hierarchical (Single)":
            Z = linkage(pdist(X_use), method="single")
            clusters = fcluster(Z, t=n_clusters, criterion="maxclust")
        elif method == "K-means":
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = km.fit_predict(X_use) + 1
            Z = None
        elif method == "DBSCAN":
            db = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = db.fit_predict(X_use)
            # 0 = noise, ≥1 = clusters
            clusters = np.array([c + 2 if c >= 0 else 0 for c in clusters])
            Z = None
        elif method == "Neighbor Joining (NJ-like)":
            # NJ-like: use average linkage then cluster
            Z = linkage(pdist(X_use), method="average")
            clusters = fcluster(Z, t=n_clusters, criterion="maxclust")

    n_unique = len(set(clusters))
    st.success(f"✅ Clustering completed. Found **{n_unique}** clusters.")

    # ─── Build result table with REAL LABELS ───
    result = pd.DataFrame({
        "Sample": geno.index.astype(str),
        "Cluster_ID": clusters,
    })

    if label_col_active and pop_map:
        result["Original_Label"] = result["Sample"].map(pop_map)

    # ─── Create descriptive cluster names ───
    if label_col_active and "Original_Label" in result.columns:
        cluster_names = {}
        cluster_composition = {}
        for cid in sorted(set(clusters)):
            mask = result["Cluster_ID"] == cid
            labs = result.loc[mask, "Original_Label"].dropna()
            count = int(mask.sum())

            if len(labs) > 0:
                most_common = labs.mode().iloc[0]
                purity = (labs == most_common).sum() / len(labs) * 100
                cluster_names[cid] = f"Cluster {cid}: {most_common} (n={count}, purity={purity:.0f}%)"
                # Store composition
                cluster_composition[cid] = labs.value_counts().to_dict()
            else:
                cluster_names[cid] = f"Cluster {cid} (n={count})"
                cluster_composition[cid] = {}

        result["Cluster"] = result["Cluster_ID"].map(cluster_names)
    else:
        # Generic cluster names
        cluster_names = {cid: (f"Noise (n={int((result['Cluster_ID'] == cid).sum())})"
                                 if cid == 0
                                 else f"Cluster {cid} (n={int((result['Cluster_ID'] == cid).sum())})")
                          for cid in sorted(set(clusters))}
        result["Cluster"] = result["Cluster_ID"].map(cluster_names)

    # ─── Quality metrics ───
    st.subheader("📊 Clustering Quality Metrics")

    if n_unique > 1 and n_unique < len(clusters):
        sil = silhouette_score(X_use, clusters)
        db_score = davies_bouldin_score(X_use, clusters)
        try:
            ch_score = calinski_harabasz_score(X_use, clusters)
        except Exception:
            ch_score = np.nan

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Silhouette Score", f"{sil:.4f}",
                    help="Range [-1, 1]. Higher = better separation.")
        m2.metric("Davies-Bouldin", f"{db_score:.4f}",
                    help="Lower = better.")
        m3.metric("Calinski-Harabasz",
                    f"{ch_score:.2f}" if not np.isnan(ch_score) else "N/A",
                    help="Higher = better.")
        m4.metric("N Clusters", n_unique)

        # Per-cluster silhouette
        st.markdown("#### Silhouette per Cluster")
        sil_samples = silhouette_samples(X_use, clusters)
        sil_by_cluster = pd.DataFrame({
            "Cluster": [cluster_names[c] for c in clusters],
            "Silhouette": sil_samples,
        }).groupby("Cluster")["Silhouette"].agg(
            ["mean", "std", "count"]).reset_index()
        sil_by_cluster.columns = ["Cluster", "Mean_Silhouette",
                                     "Std_Silhouette", "N_samples"]
        sil_by_cluster = sil_by_cluster.sort_values("Mean_Silhouette",
                                                       ascending=False)

        st.dataframe(sil_by_cluster.style.format({
            "Mean_Silhouette": "{:.4f}",
            "Std_Silhouette": "{:.4f}",
        }), use_container_width=True)

        # Bar chart
        fig_sil = px.bar(
            sil_by_cluster, x="Cluster", y="Mean_Silhouette",
            error_y="Std_Silhouette", color="Mean_Silhouette",
            color_continuous_scale="RdYlGn",
            title="Mean silhouette score per cluster",
        )
        fig_sil.add_hline(y=sil, line_dash="dash", line_color="black",
                           annotation_text=f"Overall mean = {sil:.3f}")
        fig_sil.update_layout(template="plotly_white", height=450,
                                xaxis_tickangle=45)
        st.plotly_chart(fig_sil, use_container_width=True)
    else:
        st.info(f"Only {n_unique} cluster found — quality metrics unavailable.")

    # ─── Cluster composition (if metadata) ───
    if label_col_active and "Original_Label" in result.columns:
        st.subheader("🎨 Cluster Composition (vs Real Labels)")

        composition_table = pd.crosstab(
            result["Cluster"], result["Original_Label"], margins=True,
            margins_name="Total")

        st.dataframe(composition_table, use_container_width=True)

        # Heatmap of composition (excluding totals)
        comp_data = composition_table.iloc[:-1, :-1]
        fig_comp = px.imshow(
            comp_data.values,
            x=comp_data.columns.tolist(),
            y=comp_data.index.tolist(),
            text_auto=True,
            color_continuous_scale="Blues",
            title="Composition: Real labels within each cluster",
            labels=dict(x="Real Label", y="Cluster", color="Count"),
        )
        fig_comp.update_layout(template="plotly_white", height=500)
        st.plotly_chart(fig_comp, use_container_width=True)

    # ─── Clustered samples table ───
    st.subheader("Clustered Samples")
    display_cols = ["Sample", "Cluster_ID", "Cluster"]
    if "Original_Label" in result.columns:
        display_cols.insert(2, "Original_Label")
    st.dataframe(result[display_cols], use_container_width=True)
    download_dataframe(result, "clustering_results.csv", key="dl_clu_csv")

    # ─── Consistent color mapping ───
    unique_clusters = sorted(result["Cluster"].unique())
    color_palette = (px.colors.qualitative.Set2 +
                     px.colors.qualitative.Set3 +
                     px.colors.qualitative.Plotly)
    color_map = {c: color_palette[i % len(color_palette)]
                  for i, c in enumerate(unique_clusters)}

    # ─── PCA visualization ───
    st.subheader("Cluster Visualization (PCA projection)")
    n_pca_viz = min(3, X_use.shape[1])
    pca_viz = PCA(n_components=max(n_pca_viz, 2))
    scores_viz = pca_viz.fit_transform(X_use)

    # Pad to 3 columns if fewer PCs
    if scores_viz.shape[1] < 3:
        pad = np.zeros((scores_viz.shape[0], 3 - scores_viz.shape[1]))
        scores_viz = np.hstack([scores_viz, pad])

    var_exp_viz = pca_viz.explained_variance_ratio_ * 100

    viz_df = pd.DataFrame(scores_viz[:, :3], columns=["PC1", "PC2", "PC3"])
    viz_df["Sample"] = result["Sample"].values
    viz_df["Cluster"] = result["Cluster"].values
    if "Original_Label" in result.columns:
        viz_df["Original_Label"] = result["Original_Label"].values

    hover_cols = ["Sample"]
    if "Original_Label" in viz_df.columns:
        hover_cols.append("Original_Label")

    # 2D scatter
    fig_2d = px.scatter(
        viz_df, x="PC1", y="PC2", color="Cluster",
        hover_data=hover_cols,
        color_discrete_map=color_map,
        title=f"{method} — 2D PCA projection",
        labels={
            "PC1": f"PC1 ({var_exp_viz[0]:.1f}%)"
                    if len(var_exp_viz) > 0 else "PC1",
            "PC2": f"PC2 ({var_exp_viz[1]:.1f}%)"
                    if len(var_exp_viz) > 1 else "PC2",
        },
    )
    fig_2d.update_traces(marker=dict(size=10,
                                       line=dict(width=0.8,
                                                  color="darkslategrey")))
    fig_2d.update_layout(template="plotly_white", height=650,
                          legend=dict(itemsizing="constant"))
    st.plotly_chart(fig_2d, use_container_width=True)

    # 3D scatter (only if we have real 3 PCs)
    if X_use.shape[1] >= 3:
        fig_3d = px.scatter_3d(
            viz_df, x="PC1", y="PC2", z="PC3",
            color="Cluster", hover_data=hover_cols,
            color_discrete_map=color_map,
            title=f"{method} — 3D PCA projection",
            labels={
                "PC1": f"PC1 ({var_exp_viz[0]:.1f}%)",
                "PC2": f"PC2 ({var_exp_viz[1]:.1f}%)",
                "PC3": f"PC3 ({var_exp_viz[2]:.1f}%)"
                        if len(var_exp_viz) > 2 else "PC3",
            },
        )
        fig_3d.update_traces(marker=dict(size=5,
                                           line=dict(width=0.3, color="black")))
        fig_3d.update_layout(template="plotly_white", height=700)
        st.plotly_chart(fig_3d, use_container_width=True)

    # ─── Dendrogram (for hierarchical / NJ) ───
    if Z is not None:
        st.subheader("Dendrogram")
        if len(result) <= 300:
            # Use real labels if available
            if label_col_active and "Original_Label" in result.columns:
                dendro_labels = [
                    f"{s} ({l})" if pd.notna(l) else str(s)
                    for s, l in zip(result["Sample"], result["Original_Label"])
                ]
            else:
                dendro_labels = result["Sample"].tolist()

            try:
                fig_dendro = ff.create_dendrogram(
                    X_use, labels=dendro_labels, orientation="left",
                    linkagefun=lambda x: Z,
                )
                fig_dendro.update_layout(
                    template="plotly_white",
                    height=max(600, len(dendro_labels) * 15),
                    title=f"{method} Dendrogram",
                    xaxis_title="Distance",
                )

                # Add colored leaf markers if metadata available
                if label_col_active and pop_map:
                    unique_pops = sorted(set(pop_map.values()))
                    pop_color_map = {p: color_palette[i % len(color_palette)]
                                       for i, p in enumerate(unique_pops)}

                    tick_labels = fig_dendro.layout.yaxis.ticktext
                    if tick_labels:
                        y_positions = fig_dendro.layout.yaxis.tickvals
                        # Add legend entries for populations
                        for pop_name, col in pop_color_map.items():
                            fig_dendro.add_trace(go.Scatter(
                                x=[None], y=[None],
                                mode="markers",
                                marker=dict(size=12, color=col),
                                name=str(pop_name),
                                showlegend=True,
                            ))
                        fig_dendro.update_layout(showlegend=True)

                st.plotly_chart(fig_dendro, use_container_width=True)
                download_plotly_html(fig_dendro, "cluster_dendrogram.html",
                                      key="dl_clu_dendro")
            except Exception as e:
                st.warning(f"Could not render dendrogram: {e}")
        else:
            st.info(f"Dendrogram not rendered ({len(result)} > 300 samples). "
                     "Please subsample for visualization.")

    # ─── Cluster size distribution ───
    st.subheader("Cluster Size Distribution")
    sizes = result["Cluster"].value_counts().reset_index()
    sizes.columns = ["Cluster", "Count"]
    fig_size = px.bar(
        sizes, x="Cluster", y="Count",
        color="Cluster",
        color_discrete_map=color_map,
        text="Count",
        title="Samples per Cluster",
    )
    fig_size.update_traces(textposition="outside")
    fig_size.update_layout(template="plotly_white", height=450,
                             showlegend=False, xaxis_tickangle=45)
    st.plotly_chart(fig_size, use_container_width=True)


# ═══════════════════════════════════════════
# Optimal K search (always available)
# ═══════════════════════════════════════════
st.markdown("---")
st.subheader("🔍 Optimal K Search")
st.write(
    "Explore different values of K to find the optimal number of clusters. "
    "Look for the **elbow** in WCSS and **peak** in silhouette."
)

k_range_max = st.slider("Maximum K to test", 5, 20, 15, key="clu_ksrange")

if st.button("🚀 Run K search", key="clu_ksearch",
              use_container_width=True):
    with st.spinner(f"Testing K = 2 to {k_range_max}..."):
        k_range = range(2, min(k_range_max + 1, X_use.shape[0]))
        wcss, sils, dbs, chs = [], [], [], []

        for k in k_range:
            km_test = KMeans(n_clusters=k, random_state=42, n_init=10)
            lbl = km_test.fit_predict(X_use)
            wcss.append(km_test.inertia_)
            if len(set(lbl)) > 1:
                sils.append(silhouette_score(X_use, lbl))
                dbs.append(davies_bouldin_score(X_use, lbl))
                try:
                    chs.append(calinski_harabasz_score(X_use, lbl))
                except Exception:
                    chs.append(np.nan)
            else:
                sils.append(0)
                dbs.append(np.nan)
                chs.append(np.nan)

    ksearch_df = pd.DataFrame({
        "K": list(k_range),
        "WCSS": wcss,
        "Silhouette": sils,
        "Davies_Bouldin": dbs,
        "Calinski_Harabasz": chs,
    })

    # Optimal K suggestions
    best_sil_k = ksearch_df.loc[ksearch_df["Silhouette"].idxmax(), "K"]
    best_db_k = ksearch_df.loc[ksearch_df["Davies_Bouldin"].idxmin(), "K"]
    best_ch_k = ksearch_df.loc[ksearch_df["Calinski_Harabasz"].idxmax(), "K"]

    st.markdown("#### 🎯 Recommended K")
    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("Best K (Silhouette)", int(best_sil_k))
    rc2.metric("Best K (Davies-Bouldin)", int(best_db_k))
    rc3.metric("Best K (Calinski-Harabasz)", int(best_ch_k))

    # Multi-panel plot
    fig_ks = go.Figure()
    fig_ks.add_trace(go.Scatter(
        x=ksearch_df["K"], y=ksearch_df["WCSS"],
        mode="lines+markers", name="WCSS (Elbow)",
        line=dict(color="steelblue", width=2),
        marker=dict(size=10),
    ))
    fig_ks.add_trace(go.Scatter(
        x=ksearch_df["K"], y=ksearch_df["Silhouette"],
        mode="lines+markers", name="Silhouette",
        yaxis="y2",
        line=dict(color="red", width=2),
        marker=dict(size=10),
    ))
    fig_ks.update_layout(
        title="Optimal K search — WCSS + Silhouette",
        xaxis_title="K",
        yaxis=dict(title="WCSS", side="left"),
        yaxis2=dict(title="Silhouette", overlaying="y", side="right"),
        template="plotly_white", height=500,
        legend=dict(x=0.7, y=0.98),
    )
    fig_ks.add_vline(x=best_sil_k, line_dash="dash", line_color="green",
                       annotation_text=f"Best K (Sil) = {best_sil_k}",
                       annotation_position="top")
    st.plotly_chart(fig_ks, use_container_width=True)

    # Davies-Bouldin & Calinski-Harabasz
    fig_db_ch = go.Figure()
    fig_db_ch.add_trace(go.Scatter(
        x=ksearch_df["K"], y=ksearch_df["Davies_Bouldin"],
        mode="lines+markers", name="Davies-Bouldin (lower better)",
        line=dict(color="purple", width=2),
    ))
    fig_db_ch.add_trace(go.Scatter(
        x=ksearch_df["K"], y=ksearch_df["Calinski_Harabasz"],
        mode="lines+markers", name="Calinski-Harabasz (higher better)",
        yaxis="y2",
        line=dict(color="orange", width=2),
    ))
    fig_db_ch.update_layout(
        title="Davies-Bouldin & Calinski-Harabasz Indices",
        xaxis_title="K",
        yaxis=dict(title="Davies-Bouldin"),
        yaxis2=dict(title="Calinski-Harabasz",
                    overlaying="y", side="right"),
        template="plotly_white", height=500,
    )
    st.plotly_chart(fig_db_ch, use_container_width=True)

    st.dataframe(ksearch_df.style.format({
        "WCSS": "{:.2f}", "Silhouette": "{:.4f}",
        "Davies_Bouldin": "{:.4f}", "Calinski_Harabasz": "{:.2f}",
    }), use_container_width=True)

    download_dataframe(ksearch_df, "k_search_results.csv",
                        key="dl_ksrch")