"""Clustering — Hierarchical (UPGMA/Ward), K-means, DBSCAN, NJ."""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

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

# ─── Preprocess ───
st.subheader("Preprocessing")
c1, c2 = st.columns(2)
with c1:
    imp_method = st.selectbox("Missing imputation",
                                ["mean", "median", "zero"],
                                key="clu_imp")
with c2:
    scale_data = st.checkbox("Standardize markers", True, key="clu_scale")

geno_imp = impute_missing(geno, method=imp_method)
X = geno_imp.values
if scale_data:
    X = StandardScaler().fit_transform(X)

# Optional PCA pre-reduction
use_pca = st.checkbox("Reduce dimensionality with PCA first "
                       "(recommended for large datasets)", True,
                       key="clu_usepca")
if use_pca:
    n_pcs = st.slider("Number of PCs", 2, min(50, X.shape[1]),
                       min(10, X.shape[1]), key="clu_npcs")
    X_use = PCA(n_components=n_pcs).fit_transform(X)
else:
    X_use = X

# ─── Method selection ───
st.subheader("Clustering Method")
method = st.selectbox(
    "Method",
    ["Hierarchical (UPGMA)", "Hierarchical (Ward)",
     "Hierarchical (Complete)", "Hierarchical (Single)",
     "K-means", "DBSCAN"],
    key="clu_method",
)

# ─── Method-specific parameters ───
if method.startswith("Hierarchical") or method == "K-means":
    n_clusters = st.slider("Number of clusters (K)", 2,
                            min(20, X_use.shape[0]), 4, key="clu_k")

if method == "DBSCAN":
    eps = st.slider("EPS (neighborhood radius)", 0.1, 10.0, 0.5, 0.1,
                     key="clu_eps")
    min_samples = st.slider("Min samples", 2, 20, 5, key="clu_minsam")

if st.button("🚀 Run Clustering", use_container_width=True,
             key="clu_run"):
    with st.spinner(f"Running {method}..."):
        if method == "Hierarchical (UPGMA)":
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
            # Convert to 1-based; -1 = noise
            clusters = np.array([c + 2 if c >= 0 else 0 for c in clusters])
            Z = None

    n_unique = len(set(clusters))
    st.success(f"✅ Clustering completed. Found **{n_unique}** clusters.")

    # ── Cluster metrics ──
    st.subheader("Clustering Quality Metrics")
    m1, m2, m3 = st.columns(3)
    if n_unique > 1 and n_unique < len(clusters):
        sil = silhouette_score(X_use, clusters)
        db_score = davies_bouldin_score(X_use, clusters)
        m1.metric("Silhouette Score",
                    f"{sil:.4f}",
                    help="Range [-1, 1]. Higher is better.")
        m2.metric("Davies-Bouldin",
                    f"{db_score:.4f}",
                    help="Lower is better.")
        m3.metric("Number of Clusters", n_unique)
    else:
        m1.metric("Number of Clusters", n_unique)

    # ── Build result table ──
    result = pd.DataFrame({
        "Sample": geno.index.astype(str),
        "Cluster_ID": clusters,
    })

    # Attach labels from metadata if available
    label_col = None
    if meta is not None:
        label_col = st.session_state.get("clu_label_col", None)
        pop_col = st.selectbox(
            "Attach real labels from metadata (optional)",
            ["None"] + meta.columns.tolist(),
            key="clu_label_sel",
        )
        sam_col = st.selectbox("Sample ID column in metadata",
                                meta.columns.tolist(),
                                key="clu_label_sam")

        if pop_col != "None":
            meta_sub = meta[[sam_col, pop_col]].drop_duplicates()
            meta_sub[sam_col] = meta_sub[sam_col].astype(str)
            result = result.merge(meta_sub, left_on="Sample",
                                    right_on=sam_col, how="left")
            result = result.rename(columns={pop_col: "Original_Label"})
            label_col = "Original_Label"

    # Build descriptive cluster names
    if label_col and label_col in result.columns:
        cluster_names = {}
        for cid in sorted(set(clusters)):
            mask = result["Cluster_ID"] == cid
            labs = result.loc[mask, label_col].dropna()
            if len(labs) > 0:
                most_common = labs.mode().iloc[0]
                count = int(mask.sum())
                cluster_names[cid] = f"Cluster {cid}: {most_common} (n={count})"
            else:
                cluster_names[cid] = f"Cluster {cid}"
        result["Cluster"] = result["Cluster_ID"].map(cluster_names)
    else:
        result["Cluster"] = result["Cluster_ID"].apply(
            lambda x: "Noise" if x == 0 else f"Cluster {x}")

    st.subheader("Clustered Samples")
    st.dataframe(result, use_container_width=True)
    download_dataframe(result, "clustering_results.csv",
                        key="dl_clu_csv")

    # ── PCA visualization of clusters ──
    st.subheader("Cluster Visualization (PCA projection)")
    pca_viz = PCA(n_components=3)
    scores_viz = pca_viz.fit_transform(X_use if X_use.shape[1] >= 3 else
                                         np.hstack([X_use, np.zeros((X_use.shape[0], 3 - X_use.shape[1]))]))

    viz_df = pd.DataFrame(scores_viz, columns=["PC1", "PC2", "PC3"])
    viz_df["Sample"] = result["Sample"].values
    viz_df["Cluster"] = result["Cluster"].values

    # 2D
    fig_2d = px.scatter(
        viz_df, x="PC1", y="PC2", color="Cluster",
        hover_data=["Sample"],
        title=f"{method} — 2D PCA projection",
    )
    fig_2d.update_traces(marker=dict(size=9,
                                       line=dict(width=0.5, color="darkslategrey")))
    fig_2d.update_layout(template="plotly_white", height=600)
    st.plotly_chart(fig_2d, use_container_width=True)

    # 3D
    fig_3d = px.scatter_3d(
        viz_df, x="PC1", y="PC2", z="PC3",
        color="Cluster", hover_data=["Sample"],
        title=f"{method} — 3D PCA projection",
    )
    fig_3d.update_layout(template="plotly_white", height=700)
    st.plotly_chart(fig_3d, use_container_width=True)

    # ── Dendrogram (for hierarchical only) ──
    if Z is not None and len(result) <= 300:
        st.subheader("Dendrogram")
        dendro_labels = (result[label_col].astype(str).tolist()
                          if label_col
                          else result["Sample"].tolist())

        fig_dendro = ff.create_dendrogram(
            X_use, labels=dendro_labels, orientation="left",
            linkagefun=lambda x: Z,
        )
        fig_dendro.update_layout(
            template="plotly_white",
            height=max(600, len(dendro_labels) * 15),
            title=f"{method} Dendrogram",
        )
        st.plotly_chart(fig_dendro, use_container_width=True)
        download_plotly_html(fig_dendro, "cluster_dendrogram.html",
                              key="dl_clu_dendro")

    # ── Cluster size distribution ──
    st.subheader("Cluster Size Distribution")
    sizes = result["Cluster"].value_counts().reset_index()
    sizes.columns = ["Cluster", "Count"]
    fig_size = px.bar(sizes, x="Cluster", y="Count",
                       color="Cluster", text="Count",
                       title="Samples per Cluster")
    fig_size.update_traces(textposition="outside")
    fig_size.update_layout(template="plotly_white", height=450,
                             showlegend=False)
    st.plotly_chart(fig_size, use_container_width=True)

    # ── Optimal K search (elbow + silhouette) ──
    st.markdown("---")
    st.subheader("🔍 Optimal K Search (for K-means / Hierarchical)")

    if st.button("Run K search (K=2..15)", key="clu_ksearch"):
        with st.spinner("Testing K values..."):
            k_range = range(2, min(16, X_use.shape[0]))
            wcss, sils = [], []
            for k in k_range:
                km_test = KMeans(n_clusters=k, random_state=42, n_init=10)
                lbl = km_test.fit_predict(X_use)
                wcss.append(km_test.inertia_)
                if len(set(lbl)) > 1:
                    sils.append(silhouette_score(X_use, lbl))
                else:
                    sils.append(0)

        ksearch_df = pd.DataFrame({"K": list(k_range),
                                     "WCSS": wcss,
                                     "Silhouette": sils})

        fig_ks = go.Figure()
        fig_ks.add_trace(go.Scatter(x=ksearch_df["K"],
                                      y=ksearch_df["WCSS"],
                                      mode="lines+markers",
                                      name="WCSS (Elbow)",
                                      line=dict(color="steelblue")))
        fig_ks.add_trace(go.Scatter(x=ksearch_df["K"],
                                      y=ksearch_df["Silhouette"],
                                      mode="lines+markers",
                                      name="Silhouette",
                                      yaxis="y2",
                                      line=dict(color="red")))
        fig_ks.update_layout(
            title="Optimal K search",
            xaxis_title="K",
            yaxis=dict(title="WCSS"),
            yaxis2=dict(title="Silhouette",
                         overlaying="y", side="right"),
            template="plotly_white", height=450,
        )
        st.plotly_chart(fig_ks, use_container_width=True)
        st.dataframe(ksearch_df, use_container_width=True)