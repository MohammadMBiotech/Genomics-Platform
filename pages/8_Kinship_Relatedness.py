"""Kinship & Relatedness — IBS, VanRaden, Astle-Balding matrices."""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.preprocessing import StandardScaler

from utils import (
    get_geno_from_session, get_meta_from_session,
    impute_missing, calc_allele_freq,
    download_plotly_html, download_dataframe,
)

st.title("👥 Kinship & Relatedness")
st.markdown("---")

geno, marker_info = get_geno_from_session()
if geno is None:
    st.stop()

meta = get_meta_from_session()

st.subheader("Kinship Matrix Construction")

kin_method = st.selectbox(
    "Kinship estimation method",
    ["VanRaden (2008)", "Astle-Balding", "IBS (Identity By State)",
     "Simple Correlation"],
    key="kin_method",
    help=(
        "• VanRaden: Standard genomic relationship matrix (GRM)\n"
        "• Astle-Balding: Similar to VanRaden with different scaling\n"
        "• IBS: Proportion of alleles shared identically\n"
        "• Correlation: Pearson correlation between individuals"
    ),
)

imp_method = st.selectbox("Missing imputation",
                            ["mean", "median", "zero"],
                            key="kin_imp")

if st.button("🚀 Compute Kinship Matrix", use_container_width=True,
             key="kin_run"):
    with st.spinner("Computing kinship matrix..."):
        # Impute
        geno_imp = impute_missing(geno, method=imp_method)
        M = geno_imp.values.astype(float)  # samples × markers
        n_samples, n_markers = M.shape

        if kin_method == "VanRaden (2008)":
            # G = (M-P)(M-P)' / (2 * sum(p*(1-p)))
            p = M.mean(axis=0) / 2  # allele freq
            P = 2 * p  # expected freq
            Z = M - P  # center
            denom = 2 * np.sum(p * (1 - p))
            if denom < 1e-9:
                denom = 1e-9
            K = Z @ Z.T / denom

        elif kin_method == "Astle-Balding":
            # Similar to VanRaden but each marker scaled by its variance
            p = M.mean(axis=0) / 2
            var_marker = 2 * p * (1 - p)
            var_marker[var_marker < 1e-9] = 1e-9
            M_scaled = (M - 2 * p) / np.sqrt(var_marker)
            K = M_scaled @ M_scaled.T / n_markers

        elif kin_method == "IBS (Identity By State)":
            # IBS(i,j) = mean over markers of (2 - |M_i - M_j|) / 2
            K = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                for j in range(i, n_samples):
                    diff = np.abs(M[i] - M[j])
                    ibs = np.mean((2 - diff) / 2)
                    K[i, j] = K[j, i] = ibs

        elif kin_method == "Simple Correlation":
            K = np.corrcoef(M)

    kin_df = pd.DataFrame(K, index=geno.index, columns=geno.index)

    st.success("✅ Kinship matrix computed.")

    # ── Summary stats ──
    st.subheader("Kinship Statistics")
    diag = np.diag(K)
    off_diag = K[np.triu_indices(n_samples, k=1)]

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Mean self-relatedness", f"{diag.mean():.4f}")
    s2.metric("Mean pairwise kinship", f"{off_diag.mean():.4f}")
    s3.metric("Max pairwise kinship", f"{off_diag.max():.4f}")
    s4.metric("Min pairwise kinship", f"{off_diag.min():.4f}")

    # ── Kinship heatmap (with hierarchical clustering) ──
    st.subheader("Kinship Heatmap")

    show_clustered = st.checkbox("Reorder rows/cols by clustering",
                                   True, key="kin_clust")

    if show_clustered and n_samples > 2 and n_samples <= 500:
        # Convert kinship to distance
        D_kin = 1 - K
        np.fill_diagonal(D_kin, 0)
        # Force symmetric distance
        D_kin = (D_kin + D_kin.T) / 2
        try:
            from scipy.spatial.distance import squareform
            condensed = squareform(D_kin, checks=False)
            Z_kin = linkage(condensed, method="average")
            order = leaves_list(Z_kin)
            K_ordered = K[np.ix_(order, order)]
            labels_ordered = [geno.index[i] for i in order]
        except Exception:
            K_ordered = K
            labels_ordered = geno.index.tolist()
    else:
        K_ordered = K
        labels_ordered = geno.index.tolist()

    fig_kin = px.imshow(
        K_ordered, x=labels_ordered, y=labels_ordered,
        color_continuous_scale="RdBu_r",
        title=f"Kinship Matrix — {kin_method}",
        aspect="auto",
    )
    fig_kin.update_layout(template="plotly_white", height=700)
    st.plotly_chart(fig_kin, use_container_width=True)
    download_plotly_html(fig_kin, "kinship_heatmap.html",
                          key="dl_kin_html")

    # ── Distribution ──
    st.subheader("Distributions")
    d1, d2 = st.columns(2)

    with d1:
        fig_diag = px.histogram(
            x=diag, nbins=40,
            title="Distribution of self-relatedness (diagonal)",
        )
        fig_diag.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig_diag, use_container_width=True)

    with d2:
        fig_off = px.histogram(
            x=off_diag, nbins=50,
            title="Distribution of pairwise kinship (off-diagonal)",
        )
        fig_off.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig_off, use_container_width=True)

    # ── Top related pairs ──
    st.subheader("Top Related Pairs")
    top_n = st.slider("Number of top pairs", 5, 100, 20,
                       key="kin_topn")

    triu_i, triu_j = np.triu_indices(n_samples, k=1)
    pair_df = pd.DataFrame({
        "Sample_1": geno.index[triu_i],
        "Sample_2": geno.index[triu_j],
        "Kinship": K[triu_i, triu_j],
    })
    pair_df = pair_df.sort_values("Kinship", ascending=False).head(top_n)
    st.dataframe(pair_df, use_container_width=True)
    download_dataframe(pair_df, "top_related_pairs.csv",
                        key="dl_kin_pairs")

    # ── Relatedness classification ──
    st.subheader("Inferred Relatedness Categories")
    all_pairs = pd.DataFrame({
        "Sample_1": geno.index[triu_i],
        "Sample_2": geno.index[triu_j],
        "Kinship": K[triu_i, triu_j],
    })

    def _class(k):
        if k > 0.354:
            return "Duplicate / MZ Twin"
        elif k > 0.177:
            return "1st-degree (parent-offspring / full-sib)"
        elif k > 0.0884:
            return "2nd-degree (half-sib / grandparent)"
        elif k > 0.0442:
            return "3rd-degree (cousin)"
        else:
            return "Unrelated"

    all_pairs["Relatedness"] = all_pairs["Kinship"].apply(_class)
    counts = all_pairs["Relatedness"].value_counts().reset_index()
    counts.columns = ["Category", "Count"]

    fig_cat = px.bar(counts, x="Category", y="Count", color="Category",
                      title="Inferred pairwise relatedness categories",
                      text="Count")
    fig_cat.update_traces(textposition="outside")
    fig_cat.update_layout(template="plotly_white", height=450,
                            showlegend=False)
    st.plotly_chart(fig_cat, use_container_width=True)

    st.dataframe(counts, use_container_width=True)

    # Download full matrix
    download_dataframe(kin_df.reset_index(),
                        f"kinship_matrix_{kin_method.split()[0]}.csv",
                        index=False, key="dl_kin_mat")