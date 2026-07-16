"""Linkage Disequilibrium — Genome-wide LD, LD decay, pairwise LD, haplotype blocks."""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

from utils import (
    get_geno_from_session, get_meta_from_session,
    impute_missing, calc_allele_freq, calc_maf,
    download_plotly_html, download_dataframe,
)

st.title("🔗 Linkage Disequilibrium")
st.markdown("---")

geno, marker_info = get_geno_from_session()
if geno is None:
    st.stop()

if marker_info is None or "Chrom" not in marker_info.columns:
    st.warning(
        "⚠️ Marker information (Chromosome + Position) is required for "
        "meaningful LD analysis. Uploading VCF or HapMap format is recommended."
    )


# ═══════════════════════════════════════════
# Core LD computation helpers
# ═══════════════════════════════════════════
def compute_r2(g1, g2):
    """
    Compute r² between two SNP columns (0/1/2 coded).
    Uses correlation-based formula: r² = corr(g1, g2)²
    """
    mask = ~(np.isnan(g1) | np.isnan(g2))
    if mask.sum() < 5:
        return np.nan
    a, b = g1[mask], g2[mask]
    if a.std() == 0 or b.std() == 0:
        return np.nan
    r = np.corrcoef(a, b)[0, 1]
    return r * r


def compute_dprime(g1, g2):
    """
    Compute D' between two SNP columns.
    Approximation using genotype dosages.
    """
    mask = ~(np.isnan(g1) | np.isnan(g2))
    if mask.sum() < 5:
        return np.nan
    a, b = g1[mask], g2[mask]

    pA = a.mean() / 2.0
    pB = b.mean() / 2.0

    if pA == 0 or pA == 1 or pB == 0 or pB == 1:
        return np.nan

    # Covariance / 2 = D
    D = np.cov(a, b)[0, 1] / 2.0
    if D >= 0:
        Dmax = min(pA * (1 - pB), (1 - pA) * pB)
    else:
        Dmax = min(pA * pB, (1 - pA) * (1 - pB))

    if Dmax < 1e-9:
        return np.nan
    return abs(D / Dmax)


def compute_pairwise_ld_matrix(geno_sub, metric="r2"):
    """Compute full pairwise LD matrix for a subset of SNPs."""
    n_snps = geno_sub.shape[1]
    G = geno_sub.values.astype(float)
    LD = np.full((n_snps, n_snps), np.nan)

    for i in range(n_snps):
        LD[i, i] = 1.0
        for j in range(i + 1, n_snps):
            if metric == "r2":
                v = compute_r2(G[:, i], G[:, j])
            else:
                v = compute_dprime(G[:, i], G[:, j])
            LD[i, j] = v
            LD[j, i] = v
    return LD


# ═══════════════════════════════════════════
# Tabs
# ═══════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌍 Genome-wide LD",
    "📉 LD Decay",
    "🧬 Chromosome LD",
    "🔗 Pairwise LD",
    "🧱 Haplotype Blocks",
])

# =========================================================
# TAB 1 — Genome-wide LD summary
# =========================================================
with tab1:
    st.subheader("Genome-wide LD Summary")
    st.write(
        "Computes summary statistics of LD (r²) across a random sample "
        "of SNP pairs to characterize the global LD landscape."
    )

    n_pairs_gw = st.slider("Number of random SNP pairs to sample",
                             500, 20000, 5000, 500, key="gwld_n")

    metric_gw = st.radio("LD metric", ["r²", "D'"], horizontal=True,
                          key="gwld_metric")

    if st.button("🚀 Compute Genome-wide LD", key="gwld_run"):
        rng = np.random.RandomState(42)
        n_snps = geno.shape[1]

        with st.spinner(f"Computing LD for {n_pairs_gw} random pairs..."):
            i_idx = rng.randint(0, n_snps, n_pairs_gw)
            j_idx = rng.randint(0, n_snps, n_pairs_gw)
            mask_ij = i_idx != j_idx
            i_idx, j_idx = i_idx[mask_ij], j_idx[mask_ij]

            G = geno.values.astype(float)
            ld_vals = []
            for i, j in zip(i_idx, j_idx):
                if metric_gw == "r²":
                    v = compute_r2(G[:, i], G[:, j])
                else:
                    v = compute_dprime(G[:, i], G[:, j])
                if not np.isnan(v):
                    ld_vals.append(v)

        ld_vals = np.array(ld_vals)

        st.success(f"✅ Computed {len(ld_vals):,} valid LD values.")

        # Summary
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mean", f"{ld_vals.mean():.4f}")
        m2.metric("Median", f"{np.median(ld_vals):.4f}")
        m3.metric("Std", f"{ld_vals.std():.4f}")
        m4.metric(f"{metric_gw} > 0.2", f"{(ld_vals > 0.2).mean()*100:.1f}%")

        # Distribution
        fig_gw = px.histogram(
            x=ld_vals, nbins=60,
            title=f"Distribution of pairwise {metric_gw}",
            labels={"x": metric_gw, "y": "Count"},
        )
        fig_gw.update_layout(template="plotly_white", height=450)
        st.plotly_chart(fig_gw, use_container_width=True)

        download_dataframe(
            pd.DataFrame({metric_gw: ld_vals}),
            "genome_wide_ld.csv", key="dl_gwld"
        )

# =========================================================
# TAB 2 — LD Decay
# =========================================================
with tab2:
    st.subheader("LD Decay with Physical Distance")
    st.write(
        "Estimates how LD (r²) decays as a function of physical distance "
        "between SNPs. Requires marker positions."
    )

    if marker_info is None or "Pos" not in marker_info.columns:
        st.error("Marker Position information is required.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            chrom_sel = st.selectbox(
                "Chromosome (for LD decay)",
                sorted(marker_info["Chrom"].astype(str).unique()),
                key="ldd_chr",
            )
        with c2:
            max_dist = st.number_input(
                "Max distance (bp) to consider",
                value=1_000_000, step=10_000, key="ldd_maxd",
            )

        n_pairs_dec = st.slider("Max SNP pairs to compute",
                                  1000, 50000, 10000, 1000,
                                  key="ldd_np")

        if st.button("🚀 Compute LD Decay", key="ldd_run"):
            # Subset markers on this chromosome
            m_sub = marker_info[marker_info["Chrom"].astype(str)
                                 == str(chrom_sel)]
            if len(m_sub) < 5:
                st.warning("Not enough markers on this chromosome.")
                st.stop()

            m_sub = m_sub.sort_values("Pos").reset_index(drop=True)
            markers_here = m_sub["Marker"].tolist()
            markers_here = [m for m in markers_here if m in geno.columns]
            geno_chr = geno[markers_here]

            positions = m_sub.set_index("Marker").loc[markers_here, "Pos"].values.astype(float)

            n_here = len(markers_here)
            G = geno_chr.values.astype(float)

            rng = np.random.RandomState(42)
            distances = []
            r2_vals = []

            with st.spinner("Computing LD-distance pairs..."):
                for _ in range(n_pairs_dec):
                    i = rng.randint(0, n_here)
                    j = rng.randint(0, n_here)
                    if i == j:
                        continue
                    d = abs(positions[i] - positions[j])
                    if d > max_dist or d == 0:
                        continue
                    r2 = compute_r2(G[:, i], G[:, j])
                    if not np.isnan(r2):
                        distances.append(d)
                        r2_vals.append(r2)

            decay_df = pd.DataFrame({"Distance_bp": distances,
                                       "r2": r2_vals})

            if len(decay_df) < 10:
                st.warning("Too few valid pairs. Try increasing sample size.")
                st.stop()

            # Fit LD decay curve: r² = 1 / (1 + C * d)
            # Nonlinear fit using scipy
            from scipy.optimize import curve_fit

            def _decay_fn(d, C):
                return 1.0 / (1.0 + C * d)

            try:
                popt, _ = curve_fit(_decay_fn, decay_df["Distance_bp"],
                                      decay_df["r2"], p0=[1e-4],
                                      maxfev=10000)
                C = popt[0]
                # Half-decay: r² = 0.5 → d = 1/C
                half_decay = 1.0 / C
            except Exception:
                C = None
                half_decay = None

            # Binned mean r² by distance
            bins = np.linspace(0, max_dist, 40)
            decay_df["bin"] = pd.cut(decay_df["Distance_bp"], bins)
            binned = decay_df.groupby("bin", observed=True)["r2"].agg(
                ["mean", "count"]).reset_index()
            binned["mid"] = binned["bin"].apply(
                lambda x: (x.left + x.right) / 2)

            fig_dec = go.Figure()
            fig_dec.add_trace(go.Scatter(
                x=decay_df["Distance_bp"], y=decay_df["r2"],
                mode="markers", name="Pairs",
                marker=dict(color="lightblue", size=3, opacity=0.3),
                hoverinfo="skip",
            ))
            fig_dec.add_trace(go.Scatter(
                x=binned["mid"], y=binned["mean"],
                mode="lines+markers", name="Binned mean r²",
                line=dict(color="red", width=3),
                marker=dict(size=8),
            ))
            if C is not None:
                x_fit = np.linspace(0, max_dist, 200)
                y_fit = _decay_fn(x_fit, C)
                fig_dec.add_trace(go.Scatter(
                    x=x_fit, y=y_fit,
                    mode="lines", name=f"Fit: 1/(1+{C:.2e}·d)",
                    line=dict(color="darkgreen", width=2, dash="dash"),
                ))

            fig_dec.update_layout(
                title=f"LD Decay — Chromosome {chrom_sel}",
                xaxis_title="Physical distance (bp)",
                yaxis_title="r²",
                template="plotly_white",
                height=550,
            )
            st.plotly_chart(fig_dec, use_container_width=True)

            # Summary
            if half_decay is not None:
                st.info(
                    f"📏 **Estimated LD half-decay distance:** "
                    f"{half_decay:,.0f} bp "
                    f"(distance at which r² drops to 0.5)"
                )

            m1, m2, m3 = st.columns(3)
            m1.metric("N pairs", f"{len(decay_df):,}")
            m2.metric("Mean r²", f"{decay_df['r2'].mean():.4f}")
            m3.metric("Half-decay", f"{half_decay:,.0f} bp" if half_decay else "N/A")

            download_dataframe(decay_df.drop(columns=["bin"]),
                                "ld_decay.csv", key="dl_ldd")

# =========================================================
# TAB 3 — Chromosome-wise LD
# =========================================================
with tab3:
    st.subheader("Per-Chromosome LD Summary")
    st.write("Compare mean LD levels across chromosomes.")

    if marker_info is None or "Chrom" not in marker_info.columns:
        st.error("Chromosome information is required.")
    else:
        n_pairs_chr = st.slider(
            "Random pairs per chromosome", 500, 10000, 2000, 500,
            key="chrld_n",
        )

        if st.button("🚀 Compute per-chromosome LD", key="chrld_run"):
            chroms = sorted(marker_info["Chrom"].astype(str).unique())
            chr_stats = []

            with st.spinner("Computing..."):
                for ch in chroms:
                    m_sub = marker_info[marker_info["Chrom"].astype(str)
                                         == str(ch)]
                    markers_ch = [m for m in m_sub["Marker"]
                                    if m in geno.columns]
                    if len(markers_ch) < 5:
                        continue
                    G_ch = geno[markers_ch].values.astype(float)
                    n_here = G_ch.shape[1]

                    rng = np.random.RandomState(42)
                    r2s = []
                    for _ in range(n_pairs_chr):
                        i = rng.randint(0, n_here)
                        j = rng.randint(0, n_here)
                        if i == j:
                            continue
                        r2 = compute_r2(G_ch[:, i], G_ch[:, j])
                        if not np.isnan(r2):
                            r2s.append(r2)

                    if r2s:
                        chr_stats.append({
                            "Chromosome": ch,
                            "N_markers": len(markers_ch),
                            "N_pairs": len(r2s),
                            "Mean_r2": np.mean(r2s),
                            "Median_r2": np.median(r2s),
                            "Prop_r2_gt_0.2": np.mean(np.array(r2s) > 0.2),
                        })

            chr_stats_df = pd.DataFrame(chr_stats)
            st.dataframe(chr_stats_df, use_container_width=True)
            download_dataframe(chr_stats_df, "per_chromosome_ld.csv",
                                key="dl_chrld")

            fig_chr = px.bar(
                chr_stats_df, x="Chromosome", y="Mean_r2",
                color="Mean_r2", color_continuous_scale="Viridis",
                title="Mean r² per chromosome",
                text=chr_stats_df["Mean_r2"].round(3),
            )
            fig_chr.update_traces(textposition="outside")
            fig_chr.update_layout(template="plotly_white", height=500)
            st.plotly_chart(fig_chr, use_container_width=True)

# =========================================================
# TAB 4 — Pairwise LD heatmap
# =========================================================
with tab4:
    st.subheader("Pairwise LD Heatmap (Region)")
    st.write(
        "Compute full pairwise LD matrix for a subset of markers "
        "(recommended ≤200 markers for readable heatmap)."
    )

    n_markers_pw = st.slider("Number of markers to display",
                               10, 500, 100, 10, key="pwld_n")

    ld_metric = st.radio("LD metric", ["r²", "D'"], horizontal=True,
                          key="pwld_metric")

    marker_source = st.radio(
        "Marker selection",
        ["Random", "Highest MAF", "Contiguous (first N)"],
        horizontal=True, key="pwld_src",
    )

    if st.button("🚀 Compute pairwise LD", key="pwld_run"):
        if marker_source == "Random":
            selected = np.random.RandomState(42).choice(
                geno.columns, min(n_markers_pw, geno.shape[1]),
                replace=False)
        elif marker_source == "Highest MAF":
            maf = calc_maf(geno)
            selected = maf.nlargest(n_markers_pw).index.tolist()
        else:
            selected = geno.columns[:n_markers_pw].tolist()

        geno_sub = geno[selected]

        with st.spinner("Computing pairwise LD..."):
            metric_key = "r2" if ld_metric == "r²" else "dprime"
            LD = compute_pairwise_ld_matrix(geno_sub, metric=metric_key)

        ld_df = pd.DataFrame(LD, index=selected, columns=selected)

        fig_pw = px.imshow(
            LD, x=selected, y=selected,
            color_continuous_scale="Reds",
            title=f"Pairwise {ld_metric} Heatmap",
            aspect="auto",
        )
        fig_pw.update_layout(template="plotly_white", height=700)
        st.plotly_chart(fig_pw, use_container_width=True)

        # Summary
        upper = LD[np.triu_indices(len(LD), k=1)]
        upper = upper[~np.isnan(upper)]

        s1, s2, s3 = st.columns(3)
        s1.metric("Mean", f"{upper.mean():.4f}")
        s2.metric("Median", f"{np.median(upper):.4f}")
        s3.metric(f"% > 0.5", f"{(upper > 0.5).mean()*100:.1f}%")

        download_plotly_html(fig_pw, "pairwise_ld.html",
                              key="dl_pwld_html")
        download_dataframe(ld_df.reset_index(),
                            "pairwise_ld_matrix.csv",
                            key="dl_pwld_csv")

# =========================================================
# TAB 5 — Haplotype blocks
# =========================================================
with tab5:
    st.subheader("Haplotype Block Detection")
    st.write(
        "Identifies contiguous blocks of high LD (r² above threshold). "
        "Simple sliding-window approach based on r² values."
    )

    if marker_info is None or "Chrom" not in marker_info.columns:
        st.error("Marker information required.")
    else:
        b1, b2, b3 = st.columns(3)
        with b1:
            chrom_blk = st.selectbox(
                "Chromosome",
                sorted(marker_info["Chrom"].astype(str).unique()),
                key="blk_chr",
            )
        with b2:
            r2_thresh = st.slider("r² threshold for block", 0.1, 1.0,
                                    0.5, 0.05, key="blk_r2")
        with b3:
            max_gap = st.number_input(
                "Max gap (bp) within block",
                value=100_000, step=10_000, key="blk_gap",
            )

        if st.button("🚀 Detect haplotype blocks", key="blk_run"):
            m_sub = marker_info[marker_info["Chrom"].astype(str)
                                 == str(chrom_blk)]
            m_sub = m_sub.sort_values("Pos").reset_index(drop=True)
            markers_here = [m for m in m_sub["Marker"]
                             if m in geno.columns]
            positions = m_sub.set_index("Marker").loc[markers_here,
                                                        "Pos"].values.astype(float)
            geno_here = geno[markers_here]

            n_here = len(markers_here)
            if n_here < 3:
                st.warning("Not enough markers on this chromosome.")
                st.stop()

            # Simple block algorithm: start a block, extend while r² >= thresh
            # and physical distance <= max_gap
            blocks = []
            i = 0
            G = geno_here.values.astype(float)

            with st.spinner("Detecting blocks..."):
                while i < n_here - 1:
                    block_start = i
                    j = i + 1
                    while j < n_here:
                        if positions[j] - positions[j - 1] > max_gap:
                            break
                        r2 = compute_r2(G[:, block_start], G[:, j])
                        if np.isnan(r2) or r2 < r2_thresh:
                            break
                        j += 1
                    block_end = j - 1
                    if block_end > block_start:
                        blocks.append({
                            "Block_ID": len(blocks) + 1,
                            "Start_marker": markers_here[block_start],
                            "End_marker": markers_here[block_end],
                            "Start_pos": positions[block_start],
                            "End_pos": positions[block_end],
                            "Length_bp": positions[block_end] - positions[block_start],
                            "N_markers": block_end - block_start + 1,
                        })
                    i = block_end + 1 if block_end > block_start else i + 1

            blocks_df = pd.DataFrame(blocks)

            if len(blocks_df) == 0:
                st.warning("No blocks detected with current settings.")
            else:
                st.success(f"✅ Detected {len(blocks_df)} haplotype blocks.")

                m1, m2, m3 = st.columns(3)
                m1.metric("N blocks", f"{len(blocks_df)}")
                m2.metric("Mean length (bp)",
                            f"{blocks_df['Length_bp'].mean():,.0f}")
                m3.metric("Mean markers/block",
                            f"{blocks_df['N_markers'].mean():.1f}")

                st.dataframe(blocks_df, use_container_width=True)

                # Visualization: blocks along the chromosome
                fig_blk = go.Figure()
                for _, row in blocks_df.iterrows():
                    fig_blk.add_shape(
                        type="rect",
                        x0=row["Start_pos"], x1=row["End_pos"],
                        y0=0, y1=1,
                        fillcolor="steelblue", opacity=0.5,
                        line=dict(width=0),
                    )
                fig_blk.update_layout(
                    title=f"Haplotype blocks on Chr {chrom_blk}",
                    xaxis_title="Position (bp)",
                    yaxis=dict(visible=False),
                    template="plotly_white",
                    height=250,
                )
                st.plotly_chart(fig_blk, use_container_width=True)

                # Length distribution
                fig_len = px.histogram(
                    blocks_df, x="Length_bp", nbins=30,
                    title="Block length distribution",
                )
                fig_len.update_layout(template="plotly_white", height=400)
                st.plotly_chart(fig_len, use_container_width=True)

                download_dataframe(blocks_df, "haplotype_blocks.csv",
                                    key="dl_blk")