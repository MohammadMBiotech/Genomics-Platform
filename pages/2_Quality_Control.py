"""Quality Control module — Missing rates, MAF filtering, HWE."""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import chisquare

from utils import (
    get_geno_from_session, calc_maf, calc_missing_rate,
    calc_het_obs, calc_het_exp, download_plotly_html, download_dataframe,
)

st.title("🧹 Quality Control")
st.markdown("---")

geno, marker_info = get_geno_from_session()
if geno is None:
    st.stop()

st.subheader("📊 Raw Data Summary")
r1, r2, r3, r4 = st.columns(4)
r1.metric("Samples", geno.shape[0])
r2.metric("Markers", geno.shape[1])
r3.metric("Total Missing %", f"{geno.isna().sum().sum() / geno.size * 100:.2f}%")
r4.metric("Monomorphic Markers", int((geno.nunique() <= 1).sum()))

# ── Missing rate per marker ──
st.subheader("Missing Rate per Marker")
miss_marker = calc_missing_rate(geno, axis=0)

fig_miss_m = px.histogram(x=miss_marker.values, nbins=50,
                           title="Distribution of Marker Missing Rate",
                           labels={"x": "Missing Rate", "y": "Count"})
fig_miss_m.update_layout(template="plotly_white", height=400)
st.plotly_chart(fig_miss_m, use_container_width=True)

# ── Missing rate per sample ──
st.subheader("Missing Rate per Sample")
miss_sample = calc_missing_rate(geno, axis=1)

fig_miss_s = px.histogram(x=miss_sample.values, nbins=50,
                           title="Distribution of Sample Missing Rate",
                           labels={"x": "Missing Rate", "y": "Count"})
fig_miss_s.update_layout(template="plotly_white", height=400)
st.plotly_chart(fig_miss_s, use_container_width=True)

# ── MAF distribution ──
st.subheader("Minor Allele Frequency (MAF)")
maf = calc_maf(geno)

fig_maf = px.histogram(x=maf.values, nbins=50,
                        title="MAF Distribution",
                        labels={"x": "MAF", "y": "Count"})
fig_maf.update_layout(template="plotly_white", height=400)
st.plotly_chart(fig_maf, use_container_width=True)

# ── Heterozygosity ──
st.subheader("Observed vs Expected Heterozygosity")
ho = calc_het_obs(geno)
he = calc_het_exp(geno)

het_df = pd.DataFrame({"Ho": ho, "He": he}).dropna()
fig_het = px.scatter(het_df, x="He", y="Ho", opacity=0.5,
                      title="Observed vs Expected Heterozygosity per Marker")
max_val = max(het_df.max().max(), 0.6)
fig_het.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                   line=dict(color="red", dash="dash"))
fig_het.update_layout(template="plotly_white", height=500)
st.plotly_chart(fig_het, use_container_width=True)

# ── HWE test ──
st.subheader("Hardy-Weinberg Equilibrium (HWE) Test")

def hwe_test_marker(col):
    """Chi-square HWE test for a single marker."""
    counts = col.dropna()
    n = len(counts)
    if n == 0:
        return np.nan
    n0 = (counts == 0).sum()
    n1 = (counts == 1).sum()
    n2 = (counts == 2).sum()
    p = (2 * n0 + n1) / (2 * n)
    q = 1 - p
    exp0 = p**2 * n
    exp1 = 2 * p * q * n
    exp2 = q**2 * n
    observed = [n0, n1, n2]
    expected = [max(exp0, 1e-10), max(exp1, 1e-10), max(exp2, 1e-10)]
    try:
        _, pval = chisquare(observed, f_exp=expected)
        return pval
    except Exception:
        return np.nan

with st.spinner("Running HWE tests..."):
    hwe_pvals = geno.apply(hwe_test_marker, axis=0)

fig_hwe = px.histogram(x=-np.log10(hwe_pvals.clip(lower=1e-300).values),
                        nbins=50,
                        title="HWE Test: -log10(p-value) Distribution",
                        labels={"x": "-log10(p)", "y": "Count"})
fig_hwe.add_vline(x=-np.log10(0.001), line_dash="dash", line_color="red",
                   annotation_text="p = 0.001")
fig_hwe.update_layout(template="plotly_white", height=400)
st.plotly_chart(fig_hwe, use_container_width=True)

n_hwe_fail = int((hwe_pvals < 0.001).sum())
st.metric("Markers failing HWE (p < 0.001)", f"{n_hwe_fail:,}")

# ── Filtering ──
st.markdown("---")
st.subheader("🔧 Apply Filters")

fc1, fc2, fc3, fc4 = st.columns(4)
with fc1:
    max_miss_marker = st.slider("Max marker missing rate", 0.0, 1.0, 0.2, 0.05, key="qc_mmm")
with fc2:
    max_miss_sample = st.slider("Max sample missing rate", 0.0, 1.0, 0.3, 0.05, key="qc_mms")
with fc3:
    min_maf = st.slider("Min MAF", 0.0, 0.5, 0.05, 0.01, key="qc_maf")
with fc4:
    hwe_threshold = st.number_input("HWE p threshold", value=0.001, format="%.6f", key="qc_hwe")

if st.button("🚀 Apply Filters", use_container_width=True, key="qc_apply"):
    # Filter markers
    keep_markers = (miss_marker <= max_miss_marker) & (maf >= min_maf) & (hwe_pvals >= hwe_threshold)
    geno_filtered = geno.loc[:, keep_markers]

    # Filter samples
    miss_s_filtered = geno_filtered.isna().mean(axis=1)
    keep_samples = miss_s_filtered <= max_miss_sample
    geno_filtered = geno_filtered.loc[keep_samples]

    st.success(
        f"✅ After filtering: **{geno_filtered.shape[0]}** samples × "
        f"**{geno_filtered.shape[1]}** markers "
        f"(removed {geno.shape[1] - geno_filtered.shape[1]} markers, "
        f"{geno.shape[0] - geno_filtered.shape[0]} samples)"
    )

    # Update session state
    st.session_state["genotype_matrix"] = geno_filtered
    if marker_info is not None:
        st.session_state["marker_info"] = marker_info[
            marker_info["Marker"].isin(geno_filtered.columns)
        ].reset_index(drop=True)

    st.info("Filtered data saved. Other modules will use the filtered dataset.")

# ── QC Summary Table ──
st.subheader("📋 QC Summary Table")
qc_summary = pd.DataFrame({
    "Marker": geno.columns,
    "Missing_Rate": miss_marker.values,
    "MAF": maf.values,
    "Ho": ho.values,
    "He": he.values,
    "HWE_pval": hwe_pvals.values,
})
st.dataframe(qc_summary.head(100), use_container_width=True)
download_dataframe(qc_summary, "qc_summary.csv", key="dl_qc")