"""Upload & Metadata page — load genotype + metadata files."""

import streamlit as st
import pandas as pd
import plotly.express as px
from parsers import load_genotype_data, load_metadata

st.title("📁 Upload & Metadata")
st.markdown("---")

st.subheader("Step 1: Upload Genotype Data")
st.write("Supported formats: **Numeric (0/1/2)**, **HapMap**, **VCF**")

geno, marker_info = load_genotype_data("upload_geno")

if geno is not None:
    st.subheader("📊 Genotype Summary")
    g1, g2, g3 = st.columns(3)
    g1.metric("Samples", geno.shape[0])
    g2.metric("Markers", geno.shape[1])
    g3.metric("Missing %", f"{geno.isna().sum().sum() / geno.size * 100:.2f}%")

    with st.expander("Preview genotype matrix (first 10 rows × 10 cols)"):
        st.dataframe(geno.iloc[:10, :10], use_container_width=True)

    if marker_info is not None:
        with st.expander("Marker information"):
            st.dataframe(marker_info.head(20), use_container_width=True)

st.markdown("---")

st.subheader("Step 2: Upload Metadata (Optional)")
st.write("CSV/Excel file with columns like: **Sample_ID**, **Species**, **Origin**, **Group**, **Latitude**, **Longitude**")

meta = load_metadata("upload_meta")

if meta is not None:
    with st.expander("Preview metadata"):
        st.dataframe(meta.head(20), use_container_width=True)

    # If genotype is loaded, validate sample overlap
    if geno is not None:
        meta_ids = set(meta.iloc[:, 0].astype(str))
        geno_ids = set(geno.index.astype(str))
        overlap = meta_ids & geno_ids
        st.info(f"Sample overlap: **{len(overlap)}** / {len(geno_ids)} genotyped samples found in metadata")