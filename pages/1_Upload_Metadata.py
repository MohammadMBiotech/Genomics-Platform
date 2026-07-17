"""
Upload & Metadata page — comprehensive data loading with detailed preview.

Features:
  - Full genotype preview with adjustable size
  - Genotype distribution (0/1/2 histogram)
  - Missing pattern visualization
  - Per-sample and per-marker statistics
  - Metadata column analysis
  - Sample overlap validation
  - Suggested column mapping
  - Chromosome distribution (if marker info available)
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from parsers import load_genotype_data, load_metadata

st.title("📁 Upload & Metadata")
st.markdown("---")


# ═══════════════════════════════════════════
# STEP 1: Upload Genotype Data
# ═══════════════════════════════════════════
st.subheader("Step 1: Upload Genotype Data")
st.write(
    "Supported formats: **Numeric (0/1/2)**, **HapMap**, **VCF**\n\n"
    "Recommended data structures:\n"
    "- **Numeric CSV**: Rows = samples, Columns = markers\n"
    "- **HapMap**: Standard hmp.txt format\n"
    "- **VCF**: Standard vcf format with GT field"
)

geno, marker_info = load_genotype_data("upload_geno")

if geno is not None:

    # ─── Basic Summary ───
    st.subheader("📊 Genotype Summary")

    g1, g2, g3, g4, g5 = st.columns(5)
    g1.metric("Samples", f"{geno.shape[0]:,}")
    g2.metric("Markers", f"{geno.shape[1]:,}")
    missing_pct = geno.isna().sum().sum() / geno.size * 100
    g3.metric("Missing %", f"{missing_pct:.2f}%")
    memory_mb = geno.memory_usage(deep=True).sum() / (1024 ** 2)
    g4.metric("Memory (MB)", f"{memory_mb:.1f}")

    # Genotype dosage distribution
    all_values = geno.values.flatten()
    all_values = all_values[~np.isnan(all_values)]
    unique_vals = np.unique(all_values)
    is_dosage = set(np.unique(np.round(all_values))).issubset({0, 1, 2})
    g5.metric("Coding", "0/1/2 dosage" if is_dosage else "Continuous")

    # ─── Genotype Preview (adjustable size) ───
    st.markdown("### 👁️ Genotype Matrix Preview")

    pc1, pc2 = st.columns(2)
    with pc1:
        n_rows_show = st.slider(
            "Rows to preview",
            5, min(500, geno.shape[0]),
            min(20, geno.shape[0]),
            key="prev_rows",
        )
    with pc2:
        n_cols_show = st.slider(
            "Columns to preview",
            5, min(200, geno.shape[1]),
            min(20, geno.shape[1]),
            key="prev_cols",
        )

    st.dataframe(geno.iloc[:n_rows_show, :n_cols_show],
                  use_container_width=True)

    with st.expander("🔎 View full matrix (may be slow for large data)"):
        show_full = st.checkbox("Show full genotype matrix",
                                  value=False, key="show_full_geno")
        if show_full:
            st.dataframe(geno, use_container_width=True)

    # ─── Genotype dosage distribution ───
    st.markdown("### 📈 Genotype Distribution")

    dc1, dc2 = st.columns(2)

    with dc1:
        val_counts = pd.Series(all_values).value_counts().sort_index()
        if len(val_counts) <= 20:
            # Discrete distribution (dosages)
            fig_dist = px.bar(
                x=val_counts.index.astype(str),
                y=val_counts.values,
                labels={"x": "Genotype dosage", "y": "Count"},
                title="Genotype dosage distribution",
                color=val_counts.values,
                color_continuous_scale="Blues",
            )
        else:
            fig_dist = px.histogram(
                x=all_values, nbins=50,
                title="Genotype value distribution",
                labels={"x": "Value", "y": "Count"},
            )
        fig_dist.update_layout(template="plotly_white", height=400,
                                 showlegend=False)
        st.plotly_chart(fig_dist, use_container_width=True)

    with dc2:
        # Missing pattern per sample
        miss_per_sample = geno.isna().mean(axis=1).sort_values(ascending=False)
        fig_miss = px.histogram(
            x=miss_per_sample.values, nbins=30,
            title="Missing rate distribution per sample",
            labels={"x": "Missing rate", "y": "Count"},
        )
        fig_miss.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig_miss, use_container_width=True)

    # ─── Per-sample statistics ───
    st.markdown("### 👤 Per-Sample Statistics")

    with st.spinner("Computing per-sample stats..."):
        sample_stats = pd.DataFrame({
            "Sample": geno.index.astype(str),
            "Missing_rate": geno.isna().mean(axis=1).values,
            "Het_rate": (geno == 1).sum(axis=1).values / geno.notna().sum(axis=1).values,
            "N_markers": geno.notna().sum(axis=1).values,
        })

    n_show_samples = st.slider("Samples to show", 10,
                                 min(200, len(sample_stats)),
                                 min(30, len(sample_stats)),
                                 key="prev_sample_stats")

    st.dataframe(sample_stats.head(n_show_samples).style.format({
        "Missing_rate": "{:.4f}",
        "Het_rate": "{:.4f}",
    }), use_container_width=True)

    # ─── Per-marker statistics ───
    st.markdown("### 🧬 Per-Marker Statistics")

    with st.spinner("Computing per-marker stats..."):
        p = geno.mean(axis=0) / 2  # allele freq
        maf = np.minimum(p, 1 - p)
        het_marker = (geno == 1).sum(axis=0) / geno.notna().sum(axis=0)
        miss_marker = geno.isna().mean(axis=0)

        marker_stats = pd.DataFrame({
            "Marker": geno.columns.astype(str),
            "MAF": maf.values,
            "Het_rate": het_marker.values,
            "Missing_rate": miss_marker.values,
        })

        if marker_info is not None:
            marker_stats = marker_stats.merge(marker_info, on="Marker",
                                                 how="left")

    n_show_markers = st.slider("Markers to show", 10,
                                 min(200, len(marker_stats)),
                                 min(30, len(marker_stats)),
                                 key="prev_marker_stats")

    st.dataframe(marker_stats.head(n_show_markers).style.format({
        "MAF": "{:.4f}",
        "Het_rate": "{:.4f}",
        "Missing_rate": "{:.4f}",
    }), use_container_width=True)

    # Quick MAF distribution
    fig_maf = px.histogram(marker_stats, x="MAF", nbins=40,
                             title="MAF distribution across markers")
    fig_maf.add_vline(x=0.05, line_dash="dash", line_color="red",
                        annotation_text="MAF = 0.05")
    fig_maf.update_layout(template="plotly_white", height=400)
    st.plotly_chart(fig_maf, use_container_width=True)

    # ─── Chromosome distribution (if marker info) ───
    if marker_info is not None and "Chrom" in marker_info.columns:
        st.markdown("### 🎯 Chromosome Distribution")

        chr_counts = marker_info["Chrom"].astype(str).value_counts().reset_index()
        chr_counts.columns = ["Chromosome", "N_markers"]

        # Sort chromosomes properly
        def _chr_key(x):
            x = str(x).replace("chr", "").replace("Chr", "")
            if x.isdigit():
                return (0, int(x))
            return (1, x)

        chr_counts["_sort"] = chr_counts["Chromosome"].apply(_chr_key)
        chr_counts = chr_counts.sort_values("_sort").drop(columns=["_sort"])

        fig_chr = px.bar(
            chr_counts, x="Chromosome", y="N_markers",
            title="Number of markers per chromosome",
            color="N_markers", color_continuous_scale="Viridis",
            text="N_markers",
        )
        fig_chr.update_traces(textposition="outside")
        fig_chr.update_layout(template="plotly_white", height=450,
                                showlegend=False)
        st.plotly_chart(fig_chr, use_container_width=True)

        # Marker info preview
        with st.expander("🔎 Marker Information Table"):
            st.dataframe(marker_info.head(50), use_container_width=True)

    # ─── Missing pattern heatmap (subsample) ───
    if geno.shape[0] > 5 and geno.shape[1] > 5:
        with st.expander("🎨 Missing Data Pattern (heatmap of a subsample)"):
            max_show = 100
            sub_s = min(max_show, geno.shape[0])
            sub_m = min(max_show, geno.shape[1])

            missing_mat = geno.iloc[:sub_s, :sub_m].isna().astype(int).values

            fig_miss_pat = px.imshow(
                missing_mat,
                labels=dict(x="Marker", y="Sample", color="Missing"),
                color_continuous_scale=["white", "red"],
                title=f"Missing pattern ({sub_s} samples × {sub_m} markers)",
                aspect="auto",
            )
            fig_miss_pat.update_layout(template="plotly_white", height=500,
                                         coloraxis_showscale=False)
            st.plotly_chart(fig_miss_pat, use_container_width=True)


st.markdown("---")

# ═══════════════════════════════════════════
# STEP 2: Upload Metadata
# ═══════════════════════════════════════════
st.subheader("Step 2: Upload Metadata (Optional)")
st.write(
    "CSV/Excel file with sample information. Recommended columns:\n"
    "- **Sample_ID** (must match genotype sample IDs)\n"
    "- **Species / Population / Group** (for grouping analyses)\n"
    "- **Origin / Country / Location** (for geographic analyses)\n"
    "- **Latitude / Longitude** (for map & IBD analyses)\n"
    "- **Environmental variables** (temperature, rainfall, etc. for IBE)\n"
    "- **Phenotypic traits** (for GWAS / feature selection)"
)

meta = load_metadata("upload_meta")

if meta is not None:

    # ─── Metadata Summary ───
    st.subheader("📋 Metadata Summary")

    ms1, ms2, ms3, ms4 = st.columns(4)
    ms1.metric("Rows", f"{meta.shape[0]:,}")
    ms2.metric("Columns", f"{meta.shape[1]:,}")
    ms3.metric("Missing cells",
                f"{meta.isna().sum().sum():,}")
    n_numeric = len(meta.select_dtypes(include=[np.number]).columns)
    ms4.metric("Numeric columns", n_numeric)

    # ─── Column analysis ───
    st.markdown("### 🔍 Column Analysis")

    col_analysis = []
    for col in meta.columns:
        col_data = meta[col]
        n_unique = col_data.nunique()
        n_missing = col_data.isna().sum()
        dtype = str(col_data.dtype)

        # Detect column purpose
        col_lower = col.lower()
        purpose = "General"
        if any(x in col_lower for x in ["id", "sample", "individual", "accession"]):
            purpose = "🆔 Sample ID"
        elif any(x in col_lower for x in ["species", "population", "pop", "group", "cluster"]):
            purpose = "🌿 Population"
        elif any(x in col_lower for x in ["lat", "latitude"]):
            purpose = "📍 Latitude"
        elif any(x in col_lower for x in ["lon", "long", "longitude"]):
            purpose = "📍 Longitude"
        elif any(x in col_lower for x in ["origin", "country", "location", "region"]):
            purpose = "🌍 Origin"
        elif any(x in col_lower for x in ["temp", "rain", "humid", "prec",
                                            "elev", "altitude", "climate"]):
            purpose = "🌡️ Environmental"
        elif any(x in col_lower for x in ["trait", "yield", "height", "weight",
                                            "phenotype"]):
            purpose = "📊 Phenotype"

        # Sample values
        sample_vals = col_data.dropna().unique()[:3]
        sample_str = ", ".join([str(v)[:20] for v in sample_vals])

        col_analysis.append({
            "Column": col,
            "Detected Purpose": purpose,
            "Type": dtype,
            "N unique": n_unique,
            "N missing": n_missing,
            "Sample values": sample_str,
        })

    col_analysis_df = pd.DataFrame(col_analysis)
    st.dataframe(col_analysis_df, use_container_width=True)

    # ─── Suggested column mapping ───
    id_cols = col_analysis_df[
        col_analysis_df["Detected Purpose"] == "🆔 Sample ID"]["Column"].tolist()
    pop_cols = col_analysis_df[
        col_analysis_df["Detected Purpose"] == "🌿 Population"]["Column"].tolist()

    if id_cols or pop_cols:
        st.markdown("### 💡 Suggested Column Mapping")
        if id_cols:
            st.info(f"**Sample ID column(s):** {', '.join(id_cols)}")
        if pop_cols:
            st.info(f"**Population column(s):** {', '.join(pop_cols)}")

    # ─── Metadata Preview ───
    st.markdown("### 👁️ Metadata Preview")

    mp1, mp2 = st.columns(2)
    with mp1:
        n_rows_meta = st.slider(
            "Rows to preview",
            5, min(500, meta.shape[0]),
            min(20, meta.shape[0]),
            key="prev_meta_rows",
        )
    with mp2:
        selected_cols = st.multiselect(
            "Columns to show (empty = all)",
            meta.columns.tolist(),
            default=meta.columns.tolist()[:min(10, meta.shape[1])],
            key="prev_meta_cols",
        )

    if selected_cols:
        preview_meta = meta[selected_cols].head(n_rows_meta)
    else:
        preview_meta = meta.head(n_rows_meta)

    st.dataframe(preview_meta, use_container_width=True)

    with st.expander("🔎 View full metadata"):
        show_full_meta = st.checkbox("Show full metadata table",
                                       value=False, key="show_full_meta")
        if show_full_meta:
            st.dataframe(meta, use_container_width=True)

    # ─── Categorical value distributions ───
    cat_cols = meta.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        st.markdown("### 🎨 Categorical Column Distributions")
        cat_col_show = st.selectbox(
            "Choose a categorical column to visualize",
            cat_cols, key="cat_col_show",
        )
        if cat_col_show:
            val_counts = meta[cat_col_show].value_counts().reset_index()
            val_counts.columns = [cat_col_show, "Count"]
            if len(val_counts) <= 50:
                fig_cat = px.bar(
                    val_counts, x=cat_col_show, y="Count",
                    color="Count", color_continuous_scale="Viridis",
                    title=f"Value distribution of '{cat_col_show}'",
                    text="Count",
                )
                fig_cat.update_traces(textposition="outside")
                fig_cat.update_layout(template="plotly_white", height=450,
                                        showlegend=False,
                                        xaxis_tickangle=45)
                st.plotly_chart(fig_cat, use_container_width=True)
            else:
                st.info(f"Column has {len(val_counts)} unique values "
                         "(too many to plot). Showing top 30:")
                st.dataframe(val_counts.head(30), use_container_width=True)

    # ─── Numeric column distributions ───
    num_cols = meta.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        st.markdown("### 📊 Numeric Column Distributions")
        num_col_show = st.selectbox(
            "Choose a numeric column to visualize",
            num_cols, key="num_col_show",
        )
        if num_col_show:
            fig_num = px.histogram(
                meta, x=num_col_show, nbins=40,
                title=f"Distribution of '{num_col_show}'",
                marginal="box",
            )
            fig_num.update_layout(template="plotly_white", height=450)
            st.plotly_chart(fig_num, use_container_width=True)

            # Summary stats
            desc = meta[num_col_show].describe()
            st.dataframe(desc.to_frame().T, use_container_width=True)

    # ─── Sample overlap validation ───
    if geno is not None:
        st.markdown("---")
        st.subheader("🔗 Sample Overlap Validation")

        st.info(
            "This section checks how many samples in your genotype file "
            "have matching entries in the metadata."
        )

        # Try each column as potential sample ID
        overlap_results = []
        geno_ids = set(geno.index.astype(str))

        for col in meta.columns:
            meta_ids = set(meta[col].astype(str))
            overlap = meta_ids & geno_ids
            overlap_results.append({
                "Metadata Column": col,
                "N unique in metadata": len(meta_ids),
                "N matching genotype IDs": len(overlap),
                "% match": len(overlap) / len(geno_ids) * 100,
            })

        overlap_df = pd.DataFrame(overlap_results).sort_values(
            "% match", ascending=False)

        st.dataframe(overlap_df.style.format({
            "% match": "{:.1f}%"
        }).background_gradient(subset=["% match"], cmap="Greens"),
                      use_container_width=True)

        # Best match column
        best_col = overlap_df.iloc[0]
        if best_col["% match"] >= 80:
            st.success(
                f"✅ **Best matching column: `{best_col['Metadata Column']}`** "
                f"({best_col['N matching genotype IDs']} / {len(geno_ids)} "
                f"samples matched, {best_col['% match']:.1f}%)"
            )
        elif best_col["% match"] >= 50:
            st.warning(
                f"⚠️ Best matching column `{best_col['Metadata Column']}` "
                f"only matches {best_col['% match']:.1f}% of samples. "
                "Check that sample IDs match exactly."
            )
        else:
            st.error(
                f"❌ Low overlap detected (best = {best_col['% match']:.1f}%). "
                "Sample IDs in metadata do NOT match genotype file. "
                "Please verify the sample identifiers."
            )

        # Show mismatched samples
        with st.expander("🔍 View unmatched sample IDs"):
            best_id_col = best_col["Metadata Column"]
            meta_ids_best = set(meta[best_id_col].astype(str))

            geno_only = geno_ids - meta_ids_best
            meta_only = meta_ids_best - geno_ids

            oc1, oc2 = st.columns(2)
            with oc1:
                st.markdown(
                    f"**In genotype but not in metadata** ({len(geno_only)}):"
                )
                if geno_only:
                    st.dataframe(
                        pd.DataFrame(sorted(geno_only)[:100],
                                       columns=["Sample_ID"]),
                        use_container_width=True,
                    )
                else:
                    st.write("✅ All genotype samples are in metadata!")

            with oc2:
                st.markdown(
                    f"**In metadata but not in genotype** ({len(meta_only)}):"
                )
                if meta_only:
                    st.dataframe(
                        pd.DataFrame(sorted(meta_only)[:100],
                                       columns=["Sample_ID"]),
                        use_container_width=True,
                    )
                else:
                    st.write("✅ All metadata samples are in genotype!")


# ═══════════════════════════════════════════
# STEP 3: Data Quality Dashboard
# ═══════════════════════════════════════════
if geno is not None:
    st.markdown("---")
    st.subheader("🎯 Data Quality Dashboard")

    quality_items = []

    # Sample size
    if geno.shape[0] >= 50:
        quality_items.append(("Sample size", "✅ OK",
                                f"{geno.shape[0]} samples"))
    elif geno.shape[0] >= 20:
        quality_items.append(("Sample size", "⚠️ Small",
                                f"{geno.shape[0]} samples - some analyses may be underpowered"))
    else:
        quality_items.append(("Sample size", "❌ Very small",
                                f"{geno.shape[0]} samples - many analyses may not work"))

    # Marker count
    if geno.shape[1] >= 500:
        quality_items.append(("Marker count", "✅ OK",
                                f"{geno.shape[1]} markers"))
    elif geno.shape[1] >= 100:
        quality_items.append(("Marker count", "⚠️ Small",
                                f"{geno.shape[1]} markers"))
    else:
        quality_items.append(("Marker count", "❌ Very small",
                                f"{geno.shape[1]} markers"))

    # Missing rate
    if missing_pct < 5:
        quality_items.append(("Missing data", "✅ Low",
                                f"{missing_pct:.2f}%"))
    elif missing_pct < 20:
        quality_items.append(("Missing data", "⚠️ Moderate",
                                f"{missing_pct:.2f}%"))
    else:
        quality_items.append(("Missing data", "❌ High",
                                f"{missing_pct:.2f}% - filtering strongly recommended"))

    # Low MAF markers
    low_maf = int((maf < 0.05).sum())
    if low_maf / len(maf) < 0.1:
        quality_items.append(("Low-MAF markers", "✅ Few",
                                f"{low_maf} ({low_maf/len(maf)*100:.1f}%)"))
    elif low_maf / len(maf) < 0.3:
        quality_items.append(("Low-MAF markers", "⚠️ Moderate",
                                f"{low_maf} ({low_maf/len(maf)*100:.1f}%)"))
    else:
        quality_items.append(("Low-MAF markers", "❌ Many",
                                f"{low_maf} ({low_maf/len(maf)*100:.1f}%) - consider MAF filter"))

    # Marker info
    if marker_info is not None and "Chrom" in marker_info.columns:
        quality_items.append(("Marker positions", "✅ Available",
                                f"{marker_info['Chrom'].nunique()} chromosomes"))
    else:
        quality_items.append(("Marker positions", "⚠️ Missing",
                                "Some analyses (LD, Manhattan plots) will be limited"))

    # Metadata
    if meta is not None:
        quality_items.append(("Metadata", "✅ Loaded",
                                f"{meta.shape[0]} rows × {meta.shape[1]} cols"))
    else:
        quality_items.append(("Metadata", "⚠️ Not loaded",
                                "Population-based analyses will not work"))

    quality_df = pd.DataFrame(quality_items,
                                columns=["Item", "Status", "Details"])
    st.dataframe(quality_df, use_container_width=True)

    # Recommendations
    st.markdown("### 💡 Recommendations")
    recs = []

    if missing_pct > 10:
        recs.append("• Run **Quality Control** to filter markers with high missing rate")
    if low_maf / len(maf) > 0.2:
        recs.append("• Apply MAF > 0.05 filter to remove rare variants")
    if geno.shape[0] < 50:
        recs.append("• Consider that some analyses (STRUCTURE, phylogenetics) work best with N > 50")
    if meta is None:
        recs.append("• Upload metadata for population-based analyses")
    if marker_info is None or "Chrom" not in (marker_info.columns if marker_info is not None else []):
        recs.append("• Use VCF or HapMap format to enable chromosome-based analyses")

    if not recs:
        st.success("✅ Your data looks great! You're ready to explore all analyses.")
    else:
        for rec in recs:
            st.write(rec)

    # Next steps
    st.markdown("### 🚀 Next Steps")
    st.markdown("""
    Now that your data is loaded, you can navigate to any analysis module:

    1. **🧹 Quality Control** — Filter markers and samples
    2. **🧬 SNP Statistics** — Allele frequencies, PIC, diversity
    3. **🌿 Genetic Diversity** — Ho, He, Fis, per-population stats
    4. **🧩 Population Structure** — PCA, STRUCTURE, fastStructure
    5. **🌳 Phylogenetics** — NJ, UPGMA, ML trees with bootstrap
    6. **🎯 Clustering** — Hierarchical, K-means, DBSCAN
    7. **👥 Kinship** — VanRaden, IBS matrices
    8. **🔗 LD Analysis** — Decay, blocks, Hill-Weir fit
    9. **🌍 Geographic Genetics** — Fst, DEST, Gst, Nm, IBD
    10. **🤖 Machine Learning** — Classification, selection detection
    11. **📑 Reports** — Comprehensive summary
    12. **💾 Export** — Download processed data
    """)