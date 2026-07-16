"""Export Results — Download processed data and analysis outputs."""

import io
import zipfile
import numpy as np
import pandas as pd
import streamlit as st

from utils import (
    get_geno_from_session, get_meta_from_session,
    calc_allele_freq, calc_maf, calc_het_obs, calc_het_exp,
    calc_pic, calc_fis, calc_missing_rate,
    download_dataframe,
)

st.title("💾 Export Results")
st.markdown("---")

geno, marker_info = get_geno_from_session()
if geno is None:
    st.stop()

meta = get_meta_from_session()

st.subheader("Available Exports")

st.markdown("### 🧬 Genotype Data")
c1, c2, c3 = st.columns(3)

with c1:
    st.download_button(
        "📥 Genotype Matrix (CSV)",
        geno.to_csv(),
        "genotype_matrix.csv", "text/csv",
        key="dl_geno_csv",
    )

with c2:
    if marker_info is not None:
        st.download_button(
            "📥 Marker Information (CSV)",
            marker_info.to_csv(index=False),
            "marker_info.csv", "text/csv",
            key="dl_mi_csv",
        )
    else:
        st.info("Marker info not available.")

with c3:
    if meta is not None:
        st.download_button(
            "📥 Metadata (CSV)",
            meta.to_csv(index=False),
            "metadata.csv", "text/csv",
            key="dl_meta_csv",
        )
    else:
        st.info("Metadata not loaded.")

# ── Convert to HapMap ──
st.markdown("### 🔄 Convert Between Formats")

fmt_out = st.selectbox("Export genotype in format",
                        ["Numeric CSV", "HapMap (simplified)",
                         "VCF (simplified)", "PLINK PED (simplified)"],
                        key="ex_fmt")

if st.button("🚀 Generate export", key="ex_gen"):
    if fmt_out == "Numeric CSV":
        content = geno.to_csv()
        mime = "text/csv"
        name = "genotype_numeric.csv"

    elif fmt_out == "HapMap (simplified)":
        # Build minimal HapMap
        n_markers = geno.shape[1]
        hmp = pd.DataFrame({
            "rs#": geno.columns,
            "alleles": "A/T",
            "chrom": marker_info["Chrom"].values if marker_info is not None else "NA",
            "pos": marker_info["Pos"].values if marker_info is not None else np.arange(n_markers),
            "strand": "+", "assembly#": "NA", "center": "NA",
            "protLSID": "NA", "assayLSID": "NA",
            "panelLSID": "NA", "QCcode": "NA",
        })
        # Convert numeric to A/T calls (simplified: 0=AA, 1=AT, 2=TT)
        def _num_to_call(v):
            if pd.isna(v): return "NN"
            if v == 0: return "AA"
            if v == 1: return "AT"
            if v == 2: return "TT"
            return "NN"

        calls = geno.T.applymap(_num_to_call)
        calls.columns = geno.index
        calls.insert(0, "rs#", geno.columns)
        full = pd.merge(hmp, calls, on="rs#")
        content = full.to_csv(sep="\t", index=False)
        mime = "text/plain"
        name = "genotype.hmp.txt"

    elif fmt_out == "VCF (simplified)":
        lines = ["##fileformat=VCFv4.2",
                 "##source=Interactive_Population_Genomics_Platform",
                 '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">']
        header = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(geno.index.astype(str))
        lines.append(header)

        chr_arr = marker_info["Chrom"].values if marker_info is not None else ["NA"]*geno.shape[1]
        pos_arr = marker_info["Pos"].values if marker_info is not None else np.arange(geno.shape[1])

        def _num_to_gt(v):
            if pd.isna(v): return "./."
            if v == 0: return "0/0"
            if v == 1: return "0/1"
            if v == 2: return "1/1"
            return "./."

        for j, snp in enumerate(geno.columns):
            row = [str(chr_arr[j]), str(int(pos_arr[j])), str(snp),
                   "A", "T", ".", "PASS", ".", "GT"]
            for s in geno.index:
                row.append(_num_to_gt(geno.loc[s, snp]))
            lines.append("\t".join(row))
        content = "\n".join(lines)
        mime = "text/plain"
        name = "genotype.vcf"

    else:  # PLINK PED
        # PED format: FID IID PID MID SEX PHENO geno1_A1 geno1_A2 ...
        def _num_to_alleles(v):
            if pd.isna(v): return ("0", "0")
            if v == 0: return ("A", "A")
            if v == 1: return ("A", "T")
            if v == 2: return ("T", "T")
            return ("0", "0")

        ped_lines = []
        for s in geno.index:
            row = [str(s), str(s), "0", "0", "0", "-9"]
            for snp in geno.columns:
                a1, a2 = _num_to_alleles(geno.loc[s, snp])
                row.extend([a1, a2])
            ped_lines.append("\t".join(row))
        content = "\n".join(ped_lines)
        mime = "text/plain"
        name = "genotype.ped"

    st.download_button(
        f"📥 Download {fmt_out}",
        content, name, mime,
        key="ex_dl_button",
    )
    st.success(f"✅ Export ready: {name}")

# ── Bundle everything ──
st.markdown("### 📦 Download Full Analysis Bundle (ZIP)")

if st.button("🚀 Build ZIP bundle", key="ex_zip"):
    with st.spinner("Building bundle..."):
        buf = io.BytesIO()

        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("genotype_matrix.csv", geno.to_csv())
            if marker_info is not None:
                zf.writestr("marker_info.csv",
                             marker_info.to_csv(index=False))
            if meta is not None:
                zf.writestr("metadata.csv", meta.to_csv(index=False))

            # Statistics
            maf = calc_maf(geno)
            ho = calc_het_obs(geno)
            he = calc_het_exp(geno)
            pic = calc_pic(geno)
            fis = calc_fis(geno)

            stats = pd.DataFrame({
                "Marker": geno.columns,
                "MAF": maf.values,
                "Ho": ho.values,
                "He": he.values,
                "PIC": pic.values,
                "Fis": fis.values,
                "MissingRate": calc_missing_rate(geno, axis=0).values,
            })
            zf.writestr("snp_statistics.csv",
                         stats.to_csv(index=False))

            # README
            readme = f"""# Population Genomics Export Bundle

Generated by Interactive Population Genomics Platform.

## Files included
- `genotype_matrix.csv`: Full 0/1/2 genotype matrix
  ({geno.shape[0]} samples × {geno.shape[1]} markers)
- `snp_statistics.csv`: Per-SNP MAF, Ho, He, PIC, Fis, missing rate
- `marker_info.csv`: Marker chromosome/position (if available)
- `metadata.csv`: Sample metadata (if available)

## Genome-wide summary
- Mean MAF: {maf.mean():.4f}
- Mean He: {he.mean():.4f}
- Mean PIC: {pic.mean():.4f}
"""
            zf.writestr("README.md", readme)

        buf.seek(0)

    st.download_button(
        "📥 Download bundle.zip",
        buf, "population_genomics_bundle.zip", "application/zip",
        key="ex_zip_dl",
    )
    st.success("✅ Bundle ready!")

# ── Session state summary ──
st.markdown("---")
st.subheader("Current Session State")

session_info = {
    "Genotype loaded": "✅ Yes" if geno is not None else "❌ No",
    "Marker info loaded": "✅ Yes" if marker_info is not None else "❌ No",
    "Metadata loaded": "✅ Yes" if meta is not None else "❌ No",
    "Samples": geno.shape[0] if geno is not None else 0,
    "Markers": geno.shape[1] if geno is not None else 0,
}
st.table(pd.DataFrame(list(session_info.items()),
                      columns=["Item", "Status"]))