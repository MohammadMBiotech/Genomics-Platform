"""Machine Learning — Classification, Clustering, Feature Selection, Selection Detection."""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Lasso, LassoCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    silhouette_score,
)

from utils import (
    get_geno_from_session, get_meta_from_session,
    impute_missing, calc_maf,
    download_plotly_html, download_dataframe,
)

st.title("🤖 Machine Learning")
st.markdown("---")

geno, marker_info = get_geno_from_session()
if geno is None:
    st.stop()

meta = get_meta_from_session()

tab_clf, tab_fs, tab_sel, tab_um = st.tabs([
    "🏷️ Classification",
    "🎯 Feature Selection",
    "🎯 Selection Detection",
    "🗺️ Unsupervised (UMAP)",
])

# =========================================================
# TAB 1 — Classification (Species/Population Prediction)
# =========================================================
with tab_clf:
    st.subheader("Classification: Predict Population/Species from SNPs")

    if meta is None:
        st.warning("Metadata required for classification.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            target_col = st.selectbox("Target column (label)",
                                        meta.columns.tolist(),
                                        key="ml_clf_target")
        with c2:
            sam_col = st.selectbox("Sample ID column",
                                     meta.columns.tolist(),
                                     key="ml_clf_sam")

        method = st.selectbox(
            "Classifier",
            ["Random Forest", "Logistic Regression", "SVM",
             "XGBoost", "Gradient Boosting"],
            key="ml_clf_method",
        )

        # Hyperparameters
        rf_n, rf_d = 100, None
        if method == "Random Forest":
            rf_n = st.slider("N estimators", 10, 500, 100, 10,
                             key="ml_rf_n")
            rf_d = st.selectbox("Max depth", [None, 5, 10, 15, 20],
                                key="ml_rf_d")

        lr_C = 1.0
        if method == "Logistic Regression":
            lr_C = st.slider("Regularization C", 0.01, 10.0, 1.0, 0.01,
                             key="ml_lr_c")

        svm_C, svm_k = 1.0, "rbf"
        if method == "SVM":
            svm_C = st.slider("C", 0.01, 10.0, 1.0, 0.01, key="ml_svm_c")
            svm_k = st.selectbox("Kernel",
                                 ["rbf", "linear", "poly"], key="ml_svm_k")

        xgb_lr, xgb_n, xgb_d = 0.1, 100, 6
        if method == "XGBoost":
            xgb_lr = st.slider("Learning rate", 0.01, 0.5, 0.1, 0.01,
                               key="ml_xgb_lr")
            xgb_n = st.slider("N estimators", 10, 500, 100, 10,
                              key="ml_xgb_n")
            xgb_d = st.slider("Max depth", 2, 15, 6, key="ml_xgb_d")

        # PCA pre-reduction
        use_pca_clf = st.checkbox("PCA pre-reduction (recommended)",
                                    True, key="ml_use_pca")
        n_pcs_clf = st.slider("N PCs", 2, 50, 20, key="ml_npcs") if use_pca_clf else None

        # Split
        c3, c4, c5 = st.columns(3)
        with c3:
            test_sz = st.slider("Test %", 10, 50, 20, 5,
                                key="ml_ts") / 100
        with c4:
            rs = int(st.number_input("Random state", value=42,
                                      key="ml_rs"))
        with c5:
            cv_k = st.slider("CV folds", 2, 10, 5, key="ml_cv")

        if st.button("🚀 Train & Evaluate Classifier",
                     use_container_width=True, key="ml_clf_run"):
            # Prepare data
            pop_map = dict(zip(meta[sam_col].astype(str),
                                meta[target_col].astype(str)))
            samples_use = [s for s in geno.index.astype(str)
                            if s in pop_map and pd.notna(pop_map.get(s))]

            X_raw = geno.loc[samples_use]
            y_raw = np.array([pop_map[s] for s in samples_use])

            X_imp = impute_missing(X_raw, "mean").values
            X_scaled = StandardScaler().fit_transform(X_imp)

            if use_pca_clf:
                pca_pre = PCA(n_components=min(n_pcs_clf, X_scaled.shape[1]))
                X_use = pca_pre.fit_transform(X_scaled)
            else:
                X_use = X_scaled

            le = LabelEncoder()
            y = le.fit_transform(y_raw)
            cnames = le.classes_.astype(str)
            nc = len(cnames)

            X_tr, X_te, y_tr, y_te = train_test_split(
                X_use, y, test_size=test_sz, random_state=rs, stratify=y,
            )

            # Build model
            if method == "Random Forest":
                model = RandomForestClassifier(
                    n_estimators=rf_n, max_depth=rf_d,
                    random_state=rs, n_jobs=-1)
            elif method == "Logistic Regression":
                model = LogisticRegression(
                    C=lr_C, max_iter=2000, random_state=rs)
            elif method == "SVM":
                model = SVC(C=svm_C, kernel=svm_k,
                            probability=True, random_state=rs)
            elif method == "Gradient Boosting":
                model = GradientBoostingClassifier(random_state=rs)
            elif method == "XGBoost":
                try:
                    from xgboost import XGBClassifier
                    model = XGBClassifier(
                        learning_rate=xgb_lr, n_estimators=xgb_n,
                        max_depth=xgb_d, random_state=rs, n_jobs=-1,
                        verbosity=0, use_label_encoder=False,
                        eval_metric="mlogloss" if nc > 2 else "logloss")
                except ImportError:
                    st.error("XGBoost not installed.")
                    st.stop()

            with st.spinner(f"Training {method}..."):
                model.fit(X_tr, y_tr)
                y_pr = model.predict(X_te)
                try:
                    y_pp = model.predict_proba(X_te)
                except Exception:
                    y_pp = None

                avg = "weighted" if nc > 2 else "binary"
                acc = accuracy_score(y_te, y_pr)
                prec = precision_score(y_te, y_pr, average=avg,
                                        zero_division=0)
                rec = recall_score(y_te, y_pr, average=avg,
                                    zero_division=0)
                f1 = f1_score(y_te, y_pr, average=avg,
                               zero_division=0)

                skf = StratifiedKFold(n_splits=cv_k, shuffle=True,
                                        random_state=rs)
                cv_acc = cross_val_score(model, X_use, y, cv=skf,
                                            scoring="accuracy")

            st.success(f"✅ {method} trained.")

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Accuracy", f"{acc:.4f}")
            m2.metric("Precision", f"{prec:.4f}")
            m3.metric("Recall", f"{rec:.4f}")
            m4.metric("F1", f"{f1:.4f}")
            m5.metric(f"CV Acc ({cv_k}-fold)",
                        f"{cv_acc.mean():.4f}±{cv_acc.std():.4f}")

            # Classification report
            st.subheader("Detailed classification report")
            rpt = classification_report(y_te, y_pr, target_names=cnames,
                                          output_dict=True)
            st.dataframe(pd.DataFrame(rpt).T.style.format("{:.4f}"),
                          use_container_width=True)

            # Confusion matrix
            cm = confusion_matrix(y_te, y_pr)
            fig_cm = px.imshow(cm, x=cnames, y=cnames, text_auto=True,
                                color_continuous_scale="Blues",
                                title="Confusion Matrix",
                                labels=dict(x="Predicted", y="Actual"))
            fig_cm.update_layout(template="plotly_white", height=500)
            st.plotly_chart(fig_cm, use_container_width=True)

            # ROC
            if y_pp is not None:
                st.subheader("ROC Curve")
                fig_roc = go.Figure()
                if nc == 2:
                    fpr, tpr, _ = roc_curve(y_te, y_pp[:, 1])
                    fig_roc.add_trace(go.Scatter(
                        x=fpr, y=tpr, mode="lines",
                        name=f"AUC={auc(fpr, tpr):.4f}",
                        line=dict(color="steelblue", width=2)))
                else:
                    for i, cn in enumerate(cnames):
                        yb = (y_te == i).astype(int)
                        if yb.sum() == 0:
                            continue
                        fpr, tpr, _ = roc_curve(yb, y_pp[:, i])
                        fig_roc.add_trace(go.Scatter(
                            x=fpr, y=tpr, mode="lines",
                            name=f"{cn} AUC={auc(fpr, tpr):.4f}"))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode="lines",
                    line=dict(dash="dash", color="gray"),
                    name="Random"))
                fig_roc.update_layout(
                    xaxis_title="FPR", yaxis_title="TPR",
                    template="plotly_white", height=500,
                    title="ROC Curves")
                st.plotly_chart(fig_roc, use_container_width=True)

            # Feature importance
            if method in ["Random Forest", "XGBoost",
                            "Gradient Boosting"]:
                st.subheader("Feature Importance (top 30)")
                if use_pca_clf:
                    feat_names = [f"PC{i+1}"
                                    for i in range(X_use.shape[1])]
                else:
                    feat_names = geno.columns.astype(str).tolist()

                imp = pd.DataFrame({
                    "Feature": feat_names,
                    "Importance": model.feature_importances_,
                }).sort_values("Importance", ascending=False).head(30)

                fig_imp = px.bar(imp, x="Importance", y="Feature",
                                  orientation="h",
                                  color="Importance",
                                  color_continuous_scale="viridis",
                                  title="Top 30 features")
                fig_imp.update_layout(template="plotly_white", height=650,
                                         yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_imp, use_container_width=True)

                download_dataframe(imp, "feature_importance.csv",
                                    key="dl_ml_imp")

# =========================================================
# TAB 2 — Feature Selection (Lasso for GWAS)
# =========================================================
with tab_fs:
    st.subheader("Feature Selection with Lasso (L1 penalty)")
    st.write(
        "Identifies most informative SNPs for predicting a "
        "phenotype/trait using Lasso regularization."
    )

    if meta is None:
        st.warning("Metadata required.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            target_fs = st.selectbox(
                "Target trait (numeric)",
                meta.select_dtypes(include=[np.number]).columns.tolist(),
                key="fs_target",
            )
        with c2:
            sam_fs = st.selectbox("Sample ID column",
                                    meta.columns.tolist(),
                                    key="fs_sam")

        alpha_choice = st.selectbox(
            "Alpha selection",
            ["Auto (LassoCV)", "Manual"],
            key="fs_alpha_ch",
        )
        if alpha_choice == "Manual":
            alpha_fs = st.slider("Alpha", 0.001, 5.0, 0.1, 0.001,
                                    key="fs_alpha")

        if st.button("🚀 Run Lasso feature selection",
                     use_container_width=True, key="fs_run"):
            trait_map = dict(zip(meta[sam_fs].astype(str),
                                   meta[target_fs].astype(float)))
            samples_use = [s for s in geno.index.astype(str)
                            if s in trait_map and pd.notna(trait_map.get(s))]

            X_raw = geno.loc[samples_use]
            y = np.array([trait_map[s] for s in samples_use])

            X_imp = impute_missing(X_raw, "mean").values
            X_scaled = StandardScaler().fit_transform(X_imp)

            with st.spinner("Fitting Lasso..."):
                if alpha_choice == "Auto (LassoCV)":
                    model_fs = LassoCV(cv=5, random_state=42,
                                        max_iter=10000, n_jobs=-1)
                else:
                    model_fs = Lasso(alpha=alpha_fs, max_iter=10000)
                model_fs.fit(X_scaled, y)

            coefs = model_fs.coef_
            selected_mask = np.abs(coefs) > 1e-8
            selected_snps = geno.columns[selected_mask]

            st.success(f"✅ Selected **{selected_mask.sum()}** SNPs "
                        f"out of {len(coefs)}.")

            if alpha_choice == "Auto (LassoCV)":
                st.info(f"Optimal α = {model_fs.alpha_:.6f}")

            # Coefficient distribution
            imp_df = pd.DataFrame({
                "SNP": geno.columns.astype(str),
                "Coefficient": coefs,
                "Abs_Coef": np.abs(coefs),
            }).sort_values("Abs_Coef", ascending=False)

            # Top 50
            top_coef = imp_df.head(50)

            fig_c = px.bar(top_coef, x="Coefficient", y="SNP",
                            orientation="h", color="Coefficient",
                            color_continuous_scale="RdBu_r",
                            title="Top 50 selected SNPs by Lasso")
            fig_c.update_layout(template="plotly_white", height=800,
                                 yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_c, use_container_width=True)

            st.dataframe(imp_df.head(200), use_container_width=True)
            download_dataframe(imp_df, "lasso_selected_snps.csv",
                                key="dl_fs")

# =========================================================
# TAB 3 — Selection detection (outlier SNPs)
# =========================================================
with tab_sel:
    st.subheader("Selection Detection")
    st.write(
        "Identifies SNPs under putative selection using Fst-outlier and "
        "PCA-based outlier detection approaches (PCAdapt-like)."
    )

    method_sel = st.selectbox(
        "Detection method",
        ["Fst outlier (population-based)",
         "PCAdapt (PC-based)",
         "Combined ROH-like windows"],
        key="sel_method",
    )

    if method_sel == "Fst outlier (population-based)":
        if meta is None:
            st.warning("Metadata required.")
        else:
            pop_sel = st.selectbox("Population column",
                                    meta.columns.tolist(),
                                    key="sel_pop")
            sam_sel = st.selectbox("Sample ID column",
                                    meta.columns.tolist(),
                                    key="sel_sam")

            fst_pct = st.slider("Top % as outliers", 1, 20, 5,
                                 key="sel_fst_pct")

            if st.button("🚀 Detect Fst outliers", key="sel_fst_run"):
                pop_map_s = dict(zip(meta[sam_sel].astype(str),
                                       meta[pop_sel].astype(str)))

                # Per-marker Fst (Weir & Cockerham approximation)
                pops_here = sorted(set(pop_map_s.values()))

                with st.spinner("Computing per-marker Fst..."):
                    # Per-marker global Fst = (Ht - Hs)/Ht
                    from utils import calc_het_exp

                    Ht_marker = calc_het_exp(geno)

                    # Weighted Hs
                    hs_list = []
                    weights = []
                    for p in pops_here:
                        samples_p = [s for s in geno.index
                                       if pop_map_s.get(str(s)) == p]
                        if len(samples_p) < 2:
                            continue
                        Hs_p = calc_het_exp(geno.loc[samples_p])
                        hs_list.append(Hs_p * len(samples_p))
                        weights.append(len(samples_p))
                    if hs_list:
                        Hs_marker = pd.concat(hs_list, axis=1).sum(axis=1) / sum(weights)
                        Fst_marker = (Ht_marker - Hs_marker) / Ht_marker.replace(0, np.nan)
                    else:
                        st.error("Not enough populations.")
                        st.stop()

                fst_df = pd.DataFrame({
                    "Marker": geno.columns,
                    "Fst": Fst_marker.values,
                })

                # Add chromosome/position if available
                if marker_info is not None:
                    fst_df = fst_df.merge(marker_info, on="Marker",
                                            how="left")

                fst_df = fst_df.dropna(subset=["Fst"])
                thresh = np.percentile(fst_df["Fst"], 100 - fst_pct)
                fst_df["Outlier"] = fst_df["Fst"] >= thresh

                n_outl = int(fst_df["Outlier"].sum())
                st.success(f"✅ Detected **{n_outl}** outlier SNPs "
                            f"(Fst ≥ {thresh:.4f}).")

                # Manhattan-style plot
                if "Chrom" in fst_df.columns:
                    fst_df_plot = fst_df.copy()
                    fst_df_plot["Chrom"] = fst_df_plot["Chrom"].astype(str)
                    chr_order = sorted(fst_df_plot["Chrom"].unique())
                    fst_df_plot["Chrom"] = pd.Categorical(
                        fst_df_plot["Chrom"], categories=chr_order,
                        ordered=True)
                    fst_df_plot = fst_df_plot.sort_values(["Chrom", "Pos"])
                    fst_df_plot["Order"] = np.arange(len(fst_df_plot))

                    fig_man = px.scatter(
                        fst_df_plot, x="Order", y="Fst", color="Chrom",
                        title="Per-marker Fst (Manhattan-style)",
                        hover_data=["Marker", "Pos"],
                    )
                    fig_man.add_hline(y=thresh, line_dash="dash",
                                       line_color="red",
                                       annotation_text=f"Outlier threshold ({fst_pct}%)")
                    fig_man.update_layout(template="plotly_white",
                                            height=550, showlegend=False)
                    st.plotly_chart(fig_man, use_container_width=True)

                # Distribution
                fig_fst_dist = px.histogram(
                    fst_df, x="Fst", nbins=60,
                    title="Fst distribution across markers")
                fig_fst_dist.add_vline(x=thresh, line_dash="dash",
                                         line_color="red")
                fig_fst_dist.update_layout(template="plotly_white",
                                              height=400)
                st.plotly_chart(fig_fst_dist, use_container_width=True)

                # Top outliers
                st.subheader("Top outlier SNPs")
                st.dataframe(fst_df[fst_df["Outlier"]].sort_values(
                    "Fst", ascending=False).head(50),
                    use_container_width=True)
                download_dataframe(fst_df, "fst_outliers.csv",
                                    key="dl_sel_fst")

    elif method_sel == "PCAdapt (PC-based)":
        st.write(
            "PCAdapt-style method: identifies SNPs strongly correlated "
            "with the first K principal components."
        )
        k_pcs = st.slider("Number of PCs to consider", 2, 20, 5,
                           key="sel_pca_k")
        pct_out = st.slider("Top % as outliers", 1, 20, 5,
                              key="sel_pca_pct")

        if st.button("🚀 Run PCAdapt-like", key="sel_pca_run"):
            with st.spinner("Running PCA + Mahalanobis..."):
                imp = impute_missing(geno, "mean").values
                X_sc = StandardScaler().fit_transform(imp)
                pca_a = PCA(n_components=k_pcs)
                pca_a.fit(X_sc)

                # Loadings (n_pcs x n_markers)
                loadings = pca_a.components_  # shape (k, m)

                # Mahalanobis-like: sum of squared standardized loadings
                loadings_z = (loadings - loadings.mean(axis=1, keepdims=True)) / \
                              loadings.std(axis=1, keepdims=True)
                stat = (loadings_z ** 2).sum(axis=0)

                # Chi-square-like p-value (df=k)
                from scipy.stats import chi2
                pvals = 1 - chi2.cdf(stat, df=k_pcs)

            adapt_df = pd.DataFrame({
                "Marker": geno.columns,
                "Statistic": stat,
                "P_value": pvals,
                "NegLog10P": -np.log10(np.clip(pvals, 1e-300, 1)),
            })

            if marker_info is not None:
                adapt_df = adapt_df.merge(marker_info,
                                            on="Marker", how="left")

            thr = np.percentile(adapt_df["Statistic"], 100 - pct_out)
            adapt_df["Outlier"] = adapt_df["Statistic"] >= thr

            n_out = int(adapt_df["Outlier"].sum())
            st.success(f"✅ Detected **{n_out}** outlier SNPs.")

            fig_pca_out = px.histogram(
                adapt_df, x="Statistic", nbins=60,
                title="PCAdapt statistic distribution")
            fig_pca_out.add_vline(x=thr, line_dash="dash",
                                    line_color="red")
            fig_pca_out.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig_pca_out, use_container_width=True)

            st.dataframe(adapt_df[adapt_df["Outlier"]].sort_values(
                "Statistic", ascending=False).head(50),
                use_container_width=True)
            download_dataframe(adapt_df, "pcadapt_outliers.csv",
                                key="dl_sel_pca")

    else:  # ROH-like
        st.write(
            "Simple ROH (Runs of Homozygosity)-like scan: identifies "
            "windows of low heterozygosity per sample."
        )
        win_size = st.slider("Window size (markers)", 10, 500, 50,
                              key="sel_roh_ws")
        min_het = st.slider("Max heterozygosity in window", 0.0, 0.5, 0.05,
                              0.01, key="sel_roh_h")

        if st.button("🚀 Scan ROH-like windows", key="sel_roh_run"):
            with st.spinner("Scanning windows..."):
                G = geno.fillna(1).values  # treat missing as het
                n_samples, n_markers = G.shape
                results = []
                for si in range(n_samples):
                    for start in range(0, n_markers - win_size, win_size):
                        window = G[si, start:start + win_size]
                        het_prop = np.mean(window == 1)
                        if het_prop <= min_het:
                            results.append({
                                "Sample": geno.index[si],
                                "Start_marker": start,
                                "End_marker": start + win_size,
                                "Het_proportion": het_prop,
                            })

            roh_df = pd.DataFrame(results)
            if len(roh_df) == 0:
                st.warning("No ROH-like windows found.")
            else:
                st.success(f"✅ Found **{len(roh_df)}** ROH-like windows.")

                # Windows per sample
                per_sample = roh_df.groupby("Sample").size().reset_index(
                    name="N_ROH_windows")
                fig_roh = px.bar(per_sample.head(50),
                                  x="Sample", y="N_ROH_windows",
                                  title="Number of ROH-like windows per sample "
                                          "(top 50)")
                fig_roh.update_layout(template="plotly_white", height=450,
                                        xaxis_tickangle=90)
                st.plotly_chart(fig_roh, use_container_width=True)

                st.dataframe(roh_df.head(200), use_container_width=True)
                download_dataframe(roh_df, "roh_windows.csv",
                                    key="dl_sel_roh")

# =========================================================
# TAB 4 — Unsupervised (UMAP)
# =========================================================
with tab_um:
    st.subheader("UMAP Non-linear Dimensionality Reduction")
    st.write(
        "Uses UMAP to visualize sample structure in a low-dimensional "
        "embedding — often reveals sub-structure invisible to PCA."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        nn = st.slider("N neighbors", 2, 100, 15, key="um_nn")
    with c2:
        md = st.slider("Min distance", 0.0, 1.0, 0.1, 0.01, key="um_md")
    with c3:
        nc_um = st.radio("Components", [2, 3], key="um_nc")

    color_col_u = None
    if meta is not None:
        color_col_u = st.selectbox("Color by (metadata)",
                                     ["None"] + meta.columns.tolist(),
                                     key="um_color")
        sam_col_u = st.selectbox("Sample ID column",
                                  meta.columns.tolist(),
                                  key="um_sam")

    if st.button("🚀 Run UMAP", use_container_width=True, key="um_run"):
        try:
            from umap import UMAP
        except ImportError:
            st.error("UMAP not installed. Add `umap-learn` to requirements.")
            st.stop()

        imp = impute_missing(geno, "mean").values
        X_sc = StandardScaler().fit_transform(imp)

        with st.spinner("Running UMAP..."):
            reducer = UMAP(n_neighbors=nn, min_dist=md,
                            n_components=nc_um, random_state=42)
            emb = reducer.fit_transform(X_sc)

        um_df = pd.DataFrame(
            emb, columns=[f"UMAP_{i+1}" for i in range(nc_um)])
        um_df["Sample"] = geno.index.astype(str)

        if color_col_u and color_col_u != "None":
            m_sub = meta[[sam_col_u, color_col_u]].drop_duplicates()
            m_sub[sam_col_u] = m_sub[sam_col_u].astype(str)
            um_df = um_df.merge(m_sub, left_on="Sample",
                                  right_on=sam_col_u, how="left")

        color_arg = color_col_u if (color_col_u and color_col_u != "None") else None

        if nc_um == 2:
            fig_um = px.scatter(um_df, x="UMAP_1", y="UMAP_2",
                                 color=color_arg, hover_data=["Sample"],
                                 title="UMAP 2D projection")
        else:
            fig_um = px.scatter_3d(um_df, x="UMAP_1", y="UMAP_2", z="UMAP_3",
                                    color=color_arg, hover_data=["Sample"],
                                    title="UMAP 3D projection")

        fig_um.update_layout(template="plotly_white", height=700)
        st.plotly_chart(fig_um, use_container_width=True)

        download_plotly_html(fig_um, "umap_projection.html",
                              key="dl_um_html")
        download_dataframe(um_df, "umap_embedding.csv",
                            key="dl_um_csv")