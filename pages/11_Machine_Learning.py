"""
Machine Learning & Selection Detection — Publication Quality
────────────────────────────────────────────────────────────
Features:
  - Classification (RF, LR, SVM, XGBoost, GB) — full metrics + CV
  - Regression (Linear, Ridge, Lasso, ElasticNet, RF, XGBoost, SVR) — full metrics + CV
  - Feature Selection (Lasso, Ridge)
  - Selection Detection (9 methods — many work without metadata)
  - Unsupervised UMAP (no metadata required)
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import (
    train_test_split, cross_val_score, cross_validate,
    StratifiedKFold, KFold, learning_curve,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import (
    LogisticRegression, Lasso, LassoCV, LinearRegression,
    Ridge, RidgeCV, ElasticNet, ElasticNetCV,
)
from sklearn.svm import SVC, SVR
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, GradientBoostingRegressor,
)
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    # Classification
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    matthews_corrcoef, balanced_accuracy_score, cohen_kappa_score,
    # Regression
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score,
    # Clustering
    silhouette_score,
)
from scipy.stats import chi2, f as f_dist, probplot

from utils import (
    get_geno_from_session, get_meta_from_session,
    impute_missing, calc_maf, calc_het_exp, calc_allele_freq,
    download_plotly_html, download_dataframe,
)

st.title("🤖 Machine Learning & Selection Detection")
st.markdown("---")

geno, marker_info = get_geno_from_session()
if geno is None:
    st.stop()

meta = get_meta_from_session()


# ═══════════════════════════════════════════
# GLOBAL: Metadata column selector (optional)
# ═══════════════════════════════════════════
global_sam_col = None
global_pop_col = None

if meta is not None:
    st.subheader("🔧 Metadata Configuration (optional)")
    gc1, gc2 = st.columns(2)
    with gc1:
        global_sam_col = st.selectbox(
            "Sample ID column",
            meta.columns.tolist(),
            key="ml_samcol_g",
        )
    with gc2:
        global_pop_col = st.selectbox(
            "Default Target / Population column",
            meta.columns.tolist(),
            key="ml_popcol_g",
        )
    st.info(f"✅ Using **{global_sam_col}** as Sample ID, "
             f"**{global_pop_col}** as default target.")
    st.markdown("---")
else:
    st.info(
        "ℹ️ **No metadata loaded.** Some analyses require metadata "
        "(Classification, Regression, Feature Selection, Fst outlier, "
        "BayeScan, LFMM). Others work without it (UMAP, PCAdapt, ROH, "
        "Tajima's D, iHS, Selective Sweep)."
    )
    st.markdown("---")


tab_clf, tab_reg, tab_fs, tab_sel, tab_um = st.tabs([
    "🏷️ Classification",
    "📈 Regression",
    "🎯 Feature Selection",
    "🎯 Selection Detection",
    "🗺️ Unsupervised (UMAP)",
])


# ═══════════════════════════════════════════
# HELPER: Comprehensive classification metrics
# ═══════════════════════════════════════════
def compute_classification_metrics(y_true, y_pred, y_proba=None,
                                     class_names=None):
    """Compute comprehensive classification metrics."""
    avg = "weighted" if len(np.unique(y_true)) > 2 else "binary"

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average=avg,
                                        zero_division=0),
        "Recall (Sensitivity)": recall_score(y_true, y_pred, average=avg,
                                                zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, average=avg, zero_division=0),
        "MCC (Matthews)": matthews_corrcoef(y_true, y_pred),
        "Cohen's Kappa": cohen_kappa_score(y_true, y_pred),
    }

    # AUC (only if probabilities available)
    if y_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:
                metrics["AUC"] = auc(*roc_curve(y_true, y_proba[:, 1])[:2])
            else:
                # Multi-class: macro-averaged one-vs-rest AUC
                aucs = []
                for i in range(len(np.unique(y_true))):
                    yb = (y_true == i).astype(int)
                    if yb.sum() > 0 and yb.sum() < len(yb):
                        fpr, tpr, _ = roc_curve(yb, y_proba[:, i])
                        aucs.append(auc(fpr, tpr))
                if aucs:
                    metrics["AUC (macro)"] = np.mean(aucs)
        except Exception:
            pass

    return metrics


# ═══════════════════════════════════════════
# HELPER: Comprehensive regression metrics
# ═══════════════════════════════════════════
def compute_regression_metrics(y_true, y_pred):
    """Compute comprehensive regression metrics."""
    return {
        "R² (Coefficient of Determination)": r2_score(y_true, y_pred),
        "Adjusted R²": 1 - (1 - r2_score(y_true, y_pred)) *
                         (len(y_true) - 1) / max(len(y_true) - 2, 1),
        "RMSE (Root Mean Squared Error)": np.sqrt(
            mean_squared_error(y_true, y_pred)),
        "MSE (Mean Squared Error)": mean_squared_error(y_true, y_pred),
        "MAE (Mean Absolute Error)": mean_absolute_error(y_true, y_pred),
        "MAPE (%)": (mean_absolute_percentage_error(y_true, y_pred) * 100
                        if (y_true != 0).all() else np.nan),
        "Explained Variance": explained_variance_score(y_true, y_pred),
        "Pearson r": np.corrcoef(y_true, y_pred)[0, 1]
                      if len(y_true) > 1 else np.nan,
    }


# =========================================================
# TAB 1 — Classification (Enhanced)
# =========================================================
with tab_clf:
    st.subheader("🏷️ Classification: Predict Categorical Target from SNPs")

    if meta is None:
        st.warning(
            "⚠️ **Metadata required.** Classification needs a target label "
            "(e.g., Species, Population, Group)."
        )
    else:
        c1, c2 = st.columns(2)
        with c1:
            target_col = st.selectbox(
                "Target column (categorical label)",
                meta.columns.tolist(),
                index=meta.columns.tolist().index(global_pop_col)
                if global_pop_col in meta.columns.tolist() else 0,
                key="ml_clf_target",
            )
        with c2:
            sam_col = st.selectbox(
                "Sample ID column",
                meta.columns.tolist(),
                index=meta.columns.tolist().index(global_sam_col)
                if global_sam_col in meta.columns.tolist() else 0,
                key="ml_clf_sam",
            )

        method = st.selectbox(
            "Classifier",
            ["Random Forest", "Logistic Regression", "SVM",
             "XGBoost", "Gradient Boosting"],
            key="ml_clf_method",
            help="All classifiers include full metrics + cross-validation",
        )

        # Hyperparameters
        st.markdown("#### 🔧 Hyperparameters")
        rf_n, rf_d = 100, None
        if method == "Random Forest":
            hc1, hc2 = st.columns(2)
            with hc1:
                rf_n = st.slider("N estimators", 10, 500, 100, 10, key="ml_rf_n")
            with hc2:
                rf_d = st.selectbox("Max depth", [None, 5, 10, 15, 20],
                                     key="ml_rf_d")

        lr_C = 1.0
        if method == "Logistic Regression":
            lr_C = st.slider("Regularization C", 0.01, 10.0, 1.0, 0.01,
                              key="ml_lr_c")

        svm_C, svm_k = 1.0, "rbf"
        if method == "SVM":
            hc1, hc2 = st.columns(2)
            with hc1:
                svm_C = st.slider("C", 0.01, 10.0, 1.0, 0.01, key="ml_svm_c")
            with hc2:
                svm_k = st.selectbox("Kernel", ["rbf", "linear", "poly"],
                                      key="ml_svm_k")

        xgb_lr, xgb_n, xgb_d = 0.1, 100, 6
        if method == "XGBoost":
            hc1, hc2, hc3 = st.columns(3)
            with hc1:
                xgb_lr = st.slider("Learning rate", 0.01, 0.5, 0.1, 0.01,
                                    key="ml_xgb_lr")
            with hc2:
                xgb_n = st.slider("N estimators", 10, 500, 100, 10,
                                   key="ml_xgb_n")
            with hc3:
                xgb_d = st.slider("Max depth", 2, 15, 6, key="ml_xgb_d")

        st.markdown("#### ⚙️ Data Splitting & CV")
        use_pca_clf = st.checkbox("PCA pre-reduction (recommended for many markers)",
                                    True, key="ml_use_pca")
        n_pcs_clf = st.slider("N PCs", 2, 50, 20,
                                key="ml_npcs") if use_pca_clf else None

        c3, c4, c5 = st.columns(3)
        with c3:
            test_sz = st.slider("Test size (%)", 10, 50, 20, 5,
                                 key="ml_ts") / 100
        with c4:
            rs = int(st.number_input("Random state", value=42, key="ml_rs"))
        with c5:
            cv_k = st.slider("CV folds", 2, 10, 5, key="ml_cv")

        if st.button("🚀 Train & Evaluate Classifier",
                     use_container_width=True, key="ml_clf_run"):
            pop_map = dict(zip(meta[sam_col].astype(str),
                                meta[target_col].astype(str)))
            samples_use = [s for s in geno.index.astype(str)
                            if s in pop_map and pd.notna(pop_map.get(s))]

            if len(samples_use) < 10:
                st.error(f"Only {len(samples_use)} samples with valid labels. "
                          "Need ≥ 10.")
                st.stop()

            X_raw = geno.loc[samples_use]
            y_raw = np.array([pop_map[s] for s in samples_use])

            X_imp = impute_missing(X_raw, "mean").values
            X_scaled = StandardScaler().fit_transform(X_imp)

            if use_pca_clf:
                pca_pre = PCA(n_components=min(n_pcs_clf, X_scaled.shape[1],
                                                  X_scaled.shape[0]))
                X_use = pca_pre.fit_transform(X_scaled)
            else:
                X_use = X_scaled

            le = LabelEncoder()
            y = le.fit_transform(y_raw)
            cnames = le.classes_.astype(str)
            nc = len(cnames)

            # Class distribution
            st.markdown("#### 📊 Class Distribution")
            class_dist = pd.Series(y_raw).value_counts().reset_index()
            class_dist.columns = ["Class", "Count"]
            fig_dist = px.bar(class_dist, x="Class", y="Count",
                                color="Class", text="Count",
                                title="Sample count per class")
            fig_dist.update_traces(textposition="outside")
            fig_dist.update_layout(template="plotly_white", height=400,
                                     showlegend=False)
            st.plotly_chart(fig_dist, use_container_width=True)

            # Train/test split
            try:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_use, y, test_size=test_sz,
                    random_state=rs, stratify=y,
                )
            except ValueError:
                st.warning("Stratification failed (some classes too small). "
                            "Using random split.")
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_use, y, test_size=test_sz, random_state=rs,
                )

            st.info(f"📊 **Split**: {len(X_tr)} training + {len(X_te)} test samples")

            # Build model
            if method == "Random Forest":
                model = RandomForestClassifier(n_estimators=rf_n, max_depth=rf_d,
                                                  random_state=rs, n_jobs=-1)
            elif method == "Logistic Regression":
                model = LogisticRegression(C=lr_C, max_iter=2000, random_state=rs)
            elif method == "SVM":
                model = SVC(C=svm_C, kernel=svm_k, probability=True,
                             random_state=rs)
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

                # Full metrics on test
                test_metrics = compute_classification_metrics(
                    y_te, y_pr, y_pp, cnames)

                # Cross-validation
                skf = StratifiedKFold(n_splits=cv_k, shuffle=True,
                                        random_state=rs)
                scoring = ["accuracy", "precision_weighted",
                            "recall_weighted", "f1_weighted"]
                cv_results = cross_validate(model, X_use, y, cv=skf,
                                              scoring=scoring, n_jobs=-1)

            st.success(f"✅ {method} trained successfully.")

            # ─── Test metrics ───
            st.markdown("### 📊 Test Set Metrics")
            n_metrics = len(test_metrics)
            n_cols = min(4, n_metrics)
            rows_needed = (n_metrics + n_cols - 1) // n_cols

            metric_items = list(test_metrics.items())
            for row_i in range(rows_needed):
                cols = st.columns(n_cols)
                for col_i in range(n_cols):
                    idx = row_i * n_cols + col_i
                    if idx < n_metrics:
                        name, val = metric_items[idx]
                        cols[col_i].metric(name, f"{val:.4f}")

            # ─── Cross-validation ───
            st.markdown(f"### 🔄 {cv_k}-Fold Cross-Validation Results")
            cv_summary = pd.DataFrame({
                "Metric": ["Accuracy", "Precision (weighted)",
                            "Recall (weighted)", "F1 (weighted)"],
                "Mean": [
                    cv_results["test_accuracy"].mean(),
                    cv_results["test_precision_weighted"].mean(),
                    cv_results["test_recall_weighted"].mean(),
                    cv_results["test_f1_weighted"].mean(),
                ],
                "Std": [
                    cv_results["test_accuracy"].std(),
                    cv_results["test_precision_weighted"].std(),
                    cv_results["test_recall_weighted"].std(),
                    cv_results["test_f1_weighted"].std(),
                ],
                "Min": [
                    cv_results["test_accuracy"].min(),
                    cv_results["test_precision_weighted"].min(),
                    cv_results["test_recall_weighted"].min(),
                    cv_results["test_f1_weighted"].min(),
                ],
                "Max": [
                    cv_results["test_accuracy"].max(),
                    cv_results["test_precision_weighted"].max(),
                    cv_results["test_recall_weighted"].max(),
                    cv_results["test_f1_weighted"].max(),
                ],
            })
            st.dataframe(cv_summary.style.format({
                c: "{:.4f}" for c in ["Mean", "Std", "Min", "Max"]
            }), use_container_width=True)

            # CV fold scores plot
            fig_cv = go.Figure()
            fig_cv.add_trace(go.Box(y=cv_results["test_accuracy"],
                                        name="Accuracy",
                                        marker_color="steelblue"))
            fig_cv.add_trace(go.Box(y=cv_results["test_f1_weighted"],
                                        name="F1",
                                        marker_color="orange"))
            fig_cv.add_trace(go.Box(y=cv_results["test_precision_weighted"],
                                        name="Precision",
                                        marker_color="green"))
            fig_cv.add_trace(go.Box(y=cv_results["test_recall_weighted"],
                                        name="Recall",
                                        marker_color="purple"))
            fig_cv.update_layout(
                title=f"Cross-validation scores ({cv_k} folds)",
                yaxis_title="Score", template="plotly_white", height=450,
            )
            st.plotly_chart(fig_cv, use_container_width=True)

            # ─── Classification report ───
            st.markdown("### 📝 Detailed Classification Report")
            rpt = classification_report(y_te, y_pr, target_names=cnames,
                                          output_dict=True)
            st.dataframe(pd.DataFrame(rpt).T.style.format("{:.4f}"),
                          use_container_width=True)

            # ─── Confusion matrix ───
            st.markdown("### 🔀 Confusion Matrix")
            cm = confusion_matrix(y_te, y_pr)
            fig_cm = px.imshow(cm, x=cnames, y=cnames, text_auto=True,
                                color_continuous_scale="Blues",
                                title="Confusion Matrix (Test Set)",
                                labels=dict(x="Predicted", y="Actual"))
            fig_cm.update_layout(template="plotly_white", height=500)
            st.plotly_chart(fig_cm, use_container_width=True)

            # Normalized CM
            cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
            fig_cm_n = px.imshow(cm_norm, x=cnames, y=cnames,
                                    text_auto=".2f",
                                    color_continuous_scale="Blues",
                                    title="Normalized Confusion Matrix",
                                    labels=dict(x="Predicted", y="Actual",
                                                color="Proportion"))
            fig_cm_n.update_layout(template="plotly_white", height=500)
            st.plotly_chart(fig_cm_n, use_container_width=True)

            # ─── ROC Curve ───
            if y_pp is not None:
                st.markdown("### 📈 ROC Curve")
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
                    line=dict(dash="dash", color="gray"), name="Random"))
                fig_roc.update_layout(
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    template="plotly_white", height=500,
                    title="ROC Curves (Test Set)")
                st.plotly_chart(fig_roc, use_container_width=True)

            # ─── Feature Importance ───
            if method in ["Random Forest", "XGBoost", "Gradient Boosting"]:
                st.markdown("### 🎯 Feature Importance (top 30)")
                feat_names = ([f"PC{i+1}" for i in range(X_use.shape[1])]
                                if use_pca_clf
                                else geno.columns.astype(str).tolist())
                imp = pd.DataFrame({
                    "Feature": feat_names,
                    "Importance": model.feature_importances_,
                }).sort_values("Importance", ascending=False).head(30)

                fig_imp = px.bar(imp, x="Importance", y="Feature",
                                  orientation="h", color="Importance",
                                  color_continuous_scale="viridis",
                                  title="Top 30 features")
                fig_imp.update_layout(template="plotly_white", height=650,
                                         yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_imp, use_container_width=True)
                download_dataframe(imp, "feature_importance.csv",
                                    key="dl_ml_imp")

            # ─── Save metrics ───
            metrics_df = pd.DataFrame({
                "Metric": list(test_metrics.keys()),
                "Value": list(test_metrics.values()),
            })
            download_dataframe(metrics_df, "classification_metrics.csv",
                                key="dl_clf_metrics")

# =========================================================
# TAB 2 — REGRESSION (NEW! Comprehensive)
# =========================================================
with tab_reg:
    st.subheader("📈 Regression: Predict Continuous Trait from SNPs")
    st.write(
        "Multiple regression algorithms to predict quantitative traits "
        "(e.g., yield, height, disease resistance) from SNP genotypes."
    )

    if meta is None:
        st.warning(
            "⚠️ **Metadata required.** Regression needs a numeric target "
            "(e.g., yield, height, biomass, disease score)."
        )
    else:
        rc1, rc2 = st.columns(2)
        with rc1:
            numeric_cols = meta.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                st.error("No numeric columns found in metadata.")
                st.stop()
            target_reg = st.selectbox(
                "Target trait (numeric)",
                numeric_cols,
                key="ml_reg_target",
            )
        with rc2:
            sam_reg = st.selectbox(
                "Sample ID column",
                meta.columns.tolist(),
                index=meta.columns.tolist().index(global_sam_col)
                if global_sam_col in meta.columns.tolist() else 0,
                key="ml_reg_sam",
            )

        method_reg = st.selectbox(
            "Regression Algorithm",
            [
                "Linear Regression (OLS)",
                "Ridge Regression (L2)",
                "Lasso Regression (L1)",
                "ElasticNet (L1 + L2)",
                "Random Forest",
                "Gradient Boosting",
                "XGBoost",
                "Support Vector Regression (SVR)",
            ],
            key="ml_reg_method",
            help="Each algorithm has different strengths for genomic prediction.",
        )

        # ─── Hyperparameters ───
        st.markdown("#### 🔧 Hyperparameters")

        # Ridge
        ridge_alpha = 1.0
        ridge_cv = False
        if method_reg == "Ridge Regression (L2)":
            hrc1, hrc2 = st.columns(2)
            with hrc1:
                ridge_cv = st.checkbox("Auto-select α (RidgeCV)", True,
                                        key="ml_ridge_cv")
            with hrc2:
                if not ridge_cv:
                    ridge_alpha = st.slider("Alpha (L2 strength)", 0.001,
                                              100.0, 1.0, 0.1,
                                              key="ml_ridge_alpha")

        # Lasso
        lasso_alpha = 0.1
        lasso_cv = False
        if method_reg == "Lasso Regression (L1)":
            hlc1, hlc2 = st.columns(2)
            with hlc1:
                lasso_cv = st.checkbox("Auto-select α (LassoCV)", True,
                                        key="ml_lasso_cv")
            with hlc2:
                if not lasso_cv:
                    lasso_alpha = st.slider("Alpha (L1 strength)", 0.001,
                                              10.0, 0.1, 0.01,
                                              key="ml_lasso_alpha")

        # ElasticNet
        en_alpha, en_l1 = 0.1, 0.5
        en_cv = False
        if method_reg == "ElasticNet (L1 + L2)":
            hec1, hec2, hec3 = st.columns(3)
            with hec1:
                en_cv = st.checkbox("Auto-select (ElasticNetCV)", True,
                                     key="ml_en_cv")
            with hec2:
                if not en_cv:
                    en_alpha = st.slider("Alpha", 0.001, 10.0, 0.1, 0.01,
                                          key="ml_en_alpha")
            with hec3:
                if not en_cv:
                    en_l1 = st.slider("L1 ratio", 0.0, 1.0, 0.5, 0.05,
                                       key="ml_en_l1")

        # Random Forest / Gradient Boosting
        rf_n_reg, rf_d_reg = 100, None
        if method_reg == "Random Forest":
            hrc1, hrc2 = st.columns(2)
            with hrc1:
                rf_n_reg = st.slider("N estimators", 10, 500, 100, 10,
                                       key="ml_rf_reg_n")
            with hrc2:
                rf_d_reg = st.selectbox("Max depth",
                                          [None, 5, 10, 15, 20],
                                          key="ml_rf_reg_d")

        gb_n_reg, gb_lr = 100, 0.1
        if method_reg == "Gradient Boosting":
            hgc1, hgc2 = st.columns(2)
            with hgc1:
                gb_n_reg = st.slider("N estimators", 10, 500, 100, 10,
                                       key="ml_gb_reg_n")
            with hgc2:
                gb_lr = st.slider("Learning rate", 0.01, 0.5, 0.1, 0.01,
                                    key="ml_gb_reg_lr")

        # XGBoost
        xgb_n_reg, xgb_lr_reg, xgb_d_reg = 100, 0.1, 6
        if method_reg == "XGBoost":
            hxc1, hxc2, hxc3 = st.columns(3)
            with hxc1:
                xgb_lr_reg = st.slider("Learning rate", 0.01, 0.5, 0.1, 0.01,
                                         key="ml_xgb_reg_lr")
            with hxc2:
                xgb_n_reg = st.slider("N estimators", 10, 500, 100, 10,
                                        key="ml_xgb_reg_n")
            with hxc3:
                xgb_d_reg = st.slider("Max depth", 2, 15, 6, key="ml_xgb_reg_d")

        # SVR
        svr_C, svr_k, svr_eps = 1.0, "rbf", 0.1
        if method_reg == "Support Vector Regression (SVR)":
            hsc1, hsc2, hsc3 = st.columns(3)
            with hsc1:
                svr_C = st.slider("C", 0.01, 10.0, 1.0, 0.01, key="ml_svr_c")
            with hsc2:
                svr_k = st.selectbox("Kernel",
                                       ["rbf", "linear", "poly"],
                                       key="ml_svr_k")
            with hsc3:
                svr_eps = st.slider("Epsilon", 0.001, 1.0, 0.1, 0.01,
                                      key="ml_svr_eps")

        # ─── Data Splitting & CV ───
        st.markdown("#### ⚙️ Data Splitting & Cross-Validation")

        use_pca_reg = st.checkbox("PCA pre-reduction (recommended for many markers)",
                                    True, key="ml_reg_pca")
        n_pcs_reg = st.slider("N PCs", 2, 50, 20,
                                key="ml_reg_npcs") if use_pca_reg else None

        rc3, rc4, rc5 = st.columns(3)
        with rc3:
            test_sz_reg = st.slider("Test size (%)", 10, 50, 20, 5,
                                      key="ml_reg_ts") / 100
        with rc4:
            rs_reg = int(st.number_input("Random state", value=42,
                                            key="ml_reg_rs"))
        with rc5:
            cv_k_reg = st.slider("CV folds", 2, 10, 5, key="ml_reg_cv")

        if st.button("🚀 Train & Evaluate Regression Model",
                     use_container_width=True, key="ml_reg_run"):

            # Prepare data
            trait_map = dict(zip(meta[sam_reg].astype(str),
                                   meta[target_reg]))
            samples_use = []
            y_list = []
            for s in geno.index.astype(str):
                if s in trait_map:
                    val = trait_map[s]
                    if pd.notna(val):
                        try:
                            y_list.append(float(val))
                            samples_use.append(s)
                        except (ValueError, TypeError):
                            pass

            if len(samples_use) < 10:
                st.error(f"Only {len(samples_use)} samples with valid target "
                          "values. Need ≥ 10.")
                st.stop()

            X_raw = geno.loc[samples_use]
            y = np.array(y_list)

            X_imp = impute_missing(X_raw, "mean").values
            X_scaled = StandardScaler().fit_transform(X_imp)

            if use_pca_reg:
                pca_pre = PCA(n_components=min(n_pcs_reg,
                                                  X_scaled.shape[1],
                                                  X_scaled.shape[0]))
                X_use = pca_pre.fit_transform(X_scaled)
            else:
                X_use = X_scaled

            # Target distribution
            st.markdown("#### 📊 Target Trait Distribution")
            trc1, trc2 = st.columns(2)
            with trc1:
                fig_target = px.histogram(x=y, nbins=30,
                                             title=f"Distribution of {target_reg}",
                                             labels={"x": target_reg,
                                                      "y": "Count"})
                fig_target.update_layout(template="plotly_white", height=400)
                st.plotly_chart(fig_target, use_container_width=True)
            with trc2:
                fig_box = px.box(y=y,
                                  title=f"Box plot of {target_reg}",
                                  labels={"y": target_reg})
                fig_box.update_layout(template="plotly_white", height=400)
                st.plotly_chart(fig_box, use_container_width=True)

            # Target statistics
            tsc1, tsc2, tsc3, tsc4, tsc5 = st.columns(5)
            tsc1.metric("N samples", len(y))
            tsc2.metric("Mean", f"{y.mean():.4f}")
            tsc3.metric("Std", f"{y.std():.4f}")
            tsc4.metric("Min", f"{y.min():.4f}")
            tsc5.metric("Max", f"{y.max():.4f}")

            # Train/test split
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_use, y, test_size=test_sz_reg, random_state=rs_reg,
            )

            st.info(f"📊 **Split**: {len(X_tr)} training + {len(X_te)} test samples")

            # Build model
            with st.spinner(f"Training {method_reg}..."):
                if method_reg == "Linear Regression (OLS)":
                    model_reg = LinearRegression()
                elif method_reg == "Ridge Regression (L2)":
                    if ridge_cv:
                        model_reg = RidgeCV(alphas=np.logspace(-3, 3, 30),
                                             cv=cv_k_reg)
                    else:
                        model_reg = Ridge(alpha=ridge_alpha,
                                           random_state=rs_reg)
                elif method_reg == "Lasso Regression (L1)":
                    if lasso_cv:
                        model_reg = LassoCV(cv=cv_k_reg,
                                             random_state=rs_reg,
                                             max_iter=10000, n_jobs=-1)
                    else:
                        model_reg = Lasso(alpha=lasso_alpha,
                                           random_state=rs_reg,
                                           max_iter=10000)
                elif method_reg == "ElasticNet (L1 + L2)":
                    if en_cv:
                        model_reg = ElasticNetCV(
                            cv=cv_k_reg,
                            random_state=rs_reg,
                            max_iter=10000, n_jobs=-1,
                            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9])
                    else:
                        model_reg = ElasticNet(alpha=en_alpha, l1_ratio=en_l1,
                                                random_state=rs_reg,
                                                max_iter=10000)
                elif method_reg == "Random Forest":
                    model_reg = RandomForestRegressor(
                        n_estimators=rf_n_reg, max_depth=rf_d_reg,
                        random_state=rs_reg, n_jobs=-1)
                elif method_reg == "Gradient Boosting":
                    model_reg = GradientBoostingRegressor(
                        n_estimators=gb_n_reg, learning_rate=gb_lr,
                        random_state=rs_reg)
                elif method_reg == "XGBoost":
                    try:
                        from xgboost import XGBRegressor
                        model_reg = XGBRegressor(
                            learning_rate=xgb_lr_reg, n_estimators=xgb_n_reg,
                            max_depth=xgb_d_reg, random_state=rs_reg,
                            n_jobs=-1, verbosity=0)
                    except ImportError:
                        st.error("XGBoost not installed.")
                        st.stop()
                elif method_reg == "Support Vector Regression (SVR)":
                    model_reg = SVR(C=svr_C, kernel=svr_k, epsilon=svr_eps)

                model_reg.fit(X_tr, y_tr)
                y_pr_tr = model_reg.predict(X_tr)
                y_pr_te = model_reg.predict(X_te)

                # Test metrics
                test_metrics_reg = compute_regression_metrics(y_te, y_pr_te)
                train_metrics_reg = compute_regression_metrics(y_tr, y_pr_tr)

                # Cross-validation
                kf = KFold(n_splits=cv_k_reg, shuffle=True,
                            random_state=rs_reg)
                scoring_reg = ["r2", "neg_root_mean_squared_error",
                                "neg_mean_absolute_error",
                                "explained_variance"]
                cv_results_reg = cross_validate(model_reg, X_use, y, cv=kf,
                                                  scoring=scoring_reg,
                                                  n_jobs=-1)

            st.success(f"✅ {method_reg} trained successfully.")

            # ─── Auto-selected alpha (if applicable) ───
            if method_reg == "Ridge Regression (L2)" and ridge_cv:
                st.info(f"🎯 Auto-selected α = {model_reg.alpha_:.6f}")
            elif method_reg == "Lasso Regression (L1)" and lasso_cv:
                st.info(f"🎯 Auto-selected α = {model_reg.alpha_:.6f}")
            elif method_reg == "ElasticNet (L1 + L2)" and en_cv:
                st.info(f"🎯 Auto-selected α = {model_reg.alpha_:.6f}, "
                         f"L1 ratio = {model_reg.l1_ratio_:.3f}")

            # ─── Test Metrics ───
            st.markdown("### 📊 Test Set Performance")
            metric_items_reg = list(test_metrics_reg.items())
            n_metrics_reg = len(metric_items_reg)
            n_cols_reg = min(4, n_metrics_reg)
            rows_needed_reg = (n_metrics_reg + n_cols_reg - 1) // n_cols_reg

            for row_i in range(rows_needed_reg):
                cols = st.columns(n_cols_reg)
                for col_i in range(n_cols_reg):
                    idx = row_i * n_cols_reg + col_i
                    if idx < n_metrics_reg:
                        name, val = metric_items_reg[idx]
                        cols[col_i].metric(name,
                                            f"{val:.4f}"
                                            if not np.isnan(val) else "N/A")

            # ─── Train vs Test comparison ───
            st.markdown("### 📊 Train vs Test Metrics")
            comparison_df = pd.DataFrame({
                "Metric": list(test_metrics_reg.keys()),
                "Train": list(train_metrics_reg.values()),
                "Test": list(test_metrics_reg.values()),
            })
            comparison_df["Overfitting"] = comparison_df["Train"] - comparison_df["Test"]

            st.dataframe(comparison_df.style.format({
                "Train": "{:.4f}", "Test": "{:.4f}", "Overfitting": "{:.4f}",
            }), use_container_width=True)

            # ─── Cross-Validation Results ───
            st.markdown(f"### 🔄 {cv_k_reg}-Fold Cross-Validation")
            cv_summary_reg = pd.DataFrame({
                "Metric": ["R²", "RMSE", "MAE", "Explained Variance"],
                "Mean": [
                    cv_results_reg["test_r2"].mean(),
                    -cv_results_reg["test_neg_root_mean_squared_error"].mean(),
                    -cv_results_reg["test_neg_mean_absolute_error"].mean(),
                    cv_results_reg["test_explained_variance"].mean(),
                ],
                "Std": [
                    cv_results_reg["test_r2"].std(),
                    cv_results_reg["test_neg_root_mean_squared_error"].std(),
                    cv_results_reg["test_neg_mean_absolute_error"].std(),
                    cv_results_reg["test_explained_variance"].std(),
                ],
                "Min": [
                    cv_results_reg["test_r2"].min(),
                    -cv_results_reg["test_neg_root_mean_squared_error"].max(),
                    -cv_results_reg["test_neg_mean_absolute_error"].max(),
                    cv_results_reg["test_explained_variance"].min(),
                ],
                "Max": [
                    cv_results_reg["test_r2"].max(),
                    -cv_results_reg["test_neg_root_mean_squared_error"].min(),
                    -cv_results_reg["test_neg_mean_absolute_error"].min(),
                    cv_results_reg["test_explained_variance"].max(),
                ],
            })
            st.dataframe(cv_summary_reg.style.format({
                c: "{:.4f}" for c in ["Mean", "Std", "Min", "Max"]
            }), use_container_width=True)

            # CV fold plot
            fig_cv_reg = go.Figure()
            fig_cv_reg.add_trace(go.Box(y=cv_results_reg["test_r2"],
                                            name="R²",
                                            marker_color="steelblue"))
            fig_cv_reg.add_trace(go.Box(
                y=-cv_results_reg["test_neg_root_mean_squared_error"],
                name="RMSE",
                marker_color="orange",
                yaxis="y2"))
            fig_cv_reg.update_layout(
                title=f"Cross-validation scores ({cv_k_reg} folds)",
                yaxis=dict(title="R²", side="left"),
                yaxis2=dict(title="RMSE", overlaying="y", side="right"),
                template="plotly_white", height=450,
            )
            st.plotly_chart(fig_cv_reg, use_container_width=True)

            # ─── Actual vs Predicted ───
            st.markdown("### 📈 Actual vs Predicted")

            av_df = pd.DataFrame({
                "Actual": np.concatenate([y_tr, y_te]),
                "Predicted": np.concatenate([y_pr_tr, y_pr_te]),
                "Set": ["Train"] * len(y_tr) + ["Test"] * len(y_te),
            })

            fig_av = px.scatter(
                av_df, x="Actual", y="Predicted", color="Set",
                color_discrete_map={"Train": "steelblue", "Test": "orange"},
                opacity=0.7,
                title=f"Actual vs Predicted {target_reg}",
                hover_data=["Set"],
            )
            # Add y=x line
            min_val = min(y.min(), av_df["Predicted"].min())
            max_val = max(y.max(), av_df["Predicted"].max())
            fig_av.add_shape(
                type="line", x0=min_val, y0=min_val,
                x1=max_val, y1=max_val,
                line=dict(color="red", dash="dash", width=2),
            )
            fig_av.update_traces(marker=dict(size=8,
                                                line=dict(width=0.5,
                                                            color="darkslategrey")))
            fig_av.update_layout(template="plotly_white", height=550)
            st.plotly_chart(fig_av, use_container_width=True)

            # ─── Residual Analysis ───
            st.markdown("### 📉 Residual Analysis (Test Set)")

            residuals = y_te - y_pr_te

            res_c1, res_c2 = st.columns(2)

            with res_c1:
                # Residuals vs Predicted
                fig_res = px.scatter(
                    x=y_pr_te, y=residuals,
                    labels={"x": "Predicted", "y": "Residuals"},
                    title="Residuals vs Predicted",
                    opacity=0.6,
                )
                fig_res.add_hline(y=0, line_dash="dash", line_color="red")
                fig_res.update_traces(marker=dict(size=8, color="steelblue"))
                fig_res.update_layout(template="plotly_white", height=400)
                st.plotly_chart(fig_res, use_container_width=True)

            with res_c2:
                # Residual distribution
                fig_res_dist = px.histogram(
                    x=residuals, nbins=30,
                    title="Residual distribution",
                    labels={"x": "Residual"},
                )
                fig_res_dist.add_vline(x=0, line_dash="dash", line_color="red")
                fig_res_dist.update_layout(template="plotly_white", height=400)
                st.plotly_chart(fig_res_dist, use_container_width=True)

            # Q-Q plot of residuals
            st.markdown("#### Q-Q Plot of Residuals (Normality Check)")
            try:
                qq_data = probplot(residuals, dist="norm")
                theoretical_q = qq_data[0][0]
                actual_q = qq_data[0][1]
                slope, intercept, r_val = qq_data[1]

                fig_qq = go.Figure()
                fig_qq.add_trace(go.Scatter(
                    x=theoretical_q, y=actual_q, mode="markers",
                    marker=dict(size=6, color="steelblue"),
                    name="Residuals",
                ))
                fig_qq.add_trace(go.Scatter(
                    x=theoretical_q, y=slope * theoretical_q + intercept,
                    mode="lines",
                    line=dict(color="red", dash="dash"),
                    name=f"Line (r={r_val:.3f})",
                ))
                fig_qq.update_layout(
                    title=f"Q-Q Plot (r={r_val:.4f})",
                    xaxis_title="Theoretical Quantiles",
                    yaxis_title="Sample Quantiles",
                    template="plotly_white", height=400,
                )
                st.plotly_chart(fig_qq, use_container_width=True)

                if abs(r_val) > 0.95:
                    st.success("✅ Residuals appear approximately normally distributed.")
                elif abs(r_val) > 0.9:
                    st.info("ℹ️ Residuals are moderately normal.")
                else:
                    st.warning(
                        "⚠️ Residuals may not be normally distributed. "
                        "Consider transforming your target variable."
                    )
            except Exception as e:
                st.warning(f"Could not generate Q-Q plot: {e}")

            # ─── Learning Curve ───
            st.markdown("### 📚 Learning Curve")
            with st.spinner("Computing learning curve..."):
                try:
                    train_sizes, train_scores, val_scores = learning_curve(
                        model_reg, X_use, y,
                        cv=min(3, cv_k_reg),
                        train_sizes=np.linspace(0.1, 1.0, 8),
                        scoring="r2",
                        n_jobs=-1,
                    )

                    fig_lc = go.Figure()
                    fig_lc.add_trace(go.Scatter(
                        x=train_sizes, y=train_scores.mean(axis=1),
                        mode="lines+markers",
                        name="Training R²",
                        line=dict(color="steelblue", width=2),
                        error_y=dict(type="data",
                                       array=train_scores.std(axis=1)),
                    ))
                    fig_lc.add_trace(go.Scatter(
                        x=train_sizes, y=val_scores.mean(axis=1),
                        mode="lines+markers",
                        name="Validation R²",
                        line=dict(color="orange", width=2),
                        error_y=dict(type="data",
                                       array=val_scores.std(axis=1)),
                    ))
                    fig_lc.update_layout(
                        title="Learning Curve",
                        xaxis_title="Training Set Size",
                        yaxis_title="R²",
                        template="plotly_white", height=450,
                    )
                    st.plotly_chart(fig_lc, use_container_width=True)
                except Exception as e:
                    st.warning(f"Learning curve failed: {e}")

            # ─── Feature Importance / Coefficients ───
            if method_reg in ["Random Forest", "Gradient Boosting", "XGBoost"]:
                st.markdown("### 🎯 Feature Importance (top 30)")
                feat_names = ([f"PC{i+1}" for i in range(X_use.shape[1])]
                                if use_pca_reg
                                else geno.columns.astype(str).tolist())
                imp_reg = pd.DataFrame({
                    "Feature": feat_names,
                    "Importance": model_reg.feature_importances_,
                }).sort_values("Importance", ascending=False).head(30)

                fig_imp_reg = px.bar(
                    imp_reg, x="Importance", y="Feature",
                    orientation="h", color="Importance",
                    color_continuous_scale="viridis",
                    title="Top 30 features by importance",
                )
                fig_imp_reg.update_layout(template="plotly_white", height=650,
                                            yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_imp_reg, use_container_width=True)
                download_dataframe(imp_reg, "regression_feature_importance.csv",
                                    key="dl_reg_imp")

            elif method_reg in ["Linear Regression (OLS)", "Ridge Regression (L2)",
                                  "Lasso Regression (L1)", "ElasticNet (L1 + L2)"]:
                st.markdown("### 🎯 Regression Coefficients (top 30 by |coefficient|)")
                feat_names = ([f"PC{i+1}" for i in range(X_use.shape[1])]
                                if use_pca_reg
                                else geno.columns.astype(str).tolist())
                coef_df = pd.DataFrame({
                    "Feature": feat_names,
                    "Coefficient": model_reg.coef_,
                    "Abs_Coef": np.abs(model_reg.coef_),
                }).sort_values("Abs_Coef", ascending=False).head(30)

                fig_coef = px.bar(
                    coef_df, x="Coefficient", y="Feature",
                    orientation="h", color="Coefficient",
                    color_continuous_scale="RdBu_r",
                    title="Top 30 features by coefficient magnitude",
                )
                fig_coef.update_layout(template="plotly_white", height=650,
                                          yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_coef, use_container_width=True)

                # For Lasso/ElasticNet: count of non-zero coefs
                if method_reg in ["Lasso Regression (L1)",
                                    "ElasticNet (L1 + L2)"]:
                    n_nonzero = int(np.sum(np.abs(model_reg.coef_) > 1e-8))
                    st.info(f"📊 **Non-zero coefficients:** {n_nonzero} out of "
                             f"{len(model_reg.coef_)} "
                             f"({n_nonzero / len(model_reg.coef_) * 100:.1f}%)")

                download_dataframe(coef_df, "regression_coefficients.csv",
                                    key="dl_reg_coef")

            # ─── Predictions download ───
            st.markdown("### 💾 Download Predictions")
            predictions_df = pd.DataFrame({
                "Sample": samples_use,
                "Actual": y,
                "Predicted": model_reg.predict(X_use),
            })
            predictions_df["Residual"] = predictions_df["Actual"] - predictions_df["Predicted"]

            st.dataframe(predictions_df.head(20).style.format({
                "Actual": "{:.4f}", "Predicted": "{:.4f}", "Residual": "{:.4f}",
            }), use_container_width=True)

            download_dataframe(predictions_df, "regression_predictions.csv",
                                key="dl_reg_preds")

            # ─── Save metrics ───
            metrics_reg_df = pd.DataFrame({
                "Metric": list(test_metrics_reg.keys()),
                "Test_Value": list(test_metrics_reg.values()),
                "Train_Value": list(train_metrics_reg.values()),
            })
            download_dataframe(metrics_reg_df, "regression_metrics.csv",
                                key="dl_reg_metrics")


# =========================================================
# TAB 3 — Feature Selection (Enhanced)
# =========================================================
with tab_fs:
    st.subheader("🎯 Feature Selection — Identify Informative SNPs")
    st.write(
        "Identifies the most informative SNPs for predicting a phenotype/trait "
        "using regularization methods."
    )

    if meta is None:
        st.warning("⚠️ **Metadata required** — need a numeric trait column.")
    else:
        fsc1, fsc2 = st.columns(2)
        with fsc1:
            numeric_cols_fs = meta.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols_fs:
                st.error("No numeric columns found in metadata.")
                st.stop()
            target_fs = st.selectbox(
                "Target trait (numeric)",
                numeric_cols_fs,
                key="fs_target",
            )
        with fsc2:
            sam_fs = st.selectbox(
                "Sample ID column",
                meta.columns.tolist(),
                index=meta.columns.tolist().index(global_sam_col)
                if global_sam_col in meta.columns.tolist() else 0,
                key="fs_sam",
            )

        fs_method = st.selectbox(
            "Feature selection method",
            ["Lasso (L1)", "ElasticNet (L1+L2)", "Ridge (L2, non-sparse)"],
            key="fs_method",
        )

        alpha_choice = st.selectbox(
            "Alpha selection",
            ["Auto (CV)", "Manual"],
            key="fs_alpha_ch",
        )
        if alpha_choice == "Manual":
            alpha_fs = st.slider("Alpha", 0.001, 5.0, 0.1, 0.001,
                                  key="fs_alpha")

        cv_k_fs = st.slider("CV folds (if Auto)", 3, 10, 5, key="fs_cv")

        if st.button("🚀 Run Feature Selection",
                     use_container_width=True, key="fs_run"):
            trait_map = dict(zip(meta[sam_fs].astype(str),
                                   meta[target_fs]))
            samples_use = []
            y_list = []
            for s in geno.index.astype(str):
                if s in trait_map and pd.notna(trait_map[s]):
                    try:
                        y_list.append(float(trait_map[s]))
                        samples_use.append(s)
                    except (ValueError, TypeError):
                        pass

            if len(samples_use) < 10:
                st.error(f"Only {len(samples_use)} samples with valid target. "
                          "Need ≥ 10.")
                st.stop()

            X_raw = geno.loc[samples_use]
            y = np.array(y_list)

            X_imp = impute_missing(X_raw, "mean").values
            X_scaled = StandardScaler().fit_transform(X_imp)

            with st.spinner(f"Fitting {fs_method}..."):
                if fs_method == "Lasso (L1)":
                    if alpha_choice == "Auto (CV)":
                        model_fs = LassoCV(cv=cv_k_fs, random_state=42,
                                            max_iter=10000, n_jobs=-1)
                    else:
                        model_fs = Lasso(alpha=alpha_fs, max_iter=10000)
                elif fs_method == "ElasticNet (L1+L2)":
                    if alpha_choice == "Auto (CV)":
                        model_fs = ElasticNetCV(
                            cv=cv_k_fs, random_state=42,
                            max_iter=10000, n_jobs=-1,
                            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9])
                    else:
                        model_fs = ElasticNet(alpha=alpha_fs, l1_ratio=0.5,
                                                max_iter=10000)
                else:  # Ridge
                    if alpha_choice == "Auto (CV)":
                        model_fs = RidgeCV(alphas=np.logspace(-3, 3, 30),
                                             cv=cv_k_fs)
                    else:
                        model_fs = Ridge(alpha=alpha_fs)

                model_fs.fit(X_scaled, y)

            coefs = model_fs.coef_
            selected_mask = np.abs(coefs) > 1e-8
            n_selected = int(selected_mask.sum())

            st.success(f"✅ Selected **{n_selected}** SNPs out of {len(coefs)} "
                        f"({n_selected/len(coefs)*100:.2f}%).")

            if alpha_choice == "Auto (CV)":
                if hasattr(model_fs, "alpha_"):
                    st.info(f"🎯 Optimal α = {model_fs.alpha_:.6f}")
                if hasattr(model_fs, "l1_ratio_"):
                    st.info(f"🎯 Optimal L1 ratio = {model_fs.l1_ratio_:.3f}")

            # Compute R² of selected model
            y_pred_fs = model_fs.predict(X_scaled)
            r2_fs = r2_score(y, y_pred_fs)
            rmse_fs = np.sqrt(mean_squared_error(y, y_pred_fs))

            fsc1, fsc2, fsc3 = st.columns(3)
            fsc1.metric("R² (training)", f"{r2_fs:.4f}")
            fsc2.metric("RMSE", f"{rmse_fs:.4f}")
            fsc3.metric("Non-zero SNPs", n_selected)

            imp_df = pd.DataFrame({
                "SNP": geno.columns.astype(str),
                "Coefficient": coefs,
                "Abs_Coef": np.abs(coefs),
            }).sort_values("Abs_Coef", ascending=False)

            if marker_info is not None:
                imp_df = imp_df.merge(marker_info,
                                        left_on="SNP", right_on="Marker",
                                        how="left")

            top_coef = imp_df.head(50)
            fig_c = px.bar(top_coef, x="Coefficient", y="SNP",
                            orientation="h", color="Coefficient",
                            color_continuous_scale="RdBu_r",
                            title=f"Top 50 selected SNPs — {fs_method}")
            fig_c.update_layout(template="plotly_white", height=800,
                                 yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_c, use_container_width=True)

            st.dataframe(imp_df.head(200), use_container_width=True)
            download_dataframe(imp_df, "feature_selection_snps.csv",
                                key="dl_fs")

# =========================================================
# TAB 4 — SELECTION DETECTION (Comprehensive, No Metadata Required for many)
# =========================================================
with tab_sel:
    st.subheader("🎯 Selection Detection")
    st.write(
        "Comprehensive suite for detecting SNPs under natural selection.\n\n"
        "**Methods that DON'T require metadata:**\n"
        "- PCAdapt, Extended ROH, Tajima's D, iHS-like, Selective Sweep\n\n"
        "**Methods that REQUIRE metadata (population/environment):**\n"
        "- Fst outlier, BayeScan-like, LFMM-like"
    )

    method_sel = st.selectbox(
        "Detection method",
        [
            "PCAdapt (PC-based) — no metadata needed",
            "Extended ROH (Runs of Homozygosity) — no metadata needed",
            "Tajima's D (sliding window) — no metadata needed",
            "iHS-like (Extended Haplotype Homozygosity) — no metadata needed",
            "Selective Sweep (CLR-like) — no metadata needed",
            "Fst outlier (population-based) — needs metadata",
            "BayeScan-like (Fst decomposition) — needs metadata",
            "LFMM-like (Genotype-Environment Association) — needs metadata",
            "🧩 Combined outlier detection",
        ],
        key="sel_method",
    )

    # ═══════════════════════════════════════════
    # METHOD 1: PCAdapt (NO METADATA REQUIRED)
    # ═══════════════════════════════════════════
    if "PCAdapt" in method_sel:
        st.info("✅ This method works **without metadata**.")
        st.write("Identifies SNPs strongly correlated with top PCs (Mahalanobis-based).")

        pc1, pc2 = st.columns(2)
        with pc1:
            k_pcs = st.slider("Number of PCs", 2, 20, 5, key="sel_pca_k")
        with pc2:
            pct_out = st.slider("Top % as outliers", 1, 20, 5, key="sel_pca_pct")

        if st.button("🚀 Run PCAdapt-like", key="sel_pca_run"):
            with st.spinner("Running PCA + Mahalanobis..."):
                imp = impute_missing(geno, "mean").values
                X_sc = StandardScaler().fit_transform(imp)
                pca_a = PCA(n_components=k_pcs)
                pca_a.fit(X_sc)

                loadings = pca_a.components_
                loadings_z = (loadings - loadings.mean(axis=1, keepdims=True)) / \
                              loadings.std(axis=1, keepdims=True)
                stat = (loadings_z ** 2).sum(axis=0)
                pvals = 1 - chi2.cdf(stat, df=k_pcs)

            adapt_df = pd.DataFrame({
                "Marker": geno.columns,
                "Statistic": stat,
                "P_value": pvals,
                "NegLog10P": -np.log10(np.clip(pvals, 1e-300, 1)),
            })

            if marker_info is not None:
                adapt_df = adapt_df.merge(marker_info, on="Marker", how="left")

            thr = np.percentile(adapt_df["Statistic"], 100 - pct_out)
            adapt_df["Outlier"] = adapt_df["Statistic"] >= thr

            n_out = int(adapt_df["Outlier"].sum())
            st.success(f"✅ Detected **{n_out}** outlier SNPs.")

            fig_pca_out = px.histogram(adapt_df, x="Statistic", nbins=60,
                                          title="PCAdapt statistic distribution")
            fig_pca_out.add_vline(x=thr, line_dash="dash", line_color="red")
            fig_pca_out.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig_pca_out, use_container_width=True)

            if "Chrom" in adapt_df.columns:
                ad_plot = adapt_df.dropna(subset=["Chrom"]).copy()
                ad_plot["Chrom"] = ad_plot["Chrom"].astype(str)
                ad_plot["Order"] = np.arange(len(ad_plot))
                fig_ad_man = px.scatter(
                    ad_plot, x="Order", y="NegLog10P", color="Chrom",
                    title="PCAdapt Manhattan plot",
                    hover_data=["Marker", "Pos"] if "Pos" in ad_plot.columns else ["Marker"],
                )
                fig_ad_man.add_hline(y=-np.log10(0.001), line_dash="dash",
                                       line_color="red")
                fig_ad_man.update_layout(template="plotly_white", height=500,
                                           showlegend=False)
                st.plotly_chart(fig_ad_man, use_container_width=True)

            st.dataframe(adapt_df[adapt_df["Outlier"]].sort_values(
                "Statistic", ascending=False).head(50),
                use_container_width=True)
            download_dataframe(adapt_df, "pcadapt_outliers.csv",
                                key="dl_sel_pca")

            st.session_state["pcadapt_outlier_df"] = adapt_df

    # ═══════════════════════════════════════════
    # METHOD 2: Extended ROH (NO METADATA REQUIRED)
    # ═══════════════════════════════════════════
    elif "Extended ROH" in method_sel:
        st.info("✅ This method works **without metadata**.")
        st.write(
            "Advanced ROH detection with **FROH** (fraction of genome in ROH), "
            "**length classes**, and **total ROH per sample**."
        )

        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            win_size = st.slider("Window size (markers)", 10, 500, 50, key="sel_roh_ws")
        with rc2:
            min_het = st.slider("Max heterozygosity", 0.0, 0.5, 0.05, 0.01, key="sel_roh_h")
        with rc3:
            min_len = st.slider("Min ROH length (markers)", 5, 100, 20, key="sel_roh_ml")

        rc4, rc5 = st.columns(2)
        with rc4:
            merge_gap = st.slider("Merge ROH within gap (markers)",
                                    0, 20, 5, key="sel_roh_mg")
        with rc5:
            classify_lengths = st.checkbox("Classify by length", True,
                                             key="sel_roh_cl")

        if st.button("🚀 Scan Extended ROH", key="sel_roh_run"):
            with st.spinner("Detecting ROH segments..."):
                G = geno.fillna(1).values
                n_samples, n_markers = G.shape

                all_roh = []

                for si in range(n_samples):
                    sample_name = geno.index[si]
                    genotype = G[si]

                    het_by_pos = np.array([
                        np.mean(genotype[i:i+win_size] == 1)
                        for i in range(0, n_markers - win_size + 1)
                    ])

                    is_low = het_by_pos <= min_het

                    runs = []
                    start = None
                    for i, low in enumerate(is_low):
                        if low and start is None:
                            start = i
                        elif not low and start is not None:
                            runs.append((start, i - 1 + win_size - 1))
                            start = None
                    if start is not None:
                        runs.append((start, len(is_low) - 1 + win_size - 1))

                    if runs:
                        merged = [runs[0]]
                        for s, e in runs[1:]:
                            if s - merged[-1][1] <= merge_gap:
                                merged[-1] = (merged[-1][0], e)
                            else:
                                merged.append((s, e))
                        runs = merged

                    for s, e in runs:
                        length_markers = e - s + 1
                        if length_markers >= min_len:
                            roh_entry = {
                                "Sample": sample_name,
                                "Start_marker_idx": s,
                                "End_marker_idx": e,
                                "N_markers": length_markers,
                                "Het_proportion": np.mean(genotype[s:e+1] == 1),
                            }

                            if marker_info is not None and "Pos" in marker_info.columns:
                                positions = marker_info["Pos"].values
                                if s < len(positions) and e < len(positions):
                                    try:
                                        pos_s = float(positions[s])
                                        pos_e = float(positions[e])
                                        roh_entry["Start_pos"] = pos_s
                                        roh_entry["End_pos"] = pos_e
                                        roh_entry["Length_bp"] = pos_e - pos_s
                                    except (ValueError, TypeError):
                                        pass

                            all_roh.append(roh_entry)

            roh_df = pd.DataFrame(all_roh)

            if len(roh_df) == 0:
                st.warning("No ROH segments detected.")
            else:
                st.success(f"✅ Detected **{len(roh_df)}** ROH segments "
                            f"across {roh_df['Sample'].nunique()} samples.")

                summary_cols = st.columns(4)
                summary_cols[0].metric("Total ROH", len(roh_df))
                summary_cols[1].metric("Samples with ROH", roh_df["Sample"].nunique())
                summary_cols[2].metric("Mean N markers",
                                         f"{roh_df['N_markers'].mean():.1f}")
                if "Length_bp" in roh_df.columns:
                    summary_cols[3].metric("Mean length (bp)",
                                             f"{roh_df['Length_bp'].mean():,.0f}")

                st.subheader("📊 F_ROH — Fraction of Genome in ROH")
                per_sample = roh_df.groupby("Sample").agg({
                    "N_markers": ["sum", "count", "mean"],
                }).reset_index()
                per_sample.columns = ["Sample", "Total_ROH_markers",
                                        "N_ROH_segments", "Mean_ROH_length"]
                per_sample["FROH"] = per_sample["Total_ROH_markers"] / n_markers

                st.dataframe(per_sample.style.format({
                    "FROH": "{:.4f}",
                    "Mean_ROH_length": "{:.1f}",
                }), use_container_width=True)

                fig_froh = px.histogram(
                    per_sample, x="FROH", nbins=30,
                    title="FROH distribution across samples")
                fig_froh.update_layout(template="plotly_white", height=400)
                st.plotly_chart(fig_froh, use_container_width=True)

                fig_froh_bar = px.bar(
                    per_sample.sort_values("FROH", ascending=False).head(50),
                    x="Sample", y="FROH",
                    title="Top 50 samples by FROH",
                    color="FROH", color_continuous_scale="Reds",
                )
                fig_froh_bar.update_layout(template="plotly_white", height=500,
                                              xaxis_tickangle=90)
                st.plotly_chart(fig_froh_bar, use_container_width=True)

                if classify_lengths:
                    st.subheader("📏 ROH Length Classes")
                    if "Length_bp" in roh_df.columns:
                        def _class(l):
                            if l < 500_000:
                                return "Short (<0.5 Mb)"
                            elif l < 1_500_000:
                                return "Medium (0.5-1.5 Mb)"
                            elif l < 5_000_000:
                                return "Long (1.5-5 Mb)"
                            else:
                                return "Very long (>5 Mb)"
                        roh_df["Length_class"] = roh_df["Length_bp"].apply(_class)
                    else:
                        def _class_m(n):
                            if n < 30:
                                return "Short (<30 markers)"
                            elif n < 100:
                                return "Medium (30-100)"
                            elif n < 300:
                                return "Long (100-300)"
                            else:
                                return "Very long (>300)"
                        roh_df["Length_class"] = roh_df["N_markers"].apply(_class_m)

                    class_counts = roh_df["Length_class"].value_counts().reset_index()
                    class_counts.columns = ["Length_class", "Count"]

                    fig_class = px.pie(
                        class_counts, values="Count", names="Length_class",
                        title="ROH length class distribution")
                    fig_class.update_layout(template="plotly_white", height=450)
                    st.plotly_chart(fig_class, use_container_width=True)

                st.subheader("N ROH segments per sample")
                per_sample_bar = per_sample.sort_values(
                    "N_ROH_segments", ascending=False).head(50)
                fig_ns = px.bar(per_sample_bar, x="Sample",
                                 y="N_ROH_segments",
                                 title="Number of ROH segments (top 50)")
                fig_ns.update_layout(template="plotly_white", height=450,
                                       xaxis_tickangle=90)
                st.plotly_chart(fig_ns, use_container_width=True)

                st.dataframe(roh_df.head(200), use_container_width=True)
                download_dataframe(roh_df, "roh_segments.csv", key="dl_sel_roh")
                download_dataframe(per_sample, "roh_per_sample.csv",
                                    key="dl_sel_roh_ps")

    # ═══════════════════════════════════════════
    # METHOD 3: Tajima's D (NO METADATA REQUIRED)
    # ═══════════════════════════════════════════
    elif "Tajima's D" in method_sel:
        st.info("✅ This method works **without metadata**.")
        st.write(
            "**Tajima's D** — neutrality test. Values ≠ 0 suggest departure "
            "from neutrality."
        )
        st.info(
            "• **D < -2**: recent selective sweep or population expansion\n"
            "• **D > +2**: balancing selection or population contraction\n"
            "• **D ≈ 0**: neutral evolution"
        )

        td1, td2 = st.columns(2)
        with td1:
            win_td = st.slider("Window size (markers)", 20, 500, 100, 10, key="td_win")
        with td2:
            step_td = st.slider("Step size (markers)", 5, 100, 25, 5, key="td_step")

        if st.button("🚀 Compute Tajima's D", key="td_run"):
            with st.spinner("Computing Tajima's D windows..."):
                G = geno.fillna(1).values.astype(float)
                n_samples, n_markers = G.shape

                def tajimas_d(G_win):
                    n = G_win.shape[0]
                    if n < 4:
                        return np.nan

                    p = np.mean(G_win, axis=0) / 2
                    seg = np.sum((p > 0) & (p < 1))
                    if seg < 3:
                        return np.nan

                    a1 = np.sum(1 / np.arange(1, n))
                    theta_w = seg / a1

                    pi = 0
                    for j in range(G_win.shape[1]):
                        pj = p[j]
                        if 0 < pj < 1:
                            pi += 2 * pj * (1 - pj) * n / (n - 1)
                    pi = pi / G_win.shape[1] if G_win.shape[1] > 0 else 0

                    a2 = np.sum(1 / np.arange(1, n) ** 2)
                    b1 = (n + 1) / (3 * (n - 1))
                    b2 = 2 * (n ** 2 + n + 3) / (9 * n * (n - 1))
                    c1 = b1 - 1 / a1
                    c2 = b2 - (n + 2) / (a1 * n) + a2 / a1 ** 2
                    e1 = c1 / a1
                    e2 = c2 / (a1 ** 2 + a2)

                    var_d = e1 * seg + e2 * seg * (seg - 1)
                    if var_d <= 0:
                        return np.nan

                    D = (pi - theta_w) / np.sqrt(var_d)
                    return D

                windows = []
                for start in range(0, n_markers - win_td + 1, step_td):
                    end = start + win_td
                    G_win = G[:, start:end]
                    D = tajimas_d(G_win)

                    entry = {
                        "Window_start": start,
                        "Window_end": end,
                        "Tajimas_D": D,
                    }

                    if marker_info is not None and "Pos" in marker_info.columns and "Chrom" in marker_info.columns:
                        if end - 1 < len(marker_info):
                            entry["Chrom"] = str(marker_info["Chrom"].iloc[start])
                            entry["Start_pos"] = float(marker_info["Pos"].iloc[start])
                            entry["End_pos"] = float(marker_info["Pos"].iloc[end - 1])
                            entry["Mid_pos"] = (entry["Start_pos"] + entry["End_pos"]) / 2

                    windows.append(entry)

            td_df = pd.DataFrame(windows).dropna(subset=["Tajimas_D"])

            if len(td_df) == 0:
                st.warning("No valid windows.")
            else:
                st.success(f"✅ Computed Tajima's D for **{len(td_df)}** windows.")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Mean D", f"{td_df['Tajimas_D'].mean():.4f}")
                m2.metric("Windows D < -2",
                            f"{(td_df['Tajimas_D'] < -2).sum()}")
                m3.metric("Windows D > +2",
                            f"{(td_df['Tajimas_D'] > 2).sum()}")
                m4.metric("Windows |D| > 1",
                            f"{(td_df['Tajimas_D'].abs() > 1).sum()}")

                if "Chrom" in td_df.columns:
                    td_df["Chrom"] = td_df["Chrom"].astype(str)
                    fig_td = px.scatter(
                        td_df, x="Mid_pos" if "Mid_pos" in td_df.columns
                        else "Window_start",
                        y="Tajimas_D", color="Chrom",
                        title="Tajima's D across genome",
                        hover_data=[c for c in ["Chrom", "Start_pos", "End_pos"]
                                     if c in td_df.columns],
                    )
                else:
                    fig_td = px.scatter(
                        td_df, x="Window_start", y="Tajimas_D",
                        title="Tajima's D across windows")

                fig_td.add_hline(y=0, line_dash="dot", line_color="black")
                fig_td.add_hline(y=-2, line_dash="dash", line_color="red",
                                   annotation_text="D=-2 (sweep)")
                fig_td.add_hline(y=2, line_dash="dash", line_color="blue",
                                   annotation_text="D=+2 (balancing)")
                fig_td.update_layout(template="plotly_white", height=500,
                                       showlegend=False)
                st.plotly_chart(fig_td, use_container_width=True)

                fig_td_dist = px.histogram(
                    td_df, x="Tajimas_D", nbins=50,
                    title="Distribution of Tajima's D values")
                fig_td_dist.add_vline(x=0, line_dash="dot", line_color="black")
                fig_td_dist.add_vline(x=-2, line_dash="dash", line_color="red")
                fig_td_dist.add_vline(x=2, line_dash="dash", line_color="blue")
                fig_td_dist.update_layout(template="plotly_white", height=400)
                st.plotly_chart(fig_td_dist, use_container_width=True)

                st.subheader("Extreme windows")
                extreme = td_df[td_df["Tajimas_D"].abs() > 2].sort_values(
                    "Tajimas_D")
                st.dataframe(extreme, use_container_width=True)
                download_dataframe(td_df, "tajimas_d_windows.csv",
                                    key="dl_td")

                st.session_state["tajimas_d_df"] = td_df

    # ═══════════════════════════════════════════
    # METHOD 4: iHS-like (NO METADATA REQUIRED)
    # ═══════════════════════════════════════════
    elif "iHS-like" in method_sel:
        st.info("✅ This method works **without metadata**.")
        st.write(
            "**iHS-like** — measures extended homozygosity around focal SNPs. "
            "Extreme values suggest recent positive selection."
        )

        ih1, ih2 = st.columns(2)
        with ih1:
            window_flank = st.slider("Flanking window (markers)", 5, 100, 25,
                                       key="ihs_win")
        with ih2:
            top_pct_ihs = st.slider("Top % as outliers", 1, 20, 5, key="ihs_pct")

        if st.button("🚀 Compute iHS-like", key="ihs_run"):
            with st.spinner("Computing extended homozygosity..."):
                G = geno.fillna(1).values.astype(float)
                n_samples, n_markers = G.shape

                ihs_scores = []
                for m in range(n_markers):
                    start = max(0, m - window_flank)
                    end = min(n_markers, m + window_flank + 1)
                    window = G[:, start:end]

                    hom_freq = np.mean((window == 0) | (window == 2))

                    p = G[:, m].mean() / 2
                    if p < 0.05 or p > 0.95:
                        ihs_scores.append(np.nan)
                        continue

                    if 0 < hom_freq < 1:
                        ihs = np.log(hom_freq / (1 - hom_freq))
                    else:
                        ihs = np.nan

                    ihs_scores.append(ihs)

                ihs_scores = np.array(ihs_scores)
                valid = ~np.isnan(ihs_scores)
                if valid.sum() > 0:
                    ihs_std = np.copy(ihs_scores)
                    ihs_std[valid] = (ihs_scores[valid] - ihs_scores[valid].mean()) / \
                                      ihs_scores[valid].std()
                else:
                    ihs_std = ihs_scores

            ihs_df = pd.DataFrame({
                "Marker": geno.columns,
                "iHS_like": ihs_std,
                "Abs_iHS": np.abs(ihs_std),
            })

            if marker_info is not None:
                ihs_df = ihs_df.merge(marker_info, on="Marker", how="left")

            ihs_df = ihs_df.dropna(subset=["iHS_like"])
            thr_ihs = np.percentile(ihs_df["Abs_iHS"], 100 - top_pct_ihs)
            ihs_df["Outlier"] = ihs_df["Abs_iHS"] >= thr_ihs

            n_ihs = int(ihs_df["Outlier"].sum())
            st.success(f"✅ Detected **{n_ihs}** iHS-like outliers.")

            fig_ihs = px.histogram(ihs_df, x="iHS_like", nbins=60,
                                     title="iHS-like distribution")
            fig_ihs.add_vline(x=-thr_ihs, line_dash="dash", line_color="red")
            fig_ihs.add_vline(x=thr_ihs, line_dash="dash", line_color="red")
            fig_ihs.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig_ihs, use_container_width=True)

            if "Chrom" in ihs_df.columns:
                ihs_plot = ihs_df.copy()
                ihs_plot["Chrom"] = ihs_plot["Chrom"].astype(str)
                ihs_plot["Order"] = np.arange(len(ihs_plot))
                fig_ihs_man = px.scatter(
                    ihs_plot, x="Order", y="Abs_iHS", color="Chrom",
                    title="iHS-like Manhattan plot",
                    hover_data=["Marker", "Pos"] if "Pos" in ihs_plot.columns else ["Marker"],
                )
                fig_ihs_man.add_hline(y=thr_ihs, line_dash="dash",
                                        line_color="red")
                fig_ihs_man.update_layout(template="plotly_white", height=500,
                                            showlegend=False)
                st.plotly_chart(fig_ihs_man, use_container_width=True)

            st.dataframe(ihs_df[ihs_df["Outlier"]].sort_values(
                "Abs_iHS", ascending=False).head(50),
                use_container_width=True)
            download_dataframe(ihs_df, "ihs_outliers.csv", key="dl_ihs")

            st.session_state["ihs_outlier_df"] = ihs_df

    # ═══════════════════════════════════════════
    # METHOD 5: Selective Sweep CLR (NO METADATA REQUIRED)
    # ═══════════════════════════════════════════
    elif "Selective Sweep" in method_sel:
        st.info("✅ This method works **without metadata**.")
        st.write(
            "**CLR-like sweep detection** — sliding-window statistic based on "
            "the deviation of allele frequency spectrum from neutral expectation."
        )

        cs1, cs2 = st.columns(2)
        with cs1:
            win_cs = st.slider("Window size (markers)", 20, 500, 100, 10,
                                 key="cs_win")
        with cs2:
            step_cs = st.slider("Step (markers)", 5, 100, 25, 5, key="cs_step")

        if st.button("🚀 Detect selective sweeps", key="cs_run"):
            with st.spinner("Scanning for selective sweeps..."):
                G = geno.fillna(1).values.astype(float)
                n_samples, n_markers = G.shape

                sweep_scores = []

                for start in range(0, n_markers - win_cs + 1, step_cs):
                    end = start + win_cs
                    G_win = G[:, start:end]

                    p_arr = np.mean(G_win, axis=0) / 2
                    maf_arr = np.minimum(p_arr, 1 - p_arr)

                    n_rare = np.sum(maf_arr < 0.1)
                    n_common = np.sum(maf_arr >= 0.3)

                    expected_ratio = 0.5
                    total_valid = n_rare + n_common
                    if total_valid == 0:
                        sweep_scores.append({
                            "Window_start": start,
                            "Window_end": end,
                            "CLR_stat": np.nan,
                        })
                        continue

                    obs_ratio = n_rare / total_valid

                    if 0 < obs_ratio < 1:
                        clr = 2 * total_valid * (
                            obs_ratio * np.log(obs_ratio / expected_ratio) +
                            (1 - obs_ratio) * np.log((1 - obs_ratio) / (1 - expected_ratio))
                        )
                    else:
                        clr = np.nan

                    entry = {
                        "Window_start": start,
                        "Window_end": end,
                        "CLR_stat": clr,
                        "N_rare": n_rare,
                        "N_common": n_common,
                    }

                    if marker_info is not None and "Chrom" in marker_info.columns:
                        if end - 1 < len(marker_info):
                            entry["Chrom"] = str(marker_info["Chrom"].iloc[start])
                            if "Pos" in marker_info.columns:
                                entry["Start_pos"] = float(marker_info["Pos"].iloc[start])
                                entry["End_pos"] = float(marker_info["Pos"].iloc[end - 1])
                                entry["Mid_pos"] = (entry["Start_pos"] + entry["End_pos"]) / 2

                    sweep_scores.append(entry)

                sweep_df = pd.DataFrame(sweep_scores).dropna(subset=["CLR_stat"])

            if len(sweep_df) == 0:
                st.warning("No valid windows.")
            else:
                thr_cs = np.percentile(sweep_df["CLR_stat"], 95)
                sweep_df["Sweep_candidate"] = sweep_df["CLR_stat"] >= thr_cs

                n_sweep = int(sweep_df["Sweep_candidate"].sum())
                st.success(f"✅ Detected **{n_sweep}** candidate sweep windows.")

                if "Chrom" in sweep_df.columns:
                    fig_cs = px.scatter(
                        sweep_df,
                        x="Mid_pos" if "Mid_pos" in sweep_df.columns else "Window_start",
                        y="CLR_stat", color="Chrom",
                        title="CLR-like sweep statistic across genome",
                        hover_data=["Window_start", "N_rare", "N_common"],
                    )
                else:
                    fig_cs = px.scatter(sweep_df, x="Window_start", y="CLR_stat",
                                          title="CLR-like sweep statistic")

                fig_cs.add_hline(y=thr_cs, line_dash="dash", line_color="red",
                                   annotation_text=f"95th percentile ({thr_cs:.2f})")
                fig_cs.update_layout(template="plotly_white", height=500,
                                       showlegend=False)
                st.plotly_chart(fig_cs, use_container_width=True)

                fig_cs_dist = px.histogram(sweep_df, x="CLR_stat", nbins=50,
                                              title="CLR distribution")
                fig_cs_dist.add_vline(x=thr_cs, line_dash="dash",
                                        line_color="red")
                fig_cs_dist.update_layout(template="plotly_white", height=400)
                st.plotly_chart(fig_cs_dist, use_container_width=True)

                st.subheader("Top candidate sweep windows")
                st.dataframe(sweep_df[sweep_df["Sweep_candidate"]].sort_values(
                    "CLR_stat", ascending=False).head(50),
                    use_container_width=True)
                download_dataframe(sweep_df, "selective_sweeps.csv",
                                    key="dl_cs")

    # ═══════════════════════════════════════════
    # METHOD 6: Fst outlier (REQUIRES METADATA)
    # ═══════════════════════════════════════════
    elif "Fst outlier" in method_sel:
        st.warning("⚠️ This method **requires metadata** with a population column.")
        if meta is None:
            st.error("Metadata required.")
        else:
            sc1, sc2 = st.columns(2)
            with sc1:
                pop_sel = st.selectbox("Population column",
                                        meta.columns.tolist(),
                                        index=meta.columns.tolist().index(global_pop_col)
                                        if global_pop_col in meta.columns.tolist() else 0,
                                        key="sel_pop")
            with sc2:
                sam_sel = st.selectbox("Sample ID column",
                                        meta.columns.tolist(),
                                        index=meta.columns.tolist().index(global_sam_col)
                                        if global_sam_col in meta.columns.tolist() else 0,
                                        key="sel_sam")

            fst_pct = st.slider("Top % as outliers", 1, 20, 5, key="sel_fst_pct")

            if st.button("🚀 Detect Fst outliers", key="sel_fst_run"):
                pop_map_s = dict(zip(meta[sam_sel].astype(str),
                                       meta[pop_sel].astype(str)))
                pops_here = sorted(set(pop_map_s.values()))

                with st.spinner("Computing per-marker Fst..."):
                    Ht_marker = calc_het_exp(geno)
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

                if marker_info is not None:
                    fst_df = fst_df.merge(marker_info, on="Marker", how="left")

                fst_df = fst_df.dropna(subset=["Fst"])
                thresh = np.percentile(fst_df["Fst"], 100 - fst_pct)
                fst_df["Outlier"] = fst_df["Fst"] >= thresh

                n_outl = int(fst_df["Outlier"].sum())
                st.success(f"✅ Detected **{n_outl}** outlier SNPs "
                            f"(Fst ≥ {thresh:.4f}).")

                if "Chrom" in fst_df.columns:
                    fst_df_plot = fst_df.copy()
                    fst_df_plot["Chrom"] = fst_df_plot["Chrom"].astype(str)
                    chr_order = sorted(fst_df_plot["Chrom"].unique())
                    fst_df_plot["Chrom"] = pd.Categorical(
                        fst_df_plot["Chrom"], categories=chr_order, ordered=True)
                    if "Pos" in fst_df_plot.columns:
                        fst_df_plot = fst_df_plot.sort_values(["Chrom", "Pos"])
                    fst_df_plot["Order"] = np.arange(len(fst_df_plot))

                    hover_cols = ["Marker"]
                    if "Pos" in fst_df_plot.columns:
                        hover_cols.append("Pos")

                    fig_man = px.scatter(
                        fst_df_plot, x="Order", y="Fst", color="Chrom",
                        title="Per-marker Fst (Manhattan-style)",
                        hover_data=hover_cols,
                    )
                    fig_man.add_hline(y=thresh, line_dash="dash",
                                       line_color="red",
                                       annotation_text=f"Outlier ({fst_pct}%)")
                    fig_man.update_layout(template="plotly_white",
                                            height=550, showlegend=False)
                    st.plotly_chart(fig_man, use_container_width=True)

                fig_fst_dist = px.histogram(
                    fst_df, x="Fst", nbins=60,
                    title="Fst distribution across markers")
                fig_fst_dist.add_vline(x=thresh, line_dash="dash", line_color="red")
                fig_fst_dist.update_layout(template="plotly_white", height=400)
                st.plotly_chart(fig_fst_dist, use_container_width=True)

                st.subheader("Top outlier SNPs")
                st.dataframe(fst_df[fst_df["Outlier"]].sort_values(
                    "Fst", ascending=False).head(50),
                    use_container_width=True)
                download_dataframe(fst_df, "fst_outliers.csv", key="dl_sel_fst")

                st.session_state["fst_outlier_df"] = fst_df

    # ═══════════════════════════════════════════
    # METHOD 7: BayeScan-like (REQUIRES METADATA)
    # ═══════════════════════════════════════════
    elif "BayeScan-like" in method_sel:
        st.warning("⚠️ This method **requires metadata** with a population column.")
        if meta is None:
            st.error("Metadata required.")
        else:
            bs1, bs2 = st.columns(2)
            with bs1:
                pop_bs = st.selectbox("Population column",
                                       meta.columns.tolist(),
                                       index=meta.columns.tolist().index(global_pop_col)
                                       if global_pop_col in meta.columns.tolist() else 0,
                                       key="bs_pop")
            with bs2:
                sam_bs = st.selectbox("Sample ID column",
                                       meta.columns.tolist(),
                                       index=meta.columns.tolist().index(global_sam_col)
                                       if global_sam_col in meta.columns.tolist() else 0,
                                       key="bs_sam")

            pct_bs = st.slider("Top % as outliers", 1, 20, 5, key="bs_pct")

            if st.button("🚀 Run BayeScan-like", key="bs_run"):
                pop_map = dict(zip(meta[sam_bs].astype(str),
                                    meta[pop_bs].astype(str)))
                pops = sorted(set(pop_map.values()))

                with st.spinner("Computing locus-specific Fst deviations..."):
                    Ht = calc_het_exp(geno)
                    hs_list = []
                    for p in pops:
                        samples_p = [s for s in geno.index
                                       if pop_map.get(str(s)) == p]
                        if len(samples_p) < 2:
                            continue
                        Hs_p = calc_het_exp(geno.loc[samples_p])
                        hs_list.append(Hs_p)

                    if len(hs_list) < 2:
                        st.error("Need ≥ 2 populations.")
                        st.stop()

                    Hs = pd.concat(hs_list, axis=1).mean(axis=1)
                    Fst_marker = (Ht - Hs) / Ht.replace(0, np.nan)

                    global_mean_fst = Fst_marker.mean()

                    alpha = np.log(Fst_marker / (1 - Fst_marker + 1e-10)) - \
                             np.log(global_mean_fst / (1 - global_mean_fst + 1e-10))

                    alpha_abs = np.abs(alpha)
                    posterior_prob = alpha_abs / (1 + alpha_abs)

                bs_df = pd.DataFrame({
                    "Marker": geno.columns,
                    "Fst": Fst_marker.values,
                    "Alpha": alpha.values,
                    "Posterior_prob": posterior_prob.values,
                })

                if marker_info is not None:
                    bs_df = bs_df.merge(marker_info, on="Marker", how="left")

                bs_df = bs_df.dropna(subset=["Alpha"])
                thr_bs = np.percentile(bs_df["Posterior_prob"], 100 - pct_bs)
                bs_df["Outlier"] = bs_df["Posterior_prob"] >= thr_bs

                bs_df["Selection_type"] = "Neutral"
                bs_df.loc[(bs_df["Outlier"]) & (bs_df["Alpha"] > 0), "Selection_type"] = "Positive (divergent)"
                bs_df.loc[(bs_df["Outlier"]) & (bs_df["Alpha"] < 0), "Selection_type"] = "Balancing"

                n_bs = int(bs_df["Outlier"].sum())
                n_pos = int((bs_df["Selection_type"] == "Positive (divergent)").sum())
                n_bal = int((bs_df["Selection_type"] == "Balancing").sum())

                st.success(f"✅ Detected **{n_bs}** candidate SNPs")
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Total outliers", n_bs)
                mc2.metric("Positive selection", n_pos)
                mc3.metric("Balancing selection", n_bal)

                fig_bs = px.scatter(
                    bs_df, x="Fst", y="Alpha", color="Selection_type",
                    title="BayeScan-like plot: α vs Fst",
                    hover_data=["Marker"] +
                                (["Chrom", "Pos"] if "Chrom" in bs_df.columns else []),
                    color_discrete_map={
                        "Neutral": "lightgray",
                        "Positive (divergent)": "red",
                        "Balancing": "blue",
                    },
                )
                fig_bs.add_hline(y=0, line_dash="dot", line_color="black")
                fig_bs.update_layout(template="plotly_white", height=550)
                st.plotly_chart(fig_bs, use_container_width=True)

                st.dataframe(bs_df[bs_df["Outlier"]].sort_values(
                    "Posterior_prob", ascending=False).head(50),
                    use_container_width=True)
                download_dataframe(bs_df, "bayescan_like_outliers.csv",
                                    key="dl_bs")

                st.session_state["bayescan_outlier_df"] = bs_df

    # ═══════════════════════════════════════════
    # METHOD 8: LFMM-like (REQUIRES METADATA)
    # ═══════════════════════════════════════════
    elif "LFMM-like" in method_sel:
        st.warning("⚠️ This method **requires metadata** with an environmental variable.")
        if meta is None:
            st.error("Metadata required.")
        else:
            numeric_env = meta.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_env:
                st.error("No numeric columns in metadata for environmental variable.")
                st.stop()

            lc1, lc2, lc3 = st.columns(3)
            with lc1:
                env_col = st.selectbox(
                    "Environmental variable",
                    numeric_env,
                    key="lfmm_env")
            with lc2:
                sam_lfmm = st.selectbox(
                    "Sample ID column",
                    meta.columns.tolist(),
                    index=meta.columns.tolist().index(global_sam_col)
                    if global_sam_col in meta.columns.tolist() else 0,
                    key="lfmm_sam")
            with lc3:
                n_latent = st.slider("Latent factors (K)", 1, 10, 3,
                                       key="lfmm_k")

            pct_lfmm = st.slider("Top % as outliers", 1, 20, 5, key="lfmm_pct")

            if st.button("🚀 Run LFMM-like", key="lfmm_run"):
                env_map = dict(zip(meta[sam_lfmm].astype(str),
                                    meta[env_col].astype(float)))
                samples_use = [s for s in geno.index.astype(str)
                                if s in env_map and pd.notna(env_map.get(s))]

                X_geno = geno.loc[samples_use].values
                y_env = np.array([env_map[s] for s in samples_use])

                with st.spinner("Fitting latent factor model..."):
                    X_imp = impute_missing(pd.DataFrame(X_geno,
                                                          columns=geno.columns),
                                            "mean").values
                    X_sc = StandardScaler().fit_transform(X_imp)

                    pca_lf = PCA(n_components=n_latent)
                    latent = pca_lf.fit_transform(X_sc)

                    pvals = []
                    betas = []
                    for j in range(X_sc.shape[1]):
                        try:
                            snp = X_sc[:, j].reshape(-1, 1)
                            covariates = np.hstack([snp, latent])
                            reg = LinearRegression()
                            reg.fit(covariates, y_env)
                            beta = reg.coef_[0]

                            y_pred = reg.predict(covariates)
                            residuals = y_env - y_pred
                            n_obs = len(y_env)
                            k = covariates.shape[1]
                            rss = np.sum(residuals ** 2)
                            se = np.sqrt(rss / (n_obs - k - 1))

                            snp_var = np.var(snp)
                            if snp_var > 0 and se > 0:
                                t_stat = beta / (se / np.sqrt(snp_var * n_obs))
                                pval = 2 * (1 - chi2.cdf(t_stat ** 2, df=1))
                            else:
                                pval = 1.0

                            pvals.append(pval)
                            betas.append(beta)
                        except Exception:
                            pvals.append(np.nan)
                            betas.append(np.nan)

                lfmm_df = pd.DataFrame({
                    "Marker": geno.columns,
                    "Beta": betas,
                    "P_value": pvals,
                    "NegLog10P": -np.log10(np.clip(pvals, 1e-300, 1)),
                })

                if marker_info is not None:
                    lfmm_df = lfmm_df.merge(marker_info, on="Marker", how="left")

                lfmm_df = lfmm_df.dropna(subset=["P_value"])
                thr_lfmm = np.percentile(lfmm_df["NegLog10P"], 100 - pct_lfmm)
                lfmm_df["Outlier"] = lfmm_df["NegLog10P"] >= thr_lfmm

                n_lfmm = int(lfmm_df["Outlier"].sum())
                st.success(f"✅ Detected **{n_lfmm}** SNPs associated with **{env_col}**.")

                if "Chrom" in lfmm_df.columns:
                    lp = lfmm_df.copy()
                    lp["Chrom"] = lp["Chrom"].astype(str)
                    lp["Order"] = np.arange(len(lp))
                    fig_lfmm = px.scatter(
                        lp, x="Order", y="NegLog10P", color="Chrom",
                        title=f"LFMM-like Manhattan plot for {env_col}",
                        hover_data=["Marker", "Pos"] if "Pos" in lp.columns else ["Marker"],
                    )
                    fig_lfmm.add_hline(y=thr_lfmm, line_dash="dash",
                                        line_color="red")
                    fig_lfmm.update_layout(template="plotly_white",
                                             height=550, showlegend=False)
                    st.plotly_chart(fig_lfmm, use_container_width=True)

                fig_lfmm_dist = px.histogram(
                    lfmm_df, x="NegLog10P", nbins=50,
                    title="-log10(p) distribution")
                fig_lfmm_dist.add_vline(x=thr_lfmm, line_dash="dash",
                                          line_color="red")
                fig_lfmm_dist.update_layout(template="plotly_white",
                                              height=400)
                st.plotly_chart(fig_lfmm_dist, use_container_width=True)

                st.dataframe(lfmm_df[lfmm_df["Outlier"]].sort_values(
                    "NegLog10P", ascending=False).head(50),
                    use_container_width=True)
                download_dataframe(lfmm_df, "lfmm_outliers.csv", key="dl_lfmm")

                st.session_state["lfmm_outlier_df"] = lfmm_df

    # ═══════════════════════════════════════════
    # METHOD 9: Combined outlier detection
    # ═══════════════════════════════════════════
    else:  # Combined
        st.write(
            "**Combined outlier detection** — combines results from previously-run "
            "methods to identify robust selection candidates."
        )

        available_methods = []
        if "fst_outlier_df" in st.session_state:
            available_methods.append("Fst")
        if "pcadapt_outlier_df" in st.session_state:
            available_methods.append("PCAdapt")
        if "ihs_outlier_df" in st.session_state:
            available_methods.append("iHS")
        if "bayescan_outlier_df" in st.session_state:
            available_methods.append("BayeScan")
        if "lfmm_outlier_df" in st.session_state:
            available_methods.append("LFMM")

        if len(available_methods) < 2:
            st.warning(
                "⚠️ Please run at least 2 different selection methods first. "
                f"Currently available: {available_methods if available_methods else 'None'}"
            )
        else:
            selected_methods = st.multiselect(
                "Methods to combine",
                available_methods,
                default=available_methods,
                key="comb_methods",
            )

            min_methods = st.slider(
                "Min number of methods for consensus",
                1, len(selected_methods), max(2, len(selected_methods) // 2),
                key="comb_min",
            )

            if st.button("🚀 Combine outliers", key="comb_run"):
                all_outliers = {}
                for m in selected_methods:
                    if m == "Fst":
                        df = st.session_state["fst_outlier_df"]
                    elif m == "PCAdapt":
                        df = st.session_state["pcadapt_outlier_df"]
                    elif m == "iHS":
                        df = st.session_state["ihs_outlier_df"]
                    elif m == "BayeScan":
                        df = st.session_state["bayescan_outlier_df"]
                    elif m == "LFMM":
                        df = st.session_state["lfmm_outlier_df"]

                    outlier_snps = set(df[df["Outlier"]]["Marker"].astype(str))
                    all_outliers[m] = outlier_snps

                all_snps = set()
                for s in all_outliers.values():
                    all_snps |= s

                snp_counts = {snp: 0 for snp in all_snps}
                snp_methods = {snp: [] for snp in all_snps}
                for m, snps in all_outliers.items():
                    for snp in snps:
                        snp_counts[snp] += 1
                        snp_methods[snp].append(m)

                combined_df = pd.DataFrame({
                    "Marker": list(snp_counts.keys()),
                    "N_methods": list(snp_counts.values()),
                    "Methods": [",".join(snp_methods[s])
                                 for s in snp_counts.keys()],
                })
                combined_df["Robust_candidate"] = combined_df["N_methods"] >= min_methods

                if marker_info is not None:
                    combined_df = combined_df.merge(
                        marker_info, on="Marker", how="left")

                combined_df = combined_df.sort_values("N_methods", ascending=False)

                n_robust = int(combined_df["Robust_candidate"].sum())
                st.success(f"✅ **{n_robust}** SNPs detected by ≥ {min_methods} methods.")

                cc1, cc2 = st.columns(2)
                cc1.metric("Total unique outliers", len(combined_df))
                cc2.metric(f"Robust (≥ {min_methods} methods)", n_robust)

                fig_meth = px.histogram(
                    combined_df, x="N_methods",
                    title="Distribution of methods detecting each SNP",
                    nbins=len(selected_methods) + 1,
                )
                fig_meth.update_layout(template="plotly_white", height=400)
                st.plotly_chart(fig_meth, use_container_width=True)

                if len(selected_methods) <= 4:
                    st.markdown("### Method Overlap Matrix")
                    overlap_mat = np.zeros((len(selected_methods),
                                              len(selected_methods)))
                    for i, m1 in enumerate(selected_methods):
                        for j, m2 in enumerate(selected_methods):
                            overlap = all_outliers[m1] & all_outliers[m2]
                            overlap_mat[i, j] = len(overlap)

                    fig_ov = px.imshow(
                        overlap_mat,
                        x=selected_methods, y=selected_methods,
                        text_auto=True, color_continuous_scale="Greens",
                        title="Number of SNPs detected by each method pair")
                    fig_ov.update_layout(template="plotly_white", height=500)
                    st.plotly_chart(fig_ov, use_container_width=True)

                st.subheader("Robust selection candidates")
                st.dataframe(combined_df[combined_df["Robust_candidate"]].head(100),
                              use_container_width=True)
                download_dataframe(combined_df,
                                    "combined_selection_outliers.csv",
                                    key="dl_comb")


# =========================================================
# TAB 5 — UMAP (NO METADATA REQUIRED)
# =========================================================
with tab_um:
    st.subheader("🗺️ UMAP Non-linear Dimensionality Reduction")
    st.info("✅ **This module works without metadata.** Metadata is only used "
             "for optional coloring.")
    st.write("Visualize sample structure with UMAP. Often reveals sub-structure "
              "invisible to PCA.")

    c1, c2, c3 = st.columns(3)
    with c1:
        nn = st.slider("N neighbors", 2, 100, 15, key="um_nn",
                        help="Local vs global structure trade-off")
    with c2:
        md = st.slider("Min distance", 0.0, 1.0, 0.1, 0.01, key="um_md",
                        help="Tightness of clusters")
    with c3:
        nc_um = st.radio("Components", [2, 3], key="um_nc")

    # Metadata coloring is OPTIONAL
    color_col_u = None
    sam_col_u = None

    if meta is not None:
        st.markdown("#### 🎨 Optional: Color by metadata")
        um_c1, um_c2 = st.columns(2)
        with um_c1:
            color_col_u = st.selectbox(
                "Color by (optional)",
                ["None"] + meta.columns.tolist(),
                index=(meta.columns.tolist().index(global_pop_col) + 1
                        if global_pop_col in meta.columns.tolist() else 0),
                key="um_color",
            )
        with um_c2:
            if color_col_u != "None":
                sam_col_u = st.selectbox(
                    "Sample ID column",
                    meta.columns.tolist(),
                    index=meta.columns.tolist().index(global_sam_col)
                    if global_sam_col in meta.columns.tolist() else 0,
                    key="um_sam",
                )

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

        um_df = pd.DataFrame(emb, columns=[f"UMAP_{i+1}" for i in range(nc_um)])
        um_df["Sample"] = geno.index.astype(str)

        # Add color from metadata if selected
        color_arg = None
        if color_col_u and color_col_u != "None" and sam_col_u:
            m_sub = meta[[sam_col_u, color_col_u]].drop_duplicates()
            m_sub[sam_col_u] = m_sub[sam_col_u].astype(str)
            um_df = um_df.merge(m_sub, left_on="Sample",
                                  right_on=sam_col_u, how="left")
            color_arg = color_col_u

        if nc_um == 2:
            fig_um = px.scatter(um_df, x="UMAP_1", y="UMAP_2",
                                 color=color_arg, hover_data=["Sample"],
                                 title="UMAP 2D projection")
            fig_um.update_traces(marker=dict(size=9,
                                               line=dict(width=0.5,
                                                          color="darkslategrey")))
        else:
            fig_um = px.scatter_3d(um_df, x="UMAP_1", y="UMAP_2", z="UMAP_3",
                                    color=color_arg, hover_data=["Sample"],
                                    title="UMAP 3D projection")
            fig_um.update_traces(marker=dict(size=5))

        fig_um.update_layout(template="plotly_white", height=700)
        st.plotly_chart(fig_um, use_container_width=True)

        # UMAP-based clustering (density-based, no metadata needed)
        st.markdown("### 🎯 Optional: Automatic Cluster Detection")
        if st.checkbox("Run DBSCAN clustering on UMAP result",
                        key="um_cluster"):
            eps_um = st.slider("DBSCAN eps", 0.1, 5.0, 0.5, 0.1,
                                key="um_eps")
            min_pts = st.slider("Min samples", 2, 20, 5, key="um_min")

            db = DBSCAN(eps=eps_um, min_samples=min_pts)
            labels = db.fit_predict(emb)
            um_df["Auto_Cluster"] = [f"Cluster_{l}" if l >= 0
                                       else "Noise" for l in labels]

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = int((labels == -1).sum())

            uc1, uc2 = st.columns(2)
            uc1.metric("Auto-detected clusters", n_clusters)
            uc2.metric("Noise points", n_noise)

            if nc_um == 2:
                fig_um_cl = px.scatter(um_df, x="UMAP_1", y="UMAP_2",
                                          color="Auto_Cluster",
                                          hover_data=["Sample"],
                                          title="UMAP with DBSCAN clusters")
                fig_um_cl.update_traces(marker=dict(size=9,
                                                       line=dict(width=0.5,
                                                                  color="darkslategrey")))
            else:
                fig_um_cl = px.scatter_3d(um_df, x="UMAP_1", y="UMAP_2",
                                             z="UMAP_3", color="Auto_Cluster",
                                             hover_data=["Sample"],
                                             title="UMAP 3D with clusters")
                fig_um_cl.update_traces(marker=dict(size=5))

            fig_um_cl.update_layout(template="plotly_white", height=700)
            st.plotly_chart(fig_um_cl, use_container_width=True)

        download_plotly_html(fig_um, "umap_projection.html", key="dl_um_html")
        download_dataframe(um_df, "umap_embedding.csv", key="dl_um_csv")