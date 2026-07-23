from pathlib import Path
import io
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
SAMPLE_DATASETS = {
    "Credit rating sample": DATA_DIR / "sample_credit_rating.csv",
    "Nonlinear customer sample": DATA_DIR / "sample_customer_segments.csv",
}

st.set_page_config(page_title="SVM and LDA", layout="wide")
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.facecolor": "#F8F9FB", "axes.facecolor": "#F8F9FB",
    "axes.edgecolor": "#D0D7DE", "axes.titleweight": "bold",
    "axes.titlesize": 12, "axes.labelsize": 10, "grid.alpha": 0.25,
    "grid.linestyle": "--", "axes.spines.top": False, "axes.spines.right": False,
})
st.markdown("""
<style>
.block-container {padding-top:2rem;padding-bottom:2rem;}
div[data-testid="stMetric"] {background:#F7F9FC;border:1px solid #E2E8F0;padding:14px;border-radius:12px;}
.small-note {font-size:.88rem;color:#475569;line-height:1.35;}
</style>
""", unsafe_allow_html=True)

st.title("Other Classifiers: SVM and LDA")
st.caption(
    "Workflow: upload a classification dataset, select the target and predictors, configure SVM and/or LDA, "
    "then compare holdout performance, decision outputs, and an enriched dataset."
)
[main_tab] = st.tabs(["Batch SVM and LDA"])

# ----------------------------- helpers -----------------------------
def coerce_numeric_like_columns(df: pd.DataFrame, min_ratio: float = 0.8) -> pd.DataFrame:
    converted = df.copy()
    for col in converted.columns:
        series = converted[col]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue
        text = series.astype("string").str.strip().replace({
            "": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA,
            "NULL": pd.NA, "NA": pd.NA, "$null$": pd.NA,
        })
        non_missing = text.notna()
        if non_missing.sum() == 0:
            continue
        if text[non_missing].str.match(r"^[+-]?[0-9\s.,]+$", na=False).mean() < min_ratio:
            continue
        candidates = [
            pd.to_numeric(text, errors="coerce"),
            pd.to_numeric(text.str.replace(",", "", regex=False), errors="coerce"),
            pd.to_numeric(text.str.replace(".", "", regex=False).str.replace(",", ".", regex=False), errors="coerce"),
        ]
        best = max(candidates, key=lambda x: x[non_missing].notna().mean())
        if best[non_missing].notna().mean() >= min_ratio:
            converted[col] = best
    return converted


def read_csv_flexible(uploaded_file) -> pd.DataFrame:
    best_df, best_score, last_error = None, (-1, -1), None
    for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        for separator in [None, ";", ",", "\t", "|"]:
            try:
                uploaded_file.seek(0)
                candidate = pd.read_csv(uploaded_file, sep=separator, engine="python", encoding=encoding)
                score = (candidate.shape[1], candidate.shape[0])
                if score > best_score:
                    best_df, best_score = candidate, score
            except Exception as exc:
                last_error = exc
    if best_df is None:
        raise ValueError(f"Could not parse CSV file. Last error: {last_error}")
    return coerce_numeric_like_columns(best_df)


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
        return coerce_numeric_like_columns(pd.read_excel(uploaded_file))
    return read_csv_flexible(uploaded_file)


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.dropna(how="all").copy()
    unnamed = [c for c in cleaned.columns if str(c).lower().startswith("unnamed")]
    return cleaned.drop(columns=unnamed, errors="ignore")


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def make_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_encoder()),
        ]), categorical))
    if not transformers:
        raise ValueError("No usable predictor columns were found.")
    return ColumnTransformer(transformers, remainder="drop")


def build_model(name: str, X: pd.DataFrame, kernel: str, C: float, gamma, lda_solver: str, shrinkage):
    preprocessor = build_preprocessor(X)
    if name == "SVM":
        estimator = SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=RANDOM_STATE)
    else:
        kwargs = {"solver": lda_solver}
        if lda_solver in {"lsqr", "eigen"}:
            kwargs["shrinkage"] = shrinkage
        estimator = LinearDiscriminantAnalysis(**kwargs)
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def metric_row(name, y_true, y_pred):
    average = "binary" if pd.Series(y_true).nunique() == 2 else "weighted"
    kwargs = {"average": average, "zero_division": 0}
    if average == "binary":
        kwargs["pos_label"] = sorted(pd.Series(y_true).unique(), key=str)[-1]
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, **kwargs),
        "Recall": recall_score(y_true, y_pred, **kwargs),
        "F1 score": f1_score(y_true, y_pred, **kwargs),
    }


def plot_class_balance(y):
    table = pd.Series(y).astype(str).value_counts().reset_index()
    table.columns = ["Class", "Count"]
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=table, x="Class", y="Count", ax=ax)
    ax.set_title("Target class balance")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return fig


def plot_confusion(y_true, y_pred, title):
    labels = sorted(pd.Series(y_true).astype(str).unique())
    cm = confusion_matrix(pd.Series(y_true).astype(str), pd.Series(y_pred).astype(str), labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"Actual {x}" for x in labels], columns=[f"Predicted {x}" for x in labels])
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(title); ax.set_xlabel("Predicted class"); ax.set_ylabel("Actual class")
    fig.tight_layout()
    return fig, cm_df


def plot_metric_comparison(metrics_df):
    long = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.barplot(data=long, x="Metric", y="Score", hue="Model", ax=ax)
    ax.set_ylim(0, 1.05); ax.set_title("Holdout metric comparison")
    fig.tight_layout()
    return fig


def plot_lda_projection(X, y, pipeline, title):
    transformed = pipeline.named_steps["preprocess"].transform(X)
    lda = pipeline.named_steps["model"]
    projected = lda.transform(transformed)
    plot_df = pd.DataFrame({"LD1": projected[:, 0], "Class": pd.Series(y).astype(str).values})
    if projected.shape[1] >= 2:
        plot_df["LD2"] = projected[:, 1]
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(data=plot_df, x="LD1", y="LD2", hue="Class", s=55, alpha=.8, ax=ax)
    else:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.histplot(data=plot_df, x="LD1", hue="Class", element="step", stat="density", common_norm=False, ax=ax)
    ax.set_title(title); fig.tight_layout()
    return fig, plot_df


def plot_two_feature_boundary(df, target, features, model_name, kernel, C, gamma, lda_solver, shrinkage):
    subset = df[features + [target]].dropna().copy()
    if subset[target].nunique() < 2:
        return None
    le = LabelEncoder(); y = le.fit_transform(subset[target].astype(str)); X = subset[features].to_numpy(float)
    scaler = StandardScaler(); Xs = scaler.fit_transform(X)
    if model_name == "SVM":
        estimator = SVC(kernel=kernel, C=C, gamma=gamma).fit(Xs, y)
    else:
        kwargs = {"solver": lda_solver}
        if lda_solver in {"lsqr", "eigen"}: kwargs["shrinkage"] = shrinkage
        estimator = LinearDiscriminantAnalysis(**kwargs).fit(Xs, y)
    x_min, x_max = Xs[:, 0].min() - .7, Xs[:, 0].max() + .7
    y_min, y_max = Xs[:, 1].min() - .7, Xs[:, 1].max() + .7
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
    zz = estimator.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(7, 5.4))
    ax.contourf(xx, yy, zz, alpha=.18, levels=np.arange(len(le.classes_)+1)-.5)
    sns.scatterplot(x=Xs[:,0], y=Xs[:,1], hue=le.inverse_transform(y), s=50, alpha=.85, ax=ax)
    if model_name == "SVM" and hasattr(estimator, "support_vectors_"):
        sv = estimator.support_vectors_
        ax.scatter(sv[:,0], sv[:,1], s=120, facecolors="none", edgecolors="black", linewidths=1.2, label="Support vectors")
    ax.set_title(f"{model_name} decision regions (two selected numeric features)")
    ax.set_xlabel(f"{features[0]} (standardized)"); ax.set_ylabel(f"{features[1]} (standardized)")
    fig.tight_layout(); return fig


# ----------------------------- interface -----------------------------
with main_tab:
    st.subheader("Batch SVM and LDA from file")
    st.caption("Download a sample or upload CSV/XLSX, configure the classifiers, and click Run analysis.")

    upload_col, sample_col, download_col, clear_col = st.columns(
        [5.5, 2.2, 1.4, 1], vertical_alignment="bottom"
    )
    with upload_col:
        uploaded_file = st.file_uploader(
            "Upload File", type=["csv", "xlsx", "xls"], key="svm_lda_upload"
        )
    with sample_col:
        sample_name = st.selectbox(
            "Sample Dataset", list(SAMPLE_DATASETS), key="svm_lda_sample"
        )

    sample_path = SAMPLE_DATASETS[sample_name]
    with download_col:
        if sample_path.exists():
            st.download_button(
                "Download Sample",
                sample_path.read_bytes(),
                file_name=sample_path.name,
                mime="text/csv",
                width="stretch",
            )
        else:
            st.button("Download Sample", disabled=True, width="stretch")

    with clear_col:
        if st.button("Clear", disabled=uploaded_file is None, width="stretch"):
            for key in list(st.session_state.keys()):
                if key.startswith("svm_lda_"):
                    del st.session_state[key]
            st.rerun()

    if uploaded_file is None:
        st.info(
            "Upload a CSV/XLSX file. You can also download one of the sample datasets above "
            "and upload it for practice."
        )
        st.stop()

    try:
        data = clean_frame(read_uploaded_file(uploaded_file))
    except Exception as exc:
        st.error(f"The file could not be read: {exc}")
        st.stop()

    if data.empty or data.shape[1] < 2:
        st.error("The dataset is empty or does not contain enough columns for classification.")
        st.stop()

    # Clear prior results when a different file is uploaded.
    file_signature = (uploaded_file.name, uploaded_file.size, tuple(data.columns), data.shape)
    if st.session_state.get("svm_lda_file_signature") != file_signature:
        st.session_state["svm_lda_file_signature"] = file_signature
        st.session_state.pop("svm_lda_results", None)

    left, right = st.columns([1, 2], gap="large")

    with left:
        st.subheader("Setup")
        columns = data.columns.tolist()
        preferred = ["target", "class", "label", "credit_rating", "outcome", "response"]
        lower = [str(c).lower() for c in columns]
        target_index = next(
            (lower.index(name) for name in preferred if name in lower), len(columns) - 1
        )
        target = st.selectbox("Target column", columns, index=target_index, key="svm_lda_target")
        predictor_options = [column for column in columns if column != target]
        predictors = st.multiselect(
            "Predictor columns",
            predictor_options,
            default=predictor_options,
            key="svm_lda_predictors",
        )
        method = st.selectbox(
            "Method",
            ["Compare SVM and LDA", "SVM only", "LDA only"],
            key="svm_lda_method",
        )

        st.subheader("Parameters")
        test_size = st.slider(
            "Test set share", 0.15, 0.40, 0.25, 0.05, key="svm_lda_test_size"
        )

        with st.expander("Advanced settings"):
            random_seed = st.number_input(
                "Random seed", min_value=0, value=RANDOM_STATE, step=1,
                key="svm_lda_random_seed",
            )
            kernel = st.selectbox(
                "SVM kernel", ["linear", "rbf", "poly", "sigmoid"], index=1,
                key="svm_lda_kernel",
            )
            C = st.number_input(
                "SVM C", min_value=0.001, value=1.0, step=0.5, format="%.3f",
                key="svm_lda_c",
            )
            gamma_choice = st.selectbox(
                "SVM gamma", ["scale", "auto", "custom"], index=0,
                key="svm_lda_gamma_choice",
            )
            gamma_value = st.number_input(
                "Custom gamma",
                min_value=0.0001,
                value=0.1,
                step=0.1,
                format="%.4f",
                disabled=gamma_choice != "custom",
                key="svm_lda_gamma_value",
            )
            lda_solver = st.selectbox(
                "LDA solver", ["svd", "lsqr", "eigen"], index=0,
                key="svm_lda_lda_solver",
            )
            shrinkage_choice = st.selectbox(
                "LDA shrinkage", ["None", "auto"],
                disabled=lda_solver == "svd",
                key="svm_lda_shrinkage",
            )

        run = st.button(
            "Run analysis", type="primary", width="stretch",
            key="svm_lda_run",
        )

        if st.session_state.get("svm_lda_results") is not None:
            st.caption("Results on the right reflect the most recent completed run.")

    # Run only on explicit button click, then save everything needed to render results.
    if run:
        if not predictors:
            st.session_state["svm_lda_run_error"] = "Select at least one predictor column."
        else:
            modeling = data[predictors + [target]].dropna(subset=[target]).copy()
            error = None
            if modeling[target].nunique() < 2:
                error = "The target must contain at least two classes."
            elif modeling[target].nunique() > 30:
                error = (
                    "The selected target has too many unique values for this classroom "
                    "classification app."
                )
            elif modeling[target].value_counts().min() < 2:
                error = "Each class needs at least two records for a stratified train-test split."

            if error:
                st.session_state["svm_lda_run_error"] = error
            else:
                X = modeling[predictors]
                y = modeling[target].astype(str)
                gamma = gamma_value if gamma_choice == "custom" else gamma_choice
                shrinkage = None if shrinkage_choice == "None" else "auto"

                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X,
                        y,
                        test_size=test_size,
                        stratify=y,
                        random_state=int(random_seed),
                    )
                    names = (
                        ["SVM", "LDA"]
                        if method == "Compare SVM and LDA"
                        else [method.replace(" only", "")]
                    )
                    fitted, predictions, rows = {}, {}, []
                    for name in names:
                        pipeline = build_model(
                            name, X_train, kernel, C, gamma, lda_solver, shrinkage
                        )
                        pipeline.fit(X_train, y_train)
                        pred = pipeline.predict(X_test)
                        fitted[name] = pipeline
                        predictions[name] = pred
                        rows.append(metric_row(name, y_test, pred))

                    metrics_df = pd.DataFrame(rows)
                    enriched = modeling.copy()
                    for name in names:
                        slug = name.lower()
                        enriched[f"predicted_{slug}"] = fitted[name].predict(X)
                        try:
                            proba = fitted[name].predict_proba(X)
                            classes = fitted[name].classes_
                            enriched[f"confidence_{slug}"] = proba.max(axis=1)
                            for idx, class_name in enumerate(classes):
                                safe = str(class_name).replace(" ", "_").replace("/", "_")
                                enriched[f"probability_{slug}_{safe}"] = proba[:, idx]
                        except Exception:
                            pass
                        enriched[f"correct_{slug}"] = (
                            enriched[target].astype(str)
                            == enriched[f"predicted_{slug}"].astype(str)
                        )

                    st.session_state["svm_lda_results"] = {
                        "metrics_df": metrics_df,
                        "fitted": fitted,
                        "predictions": predictions,
                        "names": names,
                        "modeling": modeling,
                        "X": X,
                        "y": y,
                        "X_test": X_test,
                        "y_test": y_test,
                        "target": target,
                        "predictors": predictors,
                        "kernel": kernel,
                        "C": C,
                        "gamma": gamma,
                        "lda_solver": lda_solver,
                        "shrinkage": shrinkage,
                        "enriched": enriched,
                    }
                    st.session_state.pop("svm_lda_run_error", None)
                except Exception as exc:
                    st.session_state["svm_lda_run_error"] = (
                        f"The selected method could not be fitted: {exc}"
                    )

    with right:
        run_error = st.session_state.get("svm_lda_run_error")
        results = st.session_state.get("svm_lda_results")

        if run_error:
            st.error(run_error)

        if results is None:
            st.subheader("Dataset preview")
            st.caption(
                f"{data.shape[0]:,} rows · {data.shape[1]:,} columns · "
                f"{data.isna().sum().sum():,} missing values"
            )
            st.dataframe(data.head(25), width="stretch", hide_index=True)
        else:
            metrics_df = results["metrics_df"]
            fitted = results["fitted"]
            predictions = results["predictions"]
            names = results["names"]
            modeling = results["modeling"]
            X = results["X"]
            y = results["y"]
            X_test = results["X_test"]
            y_test = results["y_test"]
            result_target = results["target"]
            result_predictors = results["predictors"]
            result_kernel = results["kernel"]
            result_C = results["C"]
            result_gamma = results["gamma"]
            result_lda_solver = results["lda_solver"]
            result_shrinkage = results["shrinkage"]
            enriched = results["enriched"]

            st.subheader("Results")
            best = metrics_df.sort_values("F1 score", ascending=False).iloc[0]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Best model", best["Model"])
            m2.metric("Accuracy", f"{best['Accuracy']:.3f}")
            m3.metric("Weighted/Binary F1", f"{best['F1 score']:.3f}")
            m4.metric("Test records", f"{len(X_test):,}")

            evaluation_tab, svm_tab, lda_tab, output_tab = st.tabs(
                ["Evaluation", "SVM", "LDA", "Processed output"]
            )

            with evaluation_tab:
                st.caption(
                    "Higher scores are better. For multiclass targets, precision, recall, "
                    "and F1 use weighted averaging."
                )
                st.dataframe(
                    metrics_df.style.format({
                        column: "{:.3f}"
                        for column in ["Accuracy", "Precision", "Recall", "F1 score"]
                    }),
                    width="stretch",
                    hide_index=True,
                )
                st.pyplot(plot_metric_comparison(metrics_df), width="stretch")

                cm_cols = st.columns(len(names))
                for col, name in zip(cm_cols, names):
                    with col:
                        fig, _ = plot_confusion(
                            y_test, predictions[name], f"{name}: confusion matrix"
                        )
                        st.pyplot(fig, width="stretch")
                        st.caption(
                            "Diagonal cells are correct predictions; off-diagonal cells are errors."
                        )

                st.pyplot(plot_class_balance(y), width="stretch")
                st.caption(
                    "Strong imbalance can make accuracy look better than practical performance."
                )

            with svm_tab:
                if "SVM" not in fitted:
                    st.info("SVM was not included in the most recent run.")
                else:
                    svc = fitted["SVM"].named_steps["model"]
                    support_total = int(np.sum(svc.n_support_))
                    s1, s2, s3 = st.columns(3)
                    s1.metric("Kernel", str(result_kernel).upper())
                    s2.metric("Support vectors", f"{support_total:,}")
                    s3.metric("C", f"{result_C:g}")
                    st.caption(
                        f"Support vectors by class: {', '.join(map(str, svc.n_support_))}. "
                        "These observations are closest to the learned margin or boundary."
                    )

                    numeric_selected = [
                        column for column in result_predictors
                        if pd.api.types.is_numeric_dtype(modeling[column])
                    ]
                    if len(numeric_selected) >= 2:
                        pair = st.multiselect(
                            "Two numeric features for the SVM decision-region view",
                            numeric_selected,
                            default=numeric_selected[:2],
                            max_selections=2,
                            key="svm_lda_svm_pair",
                        )
                        if len(pair) == 2:
                            fig = plot_two_feature_boundary(
                                modeling,
                                result_target,
                                pair,
                                "SVM",
                                result_kernel,
                                result_C,
                                result_gamma,
                                result_lda_solver,
                                result_shrinkage,
                            )
                            if fig is not None:
                                st.pyplot(fig, width="stretch")
                                st.caption(
                                    "This chart refits SVM using only the two displayed features. "
                                    "It is an interpretation aid rather than the full fitted model."
                                )
                    else:
                        st.info(
                            "At least two numeric predictors are needed for a two-dimensional "
                            "decision-region chart."
                        )

            with lda_tab:
                if "LDA" not in fitted:
                    st.info("LDA was not included in the most recent run.")
                else:
                    l1, l2 = st.columns(2)
                    l1.metric("Solver", str(result_lda_solver).upper())
                    l2.metric(
                        "Shrinkage",
                        "None" if result_shrinkage is None else str(result_shrinkage),
                    )
                    try:
                        fig, projection = plot_lda_projection(
                            X, y, fitted["LDA"], "LDA supervised projection"
                        )
                        st.pyplot(fig, width="stretch")
                        st.caption(
                            "LDA projects observations toward directions that separate known "
                            "classes. It is supervised and therefore differs from PCA."
                        )
                        with st.expander("Projection preview"):
                            st.dataframe(
                                projection.head(50), width="stretch", hide_index=True
                            )
                    except Exception as exc:
                        st.info(f"LDA projection is not available for this configuration: {exc}")

            with output_tab:
                st.caption(
                    "Predictions below are fitted-model outputs for the complete uploaded dataset. "
                    "Holdout metrics remain the unbiased evaluation view."
                )
                st.dataframe(enriched.head(100), width="stretch", hide_index=True)
                st.download_button(
                    "Download enriched dataset",
                    to_csv_bytes(enriched),
                    file_name="svm_lda_enriched_output.csv",
                    mime="text/csv",
                    width="stretch",
                )

            st.info(
                "Practical cautions: scale SVM inputs inside a pipeline; tune C and gamma with "
                "cross-validation; do not assume a nonlinear kernel is automatically better; "
                "and remember that LDA works best when class distributions are reasonably "
                "Gaussian with similar covariance structure."
            )
