from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
SAMPLE_DATASETS = {
    "Loan approval sample": DATA_DIR / "sample_loan_cross_validation.csv",
    "Subscription renewal sample": DATA_DIR / "sample_subscription_cross_validation.csv",
}

st.set_page_config(page_title="Cross Validation", layout="wide")
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.facecolor": "#F8F9FB",
    "axes.facecolor": "#F8F9FB",
    "axes.edgecolor": "#D0D7DE",
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
div[data-testid="stMetric"] {background:#F7F9FC;border:1px solid #E2E8F0;padding:14px;border-radius:12px;}
.small-note {font-size:0.88rem;color:#475569; line-height:1.35;}
.section-card {background:#FFFFFF;border:1px solid #E5E7EB;border-radius:14px;padding:1rem;margin-bottom:1rem;}
</style>
""", unsafe_allow_html=True)

st.title("Cross Validation")
st.caption("Workflow: load a dataset, select a target and features, then run a train-test and K-Fold Cross Validation comparison.")

[tab_cross_validation] = st.tabs(["Batch cross-validation"])
    
# ----------------------------- helpers -----------------------------
def _coerce_numeric_like_columns(df: pd.DataFrame, min_ratio: float = 0.8) -> pd.DataFrame:
    """Convert numeric-looking text columns, including common US/EU formats, into numeric dtype."""
    converted = df.copy()
    for col in converted.columns:
        series = converted[col]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue

        s = series.astype("string").str.strip()
        s = s.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA, "NULL": pd.NA, "NA": pd.NA})
        non_missing = s.notna()
        if int(non_missing.sum()) == 0:
            continue

        numeric_like_ratio = s[non_missing].str.match(r"^[+-]?[0-9\s.,]+$", na=False).mean()
        if numeric_like_ratio < min_ratio:
            continue

        candidates = [
            pd.to_numeric(s, errors="coerce"),
            pd.to_numeric(s.str.replace(",", "", regex=False), errors="coerce"),
            pd.to_numeric(s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False), errors="coerce"),
        ]
        best = max(candidates, key=lambda values: values[non_missing].notna().mean())
        if best[non_missing].notna().mean() >= min_ratio:
            converted[col] = best
    return converted


def read_csv_flexible(uploaded_file) -> pd.DataFrame:
    """Read CSV files with delimiter and encoding fallback."""
    separators = [None, ";", ",", "\t", "|"]
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    best_df, best_score, last_exc = None, (-1, -1), None

    for encoding in encodings:
        for sep in separators:
            try:
                uploaded_file.seek(0)
                candidate = pd.read_csv(uploaded_file, sep=sep, engine="python", encoding=encoding)
                score = (candidate.shape[1], candidate.shape[0])
                if score > best_score:
                    best_df, best_score = candidate, score
            except Exception as exc:
                last_exc = exc

    if best_df is None:
        raise ValueError(f"Could not parse CSV file. Last error: {last_exc}")
    return _coerce_numeric_like_columns(best_df)


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
        return _coerce_numeric_like_columns(pd.read_excel(uploaded_file))
    return read_csv_flexible(uploaded_file)


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.dropna(how="all").copy()
    unnamed_cols = [col for col in cleaned.columns if str(col).lower().startswith("unnamed")]
    return cleaned.drop(columns=unnamed_cols, errors="ignore")


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def get_binary_target_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if df[col].dropna().nunique() == 2]


def class_summary(y: pd.Series) -> pd.DataFrame:
    counts = y.value_counts(dropna=False).sort_index()
    return pd.DataFrame({
        "Class": counts.index.astype(str),
        "Count": counts.values,
        "Share %": (counts.values / counts.sum() * 100).round(2),
    })


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [col for col in X.columns if col not in numeric_columns]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, numeric_columns),
        ("cat", categorical_pipeline, categorical_columns),
    ], remainder="drop")


def build_model(name: str, X: pd.DataFrame) -> Pipeline:
    if name == "Logistic Regression":
        estimator = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    elif name == "Decision Tree":
        estimator = DecisionTreeClassifier(max_depth=4, min_samples_leaf=8, class_weight="balanced", random_state=RANDOM_STATE)
    else:
        estimator = RandomForestClassifier(
            n_estimators=160,
            max_depth=6,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    return Pipeline([("preprocess", build_preprocessor(X)), ("model", estimator)])


def make_cv(cv_type: str, n_splits: int, shuffle: bool, random_state: int):
    if cv_type == "Stratified K-Fold":
        return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state if shuffle else None)
    return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state if shuffle else None)


def safe_scorings() -> dict[str, str]:
    return {
        "Accuracy": "accuracy",
        "Precision": "precision",
        "Recall": "recall",
        "F1 score": "f1",
        "ROC AUC": "roc_auc",
    }


def holdout_metric_table(y_true, y_pred, y_proba=None) -> pd.DataFrame:
    values = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 score": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None and len(np.unique(y_true)) == 2:
        values["ROC AUC"] = roc_auc_score(y_true, y_proba)
    return pd.DataFrame({"Metric": values.keys(), "Value": [round(v, 3) for v in values.values()]})


def plot_class_balance(y: pd.Series):
    table = class_summary(y)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=table, x="Class", y="Count", ax=ax)
    ax.set_title("Target class balance")
    for i, row in table.iterrows():
        ax.text(i, row["Count"], f"{row['Share %']:.1f}%", ha="center", va="bottom")
    fig.tight_layout()
    return fig


def plot_cv_scores(scores: np.ndarray, title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    folds = np.arange(1, len(scores) + 1)
    ax.bar(folds, scores)
    ax.axhline(scores.mean(), linestyle="--", label=f"Mean = {scores.mean():.3f}")
    ax.set_title(title)
    ax.set_xlabel("Fold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(folds)
    ax.legend()
    bottom = max(0, float(scores.min()) - 0.08)
    top = min(1, float(scores.max()) + 0.08)
    if bottom < top:
        ax.set_ylim(bottom, top)
    fig.tight_layout()
    return fig


def plot_confusion(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Holdout confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    return fig


def plot_split_stability(results: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.lineplot(data=results, x="Random state", y="Test score", marker="o", ax=ax)
    ax.axhline(results["Test score"].mean(), linestyle="--", label=f"Mean = {results['Test score'].mean():.3f}")
    ax.set_title("Same model, different train-test splits")
    ax.legend()
    fig.tight_layout()
    return fig


def create_enriched_dataset(df, X_train, X_test, y_test, fitted_model):
    enriched = df.copy()
    enriched["cv_split_set"] = "not used"
    enriched.loc[X_train.index, "cv_split_set"] = "train"
    enriched.loc[X_test.index, "cv_split_set"] = "test"
    try:
        probabilities = fitted_model.predict_proba(X_test)[:, 1]
        enriched.loc[X_test.index, "holdout_predicted_probability"] = probabilities
        enriched.loc[X_test.index, "holdout_prediction"] = (probabilities >= 0.5).astype(int)
    except Exception:
        enriched.loc[X_test.index, "holdout_prediction"] = fitted_model.predict(X_test)
    enriched.loc[X_test.index, "holdout_actual"] = y_test
    return enriched


def score_by_label(metric_label: str, y_true, y_pred, model, X_test) -> float:
    if metric_label == "Accuracy":
        return accuracy_score(y_true, y_pred)
    if metric_label == "Precision":
        return precision_score(y_true, y_pred, zero_division=0)
    if metric_label == "Recall":
        return recall_score(y_true, y_pred, zero_division=0)
    if metric_label == "F1 score":
        return f1_score(y_true, y_pred, zero_division=0)
    return roc_auc_score(y_true, model.predict_proba(X_test)[:, 1])


with tab_cross_validation:
    st.subheader("Batch cross-validation from file")
    st.caption(
        "Workflow: upload a dataset, choose the target and feature columns, configure the validation strategy, then compare holdout and cross-validation performance."
    )
    # ----------------------------- input area -----------------------------
    if "uploaded_key" not in st.session_state:
        st.session_state["uploaded_key"] = 0
    if "cv_analysis_requested" not in st.session_state:
        st.session_state["cv_analysis_requested"] = False

    # Match the classroom app pattern: upload controls first; sample is downloadable only.
    # The sample is not loaded automatically. Download it, then upload it like any other file.
    input_upload, input_sample, input_download, input_clear = st.columns(
        [5.5, 2.2, 1.4, 1], gap="small", vertical_alignment="bottom"
    )

    with input_upload:
        uploaded = st.file_uploader(
            "Upload a CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            key=f"uploader_{st.session_state['uploaded_key']}",
        )

    with input_sample:
        sample_name = st.selectbox("Sample dataset", list(SAMPLE_DATASETS.keys()))
        sample_path = SAMPLE_DATASETS[sample_name]

    with input_download:
        st.write("")
        st.download_button(
            "Download sample",
            sample_path.read_bytes() if sample_path.exists() else b"",
            file_name=sample_path.name,
            mime="text/csv",
            width="stretch",
            disabled=not sample_path.exists(),
            icon=":material/download:",
        )

    with input_clear:
        st.write("")
        clear_file = st.button(
            "Clear",
            width="stretch",
            icon=":material/clear:",
            disabled=uploaded is None,
        )

    if clear_file:
        uploader_key = f"uploader_{st.session_state['uploaded_key']}"
        if uploader_key in st.session_state:
            del st.session_state[uploader_key]
        st.session_state["uploaded_key"] += 1
        st.session_state["cv_analysis_requested"] = False
        st.rerun()

    # The app intentionally stops here until the user uploads a file.
    if uploaded is None:
        st.info("Upload a CSV/XLSX file. You can also download one of the sample datasets above and upload it for practice.")
        st.stop()

    try:
        df = clean_frame(read_uploaded_file(uploaded))
        source_label = uploaded.name
    except Exception as exc:
        st.error(f"The dataset could not be loaded: {exc}")
        st.stop()

    if df.empty:
        st.error("The dataset is empty after removing blank rows.")
        st.stop()

    binary_targets = get_binary_target_columns(df)

    # ----------------------------- main layout -----------------------------
    control_col, result_col = st.columns([1, 2.25], gap="large")

    with control_col:
        st.subheader("Setup")
        st.caption("Select the modeling setup. No processing runs until the button is clicked.")

        if not binary_targets:
            st.warning("No suitable binary target column was found. The app expects a binary classification target.")
            st.stop()

        default_target = "loan_approved" if "loan_approved" in binary_targets else binary_targets[0]
        target_col = st.selectbox("Binary target column", binary_targets, index=binary_targets.index(default_target))

        candidate_features = [col for col in df.columns if col != target_col]
        default_features = [col for col in candidate_features if not str(col).lower().endswith("id")]
        features = st.multiselect("Feature columns", candidate_features, default=default_features)

        model_name = st.selectbox("Model", ["Logistic Regression", "Decision Tree", "Random Forest"])
        scoring_options = safe_scorings()
        scoring_label = st.selectbox("Scoring metric", list(scoring_options.keys()))

        with st.expander("Advanced settings", expanded=False):
            test_size = st.slider("Holdout test size", 0.15, 0.40, 0.25, 0.05)
            cv_type = st.selectbox("CV strategy", ["Stratified K-Fold", "K-Fold"])
            n_splits = st.slider("Number of folds", 3, 10, 5)
            shuffle = st.checkbox("Shuffle folds", value=True)
            compare_models = st.checkbox("Compare three simple models", value=True)

        st.markdown(
            "<div class='small-note'><b>Caution:</b> preprocessing is inside a Pipeline, so imputation, scaling and encoding are fitted inside each training fold.</div>",
            unsafe_allow_html=True,
        )
        run = st.button("Run cross-validation workflow", type="primary", width="stretch")

    if run:
        st.session_state["cv_analysis_requested"] = True

    with result_col:
        st.subheader("Dataset preview")
        st.caption(f"Source: {source_label} · Rows: {df.shape[0]:,} · Columns: {df.shape[1]:,}")
        st.dataframe(df.head(20), width="stretch")

        if not st.session_state["cv_analysis_requested"]:
            st.stop()

        if len(features) == 0:
            st.error("Select at least one feature column.")
            st.stop()

        work = df[features + [target_col]].dropna(subset=[target_col]).copy()
        if work[target_col].nunique() != 2:
            st.error("The selected target must have exactly two non-missing classes.")
            st.stop()
        if len(work) < max(30, n_splits * 4):
            st.error("The dataset is too small for the selected number of folds. Use fewer folds or a larger dataset.")
            st.stop()
        if cv_type == "Stratified K-Fold" and work[target_col].value_counts().min() < n_splits:
            st.error("The smallest target class has fewer records than the number of folds. Reduce the number of folds.")
            st.stop()

        y_raw = work[target_col]
        classes = sorted(y_raw.dropna().unique().tolist())
        class_map = {classes[0]: 0, classes[1]: 1}
        X = work[features]
        y = y_raw.map(class_map).astype(int)

        stratify_arg = y if y.value_counts().min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=stratify_arg
        )

        model = build_model(model_name, X)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        holdout_metrics = holdout_metric_table(y_test, y_pred, y_proba)

        cv = make_cv(cv_type, n_splits, shuffle, RANDOM_STATE)
        scoring = scoring_options[scoring_label]
        scores = cross_val_score(build_model(model_name, X), X, y, cv=cv, scoring=scoring, error_score="raise")

        st.subheader("Results")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        selected_holdout = holdout_metrics.loc[holdout_metrics["Metric"] == scoring_label, "Value"].iloc[0]
        kpi1.metric("Holdout score", f"{selected_holdout:.3f}")
        kpi2.metric("CV mean", f"{scores.mean():.3f}")
        kpi3.metric("CV std", f"{scores.std():.3f}")
        kpi4.metric("Folds", n_splits)

        tab_eval, tab_stability, tab_output = st.tabs(["Evaluation", "Split stability", "Processed output"])

        with tab_eval:
            top_left, top_right = st.columns([1, 1])
            with top_left:
                st.markdown("**Target distribution**")
                st.pyplot(plot_class_balance(y_raw), width="stretch")
                st.dataframe(class_summary(y_raw), width="stretch", hide_index=True)
            with top_right:
                st.markdown("**Holdout test metrics**")
                st.dataframe(holdout_metrics, width="stretch", hide_index=True)
                st.pyplot(plot_confusion(y_test, y_pred), width="stretch")

            st.markdown("**Cross-validation fold scores**")
            fold_table = pd.DataFrame({"Fold": np.arange(1, n_splits + 1), scoring_label: np.round(scores, 3)})
            cv_left, cv_right = st.columns([1, 1])
            with cv_left:
                st.dataframe(fold_table, width="stretch", hide_index=True)
            with cv_right:
                st.pyplot(plot_cv_scores(scores, f"{cv_type}: {model_name}", scoring_label), width="stretch")

            if compare_models:
                st.markdown("**Model comparison with the same CV strategy**")
                rows = []
                for candidate in ["Logistic Regression", "Decision Tree", "Random Forest"]:
                    candidate_scores = cross_val_score(build_model(candidate, X), X, y, cv=cv, scoring=scoring, error_score="raise")
                    rows.append({
                        "Model": candidate,
                        "Mean score": candidate_scores.mean(),
                        "Std score": candidate_scores.std(),
                        "Min score": candidate_scores.min(),
                        "Max score": candidate_scores.max(),
                    })
                compare_df = pd.DataFrame(rows).sort_values("Mean score", ascending=False).round(3)
                st.dataframe(compare_df, width="stretch", hide_index=True)
                fig, ax = plt.subplots(figsize=(8, 4.5))
                sns.barplot(data=compare_df, x="Model", y="Mean score", ax=ax)
                ax.set_title(f"Model comparison using {cv_type}")
                ax.set_ylim(max(0, compare_df["Mean score"].min() - .08), min(1, compare_df["Mean score"].max() + .08))
                fig.tight_layout()
                st.pyplot(fig, width="stretch")

        with tab_stability:
            st.caption("A single train-test split can be unstable. This view repeats the split with different random states.")
            stability_rows = []
            for seed in range(1, 16):
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=stratify_arg)
                candidate_model = build_model(model_name, X)
                candidate_model.fit(X_tr, y_tr)
                pred = candidate_model.predict(X_te)
                stability_rows.append({
                    "Random state": seed,
                    "Test score": score_by_label(scoring_label, y_te, pred, candidate_model, X_te),
                })
            stability_df = pd.DataFrame(stability_rows).round(3)
            stability_left, stability_right = st.columns([1, 1])
            with stability_left:
                st.pyplot(plot_split_stability(stability_df), width="stretch")
            with stability_right:
                st.dataframe(stability_df, width="stretch", hide_index=True)
                st.warning("Do not tune model settings on the final test set. Use validation or cross-validation for model selection.")

        with tab_output:
            enriched = create_enriched_dataset(work, X_train, X_test, y_test, model)
            st.caption("The enriched output marks train/test records and adds holdout predictions for the test rows.")
            st.dataframe(enriched.head(30), width="stretch")
            st.download_button(
                "Download enriched dataset",
                data=to_csv_bytes(enriched),
                file_name="cross_validation_enriched_output.csv",
                mime="text/csv",
                width="stretch",
            )

        with st.expander("Interpretation notes and common mistakes", expanded=False):
            st.markdown("""
    - **Train-test split** gives one estimate of unseen-data performance, but it depends on one random partition.
    - **K-Fold Cross Validation** repeats evaluation across folds and summarizes the mean and variation.
    - **Stratified K-Fold** is usually preferred for classification when class ratios matter.
    - **Scoring** defines what is compared. Accuracy can be misleading when the classes are imbalanced.
    - **Leakage caution:** fit preprocessing only on the training fold. A Pipeline helps keep this correct.
    """)
