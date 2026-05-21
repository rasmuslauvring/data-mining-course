from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE
    from imblearn.under_sampling import RandomUnderSampler
except Exception:  # pragma: no cover - shown in the app if dependency is missing
    ADASYN = None
    RandomOverSampler = None
    RandomUnderSampler = None
    SMOTE = None


APP_DIR = Path(__file__).parent
SAMPLE_DATASETS = {
    "Transaction fraud sample": APP_DIR / "data" / "synthetic_balancing_resampling_dataset.csv",
}

st.set_page_config(page_title="Balancing and Resampling", layout="wide")

# =========================================================
# Global visualization style
# =========================================================
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

st.title("Balancing and Resampling")
st.caption(
    "Workflow: upload or use a sample dataset, select a binary target, split train/test, "
    "balance only the training data, then compare model performance on the original test data."
)


# =========================================================
# Helper functions
# =========================================================
def _coerce_numeric_like_columns(df: pd.DataFrame, min_ratio: float = 0.8) -> pd.DataFrame:
    """Convert numeric-like text columns, including common US/EU formats, into numeric dtype."""
    converted = df.copy()

    for col in converted.columns:
        series = converted[col]
        if not pd.api.types.is_object_dtype(series) and not pd.api.types.is_string_dtype(series):
            continue

        s = series.astype("string").str.strip()
        s = s.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA, "NULL": pd.NA, "NA": pd.NA})
        non_missing = s.notna()
        if int(non_missing.sum()) == 0:
            continue

        numeric_like_ratio = s[non_missing].str.match(r"^[+-]?[0-9\s.,]+$", na=False).mean()
        if numeric_like_ratio < min_ratio:
            continue

        candidates = {
            "plain": pd.to_numeric(s, errors="coerce"),
            "us": pd.to_numeric(s.str.replace(",", "", regex=False), errors="coerce"),
            "eu": pd.to_numeric(
                s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
                errors="coerce",
            ),
        }

        best_values, best_ratio = None, -1.0
        for values in candidates.values():
            ratio = values[non_missing].notna().mean()
            if ratio > best_ratio:
                best_values, best_ratio = values, ratio

        if best_values is not None and best_ratio >= min_ratio:
            converted[col] = best_values

    return converted


def read_csv_flexible(uploaded_file) -> pd.DataFrame:
    """Read CSV files with delimiter and encoding fallback."""
    separators = [None, ";", ",", "\t", "|"]
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

    best_df = None
    best_score = (-1, -1)
    last_exc = None

    for encoding in encodings:
        for sep in separators:
            try:
                uploaded_file.seek(0)
                kwargs = {"engine": "python", "encoding": encoding}
                kwargs["sep"] = sep if sep is not None else None
                candidate = pd.read_csv(uploaded_file, **kwargs)
                score = (candidate.shape[1], candidate.shape[0])
                if score > best_score:
                    best_df = candidate
                    best_score = score
            except Exception as exc:
                last_exc = exc

    if best_df is None:
        raise ValueError(f"Could not parse CSV file. Last error: {last_exc}")

    return _coerce_numeric_like_columns(best_df)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def highlight_missing_values(df: pd.DataFrame):
    """Highlight missing values in a preview table."""
    style_fn = lambda value: "background-color: #FFE0E0" if pd.isna(value) else ""
    styler = df.style
    if hasattr(styler, "map"):
        return styler.map(style_fn)
    return styler.applymap(style_fn)


def class_summary(y: pd.Series) -> pd.DataFrame:
    counts = y.value_counts(dropna=False).sort_index()
    summary = pd.DataFrame({"Class": counts.index.astype(str), "Count": counts.values})
    summary["Share %"] = (summary["Count"] / summary["Count"].sum() * 100).round(2)
    return summary


def get_binary_target_columns(df: pd.DataFrame) -> list[str]:
    candidates = []
    for col in df.columns:
        values = df[col].dropna().unique()
        if len(values) == 2:
            candidates.append(col)
    return candidates


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [col for col in X.columns if col not in numeric_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return output feature names from a fitted ColumnTransformer.

    Important: ColumnTransformer keeps the original, unfitted pipeline objects in
    ``transformers_`` in some scikit-learn versions. The fitted clones live in
    ``named_transformers_``. Reading the encoder from ``transformers_`` can raise
    NotFittedError, even after ``preprocessor.fit_transform(...)`` has run.
    """
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        pass

    names = []
    for name, _transformer, columns in preprocessor.transformers_:
        if name == "remainder":
            continue

        fitted_transformer = preprocessor.named_transformers_.get(name)
        if fitted_transformer is None or fitted_transformer == "drop":
            continue

        if name == "num":
            names.extend([str(col) for col in columns])

        elif name == "cat":
            encoder = fitted_transformer.named_steps["onehot"]
            names.extend(encoder.get_feature_names_out(columns).tolist())

    return names


def simple_rose_resample(X: np.ndarray, y: np.ndarray, random_state: int = 42, noise_scale: float = 0.08):
    """Classroom-friendly ROSE approximation using smoothed bootstrap on encoded features."""
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2:
        raise ValueError("ROSE approximation requires a binary target.")

    minority_class = classes[np.argmin(counts)]
    majority_class = classes[np.argmax(counts)]
    minority_idx = np.where(y == minority_class)[0]
    majority_idx = np.where(y == majority_class)[0]

    n_to_generate = len(majority_idx) - len(minority_idx)
    if n_to_generate <= 0:
        return X.copy(), y.copy()

    sampled_idx = rng.choice(minority_idx, size=n_to_generate, replace=True)
    base = X[sampled_idx].copy()

    feature_std = np.nanstd(X[minority_idx], axis=0)
    feature_std = np.where(feature_std == 0, 0.01, feature_std)
    noise = rng.normal(0, noise_scale * feature_std, size=base.shape)
    synthetic_X = base + noise
    synthetic_y = np.repeat(minority_class, n_to_generate)

    X_resampled = np.vstack([X, synthetic_X])
    y_resampled = np.concatenate([y, synthetic_y])
    return X_resampled, y_resampled


def apply_resampling(method: str, X_train_encoded: np.ndarray, y_train: pd.Series, random_state: int, k_neighbors: int):
    y_array = y_train.to_numpy()
    minority_count = int(pd.Series(y_array).value_counts().min())
    safe_k = max(1, min(k_neighbors, minority_count - 1))

    if method == "No balancing":
        return X_train_encoded, y_array, "Original training distribution kept."

    if RandomUnderSampler is None:
        raise ImportError("imbalanced-learn is required for this resampling method. Install it from requirements.txt.")

    if method == "Random undersampling":
        sampler = RandomUnderSampler(random_state=random_state)
        X_resampled, y_resampled = sampler.fit_resample(X_train_encoded, y_array)
        note = "Removed majority-class observations from the training data."

    elif method == "Random oversampling":
        sampler = RandomOverSampler(random_state=random_state)
        X_resampled, y_resampled = sampler.fit_resample(X_train_encoded, y_array)
        note = "Duplicated minority-class observations in the training data."

    elif method == "Hybrid sampling":
        # Meet in the middle: under-sample majority and over-sample minority to an intermediate size.
        counts = pd.Series(y_array).value_counts()
        minority_class = counts.idxmin()
        majority_class = counts.idxmax()
        target_n = int(round((counts.min() + counts.max()) / 2))
        under = RandomUnderSampler(sampling_strategy={majority_class: target_n}, random_state=random_state)
        X_tmp, y_tmp = under.fit_resample(X_train_encoded, y_array)
        over = RandomOverSampler(sampling_strategy={minority_class: target_n}, random_state=random_state)
        X_resampled, y_resampled = over.fit_resample(X_tmp, y_tmp)
        note = "Combined majority-class reduction with minority-class duplication."

    elif method == "ROSE approximation":
        X_resampled, y_resampled = simple_rose_resample(X_train_encoded, y_array, random_state=random_state)
        note = "Generated smoothed minority examples by adding controlled noise around bootstrapped cases."

    elif method == "SMOTE":
        sampler = SMOTE(random_state=random_state, k_neighbors=safe_k)
        X_resampled, y_resampled = sampler.fit_resample(X_train_encoded, y_array)
        note = f"Created synthetic minority examples by interpolation. k_neighbors used: {safe_k}."

    elif method == "ADASYN":
        sampler = ADASYN(random_state=random_state, n_neighbors=safe_k)
        X_resampled, y_resampled = sampler.fit_resample(X_train_encoded, y_array)
        note = f"Generated more synthetic examples in harder-to-learn regions. n_neighbors used: {safe_k}."

    else:
        raise ValueError(f"Unknown resampling method: {method}")

    return X_resampled, y_resampled, note


def evaluate_logistic_model(X_train, y_train, X_test, y_test, random_state: int):
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, zero_division=0),
        "ROC AUC": roc_auc_score(y_test, y_score),
    }
    return model, y_pred, y_score, metrics


def plot_class_distribution(y_values: pd.Series | np.ndarray, title: str, bar_color: str = "#4C78A8"):
    """Plot one target distribution at a time so before/after can be shown side by side."""
    chart_df = class_summary(pd.Series(y_values))

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    sns.barplot(data=chart_df, x="Class", y="Count", ax=ax, color=bar_color)
    ax.set_title(title)
    ax.set_xlabel("Target class")
    ax.set_ylabel("Count")

    total = chart_df["Count"].sum()
    for patch, (_, row) in zip(ax.patches, chart_df.iterrows()):
        share = row["Count"] / total * 100 if total else 0
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height(),
            f"{int(row['Count']):,}\n{share:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ymax = chart_df["Count"].max() if not chart_df.empty else 1
    ax.set_ylim(0, ymax * 1.22)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_pred, title: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    fig.tight_layout()
    return fig


def plot_metric_comparison(metrics_base: dict, metrics_balanced: dict):
    rows = []
    for name, value in metrics_base.items():
        rows.append({"Metric": name, "Model": "Original train", "Value": value})
    for name, value in metrics_balanced.items():
        rows.append({"Metric": name, "Model": "Balanced train", "Value": value})
    chart_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=chart_df, x="Metric", y="Value", hue="Model", ax=ax, palette=["#4C78A8", "#72B7B2"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Metrics on Original Test Data")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="")
    fig.tight_layout()
    return fig


def plot_roc_curves(y_test, base_score, balanced_score):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for label, scores, color in [
        ("Original train", base_score, "#4C78A8"),
        ("Balanced train", balanced_score, "#72B7B2"),
    ]:
        fpr, tpr, _ = roc_curve(y_test, scores)
        auc = roc_auc_score(y_test, scores)
        ax.plot(fpr, tpr, label=f"{label} AUC={auc:.3f}", color=color, linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9AA4AF", linewidth=1)
    ax.set_title("ROC Curve")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_resampled_scatter(X_resampled: np.ndarray, y_resampled: np.ndarray, feature_names: list[str]):
    # Show the first two encoded features as a lightweight visual approximation.
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    if X_resampled.shape[1] < 2:
        ax.text(0.5, 0.5, "Need at least two encoded features", ha="center", va="center")
        ax.set_axis_off()
    else:
        plot_df = pd.DataFrame({
            "x": X_resampled[:, 0],
            "y": X_resampled[:, 1],
            "class": pd.Series(y_resampled).astype(str),
        })
        sns.scatterplot(data=plot_df, x="x", y="y", hue="class", ax=ax, alpha=0.65, s=35)
        x_label = feature_names[0] if len(feature_names) > 0 else "Feature 1"
        y_label = feature_names[1] if len(feature_names) > 1 else "Feature 2"
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title("Resampled Training Data: First Two Encoded Features")
        ax.legend(title="Class")
    fig.tight_layout()
    return fig


# =========================================================
# Session state
# =========================================================
if "balance_uploader_key_version" not in st.session_state:
    st.session_state["balance_uploader_key_version"] = 0
if "balance_analysis_requested" not in st.session_state:
    st.session_state["balance_analysis_requested"] = False

# =========================================================
# App layout
# =========================================================
[tab_resampling] = st.tabs(["Batch balancing and model check"])

with tab_resampling:
    st.subheader("Batch balancing from file")
    st.caption(
        "Methods: random undersampling, random oversampling, hybrid sampling, ROSE approximation, SMOTE, ADASYN, "
        "and class-weighted logistic regression as a simple algorithm-level comparison."
    )

    uploader_key = f"balance_uploader_{st.session_state['balance_uploader_key_version']}"

    uploader_col, sample_select_col, sample_download_col, clear_col = st.columns(
        [5.5, 2.2, 1.4, 1], gap="small", vertical_alignment="bottom"
    )

    with uploader_col:
        uploaded_file = st.file_uploader(
            "Upload a CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            key=uploader_key,
        )

    with sample_select_col:
        sample_choice = st.selectbox("Sample dataset", list(SAMPLE_DATASETS.keys()))
        sample_path = SAMPLE_DATASETS[sample_choice]
        if not sample_path.exists():
            same_name_fallback = APP_DIR / sample_path.name
            app_dataset_fallback = APP_DIR / "synthetic_balancing_resampling_app_dataset.csv"
            if same_name_fallback.exists():
                sample_path = same_name_fallback
            elif app_dataset_fallback.exists():
                sample_path = app_dataset_fallback

    with sample_download_col:
        st.write("")
        st.download_button(
            "Download sample",
            data=sample_path.read_bytes() if sample_path.exists() else b"",
            file_name=sample_path.name,
            mime="text/csv",
            width="stretch",
            disabled=not sample_path.exists(),
            icon=":material/download:",
        )

    with clear_col:
        st.write("")
        clear_file = st.button(
            "Clear",
            width="stretch",
            icon=":material/clear:",
            disabled=uploaded_file is None,
        )

    if clear_file:
        if uploader_key in st.session_state:
            del st.session_state[uploader_key]
        st.session_state["balance_uploader_key_version"] += 1
        st.session_state["balance_analysis_requested"] = False
        st.rerun()

    if uploaded_file is None:
        st.info("Upload a CSV/XLSX file. You can also download the sample dataset above and upload it for practice.")
        st.stop()

with tab_resampling:
    file_suffix = Path(uploaded_file.name).suffix.lower()

    try:
        if file_suffix in [".xlsx", ".xls"]:
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_name = st.selectbox("Select sheet", excel_file.sheet_names)
            df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
            df = _coerce_numeric_like_columns(df)
        else:
            df = read_csv_flexible(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read the uploaded file: {exc}")
        st.stop()

    if df.empty:
        st.warning("The uploaded file contains no rows.")
        st.stop()

    binary_target_columns = get_binary_target_columns(df)
    if not binary_target_columns:
        st.warning("No binary target column was found. This app expects a target with exactly two classes.")
        st.dataframe(highlight_missing_values(df.head(20)), width="stretch")
        st.stop()

    left_col, right_col = st.columns([1, 2], gap="large")

    with left_col:
        st.subheader("Setup")
        st.write(f"**File name:** {uploaded_file.name}")
        st.write(f"**Rows:** {len(df):,}")
        st.write(f"**Columns:** {len(df.columns):,}")
        st.write(f"**Missing cells:** {int(df.isna().sum().sum()):,}")

        default_target = "fraud" if "fraud" in binary_target_columns else binary_target_columns[0]
        target_column = st.selectbox(
            "Binary target column",
            binary_target_columns,
            index=binary_target_columns.index(default_target),
        )

        id_like_defaults = [col for col in df.columns if col.lower().endswith("id") or col.lower() in ["id", "transaction_id"]]
        excluded_columns = st.multiselect(
            "Columns to exclude from modeling",
            [col for col in df.columns if col != target_column],
            default=id_like_defaults,
            help="Use this for identifiers or free-text fields that should not be used as model features.",
        )

        positive_class = st.selectbox(
            "Class to evaluate as positive",
            sorted(df[target_column].dropna().unique().tolist()),
            index=sorted(df[target_column].dropna().unique().tolist()).index(1) if 1 in df[target_column].dropna().unique().tolist() else 0,
            help=(
                "The selected class is encoded as 1 for model evaluation. "
                "Precision, recall, F1-score, ROC AUC, and the confusion matrix focus on detecting this class. "
                "In imbalance problems this is usually the minority class, such as fraud, defect, or disease."
            ),
        )
        st.caption(
            "Interpretation: choose the class you care most about detecting. "
            "The other class is treated as the normal / negative class."
        )

        st.divider()
        st.write("**Resampling controls**")
        method = st.selectbox(
            "Balancing method",
            [
                "No balancing",
                "Random undersampling",
                "Random oversampling",
                "Hybrid sampling",
                "ROSE approximation",
                "SMOTE",
                "ADASYN",
                "Class-weighted model only",
            ],
            index=4,
        )

        test_size = st.slider("Test size", min_value=0.20, max_value=0.50, value=0.30, step=0.05)
        random_state = st.number_input("Random seed", min_value=0, value=42, step=1)
        k_neighbors = st.slider("Neighbors for SMOTE / ADASYN", min_value=1, max_value=10, value=5)

        st.caption("Important workflow rule: the app splits train/test first and balances only the training data.")

        st.subheader("Dataset preview")
        st.caption("Light red cells indicate missing values.")
        st.dataframe(highlight_missing_values(df.head(10)), width="stretch")

        run_clicked = st.button("Run balancing", type="primary", width="stretch")

    if run_clicked:
        st.session_state["balance_analysis_requested"] = True

    if not st.session_state["balance_analysis_requested"]:
        st.stop()

    # =========================================================
    # Processing
    # =========================================================
    analysis_df = df.dropna(subset=[target_column]).copy()
    if analysis_df[target_column].nunique() != 2:
        st.warning("After removing missing target values, the target must still contain exactly two classes.")
        st.stop()

    # Encode the selected positive class as 1 and the other class as 0.
    y = (analysis_df[target_column] == positive_class).astype(int)
    feature_columns = [col for col in analysis_df.columns if col not in [target_column] + excluded_columns]
    X = analysis_df[feature_columns].copy()

    if X.empty:
        st.warning("No feature columns remain after excluding the target and selected columns.")
        st.stop()

    if y.value_counts().min() < 2:
        st.warning("The minority class has fewer than two rows. Resampling and stratified splitting cannot be applied reliably.")
        st.stop()

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=int(random_state),
            stratify=y,
        )
    except Exception as exc:
        st.error(f"Could not split the dataset: {exc}")
        st.stop()

    preprocessor = build_preprocessor(X_train)
    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)
    feature_names = get_feature_names(preprocessor)

    if method == "Class-weighted model only":
        X_resampled, y_resampled = X_train_encoded, y_train.to_numpy()
        method_note = "No rows were added or removed. The comparison model uses class_weight='balanced'."
    else:
        try:
            X_resampled, y_resampled, method_note = apply_resampling(
                method,
                X_train_encoded,
                y_train,
                random_state=int(random_state),
                k_neighbors=k_neighbors,
            )
        except Exception as exc:
            st.error(f"The selected method could not be applied: {exc}")
            st.stop()

    try:
        _, base_pred, base_score, base_metrics = evaluate_logistic_model(
            X_train_encoded, y_train, X_test_encoded, y_test, random_state=int(random_state)
        )

        if method == "Class-weighted model only":
            balanced_model = LogisticRegression(max_iter=1000, random_state=int(random_state), class_weight="balanced")
            balanced_model.fit(X_train_encoded, y_train)
            balanced_pred = balanced_model.predict(X_test_encoded)
            balanced_score = balanced_model.predict_proba(X_test_encoded)[:, 1]
            balanced_metrics = {
                "Accuracy": accuracy_score(y_test, balanced_pred),
                "Precision": precision_score(y_test, balanced_pred, zero_division=0),
                "Recall": recall_score(y_test, balanced_pred, zero_division=0),
                "F1-score": f1_score(y_test, balanced_pred, zero_division=0),
                "ROC AUC": roc_auc_score(y_test, balanced_score),
            }
        else:
            _, balanced_pred, balanced_score, balanced_metrics = evaluate_logistic_model(
                X_resampled, y_resampled, X_test_encoded, y_test, random_state=int(random_state)
            )
    except Exception as exc:
        st.error(f"Model comparison could not be completed: {exc}")
        st.stop()

    resampled_df = pd.DataFrame(X_resampled, columns=feature_names)
    resampled_df[target_column] = y_resampled
    resampled_df["resampling_method"] = method

    metrics_df = pd.DataFrame(
        [
            {"Model": "Original training data", **{k: round(v, 4) for k, v in base_metrics.items()}},
            {"Model": method, **{k: round(v, 4) for k, v in balanced_metrics.items()}},
        ]
    )

    train_summary = class_summary(y_train)
    resampled_summary = class_summary(pd.Series(y_resampled))
    test_summary = class_summary(y_test)

    original_minority_share = train_summary["Share %"].min()
    resampled_minority_share = resampled_summary["Share %"].min()

    with right_col:
        st.subheader("Results")

        r1 = st.columns(4)
        r1[0].metric("Train Rows", f"{len(y_train):,}")
        r1[1].metric("Resampled Train Rows", f"{len(y_resampled):,}")
        r1[2].metric("Original Minority Share", f"{original_minority_share:.2f}%")
        r1[3].metric("Resampled Minority Share", f"{resampled_minority_share:.2f}%")

        r2 = st.columns(4)
        r2[0].metric("Test Rows", f"{len(y_test):,}")
        r2[1].metric("Recall", f"{balanced_metrics['Recall']:.3f}")
        r2[2].metric("Precision", f"{balanced_metrics['Precision']:.3f}")
        r2[3].metric("F1-score", f"{balanced_metrics['F1-score']:.3f}")

        st.info(
            f"Method interpretation: {method_note} Evaluation remains on the original, unbalanced test dataset."
        )

        st.write("**Original split distributions used for evaluation**")
        split_col_1, split_col_2 = st.columns(2, gap="large")
        with split_col_2:
            st.caption("Test data stays original and is never balanced")
            fig_test = plot_class_distribution(
                y_test,
                "Original Test Distribution",
                bar_color="#4C78A8",
            )
            st.pyplot(fig_test)
            plt.close(fig_test)

        with split_col_1:
            st.caption("Training data before applying the selected method")
            fig_train_split = plot_class_distribution(
                y_train,
                "Original Training Distribution",
                bar_color="#4C78A8",
            )
            st.pyplot(fig_train_split)
            plt.close(fig_train_split)

        st.write("**Class distribution before and after balancing**")
        before_col, after_col = st.columns(2, gap="large")
        with before_col:
            st.caption("Before: original training target distribution")
            fig_train_before = plot_class_distribution(
                y_train,
                "Training Data Before Balancing",
                bar_color="#4C78A8",
            )
            st.pyplot(fig_train_before)
            plt.close(fig_train_before)

        with after_col:
            st.caption("After: resampled training target distribution")
            fig_train_after = plot_class_distribution(
                y_resampled,
                "Training Data After Balancing / Resampling",
                bar_color="#72B7B2",
            )
            st.pyplot(fig_train_after)
            plt.close(fig_train_after)


        st.write("**Model metric comparison**")
        st.dataframe(metrics_df, width="stretch")
        fig_metrics = plot_metric_comparison(base_metrics, balanced_metrics)
        st.pyplot(fig_metrics, width="stretch")
        plt.close(fig_metrics)

        cm_col_1, cm_col_2 = st.columns(2, gap="large")
        with cm_col_1:
            st.write("**Original training data model**")
            fig_cm_base = plot_confusion_matrix(y_test, base_pred, "Confusion Matrix: Original Train")
            st.pyplot(fig_cm_base)
            plt.close(fig_cm_base)

        with cm_col_2:
            st.write(f"**{method} model**")
            fig_cm_balanced = plot_confusion_matrix(y_test, balanced_pred, f"Confusion Matrix: {method}")
            st.pyplot(fig_cm_balanced)
            plt.close(fig_cm_balanced)

        roc_col, scatter_col = st.columns(2, gap="large")
        with roc_col:
            st.write("**ROC comparison**")
            fig_roc = plot_roc_curves(y_test, base_score, balanced_score)
            st.pyplot(fig_roc)
            plt.close(fig_roc)

        with scatter_col:
            st.write("**Resampled feature space preview**")
            fig_scatter = plot_resampled_scatter(X_resampled, y_resampled, feature_names)
            st.pyplot(fig_scatter)
            plt.close(fig_scatter)
            st.caption("This preview uses the first two encoded features, so it is a quick visual check rather than a full model explanation.")

        st.write("**Class summary tables**")
        summary_col_1, summary_col_2, summary_col_3 = st.columns(3, gap="large")
        with summary_col_1:
            st.caption("Training before")
            st.dataframe(train_summary, width="stretch")
        with summary_col_2:
            st.caption("Training after")
            st.dataframe(resampled_summary, width="stretch")
        with summary_col_3:
            st.caption("Test data")
            st.dataframe(test_summary, width="stretch")

        st.write("**Practical cautions**")
        st.warning(
            "Do not balance the test set. Accuracy may still look high or low depending on the business context, "
            "so inspect recall, precision, F1-score and the confusion matrix. Synthetic methods can create unrealistic points, "
            "especially when the minority class is very small or noisy."
        )

        st.write("**Processed / resampled training data preview**")
        st.dataframe(resampled_df.head(20), width="stretch")

        st.download_button(
            label="Download resampled training CSV",
            data=to_csv_bytes(resampled_df),
            file_name="balancing_resampling_training_output.csv",
            mime="text/csv",
            width="stretch",
        )

        st.download_button(
            label="Download model metrics CSV",
            data=to_csv_bytes(metrics_df),
            file_name="balancing_resampling_model_metrics.csv",
            mime="text/csv",
            width="stretch",
        )
