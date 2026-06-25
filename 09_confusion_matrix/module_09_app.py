from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
SAMPLE_DATASETS = {
    "Fraud detection sample with probabilities": DATA_DIR / "sample_fraud_scores.csv",
}

st.set_page_config(page_title="Confusion Matrix, ROC-AUC and PR Curve", layout="wide")

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

st.markdown(
    """
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
div[data-testid="stMetric"] {background: #F7F9FC; border: 1px solid #E2E8F0; padding: 14px; border-radius: 12px;}
.small-note {font-size: 0.88rem; color: #475569;}
.section-card {background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px; padding: 1rem;}
</style>
""",
    unsafe_allow_html=True,
)

st.title("Confusion Matrix, ROC-AUC and Precision-Recall Curves")
st.caption(
    "Workflow: upload or use a sample dataset, select actual and predicted outputs, adjust the threshold, "
    "then inspect the confusion matrix, metrics, ROC curve and precision-recall curve."
)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def coerce_numeric_like_columns(df: pd.DataFrame, min_ratio: float = 0.8) -> pd.DataFrame:
    """Convert numeric-looking text columns, including common US/EU formats."""
    converted = df.copy()
    for col in converted.columns:
        s0 = converted[col]
        if not (pd.api.types.is_object_dtype(s0) or pd.api.types.is_string_dtype(s0)):
            continue
        s = s0.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA, "NULL": pd.NA, "NA": pd.NA})
        non_missing = s.notna()
        if non_missing.sum() == 0:
            continue
        numeric_like = s[non_missing].str.match(r"^[+-]?[0-9\s.,]+$", na=False).mean()
        if numeric_like < min_ratio:
            continue
        candidates = [
            pd.to_numeric(s, errors="coerce"),
            pd.to_numeric(s.str.replace(",", "", regex=False), errors="coerce"),
            pd.to_numeric(s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False), errors="coerce"),
        ]
        best = max(candidates, key=lambda x: x[non_missing].notna().mean())
        if best[non_missing].notna().mean() >= min_ratio:
            converted[col] = best
    return converted


def read_csv_flexible(uploaded_file) -> pd.DataFrame:
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
    return coerce_numeric_like_columns(best_df)


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
        return coerce_numeric_like_columns(pd.read_excel(uploaded_file))
    return read_csv_flexible(uploaded_file)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def get_class_columns(df: pd.DataFrame) -> list[str]:
    candidates = []
    for col in df.columns:
        if df[col].dropna().nunique() >= 2 and df[col].dropna().nunique() <= 20:
            candidates.append(col)
    return candidates


def get_score_columns(df: pd.DataFrame) -> list[str]:
    candidates = []
    for col in df.select_dtypes(include=[np.number]).columns:
        s = df[col].dropna()
        if len(s) > 0 and s.between(0, 1).mean() >= 0.95 and s.nunique() > 2:
            candidates.append(col)
    return candidates


def binary_metrics(y_true, y_pred, positive_label):
    labels = [positive_label, *[x for x in pd.unique(pd.Series(y_true)) if x != positive_label]]
    # Convert to 1/0 so formulas are explicit and stable.
    y_true_bin = (pd.Series(y_true).astype(str) == str(positive_label)).astype(int)
    y_pred_bin = (pd.Series(y_pred).astype(str) == str(positive_label)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
    total = tp + tn + fp + fn
    safe = lambda num, den: np.nan if den == 0 else num / den
    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Accuracy": safe(tp + tn, total),
        "Precision": safe(tp, tp + fp),
        "Recall / Sensitivity": safe(tp, tp + fn),
        "F1-score": safe(2 * tp, 2 * tp + fp + fn),
        "Specificity": safe(tn, tn + fp),
        "Type I error rate / FPR": safe(fp, fp + tn),
        "Type II error rate / FNR": safe(fn, tp + fn),
    }


def make_predictions_from_score(scores: pd.Series, threshold: float, positive_label, negative_label) -> pd.Series:
    return pd.Series(np.where(scores >= threshold, positive_label, negative_label), index=scores.index)


def add_error_labels(df: pd.DataFrame, actual_col: str, pred_col: str, positive_label) -> pd.DataFrame:
    result = df.copy()
    actual_pos = result[actual_col].astype(str) == str(positive_label)
    pred_pos = result[pred_col].astype(str) == str(positive_label)
    conditions = [actual_pos & pred_pos, ~actual_pos & ~pred_pos, ~actual_pos & pred_pos, actual_pos & ~pred_pos]
    choices = ["True Positive", "True Negative", "False Positive", "False Negative"]
    result["cm_result"] = np.select(conditions, choices, default="Other / multiclass")
    result["is_correct"] = result[actual_col].astype(str) == result[pred_col].astype(str)
    return result


def plot_confusion_matrix(cm_df: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    sns.heatmap(cm_df, annot=True, fmt="g", cmap="Blues", cbar=False, ax=ax, linewidths=0.5, linecolor="#FFFFFF")
    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    return fig


def plot_class_distribution(y: pd.Series, title: str):
    counts = y.value_counts(dropna=False).reset_index()
    counts.columns = ["Class", "Count"]
    counts["Class"] = counts["Class"].astype(str)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=counts, x="Class", y="Count", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of records")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return fig


def plot_score_distribution(df: pd.DataFrame, score_col: str, actual_col: str, threshold: float):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(data=df, x=score_col, hue=actual_col, bins=25, kde=True, element="step", ax=ax)
    ax.axvline(threshold, linestyle="--", linewidth=1.8, label=f"Threshold = {threshold:.2f}")
    ax.set_title("Predicted probability distribution by actual class")
    ax.set_xlabel("Predicted probability / score")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_roc(y_true_bin: pd.Series, scores: pd.Series):
    fpr, tpr, _ = roc_curve(y_true_bin, scores)
    auc_value = roc_auc_score(y_true_bin, scores)
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.plot(fpr, tpr, linewidth=2, label=f"ROC-AUC = {auc_value:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Random guess")
    ax.set_title("ROC curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate / Recall")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig, auc_value


def plot_precision_recall(y_true_bin: pd.Series, scores: pd.Series):
    precision, recall, _ = precision_recall_curve(y_true_bin, scores)
    ap_value = average_precision_score(y_true_bin, scores)
    baseline = float(y_true_bin.mean())
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.plot(recall, precision, linewidth=2, label=f"Average precision = {ap_value:.3f}")
    ax.axhline(baseline, linestyle="--", linewidth=1, label=f"Baseline = {baseline:.3f}")
    ax.set_title("Precision-Recall curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig, ap_value


def threshold_table(y_true_bin: pd.Series, scores: pd.Series, thresholds: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        pred = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true_bin, pred, labels=[0, 1]).ravel()
        rows.append({
            "threshold": threshold,
            "accuracy": accuracy_score(y_true_bin, pred),
            "precision": precision_score(y_true_bin, pred, zero_division=0),
            "recall": recall_score(y_true_bin, pred, zero_division=0),
            "f1": f1_score(y_true_bin, pred, zero_division=0),
            "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
            "false_positives": fp,
            "false_negatives": fn,
        })
    return pd.DataFrame(rows).round(3)


def highlight_results(df: pd.DataFrame):
    def style_row(row):
        color_map = {
            "True Positive": "background-color: #E8F5E9; color: #166534; font-weight: 600;",
            "True Negative": "background-color: #E8F5E9; color: #166534; font-weight: 600;",
            "False Positive": "background-color: #FEE2E2; color: #991B1B; font-weight: 600;",
            "False Negative": "background-color: #FEE2E2; color: #991B1B; font-weight: 600;",
        }
        return [color_map.get(row.get("cm_result"), "") if col == "cm_result" else "" for col in df.columns]
    return df.style.apply(style_row, axis=1)


# -----------------------------------------------------------------------------
# Data input and app layout
# -----------------------------------------------------------------------------
if "cm_uploader_key_version" not in st.session_state:
    st.session_state["cm_uploader_key_version"] = 0
if "cm_run_requested" not in st.session_state:
    st.session_state["cm_run_requested"] = False

[tab_evaluation] = st.tabs(["Batch classification evaluation"])

with tab_evaluation:
    st.subheader("Batch confusion-matrix evaluation from file")
    st.caption(
        "Workflow: upload a dataset with actual labels and either predicted labels or probabilities. "
        "Then choose the positive class, adjust the threshold when relevant, and review the evaluation outputs."
    )

    uploader_key = f"cm_uploader_{st.session_state['cm_uploader_key_version']}"
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
            sample_path = APP_DIR / sample_path.name

    with sample_download_col:
        st.write("")
        st.download_button(
            "Download sample",
            data=sample_path.read_bytes() if sample_path.exists() else b"",
            file_name=sample_path.name,
            mime="text/csv",
            width='stretch',
            disabled=not sample_path.exists(),
        )

    with clear_col:
        st.write("")
        clear_file = st.button(
            "Clear",
            width='stretch',
            disabled=uploaded_file is None,
        )

    if clear_file:
        if uploader_key in st.session_state:
            del st.session_state[uploader_key]
        st.session_state["cm_uploader_key_version"] += 1
        st.session_state["cm_run_requested"] = False
        st.rerun()

    if uploaded_file is None:
        st.info("Upload a CSV/XLSX file. You can also download the sample dataset above and upload it for practice.")
        st.stop()

    # Read the uploaded file only after a file is available. Nothing else is shown before this point.
    try:
        file_suffix = Path(uploaded_file.name).suffix.lower()
        if file_suffix in [".xlsx", ".xls"]:
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_name = st.selectbox("Select sheet", excel_file.sheet_names)
            df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
            df = coerce_numeric_like_columns(df)
        else:
            df = read_csv_flexible(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read the uploaded file: {exc}")
        st.stop()

    index_like = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    df = df.drop(columns=index_like, errors="ignore").dropna(how="all")

    if df.empty:
        st.warning("The uploaded file contains no rows.")
        st.stop()

    class_columns = get_class_columns(df)
    score_columns = get_score_columns(df)

    if len(class_columns) < 1:
        st.warning(
            "No suitable class column was found. The app needs at least one categorical or low-cardinality actual class column."
        )
        st.dataframe(df.head(20), width='stretch')
        st.stop()

    left_col, right_col = st.columns([1, 2], gap="large")

    with left_col:
        st.subheader("Setup")
        st.write(f"**File name:** {uploaded_file.name}")
        st.write(f"**Rows:** {len(df):,}")
        st.write(f"**Columns:** {len(df.columns):,}")
        st.write(f"**Missing cells:** {int(df.isna().sum().sum()):,}")

        default_actual = "actual_fraud" if "actual_fraud" in class_columns else class_columns[0]
        actual_col = st.selectbox(
            "Actual class column",
            class_columns,
            index=class_columns.index(default_actual),
        )

        actual_values = sorted(df[actual_col].dropna().unique().tolist(), key=lambda x: str(x))
        is_binary = len(actual_values) == 2

        if is_binary:
            positive_default = actual_values.index(1) if 1 in actual_values else len(actual_values) - 1
            positive_label = st.selectbox(
                "Positive class / class of interest",
                actual_values,
                index=positive_default,
            )
            negative_label = [x for x in actual_values if x != positive_label][0]
            st.caption("The positive class defines the meaning of TP, FP, FN and TN.")
        else:
            positive_label = None
            negative_label = None
            st.caption("Multi-class mode: diagonal cells are correct predictions; off-diagonal cells are errors.")

        st.divider()
        st.write("**Prediction input**")
        evaluation_mode = st.radio(
            "Select how predictions are available",
            ["Use probability / score column and threshold", "Use existing predicted class column"],
            horizontal=False,
        )

        if evaluation_mode.startswith("Use probability"):
            if not is_binary:
                st.warning("Threshold-based evaluation requires a binary actual class column.")
            if not score_columns:
                st.warning("No probability-like numeric column between 0 and 1 was found.")
                score_col = None
            else:
                score_default = "fraud_probability" if "fraud_probability" in score_columns else score_columns[0]
                score_col = st.selectbox(
                    "Probability / score column",
                    score_columns,
                    index=score_columns.index(score_default),
                )
            threshold = st.slider("Decision threshold", 0.01, 0.99, 0.50, 0.01)
            pred_col = "predicted_class"
            st.caption("A higher threshold usually reduces false positives but may increase false negatives.")
        else:
            pred_candidates = [c for c in class_columns if c != actual_col]
            if not pred_candidates:
                st.warning("No existing predicted class column was found. Use a score column or upload predictions.")
                pred_col = None
            else:
                default_pred = "predicted_class" if "predicted_class" in pred_candidates else pred_candidates[0]
                pred_col = st.selectbox(
                    "Predicted class column",
                    pred_candidates,
                    index=pred_candidates.index(default_pred),
                )
            threshold = None
            score_col = None

        st.caption("Rows are treated as actual classes and columns as predicted classes in the matrix.")
        run_clicked = st.button("Run evaluation", type="primary", width='stretch')
        if run_clicked:
            st.session_state["cm_run_requested"] = True

    with right_col:
        st.subheader("Dataset preview")
        meta1, meta2, meta3, meta4 = st.columns(4)
        meta1.metric("Rows", f"{df.shape[0]:,}")
        meta2.metric("Columns", f"{df.shape[1]:,}")
        meta3.metric("Class columns", f"{len(class_columns):,}")
        meta4.metric("Score columns", f"{len(score_columns):,}")
        st.dataframe(df.head(20), width='stretch')
        st.caption("Only a preview is shown. The full dataset is used when the evaluation runs.")

    if not st.session_state["cm_run_requested"]:
        st.stop()

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------
    work_df = df.copy()
    if evaluation_mode.startswith("Use probability"):
        if not is_binary or score_col is None:
            st.error("This method cannot be applied. Select a binary actual class and a probability / score column.")
            st.stop()
        valid_mask = work_df[[actual_col, score_col]].notna().all(axis=1)
        work_df = work_df.loc[valid_mask].copy()
        work_df[pred_col] = make_predictions_from_score(work_df[score_col], threshold, positive_label, negative_label)
    else:
        if pred_col is None:
            st.error("This method cannot be applied. Select or upload a predicted class column.")
            st.stop()
        valid_mask = work_df[[actual_col, pred_col]].notna().all(axis=1)
        work_df = work_df.loc[valid_mask].copy()

    if work_df.empty:
        st.error("No valid rows remain after removing records with missing actual/predicted values.")
        st.stop()

    labels = sorted(pd.unique(pd.concat([work_df[actual_col], work_df[pred_col]], ignore_index=True)), key=lambda x: str(x))
    cm = confusion_matrix(work_df[actual_col], work_df[pred_col], labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"Actual {x}" for x in labels], columns=[f"Predicted {x}" for x in labels])
    processed_df = add_error_labels(work_df, actual_col, pred_col, positive_label) if is_binary else work_df.assign(is_correct=work_df[actual_col].astype(str) == work_df[pred_col].astype(str))

    st.subheader("Results")
    if is_binary:
        metrics = binary_metrics(work_df[actual_col], work_df[pred_col], positive_label)
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
        k2.metric("Precision", f"{metrics['Precision']:.3f}")
        k3.metric("Recall", f"{metrics['Recall / Sensitivity']:.3f}")
        k4.metric("F1-score", f"{metrics['F1-score']:.3f}")
        k5.metric("Specificity", f"{metrics['Specificity']:.3f}")

        e1, e2, e3, e4 = st.columns(4)
        e1.metric("False positives", f"{metrics['FP']:,}")
        e2.metric("False negatives", f"{metrics['FN']:,}")
        e3.metric("Type I / FPR", f"{metrics['Type I error rate / FPR']:.3f}")
        e4.metric("Type II / FNR", f"{metrics['Type II error rate / FNR']:.3f}")
    else:
        acc = accuracy_score(work_df[actual_col], work_df[pred_col])
        st.metric("Overall accuracy", f"{acc:.3f}")
        st.caption("For multi-class evaluation, inspect the diagonal and off-diagonal patterns by class.")

    tab1, tab2, tab3, tab4 = st.tabs(["Confusion matrix", "Curves and threshold", "Reports", "Processed output"])

    with tab1:
        c1, c2 = st.columns([1.1, 1], gap="large")
        with c1:
            st.pyplot(plot_confusion_matrix(cm_df, "Confusion matrix"), width='stretch')
            st.caption("Diagonal cells are correct predictions. Off-diagonal cells are errors.")
        with c2:
            st.dataframe(cm_df, width='stretch')
            st.pyplot(plot_class_distribution(work_df[actual_col], "Actual class distribution"), width='stretch')
            if is_binary:
                st.info(
                    f"Positive class: {positive_label}. False positives are unnecessary positive actions. "
                    f"False negatives are missed positive cases. The worse error depends on the business context."
                )

    with tab2:
        if evaluation_mode.startswith("Use probability") and is_binary and score_col is not None:
            y_true_bin = (work_df[actual_col].astype(str) == str(positive_label)).astype(int)
            scores = work_df[score_col].astype(float)
            curve1, curve2 = st.columns(2, gap="large")
            with curve1:
                fig_roc, auc_value = plot_roc(y_true_bin, scores)
                st.pyplot(fig_roc, width='stretch')
                st.caption("ROC compares recall against the false positive rate across thresholds.")
            with curve2:
                fig_pr, ap_value = plot_precision_recall(y_true_bin, scores)
                st.pyplot(fig_pr, width='stretch')
                st.caption("Precision-recall focuses on the positive class and is useful for rare events.")

            k1, k2 = st.columns(2)
            k1.metric("ROC-AUC", f"{auc_value:.3f}")
            k2.metric("Average precision", f"{ap_value:.3f}")

            st.pyplot(plot_score_distribution(work_df, score_col, actual_col, threshold), width='stretch')
            thresholds = np.round(np.arange(0.10, 0.91, 0.10), 2)
            tdf = threshold_table(y_true_bin, scores, thresholds)
            st.dataframe(tdf, width='stretch')
            st.caption("Use this table to see how a higher or lower threshold changes precision, recall, F1 and error counts.")
        else:
            st.info("ROC and precision-recall curves require a binary actual class and a probability / score column.")

    with tab3:
        if is_binary:
            formula_table = pd.DataFrame({
                "Metric": ["Accuracy", "Precision", "Recall / Sensitivity", "F1-score", "Specificity", "Type I error / FPR", "Type II error / FNR"],
                "Value": [metrics["Accuracy"], metrics["Precision"], metrics["Recall / Sensitivity"], metrics["F1-score"], metrics["Specificity"], metrics["Type I error rate / FPR"], metrics["Type II error rate / FNR"]],
                "Practical reading": [
                    "Overall share of correct predictions.",
                    "When the model predicts positive, how often it is correct.",
                    "How many actual positives the model found.",
                    "Balanced view of precision and recall.",
                    "How many actual negatives were correctly identified.",
                    "Share of actual negatives incorrectly predicted positive.",
                    "Share of actual positives missed by the model.",
                ],
            })
            st.dataframe(formula_table.round(3), width='stretch')
        report_dict = classification_report(work_df[actual_col], work_df[pred_col], zero_division=0, output_dict=True)
        report_df = pd.DataFrame(report_dict).T.round(3)
        st.dataframe(report_df, width='stretch')
        st.warning("Common mistake: high accuracy can still hide poor detection of a rare but important class.")

    with tab4:
        st.dataframe(highlight_results(processed_df.head(200)) if "cm_result" in processed_df.columns else processed_df.head(200), width='stretch')
        st.download_button(
            "Download evaluated dataset",
            data=to_csv_bytes(processed_df),
            file_name="confusion_matrix_evaluated_dataset.csv",
            mime="text/csv",
            width='stretch',
        )
        st.caption("The downloaded dataset includes the threshold-based prediction, correctness flag and error type when binary evaluation is used.")
