from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import (
    DecisionTreeClassifier, DecisionTreeRegressor, export_text, plot_tree,
)

warnings.filterwarnings("ignore")

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
SAMPLE_DATASETS = {
    "Bank default — classification": DATA_DIR / "sample_bank_default_classification.csv",
    "Customer income — regression": DATA_DIR / "sample_customer_income_regression.csv",
}

st.set_page_config(page_title="Decision Trees", layout="wide")
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

st.title("Decision Trees")
st.caption(
    "Workflow: upload a dataset, choose classification or regression, configure the tree, "
    "then inspect evaluation, learned rules, feature importance, and enriched output."
)
[main_tab] = st.tabs(["Batch Decision Trees"])


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
            pd.to_numeric(
                text.str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
                errors="coerce",
            ),
        ]
        best = max(candidates, key=lambda values: values[non_missing].notna().mean())
        if best[non_missing].notna().mean() >= min_ratio:
            converted[col] = best
    return converted


def read_csv_flexible(uploaded_file) -> pd.DataFrame:
    best_df, best_score, last_error = None, (-1, -1), None
    for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        for separator in [None, ";", ",", "\t", "|"]:
            try:
                uploaded_file.seek(0)
                candidate = pd.read_csv(
                    uploaded_file, sep=separator, engine="python", encoding=encoding
                )
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
            ("imputer", SimpleImputer(strategy="median"))
        ]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_encoder()),
        ]), categorical))
    if not transformers:
        raise ValueError("No usable predictor columns were found.")
    return ColumnTransformer(transformers, remainder="drop")


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        names = []
        for name, transformer, columns in preprocessor.transformers_:
            if name == "remainder":
                continue
            if name == "num":
                names.extend(map(str, columns))
            else:
                names.extend(
                    transformer.named_steps["onehot"].get_feature_names_out(columns).tolist()
                )
        return names


def infer_task(df: pd.DataFrame, target: str) -> str:
    y = df[target].dropna()
    categorical = (
        pd.api.types.is_object_dtype(y) or pd.api.types.is_string_dtype(y)
        or pd.api.types.is_bool_dtype(y) or pd.api.types.is_categorical_dtype(y)
    )
    return "Classification" if categorical or y.nunique() <= min(20, max(2, int(len(y) * .05))) else "Regression"


def default_target_index(columns: list[str]) -> int:
    preferred = ["target", "class", "label", "response", "defaulted", "credit_rating", "outcome", "income"]
    lowered = [str(c).lower() for c in columns]
    for name in preferred:
        if name in lowered:
            return lowered.index(name)
    return len(columns) - 1


def classification_metrics(y_true, y_pred, probability=None) -> dict:
    labels = pd.Series(y_true).dropna().unique()
    average = "binary" if len(labels) == 2 else "weighted"
    kwargs = {"average": average, "zero_division": 0}
    if average == "binary":
        kwargs["pos_label"] = labels[-1]
    result = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, **kwargs),
        "Recall": recall_score(y_true, y_pred, **kwargs),
        "F1 score": f1_score(y_true, y_pred, **kwargs),
    }
    if probability is not None and len(labels) == 2:
        y_bin = (pd.Series(y_true).astype(str) == str(labels[-1])).astype(int)
        result["ROC AUC"] = roc_auc_score(y_bin, probability)
    return result


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** .5,
        "R²": r2_score(y_true, y_pred),
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


def plot_confusion(y_true, y_pred):
    labels = sorted(pd.Series(y_true).astype(str).unique())
    cm = confusion_matrix(pd.Series(y_true).astype(str), pd.Series(y_pred).astype(str), labels=labels)
    cm_df = pd.DataFrame(
        cm, index=[f"Actual {x}" for x in labels], columns=[f"Predicted {x}" for x in labels]
    )
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Confusion matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    fig.tight_layout()
    return fig, cm_df


def plot_roc(y_true, probability, positive_label):
    y_bin = (pd.Series(y_true).astype(str) == str(positive_label)).astype(int)
    fpr, tpr, _ = roc_curve(y_bin, probability)
    auc_value = roc_auc_score(y_bin, probability)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(fpr, tpr, linewidth=2, label=f"ROC AUC = {auc_value:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random guess")
    ax.set_title("ROC curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate / Recall")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_actual_predicted(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_true, y_pred, alpha=.75)
    low = min(float(np.min(y_true)), float(np.min(y_pred)))
    high = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax.plot([low, high], [low, high], linestyle="--", label="Perfect prediction")
    ax.set_title("Actual versus predicted")
    ax.set_xlabel("Actual target")
    ax.set_ylabel("Predicted target")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_residuals(y_true, y_pred):
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, residuals, alpha=.75)
    ax.axhline(0, linestyle="--")
    ax.set_title("Residual plot")
    ax.set_xlabel("Predicted target")
    ax.set_ylabel("Residual: actual − predicted")
    fig.tight_layout()
    return fig


def plot_importance(names, values):
    table = pd.DataFrame({"Feature": names, "Importance": values}).sort_values(
        "Importance", ascending=False
    ).reset_index(drop=True)
    chart = table.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(chart["Feature"], chart["Importance"])
    ax.set_title("Decision Tree feature importance")
    ax.set_xlabel("Impurity-based importance")
    fig.tight_layout()
    return fig, table


def plot_tree_chart(model, feature_names, class_names, display_depth):
    width = min(28, max(12, 8 + model.get_depth() * 2.2))
    height = min(18, max(6, 4 + model.get_depth() * 1.35))
    fig, ax = plt.subplots(figsize=(width, height))
    plot_tree(
        model, feature_names=feature_names, class_names=class_names,
        filled=True, rounded=True, proportion=True, precision=2,
        max_depth=display_depth, fontsize=7, ax=ax,
    )
    ax.set_title("Learned Decision Tree")
    fig.tight_layout()
    return fig


# ----------------------------- state -----------------------------
for key, value in {
    "dt_uploader_version": 0, "dt_results": None, "dt_file_signature": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = value


with main_tab:
    st.subheader("Batch Decision Trees from file")
    st.caption(
        "Download a sample or upload your own file. Configure the model on the left, "
        "then click Run Decision Tree."
    )

    upload_col, sample_col, download_col, clear_col = st.columns(
        [5.5, 2.2, 1.4, 1], gap="small", vertical_alignment="bottom"
    )
    with upload_col:
        uploaded_file = st.file_uploader(
            "Upload a CSV or Excel file", type=["csv", "xlsx", "xls"],
            key=f"dt_uploader_{st.session_state['dt_uploader_version']}",
        )
    with sample_col:
        sample_choice = st.selectbox("Sample dataset", list(SAMPLE_DATASETS.keys()))
    with download_col:
        sample_path = SAMPLE_DATASETS[sample_choice]
        st.download_button(
            "Download sample", data=sample_path.read_bytes(), file_name=sample_path.name,
            mime="text/csv", width="stretch",
        )
    with clear_col:
        clear = st.button("Clear", disabled=uploaded_file is None, width="stretch")

    if clear:
        st.session_state["dt_uploader_version"] += 1
        st.session_state["dt_results"] = None
        st.session_state["dt_file_signature"] = None
        st.rerun()

    if uploaded_file is None:
        st.info(
            "Upload a CSV/XLSX file. You can also download one of the sample datasets "
            "above and upload it for practice."
        )
        st.stop()

    try:
        raw_df = clean_frame(read_uploaded_file(uploaded_file))
    except Exception as exc:
        st.error(f"The file could not be read: {exc}")
        st.stop()

    if raw_df.empty:
        st.error("The uploaded dataset is empty after removing blank rows.")
        st.stop()
    if raw_df.shape[1] < 2:
        st.error("The dataset needs at least one predictor and one target column.")
        st.stop()

    signature = (uploaded_file.name, getattr(uploaded_file, "size", None), raw_df.shape, tuple(raw_df.columns))
    if st.session_state["dt_file_signature"] != signature:
        st.session_state["dt_results"] = None
        st.session_state["dt_file_signature"] = signature

    left, right = st.columns([1, 2], gap="large")

    with left:
        st.markdown("### Setup")
        target = st.selectbox(
            "Target column", raw_df.columns.tolist(),
            index=default_target_index(raw_df.columns.tolist()),
            help="The variable the tree should predict.",
        )
        automatic_task = infer_task(raw_df, target)
        task = st.selectbox(
            "Problem type", ["Classification", "Regression"],
            index=0 if automatic_task == "Classification" else 1,
        )

        available_features = [c for c in raw_df.columns if c != target]
        defaults = [
            c for c in available_features
            if str(c).lower() not in {"id", "customerid", "customer_id", "address", "name"}
        ] or available_features
        features = st.multiselect(
            "Predictor columns", available_features, default=defaults,
            help="Avoid identifiers and columns that reveal the target.",
        )

        st.markdown("### Parameters")
        if task == "Classification":
            criterion = st.selectbox("Split criterion", ["gini", "entropy", "log_loss"])
            class_weight = st.selectbox("Class weighting", ["None", "Balanced"])
        else:
            criterion = st.selectbox(
                "Split criterion", ["squared_error", "friedman_mse", "absolute_error", "poisson"]
            )
            class_weight = "None"

        max_depth_choice = st.selectbox(
            "Maximum depth", ["No limit", 2, 3, 4, 5, 6, 8, 10], index=3
        )
        min_leaf_max = max(2, min(50, len(raw_df) // 5))
        min_leaf_default = min(8, max(1, len(raw_df) // 50))
        min_leaf = st.slider(
            "Minimum records per leaf", 1, min_leaf_max, min_leaf_default,
            help="Larger leaves can reduce overfitting and overly specific rules.",
        )

        with st.expander("Advanced settings"):
            test_size = st.slider("Test set size", .15, .40, .25, .05)
            split_max = max(3, min(50, len(raw_df) // 4))
            min_split = st.slider("Minimum records required to split", 2, split_max, 2)
            max_features_choice = st.selectbox("Features considered at each split", ["All", "sqrt", "log2"])
            ccp_alpha = st.number_input(
                "Cost-complexity pruning (ccp_alpha)", min_value=0.0, max_value=1.0,
                value=0.0, step=.001, format="%.3f",
            )
            seed = st.number_input("Random seed", 0, 100000, 42, 1)
            display_depth_choice = st.selectbox(
                "Tree visualization depth", ["Full tree", 2, 3, 4, 5], index=2,
                help="This affects only the diagram.",
            )

        run = st.button("Run Decision Tree", type="primary", width="stretch")

    with right:
        if st.session_state["dt_results"] is None:
            st.markdown("### Dataset preview")
            st.caption(
                f"{raw_df.shape[0]:,} rows × {raw_df.shape[1]:,} columns. "
                "The analysis starts only after clicking Run."
            )
            st.dataframe(raw_df.head(100), width="stretch", height=430)

    if run:
        try:
            if not features:
                raise ValueError("Select at least one predictor column.")

            model_df = raw_df[[*features, target]].copy().dropna(subset=[target])
            if task == "Regression":
                model_df[target] = pd.to_numeric(model_df[target], errors="coerce")
                model_df = model_df.dropna(subset=[target])
                if model_df[target].nunique() < 3:
                    raise ValueError("Regression requires a numeric target with at least three distinct values.")
            elif model_df[target].nunique() < 2:
                raise ValueError("Classification requires at least two target classes.")

            X, y = model_df[features], model_df[target]
            max_depth = None if max_depth_choice == "No limit" else int(max_depth_choice)
            max_features = None if max_features_choice == "All" else max_features_choice

            stratify = None
            if task == "Classification" and y.value_counts().min() >= 2:
                stratify = y

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=float(test_size), random_state=int(seed), stratify=stratify
            )

            if task == "Classification":
                estimator = DecisionTreeClassifier(
                    criterion=criterion, max_depth=max_depth,
                    min_samples_split=int(min_split), min_samples_leaf=int(min_leaf),
                    max_features=max_features,
                    class_weight="balanced" if class_weight == "Balanced" else None,
                    ccp_alpha=float(ccp_alpha), random_state=int(seed),
                )
            else:
                estimator = DecisionTreeRegressor(
                    criterion=criterion, max_depth=max_depth,
                    min_samples_split=int(min_split), min_samples_leaf=int(min_leaf),
                    max_features=max_features, ccp_alpha=float(ccp_alpha),
                    random_state=int(seed),
                )

            pipe = Pipeline([("preprocess", build_preprocessor(X_train)), ("model", estimator)])
            pipe.fit(X_train, y_train)
            tree_model = pipe.named_steps["model"]
            feature_names = get_feature_names(pipe.named_steps["preprocess"])

            train_pred = pipe.predict(X_train)
            test_pred = pipe.predict(X_test)
            test_probability = None

            if task == "Classification":
                if len(tree_model.classes_) == 2:
                    train_probability = pipe.predict_proba(X_train)[:, 1]
                    test_probability = pipe.predict_proba(X_test)[:, 1]
                else:
                    train_probability = None
                train_metrics = classification_metrics(y_train, train_pred, train_probability)
                test_metrics = classification_metrics(y_test, test_pred, test_probability)
            else:
                train_metrics = regression_metrics(y_train, train_pred)
                test_metrics = regression_metrics(y_test, test_pred)

            enriched = raw_df.copy()
            enriched["dt_split_set"] = "not modeled"
            enriched["dt_prediction"] = pd.NA
            enriched.loc[X_train.index, "dt_split_set"] = "train"
            enriched.loc[X_test.index, "dt_split_set"] = "test"
            enriched.loc[X_train.index, "dt_prediction"] = train_pred
            enriched.loc[X_test.index, "dt_prediction"] = test_pred

            if task == "Classification":
                enriched["dt_correct"] = pd.NA
                enriched.loc[X_train.index, "dt_correct"] = (
                    pd.Series(train_pred, index=X_train.index).astype(str) == y_train.astype(str)
                )
                enriched.loc[X_test.index, "dt_correct"] = (
                    pd.Series(test_pred, index=X_test.index).astype(str) == y_test.astype(str)
                )
                all_prob = pipe.predict_proba(X)
                for i, class_label in enumerate(tree_model.classes_):
                    safe = str(class_label).strip().replace(" ", "_")
                    enriched[f"dt_probability_{safe}"] = np.nan
                    enriched.loc[X.index, f"dt_probability_{safe}"] = all_prob[:, i]
            else:
                enriched["dt_residual"] = np.nan
                enriched.loc[X_train.index, "dt_residual"] = y_train - pd.Series(train_pred, index=X_train.index)
                enriched.loc[X_test.index, "dt_residual"] = y_test - pd.Series(test_pred, index=X_test.index)

            importance_fig, importance_table = plot_importance(
                feature_names, tree_model.feature_importances_
            )
            display_depth = None if display_depth_choice == "Full tree" else int(display_depth_choice)
            class_names = [str(v) for v in tree_model.classes_] if task == "Classification" else None
            tree_fig = plot_tree_chart(tree_model, feature_names, class_names, display_depth)
            rules = export_text(
                tree_model, feature_names=feature_names,
                max_depth=display_depth if display_depth is not None else 10,
                decimals=3,
            )

            st.session_state["dt_results"] = {
                "task": task, "target": target, "features": features,
                "modeled_rows": len(model_df), "train_rows": len(X_train), "test_rows": len(X_test),
                "train_metrics": train_metrics, "test_metrics": test_metrics,
                "depth": tree_model.get_depth(), "leaves": tree_model.get_n_leaves(),
                "nodes": tree_model.tree_.node_count, "tree_fig": tree_fig,
                "importance_fig": importance_fig, "importance_table": importance_table,
                "rules": rules, "enriched": enriched, "y_train": y_train, "y_test": y_test,
                "test_pred": test_pred, "test_probability": test_probability,
                "classes": tree_model.classes_ if task == "Classification" else None,
            }
        except Exception as exc:
            st.session_state["dt_results"] = None
            st.error(f"The Decision Tree could not be fitted: {exc}")

    results = st.session_state["dt_results"]
    if results is not None:
        with right:
            st.markdown("### Results")
            names = ["Accuracy", "Precision", "Recall", "F1 score"] if results["task"] == "Classification" else ["MAE", "RMSE", "R²"]
            metric_cols = st.columns(len(names) + 2)
            for col, name in zip(metric_cols, names):
                col.metric(name, f"{results['test_metrics'][name]:.3f}")
            metric_cols[-2].metric("Tree depth", results["depth"])
            metric_cols[-1].metric("Leaves", results["leaves"])

            overview_tab, evaluation_tab, tree_tab, importance_tab, output_tab = st.tabs([
                "Overview", "Evaluation", "Tree & rules", "Feature importance", "Processed output"
            ])

            with overview_tab:
                a, b = st.columns(2)
                with a:
                    st.markdown("#### Model summary")
                    summary_df = pd.DataFrame({
                        "Item": ["Problem type", "Target", "Predictors", "Modeled rows", "Training rows", "Test rows", "Tree nodes"],
                        "Value": [results["task"], results["target"], len(results["features"]), results["modeled_rows"], results["train_rows"], results["test_rows"], results["nodes"]],
                    })
                    summary_df["Value"] = summary_df["Value"].astype("string")
                    st.dataframe(summary_df, hide_index=True, width="stretch")
                with b:
                    st.markdown("#### Train versus test")
                    comparison = pd.DataFrame({
                        "Metric": list(results["test_metrics"].keys()),
                        "Training": [results["train_metrics"][m] for m in results["test_metrics"]],
                        "Test": [results["test_metrics"][m] for m in results["test_metrics"]],
                    }).round(3)
                    st.dataframe(comparison, hide_index=True, width="stretch")
                    key_metric = "Accuracy" if results["task"] == "Classification" else "R²"
                    gap = results["train_metrics"][key_metric] - results["test_metrics"][key_metric]
                    if gap > (.10 if results["task"] == "Classification" else .15):
                        st.warning(
                            "Training performance is notably higher than test performance. "
                            "Try a smaller depth, larger leaves, or a higher ccp_alpha."
                        )
                    else:
                        st.info(
                            "A modest train–test gap is preferable. Review the detailed evaluation "
                            "before accepting the model."
                        )

            with evaluation_tab:
                if results["task"] == "Classification":
                    c1, c2 = st.columns(2)
                    with c1:
                        st.pyplot(plot_class_balance(results["y_train"]), width="stretch")
                    with c2:
                        cm_fig, cm_table = plot_confusion(results["y_test"], results["test_pred"])
                        st.pyplot(cm_fig, width="stretch")
                    if results["test_probability"] is not None and len(results["classes"]) == 2:
                        st.pyplot(
                            plot_roc(results["y_test"], results["test_probability"], results["classes"][1]),
                            width="stretch",
                        )
                    st.caption(
                        "Diagonal cells are correct predictions. Off-diagonal cells are errors."
                    )
                    st.dataframe(cm_table, width="stretch")
                else:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.pyplot(plot_actual_predicted(results["y_test"], results["test_pred"]), width="stretch")
                    with c2:
                        st.pyplot(plot_residuals(results["y_test"], results["test_pred"]), width="stretch")
                    st.caption(
                        "Points near the diagonal indicate better predictions. Residuals should "
                        "ideally remain near zero without a strong pattern."
                    )

            with tree_tab:
                st.pyplot(results["tree_fig"], width="stretch")
                st.caption(
                    "Read from top to bottom. Every internal node asks one question; "
                    "each leaf contains the final prediction."
                )
                with st.expander("Text version of the learned rules"):
                    st.code(results["rules"], language="text")
                st.info(
                    "Very deep trees can fit noise and become difficult to explain. "
                    "Control complexity with depth, minimum leaf size, and ccp_alpha."
                )

            with importance_tab:
                st.pyplot(results["importance_fig"], width="stretch")
                st.dataframe(results["importance_table"], hide_index=True, width="stretch", height=430)
                st.caption(
                    "Importance measures impurity reduction. It supports interpretation but does not prove causality."
                )

            with output_tab:
                st.dataframe(results["enriched"].head(200), width="stretch", height=440)
                st.download_button(
                    "Download enriched dataset", data=to_csv_bytes(results["enriched"]),
                    file_name="decision_tree_enriched_output.csv",
                    mime="text/csv", width="stretch",
                )
                st.caption(
                    "The output contains the original data plus split membership, predictions, "
                    "and classification probabilities or regression residuals."
                )
