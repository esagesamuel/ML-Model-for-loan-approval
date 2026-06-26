from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_DATA_PATH = Path(
    r"C:\Users\Dell_User\Documents\flatiron\module-4\First Summative Lab\financial_loan_data.csv"
)
LEAKAGE_COLUMNS = [
    "LoanApproved",
    "RiskScore",
    "BaseInterestRate",
    "InterestRate",
    "MonthlyLoanPayment",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    currency_columns = ["AnnualIncome"]
    for column in currency_columns:
        df[column] = df[column].replace(r"[$,]", "", regex=True).astype(float)
    return df


def build_preprocessors(X: pd.DataFrame) -> tuple[ColumnTransformer, ColumnTransformer]:
    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = X.select_dtypes(exclude="number").columns.tolist()

    numeric_linear = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    linear_preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_linear, numeric_columns),
            ("cat", categorical, categorical_columns),
        ]
    )
    tree_preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_columns),
            ("cat", categorical, categorical_columns),
        ]
    )
    return linear_preprocessor, tree_preprocessor


def build_models(
    linear_preprocessor: ColumnTransformer,
    tree_preprocessor: ColumnTransformer,
    random_state: int,
) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocess", linear_preprocessor),
                (
                    "model",
                    LogisticRegression(max_iter=2000, class_weight="balanced"),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocess", tree_preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=250,
                        min_samples_leaf=5,
                        random_state=random_state,
                        class_weight="balanced",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            steps=[
                ("preprocess", tree_preprocessor),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.08,
                        max_iter=250,
                        random_state=random_state,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
    }


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True
        ),
    }


def save_confusion_matrix(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Path,
) -> None:
    display = ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        display_labels=["Denied", "Approved"],
        cmap="Blues",
        colorbar=False,
    )
    display.ax_.set_title("Loan Approval Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_feature_importance(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_csv: Path,
    output_png: Path,
    random_state: int,
) -> pd.DataFrame:
    importance = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=5,
        random_state=random_state,
        scoring="roc_auc",
        n_jobs=-1,
    )
    importance_df = (
        pd.DataFrame(
            {
                "feature": X_test.columns,
                "importance": importance.importances_mean,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.to_csv(output_csv, index=False)

    top_features = importance_df.head(12).sort_values("importance")
    plt.figure(figsize=(9, 6))
    plt.barh(top_features["feature"], top_features["importance"], color="#2f6f73")
    plt.title("Top Permutation Importances")
    plt.xlabel("Mean ROC-AUC decrease")
    plt.tight_layout()
    plt.savefig(output_png, dpi=160)
    plt.close()
    return importance_df


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data_path)
    X = df.drop(columns=LEAKAGE_COLUMNS)
    y = df["LoanApproved"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=args.random_state,
        stratify=y,
    )
    linear_preprocessor, tree_preprocessor = build_preprocessors(X)
    models = build_models(linear_preprocessor, tree_preprocessor, args.random_state)

    metrics = {
        "data_path": str(args.data_path),
        "row_count": int(len(df)),
        "feature_count_after_leakage_removal": int(X.shape[1]),
        "target_distribution": y.value_counts(normalize=True).sort_index().to_dict(),
        "dropped_leakage_columns": LEAKAGE_COLUMNS,
        "models": {},
    }

    fitted_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model
        metrics["models"][name] = evaluate_model(model, X_test, y_test)

    best_model_name = max(
        metrics["models"], key=lambda name: metrics["models"][name]["roc_auc"]
    )
    best_model = fitted_models[best_model_name]
    metrics["best_model"] = best_model_name

    save_confusion_matrix(
        best_model,
        X_test,
        y_test,
        args.output_dir / "confusion_matrix.png",
    )
    save_feature_importance(
        best_model,
        X_test,
        y_test,
        args.output_dir / "feature_importance.csv",
        args.output_dir / "feature_importance.png",
        args.random_state,
    )
    joblib.dump(best_model, args.output_dir / "loan_approval_model.joblib")

    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Best model: {best_model_name}")
    print(json.dumps(metrics["models"][best_model_name], indent=2))


if __name__ == "__main__":
    main()
