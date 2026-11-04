from pathlib import Path

import joblib
from loguru import logger
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer, StandardScaler
from sklearn.tree import DecisionTreeClassifier
import typer

from wisconsin_cancer_analysis.config import MODELS_DIR, RAW_DATA_DIR, REPORTS_DIR
from wisconsin_cancer_analysis.dataset import CLASS_MAP, FEATURE_COLS, ID_COL, TARGET_COL
from wisconsin_cancer_analysis.features import (
    REP3_BASE_COLS,
    binning_rep2,
    binning_rep3,
)

app = typer.Typer()

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_SPLITS = 5
METRICS = ["accuracy", "precision", "recall", "f1"]

REP3_NAME = "Rep. 3: Binning Agrupado (7 vars)"
REP3_FEATURES = REP3_BASE_COLS + ["Atipia_Celular"]


def make_pipeline(transformer) -> Pipeline:
    # La mediana de Bare_nuclei se aprende solo del train de cada pliegue
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("repr", transformer),
            ("tree", DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE)),
        ]
    )


def build_pipelines() -> dict[str, Pipeline]:
    return {
        "Rep. 1: Z-Score (9 vars)": make_pipeline(StandardScaler()),
        "Rep. 2: Binning (9 vars)": make_pipeline(FunctionTransformer(binning_rep2)),
        "Rep. 2b: Binning KMeans (9 vars)": make_pipeline(
            KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="kmeans")
        ),
        REP3_NAME: make_pipeline(FunctionTransformer(binning_rep3)),
    }


def prepare_train_test(
    input_path: Path = RAW_DATA_DIR / "breast_cancer_original.csv",
):
    """Train/test estratificado desde los datos crudos.

    X conserva los NaN: la imputacion ocurre dentro del pipeline para evitar
    fuga de datos entre pliegues.
    """
    df = pd.read_csv(input_path)
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    X = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    y = df[TARGET_COL].map(CLASS_MAP).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    logger.info(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


def evaluate_representations(X_train, X_test, y_train, y_test):
    """5-fold CV estratificada sobre train + evaluacion en test independiente."""
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    pipelines = build_pipelines()

    filas_cv, filas_test, raw_cv = {}, {}, {}
    for nombre, pipe in pipelines.items():
        res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=METRICS)
        raw_cv[nombre] = res
        filas_cv[nombre] = {
            "Accuracy": f"{res['test_accuracy'].mean():.4f} ± {res['test_accuracy'].std():.4f}",
            "Precision": f"{res['test_precision'].mean():.4f}",
            "Recall": f"{res['test_recall'].mean():.4f}",
            "F1": f"{res['test_f1'].mean():.4f}",
        }
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        filas_test[nombre] = {
            "Accuracy": f"{accuracy_score(y_test, pred):.4f}",
            "Precision": f"{precision_score(y_test, pred):.4f}",
            "Recall": f"{recall_score(y_test, pred):.4f}",
            "F1": f"{f1_score(y_test, pred):.4f}",
        }

    df_cv = pd.DataFrame(filas_cv)
    df_test = pd.DataFrame(filas_test)
    logger.info(f"5-Fold CV sobre TRAIN:\n{df_cv.to_string()}")
    logger.info(f"TEST independiente:\n{df_test.to_string()}")
    return pipelines, df_cv, df_test, raw_cv


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "breast_cancer_original.csv",
    metrics_dir: Path = REPORTS_DIR,
    model_path: Path = MODELS_DIR / "decision_tree_atipia.pkl",
):
    """Entrena y evalua el arbol de decision con cada representacion."""
    X_train, X_test, y_train, y_test = prepare_train_test(input_path)
    pipelines, df_cv, df_test, _ = evaluate_representations(X_train, X_test, y_train, y_test)

    metrics_dir.mkdir(parents=True, exist_ok=True)
    df_cv.to_csv(metrics_dir / "metricas_cv.csv")
    df_test.to_csv(metrics_dir / "metricas_test.csv")
    logger.success(f"Metricas exportadas a {metrics_dir}")

    # La Rep. 3 se conserva como modelo final: rendimiento comparable con menos
    # variables y reglas directamente interpretables en terminos clinicos.
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipelines[REP3_NAME], model_path)
    logger.success(f"Modelo final ({REP3_NAME}) -> {model_path}")


if __name__ == "__main__":
    app()
