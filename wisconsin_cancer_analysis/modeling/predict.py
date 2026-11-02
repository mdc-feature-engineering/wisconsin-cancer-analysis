from pathlib import Path

import joblib
from loguru import logger
import pandas as pd
import typer

from wisconsin_cancer_analysis.config import MODELS_DIR, PROCESSED_DATA_DIR
from wisconsin_cancer_analysis.dataset import FEATURE_COLS

app = typer.Typer()


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "breast_cancer_limpio.csv",
    model_path: Path = MODELS_DIR / "decision_tree_atipia.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "predicciones.csv",
):
    """Corre inferencia con el modelo entrenado (Rep. 3).

    El archivo de entrada debe contener las 9 variables clinicas originales;
    la imputacion y la discretizacion ocurren dentro del pipeline.
    """
    df = pd.read_csv(features_path)
    X = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")

    model = joblib.load(model_path)
    pred = model.predict(X)

    df_out = df.copy()
    df_out["Prediccion"] = pred
    df_out["Diagnostico"] = df_out["Prediccion"].map({0: "Benigno", 1: "Maligno"})

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(predictions_path, index=False)
    logger.success(f"Predicciones ({df_out.shape[0]} filas) -> {predictions_path}")


if __name__ == "__main__":
    app()
