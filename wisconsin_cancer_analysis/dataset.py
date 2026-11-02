from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import typer

from wisconsin_cancer_analysis.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

ID_COL = "Sample_code_number"
TARGET_COL = "Class"
CLASS_MAP = {2: 0, 4: 1}  # 2 = Benigno, 4 = Maligno

FEATURE_COLS = [
    "Clump_thickness",
    "Uniformity_of_cell_size",
    "Uniformity_of_cell_shape",
    "Marginal_adhesion",
    "Single_epithelial_cell_size",
    "Bare_nuclei",
    "Bland_chromatin",
    "Normal_nucleoli",
    "Mitoses",
]


def fetch_dataset(output_path: Path = RAW_DATA_DIR / "breast_cancer_original.csv") -> pd.DataFrame:
    """Descarga el dataset desde UCI ML Repo (id=15) y lo guarda en data/raw."""
    from ucimlrepo import fetch_ucirepo

    repo = fetch_ucirepo(id=15)
    df = pd.concat([repo.data.features, repo.data.targets], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.success(f"Dataset descargado: {df.shape} -> {output_path}")
    return df


def load_raw(input_path: Path = RAW_DATA_DIR / "breast_cancer_original.csv") -> pd.DataFrame:
    df = pd.read_csv(input_path)
    logger.info(f"Datos crudos cargados: {df.shape} desde {input_path}")
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza comun a todas las representaciones.

    - Elimina el identificador de paciente (sin valor predictivo).
    - Elimina filas duplicadas.
    - Convierte '?' a NaN y coacciona las variables a numericas.
    - Imputa valores faltantes con la mediana (robusta al sesgo del 1-10).
    - Recodifica Class: 2 (Benigno) -> 0, 4 (Maligno) -> 1.
    """
    df = df.copy()
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    duplicados = int(df.duplicated().sum())
    df = df.drop_duplicates()
    logger.info(f"Filas duplicadas eliminadas: {duplicados}")

    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col].replace("?", np.nan))

    imputer = SimpleImputer(strategy="median")
    df[FEATURE_COLS] = imputer.fit_transform(df[FEATURE_COLS])
    logger.info(f"Valores nulos restantes: {int(df.isnull().sum().sum())}")

    df[TARGET_COL] = df[TARGET_COL].map(CLASS_MAP)
    return df


def export_descriptive_stats(
    df: pd.DataFrame,
    output_path: Path = PROCESSED_DATA_DIR / "estadisticas_descriptivas_cancer.csv",
) -> pd.DataFrame:
    """Tabla mean/std/median por clase sobre los datos crudos (Class 2/4)."""
    df = df.copy()
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    resumen = df.groupby(TARGET_COL)[FEATURE_COLS].agg(["mean", "std", "median"]).T
    resumen = resumen.reset_index()
    resumen.columns = ["Variable", "Estadistico", "Benigno", "Maligno"]
    resumen[["Benigno", "Maligno"]] = resumen[["Benigno", "Maligno"]].round(2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    resumen.to_csv(output_path, index=False)
    logger.success(f"Estadisticas descriptivas -> {output_path}")
    return resumen


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "breast_cancer_original.csv",
    output_path: Path = PROCESSED_DATA_DIR / "breast_cancer_limpio.csv",
    fetch: bool = False,
):
    """Genera el dataset limpio y la tabla de estadisticas descriptivas."""
    df = fetch_dataset(input_path) if fetch or not input_path.exists() else load_raw(input_path)
    export_descriptive_stats(df)

    df_clean = clean_dataset(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    logger.success(f"Dataset limpio: {df_clean.shape} -> {output_path}")


if __name__ == "__main__":
    app()
