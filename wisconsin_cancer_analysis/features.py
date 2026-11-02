from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import typer

from wisconsin_cancer_analysis.config import PROCESSED_DATA_DIR
from wisconsin_cancer_analysis.dataset import FEATURE_COLS, TARGET_COL

app = typer.Typer()

# Variables redundantes y altamente correlacionadas que se fusionan en la Rep. 3
ATYPIA_COLS = [
    "Uniformity_of_cell_size",
    "Uniformity_of_cell_shape",
    "Bare_nuclei",
]
ATYPIA_INDEX = "Indice_Atipia_Celular"
REP3_BASE_COLS = [c for c in FEATURE_COLS if c not in ATYPIA_COLS]

# Rangos clinicos de severidad: (0,3]->Bajo, (3,6]->Moderado, (6,10]->Alto
BIN_EDGES = [0, 3, 6, 10]
BIN_LABELS = [0, 1, 2]


def load_clean(input_path: Path = PROCESSED_DATA_DIR / "breast_cancer_limpio.csv") -> pd.DataFrame:
    df = pd.read_csv(input_path)
    logger.info(f"Dataset limpio cargado: {df.shape} desde {input_path}")
    return df


def build_rep1_zscore(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Rep. 1: Numerica estandarizada (Z-score), 9 variables."""
    X = StandardScaler().fit_transform(df_clean[FEATURE_COLS])
    df_rep = pd.DataFrame(X, columns=[f"{c}_zscore" for c in FEATURE_COLS])
    df_rep[TARGET_COL] = df_clean[TARGET_COL].values
    return df_rep


def _bin_series(s: pd.Series) -> pd.Series:
    return pd.cut(s, bins=BIN_EDGES, labels=BIN_LABELS).astype("int64")


def build_rep2_binning(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Rep. 2: Discretizacion por rangos clinicos (binning), 9 variables."""
    df_rep = pd.DataFrame({f"{c}_bin": _bin_series(df_clean[c]) for c in FEATURE_COLS})
    df_rep[TARGET_COL] = df_clean[TARGET_COL].values
    return df_rep


def build_rep3_atipia(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Rep. 3: Binning + indice de atipia celular agrupado, 7 variables."""
    df = df_clean.copy()
    df[ATYPIA_INDEX] = df[ATYPIA_COLS].mean(axis=1)

    cols = REP3_BASE_COLS + [ATYPIA_INDEX]
    df_rep = pd.DataFrame({f"{c}_bin": _bin_series(df[c]) for c in cols})
    df_rep = df_rep.rename(columns={f"{ATYPIA_INDEX}_bin": "Atipia_Celular_bin"})
    df_rep[TARGET_COL] = df_clean[TARGET_COL].astype("int64").values
    return df_rep


REPRESENTATIONS = {
    "representacion_1_zscore.csv": build_rep1_zscore,
    "representacion_2_binning.csv": build_rep2_binning,
    "representacion_3_atipia.csv": build_rep3_atipia,
}


# --- Transformadores para pipelines de sklearn --------------------------------
# Viven en este modulo (y no en modeling/train.py) para que los pipelines
# serializados con joblib puedan reimportarlos por su ruta de modulo real.

_IDX = {c: i for i, c in enumerate(FEATURE_COLS)}


def _bin_array(a: np.ndarray) -> np.ndarray:
    # Equivalente a pd.cut con cortes en 3 y 6 -> categorias 0, 1, 2
    return (a > 3).astype(int) + (a > 6).astype(int)


def binning_rep2(a: np.ndarray) -> np.ndarray:
    """Discretizacion de las 9 variables originales (Rep. 2)."""
    return _bin_array(a)


def binning_rep3(a: np.ndarray) -> np.ndarray:
    """Binning de 6 variables + indice de atipia agrupado (Rep. 3)."""
    atipia = a[:, [_IDX[c] for c in ATYPIA_COLS]].mean(axis=1, keepdims=True)
    resto = a[:, [_IDX[c] for c in REP3_BASE_COLS]]
    return _bin_array(np.hstack([resto, atipia]))


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "breast_cancer_limpio.csv",
    output_dir: Path = PROCESSED_DATA_DIR,
):
    """Genera las tres representaciones del dataset limpio."""
    df_clean = load_clean(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, builder in REPRESENTATIONS.items():
        df_rep = builder(df_clean)
        out = output_dir / filename
        df_rep.to_csv(out, index=False)
        logger.success(f"{filename}: {df_rep.shape}")


if __name__ == "__main__":
    app()
