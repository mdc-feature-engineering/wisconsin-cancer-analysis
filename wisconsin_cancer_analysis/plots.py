from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
from sklearn.tree import plot_tree
import typer

from wisconsin_cancer_analysis.config import (
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from wisconsin_cancer_analysis.dataset import CLASS_MAP, FEATURE_COLS, ID_COL, TARGET_COL
from wisconsin_cancer_analysis.features import ATYPIA_COLS, REP3_BASE_COLS
from wisconsin_cancer_analysis.modeling.train import (
    REP3_FEATURES,
    REP3_NAME,
    evaluate_representations,
    prepare_train_test,
)

app = typer.Typer()

CLASS_NAMES = {0: "Benigno", 1: "Maligno"}
CLASS_COLORS = {0: "#4C9F70", 1: "#C0392B"}


def plot_distribucion_por_clase(
    input_path: Path = RAW_DATA_DIR / "breast_cancer_original.csv",
    output_path: Path = FIGURES_DIR / "distribucion_por_clase.png",
):
    """Histogramas de las 9 variables clinicas separados por diagnostico."""
    df = pd.read_csv(input_path)
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Diagnostico"] = df[TARGET_COL].map(CLASS_MAP).map(CLASS_NAMES)

    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    for ax, col in zip(axes.flat, FEATURE_COLS):
        for code, nombre in CLASS_NAMES.items():
            ax.hist(
                df.loc[df[TARGET_COL].map(CLASS_MAP) == code, col].dropna(),
                bins=np.arange(0.5, 11.5),
                alpha=0.65,
                label=nombre,
                color=CLASS_COLORS[code],
            )
        ax.set_title(col, fontsize=10)
        ax.set_xticks(range(1, 11))
    for ax in axes[:, 0]:
        ax.set_ylabel("N° muestras")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(
        "Distribución de las 9 variables clínicas por diagnóstico",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Figura -> {output_path}")


def plot_binning_vs_clase(
    input_path: Path = PROCESSED_DATA_DIR / "representacion_3_atipia.csv",
    output_path: Path = FIGURES_DIR / "binning_vs_clase.png",
    feature: str = "Atipia_Celular_bin",
):
    """Barras apiladas (%) de las categorias de severidad por diagnostico."""
    df = pd.read_csv(input_path)
    conteos = pd.crosstab(df[feature], df[TARGET_COL])
    tabla = conteos / conteos.sum(axis=0) * 100

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(tabla.columns))
    bottom = np.zeros(len(tabla.columns))
    etiquetas = {0: "Bajo / Normal", 1: "Moderado", 2: "Alto / Atípico"}
    colores = ["#4C9F70", "#E8B44A", "#C0392B"]
    for i, cat in enumerate(tabla.index):
        vals = tabla.loc[cat].values
        ax.bar(
            x,
            vals,
            bottom=bottom,
            color=colores[i],
            label=etiquetas.get(cat, str(cat)),
        )
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v > 4:
                n = conteos.loc[cat].iloc[xi]
                ax.text(
                    xi,
                    b + v / 2,
                    f"{v:.0f}%\n(n={n})",
                    ha="center",
                    va="center",
                    fontsize=9,
                )
        bottom += vals
    ax.set_xticks(x, [CLASS_NAMES.get(c, str(c)) for c in tabla.columns])
    ax.set_ylabel("% dentro de cada clase")
    ax.set_title(f"{feature}: severidad clínica por diagnóstico")
    ax.legend(title="Severidad", loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Figura -> {output_path}")


def plot_metricas_comparacion(
    df_test: pd.DataFrame,
    output_path: Path = FIGURES_DIR / "metricas_comparacion.png",
):
    """Barras agrupadas de Accuracy/Precision/Recall/F1 por representacion."""
    df = df_test.astype(float).T
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(df.columns))
    n_reps = len(df.index)
    width = 0.8 / n_reps
    for i, rep in enumerate(df.index):
        vals = df.loc[rep].values
        bars = ax.bar(x + (i - (n_reps - 1) / 2) * width, vals, width, label=rep)
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(x, df.columns)
    ax.set_ylim(0.85, 1.02)
    ax.set_ylabel("Puntaje (test independiente)")
    ax.set_title("Desempeño del Árbol de Decisión por representación")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Figura -> {output_path}")


def plot_arbol(
    pipeline,
    output_path: Path = FIGURES_DIR / "arbol_decision_atipia_agrupada.png",
):
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        pipeline.named_steps["tree"],
        feature_names=REP3_FEATURES,
        class_names=["Benigno", "Maligno"],
        filled=True,
        rounded=True,
        fontsize=10,
        ax=ax,
    )
    ax.set_title(
        "Árbol de Decisión: Representación con Atipia Celular Agrupada (7 Variables)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Figura -> {output_path}")


def compute_importances(pipeline, X_test, y_test) -> pd.DataFrame:
    """Importancia Gini (train) vs permutacion (test, delta F1)."""
    imp_gini = pd.Series(pipeline.named_steps["tree"].feature_importances_, index=REP3_FEATURES)
    perm = permutation_importance(
        pipeline, X_test, y_test, scoring="f1", n_repeats=30, random_state=42
    )
    perm_orig = pd.Series(perm.importances_mean, index=list(X_test.columns))
    imp_perm = pd.concat(
        [
            perm_orig[REP3_BASE_COLS],
            pd.Series({"Atipia_Celular": perm_orig[ATYPIA_COLS].sum()}),
        ]
    )
    df_imp = pd.DataFrame(
        {
            "Importancia Gini (train)": imp_gini,
            "Importancia permutacion (test, dF1)": imp_perm,
        }
    ).sort_values("Importancia Gini (train)", ascending=False)
    return df_imp


def plot_importancias(
    df_imp: pd.DataFrame,
    output_path: Path = FIGURES_DIR / "importancia_features_rep3.png",
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    df_imp["Importancia Gini (train)"].sort_values().plot.barh(ax=axes[0])
    axes[0].set_title("Importancia por impureza (Gini)")
    df_imp["Importancia permutacion (test, dF1)"].sort_values().plot.barh(ax=axes[1])
    axes[1].set_title("Importancia por permutacion (test)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Figura -> {output_path}")


def plot_correlacion(
    input_path: Path = RAW_DATA_DIR / "breast_cancer_original.csv",
    output_path: Path = FIGURES_DIR / "correlacion_vars.png",
):
    """Heatmap de la matriz de correlacion: justifica el agrupamiento de la Rep. 3."""
    df = pd.read_csv(input_path)
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    corr = df[FEATURE_COLS].corr()

    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=9)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.iloc[i, j]
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(v) > 0.6 else "black",
            )
    fig.colorbar(im, ax=ax, label="Correlación de Pearson", shrink=0.8)
    ax.set_title("Correlación entre las 9 variables clínicas", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Figura -> {output_path}")


def plot_boxplots_por_clase(
    input_path: Path = RAW_DATA_DIR / "breast_cancer_original.csv",
    output_path: Path = FIGURES_DIR / "boxplots_por_clase.png",
):
    """Boxplots por diagnostico: evidencia la separacion de medianas (1.0 vs 6-8)."""
    df = pd.read_csv(input_path)
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    y = df[TARGET_COL].map(CLASS_MAP)

    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    for ax, col in zip(axes.flat, FEATURE_COLS):
        grupos = [df.loc[y == c, col].dropna() for c in (0, 1)]
        bp = ax.boxplot(
            grupos,
            tick_labels=["Benigno", "Maligno"],
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 1.5},
        )
        for patch, c in zip(bp["boxes"], (0, 1)):
            patch.set_facecolor(CLASS_COLORS[c])
            patch.set_alpha(0.6)
        ax.set_title(col, fontsize=10)
        ax.set_ylim(0, 11)
    for ax in axes[:, 0]:
        ax.set_ylabel("Valor (escala 1-10)")
    fig.suptitle("Separación de medianas por diagnóstico", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Figura -> {output_path}")


def plot_scatter_atipia(
    input_path: Path = RAW_DATA_DIR / "breast_cancer_original.csv",
    output_path: Path = FIGURES_DIR / "scatter_atipia.png",
    x_col: str = "Uniformity_of_cell_size",
    y_col: str = "Bare_nuclei",
):
    """Dispersion 2D (con jitter) de dos variables de atipia por diagnostico."""
    df = pd.read_csv(input_path)
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[x_col, y_col])
    y = df[TARGET_COL].map(CLASS_MAP)
    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for code, nombre in CLASS_NAMES.items():
        mask = y == code
        ax.scatter(
            df.loc[mask, x_col] + rng.uniform(-0.3, 0.3, mask.sum()),
            df.loc[mask, y_col] + rng.uniform(-0.3, 0.3, mask.sum()),
            alpha=0.5,
            s=22,
            color=CLASS_COLORS[code],
            label=nombre,
        )
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_xticks(range(1, 11))
    ax.set_yticks(range(1, 11))
    ax.set_title(
        f"{x_col} vs. {y_col}: redundancia y separación de clases",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Figura -> {output_path}")


def plot_confusion_matrix(
    pipeline,
    X_test,
    y_test,
    output_path: Path = FIGURES_DIR / "confusion_matrix_rep3.png",
):
    """Matriz de confusion del modelo final (Rep. 3) sobre el test."""
    pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, pred)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    nombres = ["Benigno", "Maligno"]
    ax.set_xticks([0, 1], [f"Pred. {n}" for n in nombres])
    ax.set_yticks([0, 1], [f"Real {n}" for n in nombres])
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(
        f"Matriz de confusión — Rep. 3 en test (n={len(y_test)})",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Figura -> {output_path}")


def plot_cv_vs_test(
    raw_cv: dict,
    df_test: pd.DataFrame,
    output_path: Path = FIGURES_DIR / "cv_vs_test.png",
):
    """CV (media +- desv. en train) vs. test independiente, por metrica y representacion."""
    metricas = ["accuracy", "precision", "recall", "f1"]
    titulos = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1"}
    reps = list(raw_cv.keys())
    test_vals = df_test.astype(float)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, m in zip(axes.flat, metricas):
        x = np.arange(len(reps))
        medias = [raw_cv[r][f"test_{m}"].mean() for r in reps]
        desv = [raw_cv[r][f"test_{m}"].std() for r in reps]
        ax.bar(x - 0.2, medias, 0.4, yerr=desv, capsize=4, label="CV (media ± desv.)")
        ax.bar(x + 0.2, test_vals.loc[titulos[m]].values, 0.4, label="Test")
        ax.set_xticks(x, [r.split(":")[0] for r in reps], fontsize=9)
        ax.set_title(titulos[m])
        ax.set_ylim(0.75, 1.05)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(
        "Validación cruzada (train) vs. test independiente — sin sobreajuste visible",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Figura -> {output_path}")


@app.command()
def main():
    """Regenera todas las figuras del reporte."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_distribucion_por_clase()
    plot_correlacion()
    plot_boxplots_por_clase()
    plot_scatter_atipia()
    plot_binning_vs_clase()

    X_train, X_test, y_train, y_test = prepare_train_test()
    pipelines, _, df_test, raw_cv = evaluate_representations(X_train, X_test, y_train, y_test)
    plot_metricas_comparacion(df_test)
    plot_cv_vs_test(raw_cv, df_test)

    pipe_rep3 = pipelines[REP3_NAME]
    plot_arbol(pipe_rep3)
    plot_confusion_matrix(pipe_rep3, X_test, y_test)
    df_imp = compute_importances(pipe_rep3, X_test, y_test)
    logger.info(f"Importancias:\n{df_imp.round(4).to_string()}")
    plot_importancias(df_imp)


if __name__ == "__main__":
    app()
