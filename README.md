# wisconsin-cancer-analysis

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Analisis de datos sobre el cancer de mama

## Ejecución

```bash
python -m venv .venv            # o: make create_environment
.venv\Scripts\activate          # Windows

make pipeline                   # instala requirements + corre data -> features -> train -> figures
```

Cada etapa tambien puede correrse por separado:

```bash
python -m wisconsin_cancer_analysis.dataset          # limpieza -> data/processed/breast_cancer_limpio.csv
python -m wisconsin_cancer_analysis.features         # 3 representaciones -> data/processed/
python -m wisconsin_cancer_analysis.modeling.train   # metricas -> reports/ ; modelo -> models/
python -m wisconsin_cancer_analysis.modeling.predict # inferencia con el modelo guardado
python -m wisconsin_cancer_analysis.plots            # figuras -> reports/figures/
```

Los notebooks en `notebooks/` usan estas mismas funciones del paquete.

## Organización del proyecto

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling
│   │                     (breast_cancer_limpio, representacion_1_zscore,
│   │                      representacion_2_binning, representacion_3_atipia,
│   │                      estadisticas_descriptivas_cancer).
│   └── raw            <- The original, immutable data dump
│                         (breast_cancer_original.csv).
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         wisconsin_cancer_analysis and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   │                    (metricas_cv.csv, metricas_test.csv)
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── wisconsin_cancer_analysis   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes wisconsin_cancer_analysis a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train and evaluate models per representation
    │
    └── plots.py                <- Code to create visualizations

