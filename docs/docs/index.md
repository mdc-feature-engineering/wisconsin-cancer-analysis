# Breast Cancer Wisconsin — Representación de Datos y Clasificación

Proyecto de la materia de **Ingeniería de Características** (maestría):
explorar y comparar propuestas de representación de datos y evaluar su impacto
en un modelo de clasificación.

## Objetivo

Detectar si un tumor de mama es **maligno o benigno** a partir de 9
características citológicas (escala 1–10) del dataset
[Breast Cancer Wisconsin (Original)](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original)
— 699 muestras.

## Pipeline del proyecto

1. **Análisis inicial** — tipos de variables, limpieza (exclusión del ID,
   imputación por mediana de `Bare_nuclei`), recodificación de `Class`
   (2→0 Benigno, 4→1 Maligno).
2. **Representaciones** —
   `R1` Z-score (9 vars continuas),
   `R2` binning clínico (9 vars en 3 niveles de severidad),
   `R3` binning + índice de `Atipia_Celular` (7 vars).
3. **Modelo** — Árbol de Decisión (`max_depth=4`), evaluado por
   representación con 5-fold CV estratificada y test independiente del 20%.
4. **Resultados** — métricas en `reports/metricas_*.csv`, figuras en
   `reports/figures/` y presentación de 2 diapositivas en
   `reports/presentacion.{pptx,pdf}`.

La lógica vive en el paquete `wisconsin_cancer_analysis/`; los notebooks en
`notebooks/` orquestan las llamadas y documentan las decisiones.
