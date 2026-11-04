# Representación 2b — Binning por Clustering (KMeans)

Nota técnica sobre la variante de discretización con umbrales aprendidos de
los datos, añadida como cuarta representación del proyecto.

## Motivación

La Representación 2 discretiza cada variable clínica (escala 1–10) en 3
categorías ordinales con **cortes fijos**: 1–3 Bajo, 4–6 Moderado, 7–10 Alto.
Es interpretable pero arbitraria — los límites vienen de una regla externa,
no de la forma de los datos.

La pregunta natural: **¿y si los propios datos decidieran dónde poner los
cortes?** Para responderla se generó una variante donde cada variable define
sus propios umbrales mediante clustering.

## Método

- **Algoritmo:** `KBinsDiscretizer` de scikit-learn con `n_bins=3`,
  `encode='ordinal'`, `strategy='kmeans'` — equivale a aplicar KMeans (k=3)
  por variable y ordenar los clusters por su centroide.
- **Mapeo:** cluster de centroide más bajo → `0` (Bajo), medio → `1`
  (Moderado), alto → `2` (Alto). Misma escala ordinal que R2, directamente
  comparable.
- **Sin fuga de datos:** el discretizador es un paso *dentro* del pipeline,
  así que los umbrales se aprenden solo del train de cada pliegue (a
  diferencia de un `FunctionTransformer` sin estado, que re-entrenaría sobre
  el test).

## Umbrales aprendidos

Cortes internos de KMeans por variable vs. la regla clínica fija
(`reports/binning_kmeans_umbrales.csv`):

| Variable | Bajo\|Moderado | Moderado\|Alto |
| :--- | :---: | :---: |
| Clump_thickness | 4.02 | 7.34 |
| Uniformity_of_cell_size | 3.40 | 7.29 |
| Uniformity_of_cell_shape | 3.49 | 7.32 |
| Marginal_adhesion | 3.36 | 7.26 |
| Single_epithelial_cell_size | 3.64 | 7.16 |
| Bare_nuclei | 3.24 | 7.35 |
| Bland_chromatin | 4.34 | 7.59 |
| Normal_nucleoli | 3.44 | 7.42 |
| Mitoses | 3.27 | 7.29 |
| **Regla clínica (R2)** | **3.00** | **6.00** |

Lectura:

- El primer corte (~3.3–4.3) casi coincide con el 3 clínico → la regla
  médica para "Bajo" quedó validada por los datos.
- El segundo corte (~7.2–7.6) queda notablemente **más alto** que el 6
  clínico → para la distribución real, el grupo "Alto" empieza ~7.3 y un
  valor de 7 todavía es "Moderado".
- `Bland_chromatin` y `Clump_thickness` tienen la banda Moderado más ancha
  (4.3–7.6): sus distribuciones intermedias son más difusas.
- Hasta `Mitoses` (muy sesgada, ~90% valor 1) produjo cortes coherentes.

## Resultados (test independiente, 140 muestras)

| Métrica | R1 Z-score | R2 Clínico | **R2b KMeans** | R3 Atipia |
| :--- | :---: | :---: | :---: | :---: |
| Accuracy | 0.929 | 0.936 | **0.957** | 0.950 |
| Precision | 0.932 | 0.933 | 0.920 | **0.936** |
| Recall | 0.854 | 0.875 | **0.958** | 0.917 |
| F1 | 0.891 | 0.903 | **0.939** | 0.926 |

En validación cruzada (5-fold sobre train), R3 sigue liderando (F1 0.950 vs.
0.919 de R2b).

## Análisis crítico

- **A favor:** los umbrales ajustados por variable capturaron mejor la
  frontera clínica real — el grupo "Alto" queda más puro y el recall sube a
  0.958, detectando ~2 tumores malignos más que R3 en test (46 de 48 vs.
  44 de 48). En diagnóstico, eso es lo que más importa.
- **Costo:** cada variable tiene fronteras distintas → un "2" ya no
  significa "7–10" en todas las columnas. Se pierde la *interpretación
  uniforme* de la regla clínica y se necesita una tabla de conversión por
  variable.
- **Matiz honesto:** el test tiene solo 48 malignos; la ventaja en recall
  puede ser parcialmente suerte del split. En CV gana R3 → la conclusión
  correcta es que R2b es *competitivo*, no objetivamente superior. Esa
  tensión test-vs-CV es un hallazgo legítimo del experimento.

## Dónde está en el código

- `wisconsin_cancer_analysis/features.py` — `build_rep2_kmeans()` (exporta
  `data/processed/representacion_2b_kmeans.csv`) y `kmeans_umbrales()`
  (exporta `reports/binning_kmeans_umbrales.csv`).
- `wisconsin_cancer_analysis/modeling/train.py` — pipeline
  `"Rep. 2b: Binning KMeans (9 vars)"`.
- Notebook `03.initial_analysis.ipynb` — celdas con la tabla de umbrales.
