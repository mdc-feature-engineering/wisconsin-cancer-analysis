# Breast Cancer Wisconsin (Original)

**Objetivo:** detectar si un tumor de mama es maligno o benigno.

**Fuente:** [UCI ML Repository](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original) —
699 muestras, 9 características citológicas (enteros de 1 a 10) y 1 variable objetivo.

---

## 1. Análisis inicial del dataset

### 1.1 Tipos de variables

| Variable | Rol | Tipo | Nulos |
| :--- | :--- | :--- | :---: |
| `Sample_code_number` | Identificador | Categórico | no |
| `Clump_thickness` | Característica | Entero | no |
| `Uniformity_of_cell_size` | Característica | Entero | no |
| `Uniformity_of_cell_shape` | Característica | Entero | no |
| `Marginal_adhesion` | Característica | Entero | no |
| `Single_epithelial_cell_size` | Característica | Entero | no |
| `Bare_nuclei` | Característica | Entero | **sí** |
| `Bland_chromatin` | Característica | Entero | no |
| `Normal_nucleoli` | Característica | Entero | no |
| `Mitoses` | Característica | Entero | no |
| `Class` | Objetivo | Categórico (2/4) | no |

**Limpieza aplicada:** se excluyó `Sample_code_number` (ID sin valor predictivo),
se eliminaron 236 filas duplicadas (699 → 463 muestras) y los `?` de
`Bare_nuclei` se imputaron por **mediana** (robusta al fuerte sesgo de la escala).
`Class` se recodificó: `2` → 0 (Benigno), `4` → 1 (Maligno).

### 1.2 Descripción clínica de las variables

Escala 1–10: **1** = lo más "benigno" (células regulares y homogéneas) y
**10** = lo más anaplásico (rasgos propios de células cancerosas).

- **`Clump_thickness`** — grosor medido en células agrupadas. Valores cercanos a
  1 indican que las células no tienden a agruparse; cercanos a 10, mayor grosor
  en células agrupadas.
- **`Uniformity_of_cell_size`** — consistencia del tamaño de las células.
  Cercano a 1: tamaño regular; cercano a 10: tamaños dispares (típico de
  células cancerosas).
- **`Uniformity_of_cell_shape`** — qué tan regulares son las células de un mismo
  tejido en su forma. Cercano a 1: forma regular; cercano a 10: forma irregular.
- **`Marginal_adhesion`** — capacidad de las células de unirse entre sí o con la
  matriz extracelular. La **pérdida de adhesión sugiere malignidad**.
  Cercano a 1: buena adhesión; cercano a 10: adhesión perdida.
- **`Single_epithelial_cell_size`** — tamaño relativo y uniformidad de células
  epiteliales. Las células normales son pequeñas y uniformes; cercano a 10:
  tamaños variables y no uniformes.
- **`Bare_nuclei`** — núcleos celulares sin citoplasma que los rodee. Las células
  sanas mantienen el núcleo dentro del citoplasma; al procesar la muestra, el
  núcleo se suele **desprender del citoplasma en células dañadas (como en
  cáncer)**. Cercano a 1: núcleo contenido; cercano a 10: núcleos desprendidos.
- **`Bland_chromatin`** — textura de la cromatina (el material que forma el ADN)
  dentro del núcleo. La cromatina uniforme es propia de células benignas;
  cercano a 10: textura irregular y abnormal.
- **`Normal_nucleoli`** — aspecto, tamaño y normalidad de los nucléolos
  celulares. En células benignas son pequeños y apenas visibles; cercano a 10:
  nucléolos grandes y deformes.
- **`Mitoses`** — número de divisiones celulares. Cercano a 1: poca actividad
  de división; cercano a 10: reproducción celular rápida y agresiva.

---

## 2. Propuestas de representación de datos

### Representación 1 — Numérica estandarizada (Z-score)

Las 9 características celulares se tratan como variables cuantitativas
continuas y se estandarizan a **media 0 y desviación estándar 1**. Conserva las
magnitudes originales y evita que las variables con mayor dispersión dominen el
aprendizaje. Orientada a modelos de distancia o probabilísticos (Naive Bayes,
KNN, regresión logística).

### Representación 2 — Discretización por rangos clínicos (Binning)

Las 9 variables se traducen de la escala 1–10 a **3 categorías ordinales de
severidad**, emulando el criterio diagnóstico:

- **0 (Bajo/Normal):** valores de 1 a 3 — tejido sano u homogéneo.
- **1 (Moderado):** valores de 4 a 6 — atipia intermedia.
- **2 (Alto/Atípico):** valores de 7 a 10 — sospecha de malignidad.

Filtra ruido de medición y produce reglas de decisión simples e interpretables
en términos clínicos. Orientada a modelos basados en reglas (Árboles de
Decisión).

### Representación 3 — Binning + índice de Atipia Celular (7 variables)

Las células cancerosas se caracterizan por un crecimiento anormal y
descontrolado, que genera una estructura irregular. En el dataset se observó
**alta correlación** entre `Uniformity_of_cell_size`, `Uniformity_of_cell_shape`
y `Bare_nuclei`, y además son las variables con **mayor diferencia entre
grupos**: medianas de 1.0 en benignos frente a 6.0, 7.0 y 8.0 en malignos.
Clínicamente, las tres describen el mismo fenómeno —**atipia celular**—:
células de tamaño y forma irregulares, y núcleos que se desprenden del
citoplasma en células dañadas.

Por esta razón, las tres variables se agruparon en un **índice de atipia
celular** (el promedio de las tres), reduciendo el conjunto de **9 a 7
variables** y evitando redundancia. Finalmente, las 7 variables resultantes se
discretizaron con los mismos rangos clínicos de la Representación 2.

---

## 3. Modelo predictivo

**Elegido: Árbol de Decisión (`max_depth=4`).**

La rúbrica pedía al menos uno entre Naive Bayes, KMeans o KNN, permitiendo otro
algoritmo con el que se tuviera experiencia. Se eligió el árbol porque:

- Produce **reglas de decisión simples e interpretables** en términos clínicos
  (ej. *"si la atipia celular es Alta, crece la probabilidad de malignidad"*).
- Se **alinea naturalmente con la representación discretizada** por severidad:
  cada corte del árbol equivale a un umbral clínico.
- `max_depth=4` limita el sobreajuste en un dataset pequeño.

**Evaluación:** split 80/20 estratificado + validación cruzada estratificada de
5 pliegues sobre el entrenamiento. La imputación por mediana ocurre **dentro**
del pipeline de cada pliegue, sin fuga de datos.

> **Nota metodológica:** los CSV de `data/processed/` se generan sobre las 463
> muestras únicas (tras deduplicar), pero la evaluación del modelo se realiza
> sobre las 699 observaciones originales (test = 140, el 20%). Las
> transformaciones de cada representación se aplican dentro del pipeline
> durante el entrenamiento.
>
> *Alternativa considerada:* unificar el criterio y evaluar también sobre
> datos deduplicados. Esto evitaría que pares duplicados caigan en train y
> test simultáneamente (lo que puede inflar las métricas), pero cambiaría
> ligeramente los resultados reportados.

---

## 4. Comparación de resultados

Métricas sobre el **test independiente** (20%):

| Métrica | R1: Z-score (9 vars) | R2: Binning (9 vars) | R3: Atipia (7 vars) |
| :--- | :---: | :---: | :---: |
| Accuracy | 0.9286 | 0.9357 | **0.9500** |
| Precision | 0.9318 | 0.9333 | **0.9362** |
| Recall | 0.8542 | 0.8750 | **0.9167** |
| F1 Score | 0.8913 | 0.9032 | **0.9263** |

**Análisis crítico:** R3 supera a R1 y R2 en *todas* las métricas usando solo 7
variables —el índice `Atipia_Celular` concentra ~92% de la importancia Gini del
árbol—. La ganancia más clara está en **recall** (0.92 vs. 0.85–0.88): con la
representación agrupada el modelo deja escapar menos tumores malignos, lo que
en diagnóstico médico es el error más costoso.

**Reflexión:** la representación no solo afecta el desempeño sino la
*interpretación*. R1 conserva magnitudes pero produce cortes poco intuitivos;
R2/R3 sacrifican resolución a cambio de reglas que un médico puede leer
directamente. Agrupar variables correlacionadas por su significado clínico
(atipia) redujo redundancia **y** mejoró el modelo —la ingeniería de
características guiada por el dominio superó a las transformaciones puramente
estadísticas.

---

## Fuentes

- [UCI ML Repository — Breast Cancer Wisconsin (Original)](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original)
- [Artículo de referencia (diagnostics-10-00136)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7151177/pdf/diagnostics-10-00136.pdf)
