### Diccionario de Datos: Breast Cancer Wisconsin (Original)

Todas las características se evalúan en una escala de **1 a 10**, donde **1** es lo más
"benigno" (células regulares, homogéneas) y **10** lo más anaplásico (rasgos típicos de
células cancerosas).

| Nombre de la Variable | Rol | Tipo de Dato | Rango / Valores | Nulos | Descripción Clínica / Significado |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **`Sample_code_number`** | Identificador (ID) | Categórico / Cadena | Único por paciente | No | Código de identificación de la muestra. **Sin valor predictivo** (debe excluirse). |
| **`Clump_thickness`** | Característica | Entero | 1 – 10 | No | **Grosor del cúmulo:** mide la tendencia de las células a amontonarse en capas. `1` = células que no tienden a agruparse; `10` = gran grosor en células agrupadas (crecimiento descontrolado). |
| **`Uniformity_of_cell_size`** | Característica | Entero | 1 – 10 | No | **Uniformidad del tamaño celular:** consistencia del tamaño de las células observadas. `1` = tamaño regular; `10` = tamaños muy dispares (las células cancerosas muestran tamaños distintos). |
| **`Uniformity_of_cell_shape`** | Característica | Entero | 1 – 10 | No | **Uniformidad de la forma celular:** regularidad de la forma de las células de un mismo tejido. `1` = forma regular; `10` = forma irregular. |
| **`Marginal_adhesion`** | Característica | Entero | 1 – 10 | No | **Adhesión marginal:** capacidad de las células de unirse entre sí o con la matriz extracelular. La **pérdida de adhesión sugiere malignidad**. `1` = buena adhesión; `10` = adhesión perdida. |
| **`Single_epithelial_cell_size`** | Característica | Entero | 1 – 10 | No | **Tamaño de la célula epitelial:** tamaño relativo y uniformidad de células epiteliales. Las células normales son pequeñas y uniformes; `10` = tamaños variables y no uniformes. |
| **`Bare_nuclei`** | Característica | Entero | 1 – 10 | **Sí** | **Núcleos desnudos:** núcleos celulares sin citoplasma que los rodee. Las células sanas mantienen el núcleo dentro del citoplasma; al procesar la muestra, **el núcleo se desprende del citoplasma en células dañadas (como en cáncer)**. `1` = núcleo contenido; `10` = núcleos desprendidos. Única variable con datos faltantes (`?`). |
| **`Bland_chromatin`** | Característica | Entero | 1 – 10 | No | **Cromatina suave/basta:** textura del material genético (ADN) dentro del núcleo. La cromatina uniforme es propia de células benignas; `10` = textura irregular y abnormal. |
| **`Normal_nucleoli`** | Característica | Entero | 1 – 10 | No | **Nucléolos normales:** aspecto y tamaño de los nucléolos dentro del núcleo. En células benignas son pequeños y apenas visibles; `10` = nucléolos grandes y deformes. |
| **`Mitoses`** | Característica | Entero | 1 – 10 | No | **Mitosis:** número de divisiones celulares observadas. `1` = poca actividad de división; `10` = reproducción celular rápida y agresiva. |
| **`Class`** | Variable Objetivo (*Target*) | Categórico | 2 (Benigno)<br>4 (Maligno) | No | **Diagnóstico clínico final:** (`2` = Tumores Benignos, `4` = Tumores Malignos). En `data/processed/` se recodifica como `0` = Benigno, `1` = Maligno. |

#### Variables derivadas (Representación 3)

| Nombre de la Variable | Rol | Tipo de Dato | Rango / Valores | Descripción |
| :--- | :--- | :--- | :--- | :--- |
| **`Atipia_Celular`** | Característica derivada | Entero (bin) | 0 – 2 | Promedio de `Uniformity_of_cell_size`, `Uniformity_of_cell_shape` y `Bare_nuclei`, discretizado con los mismos rangos clínicos (0: Bajo, 1: Moderado, 2: Alto). Las tres variables miden el mismo fenómeno clínico —**atipia celular** (irregularidad celular y núcleos desprendidos del citoplasma)—, por lo que se fusionan para reducir redundancia. |
| **`*_bin`** | Característica transformada | Entero | 0 – 2 | Versión discretizada de cada variable clínica: `(0,3]` → 0 (Bajo/Normal), `(3,6]` → 1 (Moderado), `(6,10]` → 2 (Alto/Atípico). |
| **`*_zscore`** | Característica transformada | Continuo | ~[-3, 3] | Versión estandarizada (media 0, desviación estándar 1) de cada variable clínica. |

*Fuentes: [UCI ML Repo — Breast Cancer Wisconsin (Original)](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original) y `references/Breast Cancer Wisconsin.pdf`.*
