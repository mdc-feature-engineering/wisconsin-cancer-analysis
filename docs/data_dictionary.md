### Diccionario de Datos: Breast Cancer Wisconsin (Original)

| Nombre de la Variable | Rol | Tipo de Dato | Rango / Valores | Nulos | Descripción Clínica / Significado |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **`Sample_code_number`** | Identificador (ID) | Categórico / Cadena | Único por paciente | No | Código de identificación de la muestra. **Sin valor predictivo** (debe excluirse). |
| **`Clump_thickness`** | Característica | Entero | 1 – 10 | No | **Grosor del cúmulo:** Mide la tendencia de las células a amontonarse en capas. |
| **`Uniformity_of_cell_size`** | Característica | Entero | 1 – 10 | No | **Uniformidad del tamaño celular:** Evalúa la consistencia en el tamaño de las células. |
| **`Uniformity_of_cell_shape`** | Característica | Entero | 1 – 10 | No | **Uniformidad de la forma celular:** Evalúa la regularidad o variabilidad en la forma celular. |
| **`Marginal_adhesion`** | Característica | Entero | 1 – 10 | No | **Adhesión marginal:** Mide qué tan unidas están las células entre sí. |
| **`Single_epithelial_cell_size`** | Característica | Entero | 1 – 10 | No | **Tamaño de la célula epitelial:** Evalúa si las células epiteliales están agrandadas. |
| **`Bare_nuclei`** | Característica | Entero | 1 – 10 | **Sí** | **Núcleos desnudos:** Presencia de núcleos celulares sin citoplasma circundante. Única variable con datos faltantes. |
| **`Bland_chromatin`** | Característica | Entero | 1 – 10 | No | **Cromatina suave/basta:** Describe la textura del material genético en el núcleo. |
| **`Normal_nucleoli`** | Característica | Entero | 1 – 10 | No | **Nucléolos normales:** Mide la visibilidad y tamaño de los nucléolos en la célula. |
| **`Mitoses`** | Característica | Entero | 1 – 10 | No | **Mitosis:** Mide la frecuencia de división celular. |
| **`Class`** | Variable Objetivo (*Target*) | Categórico | 2 (Benigno)<br>4 (Maligno) | No | **Diagnóstico clínico final:** (`2` = Tumores Benignos, `4` = Tumores Malignos). |
