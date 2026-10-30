# 🧾 Similitud de documentos con Bag-of-Words y similitud del coseno

Esta práctica de **Procesamiento de Lenguaje Natural** estudia la similitud entre 12 documentos relacionados con la **tecnología móvil** y los **vehículos eléctricos**.

Cada documento se preprocesa y se representa como un vector numérico mediante el modelo **Bag-of-Words**. Posteriormente, se calcula la similitud del coseno entre todos los pares de documentos utilizando **PyTorch**. Como ampliación opcional, también se calculan vectores **TF-IDF** y se comparan sus resultados con los obtenidos mediante Bag-of-Words.

```text
documentos originales
        ↓
preprocesamiento
        ↓
vocabulario global
        ↓
vectores BoW / TF-IDF
        ↓
similitud del coseno
        ↓
análisis de resultados
```

## 🎯 Objetivos

- Preprocesar documentos escritos en español.
- Construir un vocabulario global con las palabras de todos los documentos.
- Representar cada documento mediante un vector Bag-of-Words.
- Calcular la similitud del coseno entre todos los pares de documentos.
- Analizar la relación entre la similitud y los temas tratados.
- Calcular, como ampliación, la similitud utilizando vectores TF-IDF.
- Comparar los resultados de Bag-of-Words y TF-IDF.

## 🧠 Conceptos principales

### Bag-of-Words

Bag-of-Words representa cada documento mediante un vector de frecuencias. Cada posición del vector corresponde a una palabra del vocabulario global y contiene el número de veces que aparece esa palabra en el documento.

El orden de las palabras no se conserva; únicamente se tiene en cuenta su frecuencia.

### Similitud del coseno

La similitud del coseno mide el ángulo entre dos vectores:

$$
\cos(\theta) =
\frac{\mathbf{a}\cdot\mathbf{b}}
{\|\mathbf{a}\|\,\|\mathbf{b}\|}
$$

En este proyecto:

- `0` significa que los documentos no comparten ninguna palabra.
- `1` significa que los vectores tienen la misma dirección.
- Los valores más altos indican una mayor similitud léxica.

El cálculo se realiza con:

```python
torch.nn.functional.cosine_similarity
```

### TF-IDF

TF-IDF asigna más peso a las palabras frecuentes en un documento pero poco frecuentes en el resto del corpus:

```math
\mathrm{tfidf}(t,d) = \frac{f_{t,d}}{|d|} \cdot \ln\left(\frac{N}{n_t}\right)
```

donde:

- $f_{t,d}$ es el número de apariciones del término en el documento.
- $|d|$ es el número total de palabras del documento.
- $N$ es el número total de documentos.
- $n_t$ es el número de documentos que contienen el término.

## 🏗️ Proceso realizado

### 1. Preprocesamiento

Para cada documento `.docx` se realizan las siguientes operaciones:

- Conversión de todo el texto a minúsculas.
- Eliminación de signos de puntuación y caracteres no alfabéticos.
- Conservación de letras acentuadas y de la letra `ñ`.
- Eliminación de *stop words* en español mediante NLTK.
- Guardado del resultado en `docs_processed/`.

### 2. Construcción de Bag-of-Words

Se crea un vocabulario global con todas las palabras únicas de los documentos. En esta ejecución se obtienen:

- **12 documentos**.
- **411 palabras únicas**.
- Un vector de **411 componentes por documento**.

Cada componente contiene la frecuencia de una palabra concreta dentro del documento.

### 3. Similitud entre documentos

Se comparan todos los pares posibles de documentos:

$$
\frac{12 \cdot 11}{2} = 66 \text{ pares}
$$

Los resultados se almacenan en una matriz de similitud simétrica y se utilizan para identificar los pares más y menos parecidos.

### 4. Ampliación TF-IDF

También se construye un vector TF-IDF para cada documento. A continuación, se repite el cálculo de similitud del coseno y se genera una tabla comparativa con:

| Documento 1 | Documento 2 | Similitud BoW | Similitud TF-IDF |
|---|---|---:|---:|
| doc09 | doc12 | 0.3226 | 0.1321 |

## 📁 Estructura del proyecto

```text
BolsaDePalabras/
├── bag_of_words.ipynb            # Notebook principal con explicaciones y análisis
├── bag_of_words_DEFENSA.ipynb    # Versión compacta para la defensa
├── docs/                         # 12 documentos originales en formato .docx
├── docs_processed/               # Documentos preprocesados en formato .txt
├── pyproject.toml                # Dependencias y configuración del proyecto
├── uv.lock                       # Versiones bloqueadas de las dependencias
├── .python-version               # Versión de Python utilizada
├── .gitignore
├── LICENSE
└── README.md
```

## 📊 Documentos y clasificación temática

| Tema | Documentos |
|---|---|
| **Tecnología móvil** | doc01, doc02, doc03, doc06, doc07 y doc10 |
| **Vehículos eléctricos** | doc04, doc05, doc08 y doc09 |
| **Mixtos** | doc11 y doc12 |

La clasificación se realiza de forma manual a partir del contenido de los documentos y se utiliza únicamente para interpretar los resultados.

## 📈 Resultados

### Similitud media entre temas con BoW

La diagonal representa la similitud media entre documentos del mismo grupo. El resto de valores representa la similitud media entre grupos diferentes.

| | Tecnología móvil | Vehículos eléctricos | Mixtos |
|---|---:|---:|---:|
| **Tecnología móvil** | **0.0814** | 0.0252 | 0.1100 |
| **Vehículos eléctricos** | 0.0252 | **0.1549** | 0.1466 |
| **Mixtos** | 0.1100 | 0.1466 | **0.3016** |

### Similitud media entre temas con TF-IDF

| | Tecnología móvil | Vehículos eléctricos | Mixtos |
|---|---:|---:|---:|
| **Tecnología móvil** | **0.0266** | 0.0112 | 0.0279 |
| **Vehículos eléctricos** | 0.0112 | **0.0532** | 0.0579 |
| **Mixtos** | 0.0279 | 0.0579 | **0.1171** |

> El valor de la categoría «mixtos» consigo misma se obtiene a partir de un único par: doc11-doc12.

### Diez pares más similares con BoW

| Posición | Par | Similitud BoW | Similitud TF-IDF | Posición TF-IDF |
|---:|---|---:|---:|---:|
| 1 | doc09 - doc12 | 0.3226 | 0.1321 | 1 |
| 2 | doc11 - doc12 | 0.3016 | 0.1171 | 2 |
| 3 | doc04 - doc12 | 0.2488 | 0.0914 | 3 |
| 4 | doc04 - doc09 | 0.2166 | 0.0691 | 7 |
| 5 | doc05 - doc09 | 0.2067 | 0.0702 | 6 |
| 6 | doc05 - doc08 | 0.1912 | 0.0720 | 4 |
| 7 | doc07 - doc11 | 0.1904 | 0.0361 | 20 |
| 8 | doc02 - doc11 | 0.1883 | 0.0670 | 8 |
| 9 | doc09 - doc11 | 0.1660 | 0.0715 | 5 |
| 10 | doc07 - doc12 | 0.1540 | 0.0377 | 19 |

### Pares sin palabras en común

De los 66 pares comparados, **12 tienen una similitud igual a 0**. Esto significa que no comparten ninguna palabra después del preprocesamiento.

Diez de esos pares pertenecen a documentos de tecnología móvil y vehículos eléctricos, mientras que los otros dos corresponden a documentos que tratan temas más específicos dentro de una misma categoría.

## 🔎 Conclusiones

1. Los documentos del mismo tema tienden a ser más similares entre sí que los documentos de temas diferentes. Con BoW, la similitud media es 0.1549 entre documentos de vehículos eléctricos y 0.0814 entre documentos de tecnología móvil, frente a 0.0252 entre ambos temas.
2. Los documentos mixtos actúan como un puente entre las dos temáticas. Su similitud media es mayor con los documentos de vehículos eléctricos que con los documentos de tecnología móvil.
3. El par más similar es **doc09-doc12**, con una similitud de 0.3226 usando BoW y 0.1321 usando TF-IDF. Ambos documentos están relacionados con la recarga de vehículos eléctricos.
4. TF-IDF reduce los valores absolutos de similitud porque disminuye la importancia de las palabras generales o repetidas en muchos documentos.
5. Aunque TF-IDF modifica algunas posiciones del ranking, mantiene una tendencia parecida a BoW: los documentos que comparten vocabulario específico suelen permanecer entre los más similares.
6. La similitud obtenida es léxica y no semántica. Dos documentos pueden tratar el mismo tema y obtener una similitud baja si utilizan palabras diferentes.

## ⚠️ Limitaciones

- Bag-of-Words no tiene en cuenta el orden de las palabras.
- No se realiza lematización ni *stemming*, por lo que `móvil` y `móviles` se consideran palabras distintas.
- La similitud depende de las palabras compartidas, no del significado profundo del texto.
- La clasificación temática es manual.
- El corpus es pequeño y los documentos son relativamente cortos.

Como posible mejora, se podrían utilizar técnicas de lematización o representaciones semánticas basadas en *embeddings*.

## ⚙️ Requisitos

- Python 3.12 o superior.
- PyTorch.
- pandas.
- NLTK.
- python-docx.
- regex.
- ipykernel.
- uv, recomendado para instalar las dependencias del proyecto.

## 🚀 Instalación y ejecución

### 1. Clonar el repositorio

```bash
git clone https://github.com/LargeLanguageMonos/BolsaDePalabras
cd BolsaDePalabras
```

### 2. Instalar las dependencias

```bash
uv sync
```

### 3. Ejecutar el notebook

Abrir [bag_of_words.ipynb](bag_of_words.ipynb) en VS Code o JupyterLab, seleccionar el entorno virtual como kernel y ejecutar las celdas en orden.

También se puede iniciar JupyterLab mediante:

```bash
uv run --with jupyter jupyter lab
```

La primera ejecución descarga las *stop words* de español de NLTK, por lo que se necesita conexión a internet.

## 📦 Entregables

- **Código fuente:** [bag_of_words.ipynb](bag_of_words.ipynb).
- **Versión compacta para la defensa:** [bag_of_words_DEFENSA.ipynb](bag_of_words_DEFENSA.ipynb).
- **Informe de hallazgos:** README.

## 🛠️ Tecnologías utilizadas

- **Python 3.12**.
- **PyTorch** para tensores y similitud del coseno.
- **pandas** para tablas y matrices de resultados.
- **NLTK** para las *stop words* en español.
- **python-docx** para leer los documentos originales.
- **regex** para limpiar texto Unicode.
- **Jupyter / ipykernel** para ejecutar el notebook.

## 👤 Autor


- **👨‍💻 Pablo Cabeza Lantigua**: [pabcablan](https://github.com/pabcablan)
- **👨‍💻 Juan Diego González Noguera**: [juand4569](https://github.com/juand4569)


## 📄 Licencia

Este proyecto se distribuye bajo la licencia indicada en [LICENSE](LICENSE).
