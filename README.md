# Pre-entrega 3 — Clasificador Supervisado con TF-IDF

## Data Science III

Implementación de un pipeline de clasificación supervisada de textos utilizando TF-IDF y Linear SVM sobre el dataset AG News.

---

## 1. Objetivo

El objetivo de esta pre-entrega es construir un pipeline completo capaz de transformar noticias en texto en predicciones de categorías utilizando técnicas clásicas de Machine Learning para NLP.

El pipeline integra:

1. Carga del dataset AG News.
2. Preprocesamiento del texto.
3. Lematización con SpaCy.
4. Vectorización mediante TF-IDF.
5. Entrenamiento de un clasificador Linear SVM.
6. Evaluación mediante Accuracy, Precision, Recall y F1-Score.
7. Generación de una matriz de confusión.

Este trabajo continúa el pipeline desarrollado en los módulos anteriores y utiliza el mismo corpus AG News.

---

## 2. Dataset

Se utiliza el dataset **AG News**, compuesto por noticias en inglés clasificadas en cuatro categorías:

- World
- Sports
- Business
- Sci_Tech

Para esta implementación se mantienen separados los conjuntos de entrenamiento y prueba:

```text
ag_news_train.csv
ag_news_test.csv
```

### Distribución del dataset

**Train**

| Categoría | Documentos |
|---|---|
| World | 2000 |
| Sports | 2000 |
| Business | 2000 |
| Sci_Tech | 2000 |
| **Total** | **8000** |

**Test**

El conjunto de prueba contiene 2000 documentos, con 500 documentos por categoría.

---

## 3. Estructura del proyecto

```
Data_Science_III_Vitcopp_preentrega3/
│
├── data/
│   ├── ag_news_train.csv
│   └── ag_news_test.csv
│
├── src/
│   ├── train.py
│   ├── eda_nlp.py
│   ├── clasificador_tfidf.py
│   └── experimentos_tfidf.py
│
├── outputs/
│   ├── histograma_longitud.png
│   ├── distribucion_clases.png
│   ├── top_20_bigramas.csv
│   ├── top_20_trigramas.csv
│   ├── top_50_palabras.csv
│   ├── corpus_preprocesado.csv
│   ├── matriz_confusion.png
│   ├── classification_report.csv
│   └── experimentos_tfidf.csv
│
├── requirements.txt
└── README.md
```

---

## 4. Preprocesamiento

El preprocesamiento reutiliza la lógica desarrollada durante el Módulo 2.

Se aplican las siguientes etapas:

**Normalización**
El texto se convierte a minúsculas para evitar diferencias entre palabras equivalentes.

**Limpieza mediante expresiones regulares**
Se eliminan:
- URLs.
- Etiquetas HTML.
- Caracteres no alfabéticos.
- Espacios innecesarios.

**Tokenización y lematización**
Se utiliza el modelo `en_core_web_sm` de SpaCy. La lematización permite reducir distintas formas de una palabra a una representación común (por ejemplo, diferentes formas verbales se normalizan a su lema correspondiente).

---

## 5. Vectorización TF-IDF

Para convertir los documentos en una representación numérica se utiliza:

```python
TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2)
)
```

**max_features**
Se estableció `max_features = 30000`. Esto limita el vocabulario utilizado por el modelo a las 20.000 características más relevantes según el vectorizador, controlando el tamaño de la matriz y el costo computacional.

**ngram_range**
Se utilizó `ngram_range = (1, 2)`, incorporando:
- Unigramas: palabras individuales.
- Bigramas: pares consecutivos de palabras.

De esta manera, el modelo puede capturar tanto información individual como determinadas relaciones entre palabras.

---

## 6. Prevención de Data Leakage

Se mantuvo estrictamente separada la información de entrenamiento y prueba. El vectorizador se ajusta exclusivamente sobre el conjunto de entrenamiento:

```python
X_train_tfidf = vectorizer.fit_transform(X_train)
```

Luego se utiliza el mismo vectorizador para transformar el conjunto de prueba:

```python
X_test_tfidf = vectorizer.transform(X_test)
```

No se realiza `fit` sobre el conjunto de test, evitando que su vocabulario influya en el entrenamiento del modelo.

---

## 7. Modelo utilizado

Se seleccionó **Linear SVM (`LinearSVC`)** como modelo de referencia.

La elección se debe a que los clasificadores lineales funcionan adecuadamente sobre representaciones TF-IDF de alta dimensionalidad y matrices dispersas, características habituales en problemas de clasificación de texto. Además, Linear SVM permite establecer un baseline sólido para comparar posteriormente con modelos de Deep Learning.

```python
model = LinearSVC()
```

---

## 8. Entrenamiento

El modelo se entrenó utilizando exclusivamente las representaciones TF-IDF del conjunto de entrenamiento:

```python
model.fit(X_train_tfidf, y_train)
```

Posteriormente se realizaron predicciones sobre el conjunto de test:

```python
y_pred = model.predict(X_test_tfidf)
```

---

## 9. Resultados

El modelo obtuvo:

**Accuracy = 0.8935**

Es decir, aproximadamente un 89% de las noticias del conjunto de prueba fueron clasificadas correctamente.

### Classification Report

| Categoría     | Precision |   Recall | F1-Score |
| ------------- | --------: | -------: | -------: |
| Business      |      0.84 |     0.87 |     0.86 |
| Sci_Tech      |      0.88 |     0.86 |     0.87 |
| Sports        |      0.94 |     0.96 |     0.95 |
| World         |      0.91 |     0.88 |     0.89 |
| **Macro avg** |  **0.89** | **0.89** | **0.89** |


---

## 10. Análisis de resultados

La categoría con mejor desempeño fue **Sports**, con Precision 0.94, Recall 0.96 y F1-Score 0.95. Esto indica que las noticias deportivas presentan patrones léxicos relativamente diferenciados del resto de las categorías.

La categoría con menor F1-Score fue **Business**, con F1-Score 0.85. Por lo tanto, Business presenta una mayor dificultad de clasificación.

---

## 11. Matriz de confusión

              Business  Sci_Tech  Sports  World

Business          435        37       8      20
Sci_Tech           46       432       9      13
Sports              4         6     482       8
World              30        17      15     438

### Interpretación

El principal foco de confusión se encuentra entre:

- Sci_Tech → Business = 46
- Business → Sci_Tech = 37

Esto indica que ambas categorías comparten determinados patrones léxicos relacionados con tecnología, empresas, productos e industria.

También se observa cierta confusión entre World y Business:

- World → Business = 29
- Business → World = 23

En contraste, **Sports** presenta pocos errores y concentra 481 de sus 500 documentos correctamente clasificados.

---

## 12. Experimentación con parámetros del vectorizador

La configuración final seleccionada es max_features=30000 y ngram_range=(1,2), ya que obtuvo el mejor desempeño entre las configuraciones evaluadas, alcanzando un Accuracy de 0.8935 y un F1 Macro de 0.8934.

| max_features | ngram_range | Características reales | Accuracy | F1 macro | Tiempo (s) |
|---|---|---|---|---|---|
| 5000 | (1, 1) | 5000 | 0.8795 | 0.8795 | 0.29 |
| 10000 | (1, 1) | 10000 | 0.8855 | 0.8855 | 0.19 |
| 10000 | (1, 2) | 10000 | 0.8860 | 0.8859 | 0.56 |
| **20000** | **(1, 2)** | **20000** | **0.8900** | **0.8899** | **0.58** |
| 30000 | (1, 2) | 30000 | 0.8935 | 0.8934 | 0.57 |

*Tabla completa disponible en `outputs/experimentos_tfidf.csv`.*

**Lectura de los resultados:**

- **Incorporar bigramas ayuda:** con `max_features=10000` fijo, agregar bigramas (`(1,2)` vs `(1,1)`) sube el accuracy de 0.8855 a 0.8860. La mejora es modesta pero consistente con la intuición de que frases como "new york" o "united states" aportan señal que los unigramas solos no capturan.
- **Más vocabulario ayuda, con retornos decrecientes:** subir de 5000 a 20000 features aporta +1.05 puntos de accuracy (0.8795 → 0.8900), pero subir de 20000 a 30000 solo suma +0.35 puntos adicionales (0.8900 → 0.8935), a cambio de un 50% más de memoria en la matriz TF-IDF y sin diferencia relevante en tiempo de entrenamiento.
- **Elección final:** se mantiene `max_features=20000, ngram_range=(1, 2)` como configuración reportada en las Secciones 9-11, por ofrecer el mejor equilibrio entre desempeño y tamaño del vocabulario. La configuración con 30000 features queda documentada como una alternativa válida si se prioriza exclusivamente el accuracy por sobre el costo de memoria — por ejemplo, si el corpus creciera significativamente en módulos posteriores.

---

## 13. Conclusión

El pipeline desarrollado permite transformar el corpus AG News en una representación numérica mediante TF-IDF y utilizarla para realizar clasificación supervisada con Linear SVM.

El modelo alcanzó un Accuracy de 89%, demostrando que TF-IDF combinado con un clasificador lineal constituye un baseline sólido para este problema.

La principal dificultad aparece en la separación entre Business y Sci_Tech, mientras que Sports presenta el mejor desempeño. Esto puede explicarse por la existencia de vocabulario compartido entre noticias empresariales y tecnológicas.

El experimento también demuestra la importancia de mantener separado el proceso de vectorización entre entrenamiento y prueba. El `TfidfVectorizer` se ajustó únicamente sobre train y posteriormente se utilizó `transform` sobre test, evitando Data Leakage.

Estos resultados proporcionan una línea base cuantitativa para comparar posteriormente el desempeño de modelos de Deep Learning.

---

## 14. Archivos de salida

El pipeline genera automáticamente:

```
outputs/matriz_confusion.png
outputs/classification_report.csv
outputs/experimentos_tfidf.csv
```

Los resultados obtenidos permiten conservar evidencia reproducible del proceso de evaluación.

---

## 15. Requisitos

Las principales librerías utilizadas son:

- Python
- pandas
- scikit-learn
- SpaCy
- matplotlib

Instalación:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## 16. Ejecución

Activar el entorno virtual:

```bash
source .venv/bin/activate
```

Ejecutar el clasificador:

```bash
python src/clasificador_tfidf.py
```

Ejecutar la comparación de hiperparámetros del vectorizador:

```bash
python src/experimentos_tfidf.py
```

El programa carga los datasets, realiza el preprocesamiento, genera las representaciones TF-IDF, entrena el modelo, calcula las métricas y genera la matriz de confusión.

---

## 17. Continuidad del proyecto

Este trabajo utiliza el mismo corpus AG News seleccionado en el Módulo 2. El dataset y el pipeline serán utilizados como base para los siguientes módulos y el Proyecto Final. La separación entre train y test se mantiene para garantizar una evaluación correcta del modelo.