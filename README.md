Pre-entrega 3 — Clasificador Supervisado con TF-IDF

Data Science III

Implementación de un pipeline de clasificación supervisada de textos utilizando TF-IDF y Linear SVM sobre el dataset AG News.

⸻

1. Objetivo

El objetivo de esta pre-entrega es construir un pipeline completo capaz de transformar noticias en texto en predicciones de categorías utilizando técnicas clásicas de Machine Learning para NLP.

El pipeline integra:

1. Carga del dataset AG News.
2. Preprocesamiento del texto.
3. Tokenización y lematización con SpaCy.
4. Vectorización mediante TF-IDF.
5. Entrenamiento de un clasificador Linear SVM.
6. Evaluación mediante Accuracy, Precision, Recall y F1-Score.
7. Generación de una matriz de confusión.
8. Experimentación con diferentes configuraciones de TF-IDF.

Este trabajo continúa el pipeline desarrollado en los módulos anteriores y utiliza el mismo corpus AG News.

⸻

2. Dataset

Se utiliza el dataset AG News, compuesto por noticias en inglés clasificadas en cuatro categorías:

* World
* Sports
* Business
* Sci_Tech

Para esta implementación se mantienen separados los conjuntos de entrenamiento y prueba:

ag_news_train.csv
ag_news_test.csv

Distribución del dataset

Train

Categoría	Documentos
World	2000
Sports	2000
Business	2000
Sci_Tech	2000
Total	8000

Test

El conjunto de prueba contiene 2000 documentos, con 500 documentos por categoría.

Categoría	Documentos
World	500
Sports	500
Business	500
Sci_Tech	500
Total	2000

⸻

3. Estructura del proyecto

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

⸻

4. Preprocesamiento

El preprocesamiento reutiliza la lógica desarrollada durante el Módulo 2.

Se aplican las siguientes etapas:

Normalización

El texto se convierte a minúsculas para evitar diferencias entre palabras equivalentes.

Limpieza mediante expresiones regulares

Se eliminan:

* URLs.
* Etiquetas HTML.
* Caracteres no alfabéticos.
* Espacios innecesarios.

Tokenización y lematización

Se utiliza el modelo en_core_web_sm de SpaCy.

La lematización permite reducir distintas formas de una palabra a una representación común, facilitando que palabras con diferentes formas gramaticales puedan ser tratadas como una misma unidad léxica.

⸻

5. Vectorización TF-IDF

Para convertir los documentos en una representación numérica se utiliza la siguiente configuración final:

TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2)
)

max_features

Se estableció:

max_features = 30000

Esto limita el vocabulario utilizado por el modelo a un máximo de 30.000 características, reduciendo la dimensionalidad de la representación TF-IDF y controlando el costo computacional.

ngram_range

Se utilizó:

ngram_range = (1, 2)

Esto incorpora:

* Unigramas: palabras individuales.
* Bigramas: pares consecutivos de palabras.

De esta manera, el modelo puede capturar tanto información asociada a palabras individuales como determinadas relaciones entre palabras.

⸻

6. Prevención de Data Leakage

Se mantuvo estrictamente separada la información de entrenamiento y prueba.

El vectorizador se ajusta exclusivamente sobre el conjunto de entrenamiento:

X_train_tfidf = vectorizer.fit_transform(X_train)

Luego se utiliza el mismo vectorizador para transformar el conjunto de prueba:

X_test_tfidf = vectorizer.transform(X_test)

No se realiza fit sobre el conjunto de test, evitando que información del conjunto de prueba influya en la construcción del vocabulario o en el entrenamiento del modelo.

⸻

7. Modelo utilizado

Se seleccionó Linear SVM (LinearSVC) como modelo de clasificación.

Los clasificadores lineales son especialmente adecuados para representaciones TF-IDF de alta dimensionalidad y matrices dispersas, características habituales en problemas de clasificación de texto.

Además, Linear SVM permite establecer un baseline sólido para comparar posteriormente con modelos de Deep Learning.

La implementación utiliza:

model = LinearSVC()

⸻

8. Entrenamiento

El modelo se entrenó utilizando exclusivamente las representaciones TF-IDF correspondientes al conjunto de entrenamiento:

model.fit(X_train_tfidf, y_train)

Posteriormente se realizaron predicciones sobre el conjunto de prueba:

y_pred = model.predict(X_test_tfidf)

⸻

9. Resultados del modelo final

La configuración final obtuvo:

* Accuracy: 0.8935
* F1 Macro: 0.8934
* Características: 30.000
* N-gramas: (1, 2)

Esto significa que el modelo clasificó correctamente aproximadamente el 89,35% de las noticias del conjunto de prueba.

Sobre 2.000 documentos de test, esto equivale a aproximadamente 1.787 documentos correctamente clasificados.

Classification Report

Categoría	Precision	Recall	F1-Score	Support
Business	0.84	0.87	0.86	500
Sci_Tech	0.88	0.86	0.87	500
Sports	0.94	0.96	0.95	500
World	0.91	0.88	0.89	500
Macro avg	0.89	0.89	0.89	2000
Weighted avg	0.89	0.89	0.89	2000

⸻

10. Análisis de resultados

La categoría con mejor desempeño fue Sports, con:

* Precision: 0.94
* Recall: 0.96
* F1-Score: 0.95

Esto indica que las noticias deportivas presentan patrones léxicos relativamente diferenciados respecto de las demás categorías, facilitando su identificación por parte del clasificador.

La categoría con menor F1-Score fue Business, con un valor de 0.86. Esto indica una mayor dificultad para diferenciar algunas noticias empresariales de categorías como Sci_Tech y World.

En términos generales, el modelo presenta un desempeño equilibrado entre las cuatro categorías, con métricas cercanas al 0.90.

⸻

11. Matriz de confusión

La matriz de confusión obtenida fue:

              Business  Sci_Tech  Sports  World
Business          435        37       8      20
Sci_Tech           46       432       9      13
Sports              4         6     482       8
World              30        17      15     438

Interpretación

La matriz muestra que:

* 435 de 500 noticias Business fueron correctamente clasificadas.
* 432 de 500 noticias Sci_Tech fueron correctamente clasificadas.
* 482 de 500 noticias Sports fueron correctamente clasificadas.
* 438 de 500 noticias World fueron correctamente clasificadas.

El principal foco de confusión corresponde a:

* Sci_Tech → Business: 46 documentos.
* Business → Sci_Tech: 37 documentos.

Esto indica que existe cierta similitud léxica entre ambas categorías. Las noticias relacionadas con tecnología, empresas, productos e industria pueden compartir vocabulario y dificultar la separación entre Business y Sci_Tech.

También se observa cierta confusión entre World y Business:

* World → Business: 30 documentos.
* Business → World: 20 documentos.

En contraste, Sports presenta el menor nivel de confusión con las demás categorías, con solo 18 documentos incorrectamente clasificados.

⸻

12. Experimentación con parámetros del vectorizador

Para seleccionar la configuración final se realizaron cinco experimentos modificando:

* Cantidad máxima de características (max_features).
* Rango de n-gramas (ngram_range).

Las configuraciones evaluadas fueron:

#	max_features	ngram_range	Características reales	Accuracy	F1 Macro	Tiempo (s)
1	5000	(1, 1)	5000	0.8795	0.8795	0.29
2	10000	(1, 1)	10000	0.8855	0.8855	0.19
3	10000	(1, 2)	10000	0.8860	0.8859	0.56
4	20000	(1, 2)	20000	0.8900	0.8899	0.58
5	30000	(1, 2)	30000	0.8935	0.8934	0.57

La tabla completa se encuentra disponible en outputs/experimentos_tfidf.csv.

Análisis de los experimentos

Incorporación de bigramas

Manteniendo max_features=10000, incorporar bigramas produjo una pequeña mejora:

Unigramas:          Accuracy = 0.8855
Unigramas + bigramas: Accuracy = 0.8860

La mejora es modesta, pero indica que los bigramas aportan información adicional respecto del uso exclusivo de palabras individuales.

Aumento de la cantidad de características

Se observa una mejora progresiva al aumentar el número de características:

5000 features  → 0.8795
10000 features → 0.8860
20000 features → 0.8900
30000 features → 0.8935

El incremento de 5.000 a 30.000 características permite mejorar el Accuracy en 1,40 puntos porcentuales.

La mejora más importante se produce al pasar de 5.000 a 10.000 características, mientras que los incrementos posteriores generan mejoras más pequeñas.

Selección de la configuración final

La mejor configuración evaluada fue:

max_features = 30000
ngram_range = (1, 2)

Esta configuración obtuvo:

Accuracy = 0.8935
F1 Macro  = 0.8934

Por lo tanto, se selecciona como configuración final del clasificador al presentar el mejor desempeño entre las alternativas evaluadas.

⸻

13. Conclusión

El pipeline desarrollado permite transformar el corpus AG News en una representación numérica mediante TF-IDF y utilizarla para realizar clasificación supervisada con Linear SVM.

El modelo final alcanzó un Accuracy de 89,35% y un F1 Macro de 89,34%, demostrando que TF-IDF combinado con un clasificador lineal constituye un baseline sólido para este problema de clasificación de textos.

La categoría con mejor desempeño fue Sports, con un F1-Score de 0.95, mientras que Business presentó el menor F1-Score, con 0.86.

La matriz de confusión muestra que las principales dificultades se encuentran en la diferenciación entre Business y Sci_Tech, debido a la existencia de vocabulario compartido entre noticias empresariales y tecnológicas.

La experimentación con diferentes configuraciones de TF-IDF permitió comprobar que tanto el aumento de la cantidad de características como la incorporación de bigramas pueden mejorar el desempeño del modelo. La mejor configuración evaluada fue de 30.000 características con unigramas y bigramas.

Además, se mantuvo correctamente separado el conjunto de entrenamiento del conjunto de prueba. El TfidfVectorizer fue ajustado únicamente sobre train mediante fit_transform() y posteriormente aplicado sobre test mediante transform(), evitando Data Leakage.

Estos resultados proporcionan una línea base cuantitativa para comparar posteriormente el desempeño de modelos de Deep Learning sobre el mismo corpus.

⸻

14. Archivos de salida

El pipeline genera automáticamente los siguientes archivos:

outputs/matriz_confusion.png
outputs/classification_report.csv
outputs/experimentos_tfidf.csv

Estos archivos permiten conservar evidencia reproducible del proceso de evaluación y de la experimentación realizada.

⸻

15. Requisitos

Las principales librerías utilizadas son:

* Python
* pandas
* scikit-learn
* SpaCy
* matplotlib

Para instalar las dependencias:

pip install -r requirements.txt

Luego instalar el modelo de idioma de SpaCy:

python -m spacy download en_core_web_sm

⸻

16. Ejecución

Activar el entorno virtual:

source .venv/bin/activate

Ejecutar el clasificador final:

python src/clasificador_tfidf.py

Ejecutar los experimentos de TF-IDF:

python src/experimentos_tfidf.py

El programa carga los datasets, realiza el preprocesamiento, genera las representaciones TF-IDF, entrena el modelo, calcula las métricas y genera los archivos de resultados.

⸻

17. Continuidad del proyecto

Este trabajo utiliza el mismo corpus AG News seleccionado en el Módulo 2.

El dataset y el pipeline desarrollado serán utilizados como base para los siguientes módulos y el Proyecto Final.

La separación entre los conjuntos de entrenamiento y prueba se mantiene para garantizar una evaluación correcta del modelo y permitir comparaciones consistentes con futuras implementaciones.

El modelo TF-IDF + Linear SVM funciona como baseline de Machine Learning clásico, sobre el cual podrán compararse posteriormente modelos de Deep Learning y otras técnicas de representación de texto.