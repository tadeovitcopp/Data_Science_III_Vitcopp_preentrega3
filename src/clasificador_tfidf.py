import re
from pathlib import Path

import pandas as pd
import spacy
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================
# BASE_DIR apunta a la raíz del proyecto (un nivel arriba de src/),
# para que las rutas de data/ y outputs/ funcionen sin importar
# desde qué directorio se ejecute el script.

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = BASE_DIR / "data" / "ag_news_train.csv"
TEST_PATH = BASE_DIR / "data" / "ag_news_test.csv"

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Configuración final del vectorizador TF-IDF, elegida a partir de la
# comparación de hiperparámetros documentada en experimentos_tfidf.py
# y en el README (Sección 12): mejor equilibrio entre accuracy y
# tamaño de vocabulario frente a valores menores (5000-10000) y
# mayores (30000).
MAX_FEATURES = 30000
NGRAM_RANGE = (1, 2)  # unigramas + bigramas


# ============================================================
# CARGAR MODELO DE SPACY
# ============================================================
# en_core_web_sm se usa solo para tokenizar y lematizar (no se
# necesitan NER ni parsing), lo que lo hace más liviano y rápido
# que otros modelos de SpaCy para este pipeline.

print("=" * 60)
print("CARGANDO MODELO DE SPACY")
print("=" * 60)

nlp = spacy.load("en_core_web_sm")

print("Modelo SpaCy cargado correctamente.")


# ============================================================
# PREPROCESAMIENTO
# ============================================================

def preprocess_text(text):
    """
    Limpia, normaliza y lematiza un texto crudo de AG News.

    Pasos aplicados en orden:
    1. Normalización a minúsculas.
    2. Eliminación de URLs.
    3. Eliminación de etiquetas HTML.
    4. Eliminación de caracteres no alfabéticos (números, signos de
       puntuación) para quedarnos solo con palabras.
    5. Colapso de espacios múltiples generados por los reemplazos.
    6. Tokenización y lematización con SpaCy, para reducir variantes
       morfológicas (plurales, tiempos verbales) a una forma común.

    Devuelve el texto reconstruido como un string de lemas separados
    por espacios, listo para ser vectorizado con TF-IDF.
    """

    # Convertir a string (por si hay NaN u otros tipos) y pasar a minúsculas
    text = str(text).lower()

    # Eliminar URLs (http, https o www)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Eliminar etiquetas HTML (por ejemplo <br>, <p>)
    text = re.sub(r"<[^>]+>", " ", text)

    # Mantener solamente letras y espacios (saca números y puntuación)
    text = re.sub(r"[^a-z\s]", " ", text)

    # Normalizar espacios: colapsa espacios múltiples y recorta bordes
    text = re.sub(r"\s+", " ", text).strip()

    # Procesamiento con SpaCy: tokeniza y calcula el lema de cada token
    doc = nlp(text)

    # Lematización: se descartan tokens vacíos o que sean solo espacio
    tokens = []

    for token in doc:
        if not token.is_space:
            lemma = token.lemma_.strip()

            if lemma:
                tokens.append(lemma)

    return " ".join(tokens)


# ============================================================
# CARGAR DATASET
# ============================================================
# Se cargan los splits ya provistos por la cátedra (train/test),
# en vez de generar un split propio con train_test_split, para
# mantener la comparabilidad con los resultados del Módulo 2.

print()
print("=" * 60)
print("CARGANDO DATASET AG NEWS")
print("=" * 60)

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"Train: {len(train_df)} documentos")
print(f"Test:  {len(test_df)} documentos")

print()
print("Columnas:")
print(train_df.columns.tolist())

print()
print("Distribución de clases en TRAIN:")
print(train_df["label"].value_counts())


# ============================================================
# PREPROCESAMIENTO
# ============================================================
# Se aplica preprocess_text a train y test por separado (nunca se
# mezclan) para que la limpieza no filtre información entre splits.

print()
print("=" * 60)
print("PREPROCESANDO TRAIN")
print("=" * 60)

train_df["text_processed"] = train_df["text"].apply(preprocess_text)

print("Train preprocesado correctamente.")


print()
print("=" * 60)
print("PREPROCESANDO TEST")
print("=" * 60)

test_df["text_processed"] = test_df["text"].apply(preprocess_text)

print("Test preprocesado correctamente.")


# ============================================================
# SEPARAR X E Y
# ============================================================

X_train = train_df["text_processed"]
y_train = train_df["label"]

X_test = test_df["text_processed"]
y_test = test_df["label"]


# ============================================================
# VECTORIZACIÓN TF-IDF
# ============================================================
# max_features limita el vocabulario a las N palabras/bigramas más
# relevantes (por frecuencia ponderada), controlando el tamaño de la
# matriz y el riesgo de overfitting con vocabulario muy raro.
# ngram_range=(1, 2) incluye unigramas y bigramas para capturar tanto
# palabras sueltas como combinaciones con significado propio
# (por ejemplo "new york").

print()
print("=" * 60)
print("VECTORIZACIÓN TF-IDF")
print("=" * 60)

print(f"max_features: {MAX_FEATURES}")
print(f"ngram_range: {NGRAM_RANGE}")

vectorizer = TfidfVectorizer(
    max_features=MAX_FEATURES,
    ngram_range=NGRAM_RANGE
)

# IMPORTANTE — Prevención de Data Leakage:
# fit_transform SOLO sobre TRAIN. El vocabulario y los pesos IDF se
# calculan exclusivamente a partir del conjunto de entrenamiento.
X_train_tfidf = vectorizer.fit_transform(X_train)

# TEST solamente se transforma con el vectorizador ya ajustado:
# no se le vuelve a hacer fit, para que su vocabulario no influya
# en el entrenamiento del modelo.
X_test_tfidf = vectorizer.transform(X_test)

print()
print(f"Train TF-IDF shape: {X_train_tfidf.shape}")
print(f"Test TF-IDF shape:  {X_test_tfidf.shape}")

print()
print(f"Cantidad de características utilizadas: {len(vectorizer.vocabulary_)}")


# ============================================================
# MODELO
# ============================================================
# Se eligió LinearSVC (Support Vector Machine lineal) como baseline
# porque funciona muy bien sobre matrices dispersas de alta
# dimensionalidad como las que genera TF-IDF, es rápido de entrenar
# y suele superar a Naive Bayes en corpus con vocabulario amplio.

print()
print("=" * 60)
print("ENTRENANDO MODELO")
print("=" * 60)

model = LinearSVC()

model.fit(X_train_tfidf, y_train)

print("Modelo entrenado correctamente.")


# ============================================================
# PREDICCIONES
# ============================================================

print()
print("=" * 60)
print("GENERANDO PREDICCIONES")
print("=" * 60)

y_pred = model.predict(X_test_tfidf)

print("Predicciones generadas correctamente.")


# ============================================================
# EVALUACIÓN
# ============================================================
# Accuracy: proporción total de predicciones correctas.
# Classification report: precision, recall y F1-score por clase,
# más los promedios macro (todas las clases pesan igual) y weighted
# (ponderado por cantidad de ejemplos de cada clase).

print()
print("=" * 60)
print("EVALUACIÓN DEL MODELO")
print("=" * 60)

accuracy = accuracy_score(y_test, y_pred)

print()
print(f"Accuracy: {accuracy:.4f}")

print()
print("CLASSIFICATION REPORT")
print("-" * 60)

report = classification_report(y_test, y_pred)

print(report)


# ============================================================
# MATRIZ DE CONFUSIÓN
# ============================================================
# Permite ver, más allá del accuracy global, qué clases se confunden
# entre sí (por ejemplo Business vs. Sci_Tech), información que el
# classification_report por sí solo no muestra.

print()
print("=" * 60)
print("MATRIZ DE CONFUSIÓN")
print("=" * 60)

labels = sorted(y_test.unique())

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

print()
print("Clases:")
print(labels)

print()
print(cm)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

fig, ax = plt.subplots(figsize=(9, 7))

disp.plot(
    ax=ax,
    cmap="Blues",
    values_format="d"
)

plt.title("Matriz de Confusión - AG News - TF-IDF + Linear SVM")
plt.tight_layout()

confusion_path = OUTPUT_DIR / "matriz_confusion.png"

plt.savefig(confusion_path, dpi=300)
plt.close()

print()
print(f"Matriz guardada en: {confusion_path}")


# ============================================================
# GUARDAR REPORTE
# ============================================================
# Se guarda el classification_report como CSV (además de imprimirlo)
# para tener evidencia reproducible de las métricas, sin depender
# de copiar la salida de consola.

report_dict = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

report_df = pd.DataFrame(report_dict).transpose()

report_path = OUTPUT_DIR / "classification_report.csv"

report_df.to_csv(report_path)

print(f"Reporte guardado en: {report_path}")


# ============================================================
# RESUMEN FINAL
# ============================================================

print()
print("=" * 60)
print("RESUMEN FINAL")
print("=" * 60)

print(f"Documentos TRAIN: {len(train_df)}")
print(f"Documentos TEST:  {len(test_df)}")
print(f"Max features:     {MAX_FEATURES}")
print(f"N-gram range:     {NGRAM_RANGE}")
print(f"Características:  {len(vectorizer.vocabulary_)}")
print(f"Accuracy:         {accuracy:.4f}")

print()
print("Archivos generados:")
print(f"- {confusion_path}")
print(f"- {report_path}")

print()
print("=" * 60)
print("CLASIFICADOR TF-IDF FINALIZADO CORRECTAMENTE")
print("=" * 60)