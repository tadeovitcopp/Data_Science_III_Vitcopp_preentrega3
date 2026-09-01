"""
experimentos_tfidf.py

Realiza experimentos con distintas configuraciones de TF-IDF
y LinearSVC para justificar la elección de hiperparámetros.

Configuraciones evaluadas:
- max_features = 5000, 10000, 20000, 30000
- ngram_range = (1,1) y (1,2)

El objetivo es comparar Accuracy, F1 Macro, cantidad real
de características y tiempo de ejecución.

Uso:
    python src/experimentos_tfidf.py
"""

import re
import time
from pathlib import Path

import pandas as pd
import spacy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score


# ============================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = BASE_DIR / "data" / "ag_news_train.csv"
TEST_PATH = BASE_DIR / "data" / "ag_news_test.csv"

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# CONFIGURACIÓN DE EXPERIMENTOS
# ============================================================

CONFIGURACIONES = [
    {
        "max_features": 5000,
        "ngram_range": (1, 1)
    },
    {
        "max_features": 10000,
        "ngram_range": (1, 1)
    },
    {
        "max_features": 10000,
        "ngram_range": (1, 2)
    },
    {
        "max_features": 20000,
        "ngram_range": (1, 2)
    },
    {
        "max_features": 30000,
        "ngram_range": (1, 2)
    },
]


# ============================================================
# CARGAR MODELO DE SPACY
# ============================================================

print("=" * 60)
print("CARGANDO MODELO DE SPACY")
print("=" * 60)

nlp = spacy.load("en_core_web_sm")

print("Modelo SpaCy cargado correctamente.")


# ============================================================
# FUNCIÓN DE PREPROCESAMIENTO
# ============================================================

def preprocess_text(text):
    """
    Limpia, normaliza y lematiza un texto.
    """

    # Convertir a string
    text = str(text)

    # Convertir a minúsculas
    text = text.lower()

    # Eliminar URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Eliminar etiquetas HTML
    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    # Mantener solamente letras y espacios
    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    # Normalizar espacios
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # Procesar con SpaCy
    doc = nlp(text)

    # Obtener lemas
    tokens = []

    for token in doc:

        if token.is_space:
            continue

        lemma = token.lemma_.strip()

        if lemma:
            tokens.append(lemma)

    # Devolver texto nuevamente como string
    return " ".join(tokens)


# ============================================================
# CARGA DEL DATASET
# ============================================================

def cargar_dataset():
    """
    Carga los datasets de entrenamiento y prueba.
    """

    print()
    print("=" * 60)
    print("CARGANDO DATASET AG NEWS")
    print("=" * 60)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print(f"Train: {len(train_df)} documentos")
    print(f"Test:  {len(test_df)} documentos")

    print()
    print("Distribución de clases:")
    print(train_df["label"].value_counts())

    return train_df, test_df


# ============================================================
# PREPROCESAMIENTO DEL DATASET
# ============================================================

def preparar_datos(train_df, test_df):
    """
    Aplica el mismo preprocesamiento utilizado
    en clasificador_tfidf.py.
    """

    print()
    print("=" * 60)
    print("PREPROCESANDO TRAIN")
    print("=" * 60)

    train_df["text_processed"] = train_df["text"].apply(
        preprocess_text
    )

    print("Train preprocesado correctamente.")

    print()
    print("=" * 60)
    print("PREPROCESANDO TEST")
    print("=" * 60)

    test_df["text_processed"] = test_df["text"].apply(
        preprocess_text
    )

    print("Test preprocesado correctamente.")

    X_train_text = train_df["text_processed"]
    X_test_text = test_df["text_processed"]

    y_train = train_df["label"]
    y_test = test_df["label"]

    return (
        X_train_text,
        X_test_text,
        y_train,
        y_test
    )


# ============================================================
# FUNCIÓN PARA CORRER EXPERIMENTOS
# ============================================================

def correr_experimentos(
    X_train_text,
    X_test_text,
    y_train,
    y_test
):
    """
    Ejecuta distintas configuraciones de TF-IDF + LinearSVC.

    Devuelve un DataFrame con los resultados de cada experimento.
    """

    resultados = []

    print()
    print("=" * 60)
    print("INICIANDO EXPERIMENTOS TF-IDF")
    print("=" * 60)

    print()
    print(
        "Se evaluarán las siguientes configuraciones:"
    )

    for cfg in CONFIGURACIONES:

        print(
            f"- max_features={cfg['max_features']}, "
            f"ngram_range={cfg['ngram_range']}"
        )

    print()

    # --------------------------------------------------------
    # Ejecutar cada configuración
    # --------------------------------------------------------

    for numero, cfg in enumerate(
        CONFIGURACIONES,
        start=1
    ):

        print()
        print("-" * 60)
        print(f"EXPERIMENTO {numero}/{len(CONFIGURACIONES)}")
        print("-" * 60)

        print(
            f"max_features = {cfg['max_features']}"
        )

        print(
            f"ngram_range = {cfg['ngram_range']}"
        )

        inicio = time.time()

        # ----------------------------------------------------
        # TF-IDF
        # ----------------------------------------------------

        vectorizer = TfidfVectorizer(
            max_features=cfg["max_features"],
            ngram_range=cfg["ngram_range"]
        )

        # IMPORTANTE:
        # FIT solamente sobre TRAIN
        X_train_tfidf = vectorizer.fit_transform(
            X_train_text
        )

        # TEST solamente se transforma
        X_test_tfidf = vectorizer.transform(
            X_test_text
        )

        # ----------------------------------------------------
        # ENTRENAMIENTO
        # ----------------------------------------------------

        modelo = LinearSVC()

        modelo.fit(
            X_train_tfidf,
            y_train
        )

        # ----------------------------------------------------
        # PREDICCIÓN
        # ----------------------------------------------------

        y_pred = modelo.predict(
            X_test_tfidf
        )

        # ----------------------------------------------------
        # MÉTRICAS
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        f1_macro = f1_score(
            y_test,
            y_pred,
            average="macro"
        )

        duracion = time.time() - inicio

        n_features_reales = (
            X_train_tfidf.shape[1]
        )

        # ----------------------------------------------------
        # GUARDAR RESULTADOS
        # ----------------------------------------------------

        resultados.append(
            {
                "max_features": cfg["max_features"],
                "ngram_range": str(
                    cfg["ngram_range"]
                ),
                "n_features_reales": n_features_reales,
                "accuracy": round(
                    accuracy,
                    4
                ),
                "f1_macro": round(
                    f1_macro,
                    4
                ),
                "tiempo_seg": round(
                    duracion,
                    2
                ),
            }
        )

        # ----------------------------------------------------
        # MOSTRAR RESULTADOS
        # ----------------------------------------------------

        print()
        print(
            f"Características reales: "
            f"{n_features_reales}"
        )

        print(
            f"Accuracy: "
            f"{accuracy:.4f}"
        )

        print(
            f"F1 Macro: "
            f"{f1_macro:.4f}"
        )

        print(
            f"Tiempo: "
            f"{duracion:.2f} segundos"
        )

    # ========================================================
    # CREAR TABLA FINAL
    # ========================================================

    tabla = pd.DataFrame(
        resultados
    )

    # Ordenar por F1 Macro
    tabla_ordenada = tabla.sort_values(
        by="f1_macro",
        ascending=False
    )

    # Guardar resultados
    output_path = (
        OUTPUT_DIR /
        "experimentos_tfidf.csv"
    )

    tabla.to_csv(
        output_path,
        index=False
    )

    # ========================================================
    # MOSTRAR TABLA
    # ========================================================

    print()
    print("=" * 60)
    print("RESULTADOS DE TODOS LOS EXPERIMENTOS")
    print("=" * 60)

    print()

    print(
        tabla.to_string(
            index=False
        )
    )

    print()
    print("=" * 60)
    print("MEJOR CONFIGURACIÓN")
    print("=" * 60)

    mejor = tabla_ordenada.iloc[0]

    print()
    print(
        f"max_features: "
        f"{mejor['max_features']}"
    )

    print(
        f"ngram_range: "
        f"{mejor['ngram_range']}"
    )

    print(
        f"Características reales: "
        f"{mejor['n_features_reales']}"
    )

    print(
        f"Accuracy: "
        f"{mejor['accuracy']}"
    )

    print(
        f"F1 Macro: "
        f"{mejor['f1_macro']}"
    )

    print()
    print(
        f"Tabla guardada en:"
        f"\n{output_path}"
    )

    return tabla


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("EXPERIMENTOS CON PARÁMETROS DE TF-IDF")
    print("=" * 60)

    # --------------------------------------------------------
    # Cargar dataset
    # --------------------------------------------------------

    train_df, test_df = cargar_dataset()

    # --------------------------------------------------------
    # Preprocesar
    # --------------------------------------------------------

    (
        X_train_text,
        X_test_text,
        y_train,
        y_test
    ) = preparar_datos(
        train_df,
        test_df
    )

    # --------------------------------------------------------
    # Ejecutar experimentos
    # --------------------------------------------------------

    resultados = correr_experimentos(
        X_train_text,
        X_test_text,
        y_train,
        y_test
    )

    print()
    print("=" * 60)
    print("EXPERIMENTOS FINALIZADOS CORRECTAMENTE")
    print("=" * 60)