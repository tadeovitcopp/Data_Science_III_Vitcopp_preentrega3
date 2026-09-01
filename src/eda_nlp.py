import os
import re
import html
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy

from sklearn.feature_extraction.text import CountVectorizer


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(
    BASE_DIR,
    "data",
    "ag_news_train.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CARGAR MODELO DE SPACY
# ============================================================

print("=" * 60)
print("CARGANDO MODELO DE SPACY")
print("=" * 60)

nlp = spacy.load("en_core_web_sm")

print("Modelo SpaCy cargado correctamente.")


# ============================================================
# LIMPIEZA DEL TEXTO
# ============================================================

def clean_text(text):
    """
    Limpieza y normalización del texto.
    
    Incluye:
    - Conversión de entidades HTML
    - Lowercase
    - Eliminación de URLs
    - Eliminación de etiquetas HTML
    - Eliminación de residuos HTML frecuentes
    - Eliminación de caracteres no alfabéticos
    - Normalización de espacios
    """

    text = str(text)

    # --------------------------------------------------------
    # 1. Convertir entidades HTML
    # --------------------------------------------------------

    text = html.unescape(text)

    # --------------------------------------------------------
    # 2. Convertir a minúsculas
    # --------------------------------------------------------

    text = text.lower()

    # --------------------------------------------------------
    # 3. Eliminar URLs
    # --------------------------------------------------------

    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    # --------------------------------------------------------
    # 4. Eliminar etiquetas HTML
    # --------------------------------------------------------

    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    # --------------------------------------------------------
    # 5. Eliminar apóstrofes posesivos
    # --------------------------------------------------------

    # company's -> company
    text = re.sub(
        r"([a-z]+)['’]s\b",
        r"\1",
        text
    )

    # users' -> users
    text = re.sub(
        r"([a-z]+)['’]\b",
        r"\1",
        text
    )

    # --------------------------------------------------------
    # 6. Eliminar residuos HTML / formato de noticias
    # --------------------------------------------------------

    text = re.sub(
        r"\b(href|target|quickinfo|fullquote|quot|lt|gt)\b",
        " ",
        text
    )

    # "ap" aparece como residuo frecuente del formato de
    # algunas noticias del corpus.
    text = re.sub(
        r"\bap\b",
        " ",
        text
    )

    # --------------------------------------------------------
    # 7. Eliminar caracteres no alfabéticos
    # --------------------------------------------------------

    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    # --------------------------------------------------------
    # 8. Normalizar espacios
    # --------------------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# TOKENIZACIÓN + LEMATIZACIÓN
# ============================================================

def preprocess_text(text):
    """
    Aplica:
    1. Limpieza
    2. Tokenización mediante SpaCy
    3. Lematización
    4. Eliminación de tokens vacíos
    5. Eliminación de tokens de una sola letra

    Se mantienen las stop-words porque el objetivo del EDA
    es medir su presencia y analizar posteriormente su impacto.
    """

    cleaned = clean_text(text)

    doc = nlp(cleaned)

    tokens = []

    for token in doc:

        # Solo tokens alfabéticos
        if not token.is_alpha:
            continue

        # Obtener lema
        lemma = token.lemma_.lower().strip()

        # Evitar tokens vacíos
        if not lemma:
            continue

        # Evitar residuos de una sola letra
        if len(lemma) < 2:
            continue

        # Evitar residuos conocidos
        if lemma in {
            "href",
            "target",
            "quickinfo",
            "fullquote",
            "quot",
            "lt",
            "gt",
            "ap"
        }:
            continue

        tokens.append(lemma)

    return tokens


# ============================================================
# CARGAR DATASET
# ============================================================

print()
print("=" * 60)
print("CARGANDO DATASET AG NEWS")
print("=" * 60)

df = pd.read_csv(TRAIN_PATH)

print(f"Cantidad de documentos: {len(df)}")
print(f"Columnas: {df.columns.tolist()}")

# Validación
if "text" not in df.columns or "label" not in df.columns:
    raise ValueError(
        "El dataset debe contener las columnas 'text' y 'label'."
    )


# ============================================================
# DISTRIBUCIÓN ORIGINAL DE CLASES
# ============================================================

print()
print("Distribución de clases:")
print(df["label"].value_counts())


# ============================================================
# PREPROCESAMIENTO
# ============================================================

print()
print("=" * 60)
print("PREPROCESANDO TEXTO")
print("=" * 60)

df["tokens"] = df["text"].apply(preprocess_text)

df["processed_text"] = df["tokens"].apply(
    lambda tokens: " ".join(tokens)
)

df["token_count"] = df["tokens"].apply(len)

print("Preprocesamiento finalizado.")


# ============================================================
# EJEMPLO
# ============================================================

print()
print("Ejemplo de documento:")
print()

print("Texto original:")
print(df.iloc[0]["text"])

print()
print("Texto procesado:")
print(df.iloc[0]["tokens"])


# ============================================================
# DISTRIBUCIÓN DE LONGITUD
# ============================================================

print()
print("=" * 60)
print("DISTRIBUCIÓN DE LONGITUD")
print("=" * 60)

longitudes = df["token_count"]

media = longitudes.mean()
mediana = longitudes.median()
minimo = longitudes.min()
maximo = longitudes.max()
percentil_95 = np.percentile(longitudes, 95)

print(f"Longitud promedio: {media:.2f}")
print(f"Longitud mediana: {mediana:.2f}")
print(f"Longitud mínima: {minimo}")
print(f"Longitud máxima: {maximo}")
print(f"Percentil 95: {percentil_95:.2f}")

max_len = int(np.ceil(percentil_95))

print()
print(f"max_len recomendado para el futuro modelo: {max_len}")


# ============================================================
# HISTOGRAMA DE LONGITUD
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    longitudes,
    bins=30,
    edgecolor="black"
)

plt.axvline(
    percentil_95,
    linestyle="--",
    linewidth=2,
    label=f"Percentil 95 = {percentil_95:.0f}"
)

plt.axvline(
    media,
    linestyle=":",
    linewidth=2,
    label=f"Media = {media:.1f}"
)

plt.title(
    "Distribución de longitud de documentos - AG News"
)

plt.xlabel(
    "Cantidad de tokens por documento"
)

plt.ylabel(
    "Cantidad de documentos"
)

plt.legend()

plt.tight_layout()

histograma_path = os.path.join(
    OUTPUT_DIR,
    "histograma_longitud.png"
)

plt.savefig(
    histograma_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# N-GRAMAS
# ============================================================

print()
print("=" * 60)
print("EXTRAYENDO N-GRAMAS")
print("=" * 60)

corpus_text = df["processed_text"].tolist()


# ------------------------------------------------------------
# BIGRAMAS
# ------------------------------------------------------------

vectorizer_bigram = CountVectorizer(
    ngram_range=(2, 2),
    min_df=2
)

X_bigram = vectorizer_bigram.fit_transform(corpus_text)

frecuencias_bigram = np.asarray(
    X_bigram.sum(axis=0)
).ravel()

bigramas = vectorizer_bigram.get_feature_names_out()

bigram_data = pd.DataFrame(
    {
        "Bigram": bigramas,
        "Frecuencia": frecuencias_bigram
    }
)

bigram_data = bigram_data.sort_values(
    "Frecuencia",
    ascending=False
).head(20)

bigram_data = bigram_data.reset_index(drop=True)


print()
print("TOP 20 BIGRAMAS")
print(bigram_data.to_string(index=False))


bigram_path = os.path.join(
    OUTPUT_DIR,
    "top_20_bigramas.csv"
)

bigram_data.to_csv(
    bigram_path,
    index=False
)


# ------------------------------------------------------------
# TRIGRAMAS
# ------------------------------------------------------------

vectorizer_trigram = CountVectorizer(
    ngram_range=(3, 3),
    min_df=2
)

X_trigram = vectorizer_trigram.fit_transform(corpus_text)

frecuencias_trigram = np.asarray(
    X_trigram.sum(axis=0)
).ravel()

trigramas = vectorizer_trigram.get_feature_names_out()

trigram_data = pd.DataFrame(
    {
        "Trigram": trigramas,
        "Frecuencia": frecuencias_trigram
    }
)

trigram_data = trigram_data.sort_values(
    "Frecuencia",
    ascending=False
).head(20)

trigram_data = trigram_data.reset_index(drop=True)


print()
print("TOP 20 TRIGRAMAS")
print(trigram_data.to_string(index=False))


trigram_path = os.path.join(
    OUTPUT_DIR,
    "top_20_trigramas.csv"
)

trigram_data.to_csv(
    trigram_path,
    index=False
)


# ============================================================
# TOP 50 PALABRAS
# ============================================================

print()
print("=" * 60)
print("TOP 50 PALABRAS")
print("=" * 60)

contador = Counter()

for tokens in df["tokens"]:
    contador.update(tokens)

top_50 = contador.most_common(50)

top_50_df = pd.DataFrame(
    top_50,
    columns=[
        "Palabra",
        "Frecuencia"
    ]
)

print(top_50_df.to_string(index=False))


top_50_path = os.path.join(
    OUTPUT_DIR,
    "top_50_palabras.csv"
)

top_50_df.to_csv(
    top_50_path,
    index=False
)


# ============================================================
# ANÁLISIS DE STOP-WORDS
# ============================================================

print()
print("=" * 60)
print("ANÁLISIS DE STOP-WORDS")
print("=" * 60)

stop_words = nlp.Defaults.stop_words

top_50_df["Es_Stop_Word"] = (
    top_50_df["Palabra"].isin(stop_words)
)

stop_words_top50 = top_50_df[
    top_50_df["Es_Stop_Word"]
].copy()

cantidad_stopwords = len(stop_words_top50)

print(
    f"Cantidad de stop-words dentro del Top 50: "
    f"{cantidad_stopwords}"
)

print()
print("Stop-words encontradas:")

if cantidad_stopwords > 0:
    print(
        stop_words_top50[
            ["Palabra", "Frecuencia"]
        ].to_string(index=False)
    )
else:
    print("No se encontraron stop-words.")


# ============================================================
# DISTRIBUCIÓN DE CLASES
# ============================================================

print()
print("=" * 60)
print("DISTRIBUCIÓN DE CLASES")
print("=" * 60)

class_counts = df["label"].value_counts()

print(class_counts)


plt.figure(figsize=(9, 6))

class_counts.plot(
    kind="bar"
)

plt.title(
    "Distribución de clases - AG News"
)

plt.xlabel(
    "Categoría"
)

plt.ylabel(
    "Cantidad de documentos"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

classes_path = os.path.join(
    OUTPUT_DIR,
    "distribucion_clases.png"
)

plt.savefig(
    classes_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# GUARDAR CORPUS PREPROCESADO
# ============================================================

corpus_output = df[
    [
        "text",
        "label",
        "processed_text",
        "token_count"
    ]
].copy()

corpus_path = os.path.join(
    OUTPUT_DIR,
    "corpus_preprocesado.csv"
)

corpus_output.to_csv(
    corpus_path,
    index=False
)


# ============================================================
# RESUMEN FINAL
# ============================================================

print()
print("=" * 60)
print("RESUMEN DEL EDA")
print("=" * 60)

print(f"Documentos analizados: {len(df)}")
print(f"Longitud promedio: {media:.2f}")
print(f"Longitud mediana: {mediana:.2f}")
print(f"Longitud mínima: {minimo}")
print(f"Longitud máxima: {maximo}")
print(f"Percentil 95: {percentil_95:.2f}")
print(f"max_len recomendado: {max_len}")
print(
    f"Stop-words en Top 50: "
    f"{cantidad_stopwords}"
)

print()
print("Distribución de clases:")

for clase, cantidad in class_counts.items():
    print(
        f"  {clase}: {cantidad}"
    )

print()
print("Archivos generados:")
print(
    "- outputs/histograma_longitud.png"
)
print(
    "- outputs/distribucion_clases.png"
)
print(
    "- outputs/top_20_bigramas.csv"
)
print(
    "- outputs/top_20_trigramas.csv"
)
print(
    "- outputs/top_50_palabras.csv"
)
print(
    "- outputs/corpus_preprocesado.csv"
)

print()
print("=" * 60)
print("EDA FINALIZADO CORRECTAMENTE")
print("=" * 60)