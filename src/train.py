import os
import ast
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURACIÓN
# ============================================================

SEED = 42

TRAIN_PATH = "data/ag_news_train.csv"
TEST_PATH = "data/ag_news_test.csv"
OUTPUT_DIR = "outputs"

MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "modelo_ag_news.pth"
)

LOSS_CURVE_PATH = os.path.join(
    OUTPUT_DIR,
    "loss_curve.png"
)

ACCURACY_CURVE_PATH = os.path.join(
    OUTPUT_DIR,
    "accuracy_curve.png"
)


# ============================================================
# REPRODUCIBILIDAD
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DISPOSITIVO
# ============================================================

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


print("=" * 60)
print("ENTRENAMIENTO DEL MODELO NLP - AG NEWS")
print("=" * 60)
print(f"Dispositivo utilizado: {device}")


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def convertir_tokens(valor):
    """
    Convierte la columna processed_text del CSV
    en una lista de tokens.
    """

    if pd.isna(valor):
        return []

    if isinstance(valor, list):
        return valor

    try:
        resultado = ast.literal_eval(str(valor))

        if isinstance(resultado, list):
            return resultado

    except (ValueError, SyntaxError):
        pass

    return str(valor).split()


def tokens_a_texto(valor):
    """
    Convierte la lista de tokens nuevamente en texto.
    """

    tokens = convertir_tokens(valor)

    return " ".join(tokens)


# ============================================================
# MODELO
# ============================================================

class AGNewsClassifier(nn.Module):
    """
    Clasificador de noticias AG News.

    Entrada:
        vector TF-IDF

    Salida:
        4 clases
    """

    def __init__(self, input_size, num_classes=4):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# FUNCIÓN DE EVALUACIÓN
# ============================================================

def evaluate(model, criterion, X, y):
    """
    Evalúa el modelo sobre un conjunto de datos.
    """

    model.eval()

    with torch.no_grad():

        outputs = model(X)

        loss = criterion(outputs, y)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        accuracy = (
            predictions == y
        ).float().mean()

    return loss.item(), accuracy.item()


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # 1. CREAR CARPETA DE SALIDA
    # ========================================================

    os.makedirs(OUTPUT_DIR, exist_ok=True)


    # ========================================================
    # 2. CARGAR DATASETS
    # ========================================================

    print("\n" + "=" * 60)
    print("CARGANDO DATASETS")
    print("=" * 60)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print(f"Documentos de entrenamiento: {len(train_df)}")
    print(f"Documentos de prueba: {len(test_df)}")

    print("\nColumnas:")
    print(train_df.columns.tolist())


    # ========================================================
    # 3. PREPARAR TEXTOS
    # ========================================================

    print("\n" + "=" * 60)
    print("PREPARANDO TEXTOS")
    print("=" * 60)

    # Utilizamos el texto preprocesado si existe.
    if "processed_text" in train_df.columns:

        train_texts = train_df["processed_text"].apply(
            tokens_a_texto
        )

    else:

        train_texts = train_df["text"].astype(str)


    test_texts = test_df["text"].astype(str)


    # ========================================================
    # 4. CONVERTIR CLASES A NÚMEROS
    # ========================================================

    print("\n" + "=" * 60)
    print("CODIFICANDO CLASES")
    print("=" * 60)

    class_names = sorted(
        train_df["label"].unique()
    )

    label_to_id = {
        label: i
        for i, label in enumerate(class_names)
    }

    id_to_label = {
        i: label
        for label, i in label_to_id.items()
    }

    print("Clases detectadas:")

    for i, label in id_to_label.items():
        print(f"{i} -> {label}")


    y_train_full = train_df["label"].map(
        label_to_id
    ).values

    y_test = test_df["label"].map(
        label_to_id
    ).values


    # ========================================================
    # 5. DIVISIÓN TRAIN / VALIDACIÓN
    # ========================================================

    print("\n" + "=" * 60)
    print("DIVISIÓN TRAIN / VALIDACIÓN")
    print("=" * 60)

    (
        X_train_text,
        X_val_text,
        y_train,
        y_val
    ) = train_test_split(
        train_texts,
        y_train_full,
        test_size=0.20,
        random_state=SEED,
        stratify=y_train_full
    )

    print(f"Entrenamiento: {len(X_train_text)}")
    print(f"Validación: {len(X_val_text)}")
    print(f"Prueba: {len(test_texts)}")


    # ========================================================
    # 6. REPRESENTACIÓN TF-IDF
    # ========================================================

    print("\n" + "=" * 60)
    print("GENERANDO REPRESENTACIÓN TF-IDF")
    print("=" * 60)

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    X_train_tfidf = vectorizer.fit_transform(
        X_train_text
    )

    X_val_tfidf = vectorizer.transform(
        X_val_text
    )

    X_test_tfidf = vectorizer.transform(
        test_texts
    )

    print(
        f"Características TF-IDF: "
        f"{X_train_tfidf.shape[1]}"
    )

    print(
        f"Matriz de entrenamiento: "
        f"{X_train_tfidf.shape}"
    )

    print(
        f"Matriz de validación: "
        f"{X_val_tfidf.shape}"
    )


    # ========================================================
    # 7. CONVERTIR A ARRAYS DENSOS
    # ========================================================

    X_train_array = X_train_tfidf.toarray().astype(
        np.float32
    )

    X_val_array = X_val_tfidf.toarray().astype(
        np.float32
    )

    X_test_array = X_test_tfidf.toarray().astype(
        np.float32
    )


    # ========================================================
    # 8. CONVERTIR A TENSORES PYTORCH
    # ========================================================

    X_train = torch.tensor(
        X_train_array,
        dtype=torch.float32
    ).to(device)

    y_train = torch.tensor(
        y_train,
        dtype=torch.long
    ).to(device)

    X_val = torch.tensor(
        X_val_array,
        dtype=torch.float32
    ).to(device)

    y_val = torch.tensor(
        y_val,
        dtype=torch.long
    ).to(device)

    X_test = torch.tensor(
        X_test_array,
        dtype=torch.float32
    ).to(device)

    y_test = torch.tensor(
        y_test,
        dtype=torch.long
    ).to(device)


    # ========================================================
    # 9. CREAR MODELO
    # ========================================================

    print("\n" + "=" * 60)
    print("ARQUITECTURA DEL MODELO")
    print("=" * 60)

    input_size = X_train.shape[1]

    num_classes = len(class_names)

    model = AGNewsClassifier(
        input_size=input_size,
        num_classes=num_classes
    ).to(device)

    print(model)


    # ========================================================
    # 10. FUNCIÓN DE PÉRDIDA Y OPTIMIZADOR
    # ========================================================

    criterion = nn.CrossEntropyLoss()

    learning_rate = 0.001

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )


    # ========================================================
    # 11. TRAINING LOOP
    # ========================================================

    print("\n" + "=" * 60)
    print("INICIANDO ENTRENAMIENTO")
    print("=" * 60)

    epochs = 30

    train_losses = []
    val_losses = []

    train_accuracies = []
    val_accuracies = []


    for epoch in range(epochs):

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.train()

        outputs = model(X_train)

        loss = criterion(
            outputs,
            y_train
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        # ----------------------------------------------------
        # TRAIN ACCURACY
        # ----------------------------------------------------

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        train_accuracy = (
            (predictions == y_train)
            .float()
            .mean()
            .item()
        )


        # ----------------------------------------------------
        # VALIDACIÓN
        # ----------------------------------------------------

        val_loss, val_accuracy = evaluate(
            model,
            criterion,
            X_val,
            y_val
        )


        # ----------------------------------------------------
        # GUARDAR MÉTRICAS
        # ----------------------------------------------------

        train_losses.append(
            loss.item()
        )

        val_losses.append(
            val_loss
        )

        train_accuracies.append(
            train_accuracy
        )

        val_accuracies.append(
            val_accuracy
        )


        # ----------------------------------------------------
        # MOSTRAR RESULTADOS
        # ----------------------------------------------------

        print(
            f"Época [{epoch + 1}/{epochs}] "
            f"| Train Loss: {loss.item():.4f} "
            f"| Train Accuracy: {train_accuracy:.4f} "
            f"| Val Loss: {val_loss:.4f} "
            f"| Val Accuracy: {val_accuracy:.4f}"
        )


    # ========================================================
    # 12. EVALUACIÓN FINAL
    # ========================================================

    final_val_loss, final_val_accuracy = evaluate(
        model,
        criterion,
        X_val,
        y_val
    )

    final_test_loss, final_test_accuracy = evaluate(
        model,
        criterion,
        X_test,
        y_test
    )

    print("\n" + "=" * 60)
    print("RESULTADO FINAL")
    print("=" * 60)

    print(
        f"Validation Loss: "
        f"{final_val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{final_val_accuracy:.4f}"
    )

    print(
        f"Test Loss: "
        f"{final_test_loss:.4f}"
    )

    print(
        f"Test Accuracy: "
        f"{final_test_accuracy:.4f}"
    )


    # ========================================================
    # 13. GUARDAR MODELO
    # ========================================================

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": input_size,
            "num_classes": num_classes,
            "class_names": class_names,
            "label_to_id": label_to_id,
            "id_to_label": id_to_label,
            "max_features": 5000
        },
        MODEL_PATH
    )

    print(
        f"\nModelo guardado en: "
        f"{MODEL_PATH}"
    )


    # ========================================================
    # 14. CURVA DE LOSS
    # ========================================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        train_losses,
        label="Train Loss"
    )

    plt.plot(
        val_losses,
        label="Validation Loss"
    )

    plt.xlabel("Época")
    plt.ylabel("Loss")

    plt.title(
        "Curva de pérdida - AG News"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        LOSS_CURVE_PATH,
        dpi=150
    )

    plt.close()

    print(
        f"Curva de pérdida guardada en: "
        f"{LOSS_CURVE_PATH}"
    )


    # ========================================================
    # 15. CURVA DE ACCURACY
    # ========================================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        train_accuracies,
        label="Train Accuracy"
    )

    plt.plot(
        val_accuracies,
        label="Validation Accuracy"
    )

    plt.xlabel("Época")
    plt.ylabel("Accuracy")

    plt.title(
        "Accuracy durante el entrenamiento - AG News"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        ACCURACY_CURVE_PATH,
        dpi=150
    )

    plt.close()

    print(
        f"Curva de accuracy guardada en: "
        f"{ACCURACY_CURVE_PATH}"
    )


    # ========================================================
    # FINAL
    # ========================================================

    print("\n" + "=" * 60)
    print("ENTRENAMIENTO FINALIZADO CORRECTAMENTE")
    print("=" * 60)


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    main()