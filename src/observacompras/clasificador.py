"""F4 - Clasificador UNSPSC con transfer learning (Sección 5.2, Tabla 3).

Base congelada (Sentence-BERT, 384 dim.) + clasificador propio:
Dense 128 ReLU -> Dropout 0,3 -> Dense n_clases softmax, con Adam y
categorical_crossentropy. Entrenamiento con 2023 (85/15) y prueba con 2024.

Línea base (H2): TF-IDF + regresión logística sobre los mismos textos.
"""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from . import config as C


def preparar_etiquetas(lic: pd.DataFrame):
    """Clases = familias UNSPSC con al menos 30 ejemplos en el año de entrenamiento."""
    tr = lic[lic["anio"] == C.ANIO_ENTRENAMIENTO]
    conteo = tr["familia"].value_counts()
    clases = sorted(conteo[conteo >= C.MIN_EJEMPLOS_CLASE].index.tolist())
    return clases


def construir_clasificador(n_clases: int, learning_rate: float = 1e-3, dim: int = C.DIM_EMBEDDING):
    from keras import layers, models, optimizers
    clf = models.Sequential([
        layers.Input(shape=(dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),                      # control de sobreajuste
        layers.Dense(n_clases, activation="softmax"),
    ], name="clasificador_unspsc")
    clf.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
                loss="categorical_crossentropy", metrics=["accuracy"])
    return clf


def _parada():
    from keras import callbacks
    return callbacks.EarlyStopping(monitor="val_loss", patience=C.CLF_PACIENCIA,
                                   restore_best_weights=True)


def busqueda_hiperparametros(E_train: np.ndarray, y_onehot: np.ndarray,
                             learning_rates=C.CLF_LEARNING_RATES, batch_sizes=C.CLF_BATCH_SIZES,
                             verbose: int = 0):
    """Partición 85/15 (validation_split) para cada combinación; se queda con el
    modelo de menor val_loss. Los datos se barajan antes porque Keras toma el
    15% final como validación."""
    import keras
    filas, mejor = [], None
    for lr, bs in product(learning_rates, batch_sizes):
        keras.utils.set_random_seed(C.SEMILLA)
        clf = construir_clasificador(y_onehot.shape[1], lr, E_train.shape[1])
        h = clf.fit(E_train, y_onehot, validation_split=C.CLF_VALIDACION, epochs=C.CLF_EPOCHS,
                    batch_size=bs, callbacks=[_parada()], verbose=verbose)
        vl = h.history["val_loss"]
        fila = {"learning_rate": lr, "batch_size": bs, "mejor_val_loss": float(np.min(vl)),
                "val_accuracy_en_mejor": float(h.history["val_accuracy"][int(np.argmin(vl))]),
                "mejor_epoch": int(np.argmin(vl)) + 1, "epochs_ejecutados": len(vl)}
        filas.append(fila)
        print(f"  lr={lr:g} batch={bs}: val_loss={fila['mejor_val_loss']:.4f} "
              f"val_acc={fila['val_accuracy_en_mejor']:.3f} (epoch {fila['mejor_epoch']})")
        if mejor is None or fila["mejor_val_loss"] < mejor[0]["mejor_val_loss"]:
            mejor = (fila, clf, h.history)
    return pd.DataFrame(filas), mejor


def top_k(probs: np.ndarray, k: int = C.TOP_K_SUGERENCIAS) -> np.ndarray:
    return np.argsort(probs, axis=1)[:, ::-1][:, :k]


def metricas(y_true: np.ndarray, probs: np.ndarray, n_clases: int) -> dict:
    pred = probs.argmax(axis=1)
    top3 = top_k(probs, 3)
    return {
        "f1_macro": float(f1_score(y_true, pred, average="macro", labels=np.arange(n_clases), zero_division=0)),
        "exactitud": float(accuracy_score(y_true, pred)),
        "top3": float(np.mean([y in t for y, t in zip(y_true, top3)])),
        "n": int(len(y_true)),
    }


class LineaBaseTfidf:
    """TF-IDF + regresión logística (línea base clásica de H2)."""

    def __init__(self, C_reg: float = 1.0):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        self.vec = TfidfVectorizer(lowercase=True, strip_accents="unicode", sublinear_tf=True,
                                   ngram_range=(1, 2), min_df=2, max_features=100_000)
        self.lr = LogisticRegression(C=C_reg, max_iter=3000)

    def fit(self, textos, y):
        self.lr.fit(self.vec.fit_transform(textos), y)
        return self

    def predict_proba(self, textos, n_clases: int) -> np.ndarray:
        p = self.lr.predict_proba(self.vec.transform(textos))
        full = np.zeros((p.shape[0], n_clases))
        full[:, self.lr.classes_] = p
        return full


def elegir_linea_base(textos_tr, y_tr, textos_va, y_va, n_clases, valores_C=(0.1, 1.0, 10.0, 100.0)):
    """Elige la regularización de la línea base con la misma partición 85/15."""
    res = []
    for c in valores_C:
        m = LineaBaseTfidf(c).fit(textos_tr, y_tr)
        res.append((metricas(y_va, m.predict_proba(textos_va, n_clases), n_clases)["f1_macro"], c))
        print(f"  línea base C={c:g}: F1-macro validación={res[-1][0]:.4f}")
    return max(res)[1]
