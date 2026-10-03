"""F3 - Detector de patrones atípicos: autoencoder (Sección 5.1, Tabla 3).

Protocolo:
1. Las 10 variables se escalan a [0, 1] con mínimos y máximos del año de
   entrenamiento (2023); en 2024 los valores fuera de rango se recortan a [0, 1]
   porque la salida sigmoide no puede superar ese intervalo.
2. Búsqueda de hiperparámetros: 3 learning rates x 3 batch sizes; cada
   combinación se valida con K-fold (K = 5) sobre los organismos de 2023, con
   early stopping (patience = 20, restore_best_weights). Se elige la combinación
   con menor pérdida de validación media.
3. Modelo final: como en el ejemplo de Boston Housing, se entrena con todos los
   organismos de 2023 durante el número de epochs que minimizó la pérdida de
   validación media en el K-fold.
4. Puntaje de atipicidad = error cuadrático medio de reconstrucción; el error de
   cada variable es la explicación.
"""
from __future__ import annotations

import json
from itertools import combinations, product

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from . import config as C


# --------------------------------------------------------------------------
# Escalamiento
# --------------------------------------------------------------------------
class EscaladorMinMax:
    def __init__(self, minimo=None, maximo=None):
        self.minimo = None if minimo is None else np.asarray(minimo, dtype="float64")
        self.maximo = None if maximo is None else np.asarray(maximo, dtype="float64")

    def fit(self, X):
        X = np.asarray(X, dtype="float64")
        self.minimo, self.maximo = X.min(axis=0), X.max(axis=0)
        return self

    def transform(self, X):
        rango = np.where(self.maximo > self.minimo, self.maximo - self.minimo, 1.0)
        return np.clip((np.asarray(X, dtype="float64") - self.minimo) / rango, 0.0, 1.0).astype("float32")

    def guardar(self, ruta):
        with open(ruta, "w") as f:
            json.dump({"variables": C.VARIABLES, "min": self.minimo.tolist(), "max": self.maximo.tolist()}, f, indent=2)

    @classmethod
    def cargar(cls, ruta):
        with open(ruta) as f:
            d = json.load(f)
        return cls(d["min"], d["max"])


# --------------------------------------------------------------------------
# Modelo (código de la Sección 5.1)
# --------------------------------------------------------------------------
def fijar_semillas(semilla: int = C.SEMILLA) -> None:
    import keras
    keras.utils.set_random_seed(semilla)


def construir_autoencoder(learning_rate: float = 1e-3, n_variables: int = len(C.VARIABLES)):
    from keras import layers, models, optimizers
    ae = models.Sequential([
        layers.Input(shape=(n_variables,)),
        layers.Dense(16, activation="relu"),
        layers.Dense(4, activation="relu"),        # espacio latente
        layers.Dense(16, activation="relu"),
        layers.Dense(n_variables, activation="sigmoid"),  # variables escaladas a [0, 1]
    ], name="autoencoder")
    ae.compile(optimizer=optimizers.RMSprop(learning_rate=learning_rate), loss="mse")
    return ae


def _parada():
    from keras import callbacks
    return callbacks.EarlyStopping(monitor="val_loss", patience=C.AE_PACIENCIA,
                                   restore_best_weights=True)


def busqueda_kfold(X: np.ndarray, learning_rates=C.AE_LEARNING_RATES, batch_sizes=C.AE_BATCH_SIZES,
                   k: int = C.AE_K_FOLDS, epochs: int = C.AE_EPOCHS, verbose: int = 0):
    """Devuelve (tabla de resultados, historiales por combinación)."""
    kf = KFold(n_splits=k, shuffle=True, random_state=C.SEMILLA)
    filas, historiales = [], {}
    for lr, bs in product(learning_rates, batch_sizes):
        hist_comb = []
        for i, (tr, va) in enumerate(kf.split(X)):
            fijar_semillas(C.SEMILLA + i)
            ae = construir_autoencoder(lr, X.shape[1])
            h = ae.fit(X[tr], X[tr], validation_data=(X[va], X[va]), epochs=epochs,
                       batch_size=bs, callbacks=[_parada()], verbose=verbose)
            vl = h.history["val_loss"]
            filas.append({"learning_rate": lr, "batch_size": bs, "fold": i,
                          "mejor_val_loss": float(np.min(vl)), "mejor_epoch": int(np.argmin(vl)) + 1,
                          "epochs_ejecutados": len(vl)})
            hist_comb.append(h.history)
        historiales[(lr, bs)] = hist_comb
        print(f"  lr={lr:g} batch={bs}: val_loss medio="
              f"{np.mean([f['mejor_val_loss'] for f in filas[-k:]]):.5f}")
    return pd.DataFrame(filas), historiales


def elegir_configuracion(tabla: pd.DataFrame) -> dict:
    res = (tabla.groupby(["learning_rate", "batch_size"])
                .agg(val_loss_media=("mejor_val_loss", "mean"), val_loss_std=("mejor_val_loss", "std"))
                .reset_index().sort_values("val_loss_media"))
    mejor = res.iloc[0]
    return {"learning_rate": float(mejor.learning_rate), "batch_size": int(mejor.batch_size),
            "val_loss_media": float(mejor.val_loss_media)}


def curva_media(historiales: list[dict]) -> pd.DataFrame:
    """Curva de pérdida media por epoch entre folds (Boston Housing).

    Como el early stopping detiene cada fold en un epoch distinto, la media
    solo se calcula en los epochs que completaron todos los folds (`n_folds`
    indica cuántos folds aportan a cada epoch)."""
    largo = max(len(h["loss"]) for h in historiales)
    def _matriz(clave):
        m = np.full((len(historiales), largo), np.nan)
        for i, h in enumerate(historiales):
            m[i, : len(h[clave])] = h[clave]
        return m
    L, V = _matriz("loss"), _matriz("val_loss")
    n = (~np.isnan(V)).sum(axis=0)
    completo = n == len(historiales)
    return pd.DataFrame({"epoch": np.arange(1, largo + 1),
                         "loss": np.where(completo, np.nanmean(L, axis=0), np.nan),
                         "val_loss": np.where(completo, np.nanmean(V, axis=0), np.nan),
                         "n_folds": n})


def epochs_optimos(tabla_config: pd.DataFrame) -> int:
    """Epochs del modelo final: mediana del mejor epoch de cada fold."""
    return int(np.median(tabla_config["mejor_epoch"]))


def entrenar_final(X: np.ndarray, learning_rate: float, batch_size: int, epochs: int):
    fijar_semillas()
    ae = construir_autoencoder(learning_rate, X.shape[1])
    h = ae.fit(X, X, epochs=epochs, batch_size=batch_size, verbose=0)
    return ae, h.history


def modelos_por_fold(X: np.ndarray, learning_rate: float, batch_size: int, k: int = C.AE_K_FOLDS):
    """Los K modelos de la configuración elegida (para la estabilidad del ranking)."""
    kf = KFold(n_splits=k, shuffle=True, random_state=C.SEMILLA)
    modelos = []
    for i, (tr, va) in enumerate(kf.split(X)):
        fijar_semillas(C.SEMILLA + i)
        ae = construir_autoencoder(learning_rate, X.shape[1])
        ae.fit(X[tr], X[tr], validation_data=(X[va], X[va]), epochs=C.AE_EPOCHS,
               batch_size=batch_size, callbacks=[_parada()], verbose=0)
        modelos.append(ae)
    return modelos


# --------------------------------------------------------------------------
# Puntaje y explicación
# --------------------------------------------------------------------------
def puntuar(ae, X: np.ndarray):
    X_hat = ae.predict(X, verbose=0)
    error_por_variable = (X_hat - X) ** 2       # explicación
    puntaje = error_por_variable.mean(axis=1)   # atipicidad
    return puntaje, error_por_variable, X_hat


def frase_explicacion(errores: np.ndarray, x: np.ndarray, x_hat: np.ndarray, n: int = 2) -> str:
    """Frase descriptiva con las variables que más aportan al error.

    Usa lenguaje de patrones ("se aparta del típico"); nunca califica conductas.
    """
    orden = np.argsort(errores)[::-1][:n]
    partes = []
    for j in orden:
        var = C.VARIABLES[j]
        sentido = "sobre" if x[j] > x_hat[j] else "bajo"
        partes.append(f"{C.FRASES_VARIABLES[var]} ({sentido} lo esperado)")
    return ("Este organismo se aparta del patrón típico principalmente por "
            + " y por ".join(partes) + ".")


def tabla_puntajes(perfiles: pd.DataFrame, X: np.ndarray, ae) -> pd.DataFrame:
    puntaje, err, X_hat = puntuar(ae, X)
    out = perfiles[["organismo", "organismo_nombre", "anio"]].copy().reset_index(drop=True)
    out["puntaje"] = puntaje
    out["percentil"] = out.groupby("anio")["puntaje"].rank(pct=True) * 100
    out["ranking"] = out.groupby("anio")["puntaje"].rank(ascending=False, method="first").astype(int)
    for j, v in enumerate(C.VARIABLES):
        out[f"err_{v}"] = err[:, j]
        out[f"x_{v}"] = X[:, j]
        out[f"xhat_{v}"] = X_hat[:, j]
    out["variable_principal"] = [C.VARIABLES[j] for j in err.argmax(axis=1)]
    out["explicacion"] = [frase_explicacion(err[i], X[i], X_hat[i]) for i in range(len(out))]
    return out


# --------------------------------------------------------------------------
# Estabilidad del ranking (Tabla 4)
# --------------------------------------------------------------------------
def estabilidad_top(modelos, X: np.ndarray, top: int = C.TOP_N_ESTABILIDAD) -> dict:
    tops = []
    for m in modelos:
        p, _, _ = puntuar(m, X)
        tops.append(set(np.argsort(p)[::-1][:top]))
    pares = [len(a & b) for a, b in combinations(tops, 2)]
    return {"coincidencia_media_pares": float(np.mean(pares)),
            "coincidencia_minima_pares": int(np.min(pares)),
            "interseccion_todos_los_folds": int(len(set.intersection(*tops))),
            "top": top}
