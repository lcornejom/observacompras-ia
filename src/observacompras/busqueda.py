"""F5 - Búsqueda semántica de licitaciones similares (Sección 5.2).

Reutiliza los embeddings de 384 dimensiones de F4: dos licitaciones son
similares si sus vectores tienen alta similitud coseno. Como mitigación del
riesgo de plazo (Tabla 5), si el modelo de lenguaje no está disponible se usa
búsqueda por palabras clave (TF-IDF).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class BuscadorSemantico:
    def __init__(self, embeddings: np.ndarray, licitaciones: pd.DataFrame):
        E = embeddings.astype("float32")
        self.E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
        self.lic = licitaciones.reset_index(drop=True)
        self._indice = {n: i for i, n in enumerate(self.lic["NroLicitacion"])}

    def _resultado(self, sims: np.ndarray, k: int, excluir: set[int], excluir_organismo: str | None):
        orden = np.argsort(sims)[::-1]
        filas = []
        for i in orden:
            if i in excluir:
                continue
            if excluir_organismo is not None and self.lic.at[i, "organismo"] == excluir_organismo:
                continue
            filas.append(i)
            if len(filas) == k:
                break
        out = self.lic.iloc[filas].copy()
        out.insert(0, "similitud", sims[filas])
        return out

    def por_vector(self, q: np.ndarray, k: int = 10, excluir_organismo: str | None = None):
        q = q.reshape(-1) / max(np.linalg.norm(q), 1e-9)
        return self._resultado(self.E @ q, k, set(), excluir_organismo)

    def por_licitacion(self, nro: str, k: int = 10, otros_organismos: bool = True):
        i = self._indice[nro]
        org = self.lic.at[i, "organismo"] if otros_organismos else None
        return self._resultado(self.E @ self.E[i], k, {i}, org)


class BuscadorPalabrasClave:
    """Respaldo: búsqueda por palabras clave con TF-IDF."""

    def __init__(self, licitaciones: pd.DataFrame):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.lic = licitaciones.reset_index(drop=True)
        self.vec = TfidfVectorizer(strip_accents="unicode", sublinear_tf=True, min_df=1)
        self.M = self.vec.fit_transform(self.lic["texto"])

    def por_texto(self, texto: str, k: int = 10):
        sims = (self.M @ self.vec.transform([texto]).T).toarray().ravel()
        orden = np.argsort(sims)[::-1][:k]
        out = self.lic.iloc[orden].copy()
        out.insert(0, "similitud", sims[orden])
        return out
