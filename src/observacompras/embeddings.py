"""Base preentrenada congelada: Sentence-BERT multilingüe (Sección 5.2).

La red convierte cada descripción en un vector de 384 dimensiones; sus pesos no
se modifican (transfer learning como extracción de características).

Para pruebas sin acceso a Hugging Face existe un codificador de respaldo
(OBSERVACOMPRAS_ENCODER=lsa: TF-IDF + SVD a 384 dimensiones). NO es el modelo
del documento; los resultados obtenidos con él quedan marcados como
"codificador de prueba" en los reportes y en la aplicación.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

import numpy as np

from . import config as C


def nombre_codificador() -> str:
    return os.environ.get("OBSERVACOMPRAS_ENCODER", "sbert")


@lru_cache(maxsize=1)
def cargar_sbert():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(C.MODELO_SBERT)   # congelada: solo se usa .encode()


def codificar(textos, batch_size: int = 64, mostrar_progreso: bool = False) -> np.ndarray:
    """Devuelve embeddings normalizados (norma 1) para usar similitud coseno."""
    textos = [str(t) for t in textos]
    if nombre_codificador() == "sbert":
        base = cargar_sbert()
        E = base.encode(textos, batch_size=batch_size, show_progress_bar=mostrar_progreso,
                        normalize_embeddings=True, convert_to_numpy=True)
        return E.astype("float32")
    return _codificador_lsa().codificar(textos)


# --------------------------------------------------------------------------
# Respaldo solo para pruebas sin conexión
# --------------------------------------------------------------------------
class _LSA:
    ruta = C.DIR_MODELOS / "codificador_lsa_prueba.joblib"

    def __init__(self):
        import joblib
        self.pipe = joblib.load(self.ruta) if self.ruta.exists() else None

    def ajustar(self, textos):
        import joblib
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import make_pipeline
        self.pipe = make_pipeline(TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=50000),
                                  TruncatedSVD(C.DIM_EMBEDDING, random_state=C.SEMILLA))
        self.pipe.fit(textos)
        joblib.dump(self.pipe, self.ruta)

    def codificar(self, textos):
        E = self.pipe.transform(textos).astype("float32")
        return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)


@lru_cache(maxsize=1)
def _codificador_lsa():
    return _LSA()


def preparar_codificador(textos_entrenamiento) -> None:
    """Solo el respaldo LSA necesita ajustarse (con textos del año de entrenamiento)."""
    if nombre_codificador() != "sbert":
        _codificador_lsa().ajustar(list(textos_entrenamiento))


def guardar_metadatos(ruta, n: int) -> None:
    with open(ruta, "w") as f:
        json.dump({"codificador": nombre_codificador(),
                   "modelo": C.MODELO_SBERT if nombre_codificador() == "sbert" else "TF-IDF+SVD (solo prueba)",
                   "dimension": C.DIM_EMBEDDING, "n": n}, f, indent=2)
