"""Prepara los datos que usa la aplicación (paso 3 del flujo de uso, Sección 6).

Para los organismos más atípicos del año de evaluación, precalcula la sugerencia
del clasificador para cada una de sus órdenes de compra, de modo que el analista
pueda ver si la categoría registrada coincide con la sugerida sin cargar el
modelo de lenguaje en la aplicación.

Salida: data/processed/oc_revision.parquet
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import clasificador as K
from . import config as C
from .ingesta import leer_oc
from . import embeddings as EMB

N_ORGANISMOS_TOP = 30


def ejecutar() -> int:
    import keras
    punt = pd.read_parquet(C.DIR_PROCESADOS / "puntajes_atipicidad.parquet")
    oc = leer_oc()
    clases = [c["codigo"] for c in json.load(open(C.DIR_MODELOS / "clases_unspsc.json"))["clases"]]
    clf = keras.models.load_model(C.DIR_MODELOS / "clasificador.keras")

    top = punt[punt["anio"] == C.ANIO_PRUEBA].nsmallest(N_ORGANISMOS_TOP, "ranking")["organismo"]
    sel = oc[(oc["anio"] == C.ANIO_PRUEBA) & oc["organismo"].isin(top)].copy()
    sel = sel[~sel["proveedor_persona_natural"] & (sel["texto"].str.len() > 10)]
    print(f"  clasificando {len(sel)} órdenes de compra de los {N_ORGANISMOS_TOP} organismos más atípicos")
    E = EMB.codificar(sel["texto"].tolist(), mostrar_progreso=True)
    probs = clf.predict(E, verbose=0)
    t = K.top_k(probs, C.TOP_K_SUGERENCIAS)
    sel["familia_registrada"] = sel["onu_principal"].astype("string").str.zfill(8).str[: C.NIVEL_UNSPSC]
    for r in range(C.TOP_K_SUGERENCIAS):
        sel[f"sugerencia_{r + 1}"] = [clases[j] for j in t[:, r]]
        sel[f"prob_{r + 1}"] = probs[np.arange(len(probs)), t[:, r]]
    sel["coincide_sugerencia"] = sel["familia_registrada"] == sel["sugerencia_1"]
    cols = ["codigoOC", "organismo", "ProveedorRUT", "familia_registrada", "sugerencia_1", "prob_1",
            "sugerencia_2", "prob_2", "sugerencia_3", "prob_3", "coincide_sugerencia"]
    sel[cols].to_parquet(C.DIR_PROCESADOS / "oc_revision.parquet", index=False)
    return len(sel)
