"""Evaluación H1: anomalías sintéticas y línea base de puntaje z (Tabla 4).

Decisión (d): 20 organismos artificiales, 5 tipos x 4 casos. Cada caso parte de
un organismo real del año de evaluación elegido al azar y se le modifican las
variables del patrón; el resto queda como en el organismo real.

Decisión (e): la línea base asigna a cada organismo el máximo |z| entre sus 10
variables, con media y desviación del año de entrenamiento.

Recall@5%: proporción de anomalías cuyo puntaje queda en el 5% superior de los
organismos reales del año de evaluación. Cada anomalía se compara con la
población real por separado; inyectarlas todas a la vez haría imposible la
meta (20 anomalías no caben en el 5% de ~240 organismos, que son 12 lugares).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

TIPOS = ["proveedor_unico", "trato_directo", "oferente_unico", "monto_extremo", "combinacion"]
DESCRIPCION_TIPOS = {
    "proveedor_unico": "Un proveedor concentra ~95% del monto (CR1 y HHI extremos)",
    "trato_directo": "85-100% de las compras por trato directo",
    "oferente_unico": "85-100% de las licitaciones con un solo oferente",
    "monto_extremo": "Monto medio por compra 10 veces el máximo observado",
    "combinacion": "Varias variables en su percentil 90, cada una normal por separado",
}


def generar_sinteticas(base: pd.DataFrame, referencia: pd.DataFrame,
                       n_por_tipo: int = C.N_ANOMALIAS_SINTETICAS // len(TIPOS),
                       semilla: int = C.SEMILLA) -> pd.DataFrame:
    """base: perfiles reales del año de evaluación; referencia: año de entrenamiento."""
    rng = np.random.default_rng(semilla)
    p90 = referencia[C.VARIABLES].quantile(0.90)
    filas = []
    for tipo in TIPOS:
        idx = rng.choice(len(base), size=n_por_tipo, replace=False)
        for k, i in enumerate(idx):
            f = base.iloc[i].copy()
            if tipo == "proveedor_unico":
                cr1 = rng.uniform(0.93, 0.98)
                f["cr1"] = cr1
                f["hhi"] = (100 * cr1) ** 2 + (100 * (1 - cr1)) ** 2 / max(f["n_proveedores"] - 1, 1)
            elif tipo == "trato_directo":
                f["pct_trato_directo"] = rng.uniform(0.85, 1.0)
            elif tipo == "oferente_unico":
                f["pct_oferente_unico"] = rng.uniform(0.85, 1.0)
            elif tipo == "monto_extremo":
                f["monto_medio_log"] = referencia["monto_medio_log"].max() + np.log(10)
                f["monto_total_log"] = np.log(f["n_compras"]) + f["monto_medio_log"]
            elif tipo == "combinacion":
                for v in ["cr1", "hhi", "pct_trato_directo", "pct_oferente_unico", "pct_proveedores_nuevos"]:
                    f[v] = p90[v]
                f["n_proveedores"] = referencia["n_proveedores"].quantile(0.10)
                f["n_rubros"] = referencia["n_rubros"].quantile(0.10)
            f["organismo"] = f"SINT-{tipo}-{k + 1}"
            f["organismo_nombre"] = f"Organismo sintético ({tipo} #{k + 1})"
            f["tipo_anomalia"] = tipo
            filas.append(f)
    return pd.DataFrame(filas).reset_index(drop=True)


class LineaBaseZ:
    """Regla de puntaje z por variable: puntaje = max_j |z_j|."""

    def fit(self, X: np.ndarray):
        self.media = X.mean(axis=0)
        self.desv = np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)
        return self

    def puntuar(self, X: np.ndarray) -> np.ndarray:
        return np.abs((X - self.media) / self.desv).max(axis=1)


def recall_en_top(puntaje_reales: np.ndarray, puntaje_sinteticas: np.ndarray,
                  percentil: float = C.PERCENTIL_ALERTA) -> tuple[float, float]:
    umbral = np.percentile(puntaje_reales, percentil)
    return float((puntaje_sinteticas > umbral).mean()), float(umbral)


def recall_por_tipo(tipos: pd.Series, puntaje_reales, puntaje_sinteticas,
                    percentil: float = C.PERCENTIL_ALERTA) -> dict:
    umbral = np.percentile(puntaje_reales, percentil)
    detect = pd.Series(puntaje_sinteticas > umbral, index=tipos.index)
    return detect.groupby(tipos).mean().to_dict()
