"""F2 - Perfil de compra: 10 variables por organismo y año (Tabla 2).

Definiciones:
    n_compras               N.º de órdenes de compra emitidas.
    monto_total_log         ln(1 + suma del monto neto en CLP).
    n_proveedores           N.º de proveedores distintos (por RUT).
    hhi                     Σ (participación % de cada proveedor en el monto)²; máx. 10.000.
    cr1                     Participación del principal proveedor en el monto (0 a 1).
    pct_trato_directo       Fracción de OC cuya procedencia es Trato Directo.
    monto_medio_log         ln(1 + monto medio por OC).
    pct_proveedores_nuevos  Fracción de proveedores del año cuya primera OC con el
                            organismo en ese año cae en la segunda mitad del año
                            (decisión a: misma definición para 2023 y 2024).
    n_rubros                N.º de rubros (nivel 1 de Mercado Público) distintos comprados.
    pct_oferente_unico      Fracción de las licitaciones adjudicadas del organismo con un
                            único oferente.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .ingesta import leer_oc


def _hhi_cr1(oc: pd.DataFrame) -> pd.DataFrame:
    por_prov = oc.groupby(["organismo", "anio", "ProveedorRUT"], observed=True)["monto"].sum()
    total = por_prov.groupby(level=[0, 1]).transform("sum")
    part = por_prov / total
    return pd.DataFrame({
        "hhi": (100 * part).pow(2).groupby(level=[0, 1]).sum(),
        "cr1": part.groupby(level=[0, 1]).max(),
    })


def _proveedores_nuevos(oc: pd.DataFrame) -> pd.Series:
    primera = (oc.sort_values("fecha")
                 .drop_duplicates(["organismo", "anio", "ProveedorRUT"]))
    return primera.groupby(["organismo", "anio"])["segunda_mitad"].mean().rename("pct_proveedores_nuevos")


def _n_rubros(oc: pd.DataFrame) -> pd.Series:
    r = oc[["organismo", "anio", "rubros"]].dropna().copy()
    r["rubros"] = r["rubros"].str.split("|")
    r = r.explode("rubros")
    return r.groupby(["organismo", "anio"])["rubros"].nunique().rename("n_rubros")


def _oferente_unico(lic: pd.DataFrame) -> pd.Series:
    return (lic.assign(unico=lic["n_oferentes"] == 1)
               .groupby(["organismo", "anio"])["unico"].mean().rename("pct_oferente_unico"))


def calcular_perfiles(oc: pd.DataFrame, lic: pd.DataFrame,
                      min_oc: int = C.MIN_OC_POR_ORGANISMO) -> pd.DataFrame:
    g = oc.groupby(["organismo", "anio"])
    base = pd.DataFrame({
        "organismo_nombre": g["organismo_nombre"].first(),
        "n_compras": g.size(),
        "monto_total": g["monto"].sum(),
        "n_proveedores": g["ProveedorRUT"].nunique(),
        "pct_trato_directo": g["ProcedenciaOC"].agg(lambda s: (s == "Trato Directo").mean()),
        "monto_medio": g["monto"].mean(),
    })
    base["monto_total_log"] = np.log1p(base["monto_total"])
    base["monto_medio_log"] = np.log1p(base["monto_medio"])
    perfil = (base.join(_hhi_cr1(oc)).join(_proveedores_nuevos(oc))
                  .join(_n_rubros(oc)).join(_oferente_unico(lic)))
    perfil["n_licitaciones"] = lic.groupby(["organismo", "anio"]).size()
    perfil["n_licitaciones"] = perfil["n_licitaciones"].fillna(0).astype(int)
    perfil = perfil[perfil["n_compras"] >= min_oc].reset_index()
    return perfil


def imputar(perfiles: pd.DataFrame, anio_ref: int = C.ANIO_ENTRENAMIENTO) -> pd.DataFrame:
    """Organismos sin licitaciones no tienen % de oferente único: se imputa la
    mediana del año de entrenamiento y se deja una marca."""
    p = perfiles.copy()
    p["oferente_unico_imputado"] = p["pct_oferente_unico"].isna()
    med = p.loc[p["anio"] == anio_ref, "pct_oferente_unico"].median()
    p["pct_oferente_unico"] = p["pct_oferente_unico"].fillna(med)
    p[C.VARIABLES] = p[C.VARIABLES].fillna(0.0)
    return p


def ejecutar() -> pd.DataFrame:
    oc = leer_oc()
    lic = pd.read_parquet(C.DIR_PROCESADOS / "licitaciones.parquet")
    perfiles = imputar(calcular_perfiles(oc, lic))
    perfiles.to_parquet(C.DIR_PROCESADOS / "perfiles.parquet", index=False)
    return perfiles
