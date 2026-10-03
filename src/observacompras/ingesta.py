"""F1 - Ingesta y limpieza (Tabla 2).

Lee los CSV de datos abiertos de ChileCompra (formato de descarga por sector:
separador ';', codificación latin-1), ya sea sueltos o dentro de archivos
.7z/.zip; filtra el sector Salud y los años de estudio; normaliza organismos y
proveedores por RUT; valida montos y fechas; anonimiza a los proveedores que
son personas naturales; y guarda el resultado en Parquet.

Salidas (data/processed):
    ordenes_compra_<anio>.parquet   una fila por orden de compra (un archivo por año)
    licitaciones.parquet     una fila por licitación adjudicada
    resumen_ingesta.json     conteos de cada paso de limpieza
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

# Columnas utilizadas de cada fuente
COLS_OC = [
    "codigoOC", "FechaEnvioOC", "NombreOC", "EstadoOC", "ProcedenciaOC",
    "MontoNetoOC_CLP", "UnidadCompra", "UnidadCompraRUT", "entCode",
    "Institucion", "Sector", "Proveedor", "ProveedorRUT", "TamanoProveedor",
    "RubroN1", "CodigoProductoONU", "DescripcionItem", "MontoNetoItemCLP",
]
COLS_LIC = [
    "NroLicitacion", "NombreLicitacion", "TipoLicitacion", "Descripcion",
    "FechaCierre", "FechaPublicacion", "EstadoLicitacion", "MontoEstimadoLicitacion",
    "MonedaLicitacion", "UnidadCompra", "UnidadCompraRUT", "entCode", "Institucion",
    "Sector", "RubroN1", "RubroN2", "CodigoProductoONU", "NombreItem",
    "ProveedorRUT", "Proveedor", "ResultadoOferta",
]


# --------------------------------------------------------------------------
# Utilidades de RUT y privacidad
# --------------------------------------------------------------------------
def normalizar_rut(rut: pd.Series) -> pd.Series:
    """'61.608.204-3' -> '61608204-3' (sin puntos, DV en mayúscula)."""
    r = rut.astype("string").str.strip().str.replace(".", "", regex=False).str.upper()
    return r.where(r.str.contains("-", na=False), None)


def numero_rut(rut_norm: pd.Series) -> pd.Series:
    return pd.to_numeric(rut_norm.str.split("-").str[0], errors="coerce")


def digito_verificador(numero: int) -> str:
    s, m = 0, 2
    for d in reversed(str(int(numero))):
        s += int(d) * m
        m = 2 if m == 7 else m + 1
    r = 11 - s % 11
    return {11: "0", 10: "K"}.get(r, str(r))


def es_persona_natural(rut_norm: pd.Series) -> pd.Series:
    """Decisión (f): RUT < 50.000.000 corresponde a una persona natural."""
    return numero_rut(rut_norm) < C.UMBRAL_RUT_PERSONA_JURIDICA


def _seudonimo(rut: str) -> str:
    h = hashlib.sha256(f"{C.SAL_ANONIMIZACION}|{rut}".encode()).hexdigest()[:10]
    return f"PN-{h}"


def anonimizar_proveedores(df: pd.DataFrame, col_rut="ProveedorRUT", col_nombre="Proveedor",
                           cols_texto=()) -> pd.DataFrame:
    """Reemplaza RUT y nombre de personas naturales por un seudónimo estable.

    El seudónimo es determinista, por lo que las variables de concentración
    (HHI, CR1, proveedores nuevos) se calculan igual que con el RUT real.
    """
    pn = es_persona_natural(df[col_rut]).fillna(False)
    seud = df.loc[pn, col_rut].map(_seudonimo)
    df["proveedor_persona_natural"] = pn
    df.loc[pn, col_rut] = seud
    df.loc[pn, col_nombre] = "Persona natural (" + seud + ")"
    for c in cols_texto:  # textos libres que pueden contener el nombre de la persona
        df.loc[pn, c] = "[texto omitido: proveedor persona natural]"
    return df


# --------------------------------------------------------------------------
# Lectura de archivos
# --------------------------------------------------------------------------
def _expandir_comprimidos(origen: Path, destino: Path) -> list[Path]:
    """Devuelve la lista de CSV, descomprimiendo .7z/.zip si es necesario."""
    csvs = []
    for p in sorted(origen.rglob("*")):
        if p.suffix.lower() == ".csv":
            csvs.append(p)
        elif p.suffix.lower() == ".zip":
            with zipfile.ZipFile(p) as z:
                z.extractall(destino / p.stem)
            csvs += sorted((destino / p.stem).rglob("*.csv"))
        elif p.suffix.lower() == ".7z":
            import py7zr
            with py7zr.SevenZipFile(p) as z:
                z.extractall(destino / p.stem)
            csvs += sorted((destino / p.stem).rglob("*.csv"))
    return csvs


def _tipo_archivo(csv: Path) -> str | None:
    with open(csv, encoding="latin-1") as f:
        cab = f.readline().strip().split(";")
    if cab[0] == "codigoOC":
        return "oc"
    if cab[0] == "NroLicitacion" and "NombreLicitacion" in cab:
        return "lic"
    return None


def _leer(csv: Path, columnas: list[str]) -> pd.DataFrame:
    return pd.read_csv(csv, sep=";", encoding="latin-1", usecols=columnas, dtype=str,
                       on_bad_lines="skip", engine="c")


def _fecha(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, format="%d-%m-%Y %H:%M:%S", errors="coerce")


def _monto(s: pd.Series) -> pd.Series:
    # Los montos vienen como enteros o con coma decimal
    return pd.to_numeric(s.str.replace(",", ".", regex=False), errors="coerce")


# --------------------------------------------------------------------------
# Periodos: año calendario (modo normal) o trimestres (modo demostración)
# --------------------------------------------------------------------------
def asignar_periodo(fecha: pd.Series, modo: str) -> tuple[pd.Series, pd.Series]:
    """Devuelve (anio, segunda_mitad).

    modo="anio": el periodo es el año calendario (modo del documento).
    modo="demo": para probar con un solo semestre, el 1er trimestre hace de
        "2023" (entrenamiento) y el 2º trimestre de "2024" (prueba).
    `segunda_mitad` indica si la fecha cae en la segunda mitad del periodo; se
    usa para la variable % de proveedores nuevos (decisión a).
    """
    if modo == "anio":
        anio = fecha.dt.year
        segunda = fecha.dt.month >= 7
    elif modo == "demo":
        trimestre = (fecha.dt.month - 1) // 3 + 1
        anio = trimestre.map({1: C.ANIO_ENTRENAMIENTO, 2: C.ANIO_PRUEBA})
        segunda = ((fecha.dt.month - 1) % 3 + fecha.dt.day / 31) >= 1.5
    else:
        raise ValueError(modo)
    return anio.astype("Int64"), segunda


# --------------------------------------------------------------------------
# Procesamiento
# --------------------------------------------------------------------------
def _sumar(resumen: dict, clave: str, valor: int) -> None:
    resumen[clave] = resumen.get(clave, 0) + int(valor)


def colapsar_oc(df: pd.DataFrame, resumen: dict) -> pd.DataFrame:
    """Líneas de ítem de un archivo de OC -> una fila por OC del sector."""
    _sumar(resumen, "oc_lineas_leidas", len(df))
    df = df[df["Sector"].str.upper() == C.SECTOR]
    _sumar(resumen, "oc_lineas_sector", len(df))

    # Rubros por OC (a nivel de ítem) antes de colapsar a una fila por OC
    rubros = (df.dropna(subset=["RubroN1"]).groupby("codigoOC")["RubroN1"]
                .agg(lambda s: "|".join(sorted(set(s)))))
    onu = (df.dropna(subset=["CodigoProductoONU"]).groupby("codigoOC")["CodigoProductoONU"]
             .agg(lambda s: s.value_counts().index[0]))

    # Una fila por OC: la Compra Ágil repite la OC por cada cotización/ítem
    # Descripción del ítem de mayor monto: el nombre de la OC suele ser solo un código interno
    it = df.dropna(subset=["DescripcionItem"]).assign(_m=_monto(df["MontoNetoItemCLP"]))
    desc = it.sort_values("_m", ascending=False).drop_duplicates("codigoOC").set_index("codigoOC")["DescripcionItem"]
    oc = df.drop_duplicates("codigoOC").drop(columns=["RubroN1", "CodigoProductoONU", "DescripcionItem",
                                                       "MontoNetoItemCLP"]).copy()
    oc["descripcion_item"] = oc["codigoOC"].map(desc)
    oc["texto"] = (oc["NombreOC"].fillna("").str.strip() + ". "
                   + oc["descripcion_item"].fillna("").str.strip()).str.strip(". ")
    oc["rubros"] = oc["codigoOC"].map(rubros)
    oc["onu_principal"] = oc["codigoOC"].map(onu)
    return oc


def procesar_oc(frames: list[pd.DataFrame], modo: str, resumen: dict) -> pd.DataFrame:
    oc = pd.concat(frames, ignore_index=True).drop_duplicates("codigoOC")
    resumen["oc_unicas"] = len(oc)

    oc = oc[~oc["EstadoOC"].isin(C.ESTADOS_OC_EXCLUIDOS)]
    resumen["oc_tras_excluir_estados"] = len(oc)

    oc["fecha"] = _fecha(oc["FechaEnvioOC"])
    oc["monto"] = _monto(oc["MontoNetoOC_CLP"])
    validos = oc["fecha"].notna() & oc["monto"].notna() & (oc["monto"] > 0)
    resumen["oc_descartadas_monto_o_fecha_invalidos"] = int((~validos).sum())
    oc = oc[validos].copy()

    oc["anio"], oc["segunda_mitad"] = asignar_periodo(oc["fecha"], modo)
    oc = oc[oc["anio"].isin(C.ANIOS)]
    resumen["oc_en_periodo"] = len(oc)

    for c in ("UnidadCompraRUT", "ProveedorRUT"):
        oc[c] = normalizar_rut(oc[c])
    oc = oc.dropna(subset=["ProveedorRUT", "entCode"])
    oc["organismo"] = oc["entCode"].astype(str)
    # Nombre canónico del organismo: el más frecuente para su código
    oc["organismo_nombre"] = oc.groupby("organismo")["Institucion"].transform(
        lambda s: s.value_counts().index[0])
    oc = anonimizar_proveedores(oc, cols_texto=("NombreOC", "descripcion_item", "texto"))
    resumen["oc_final"] = len(oc)
    resumen["oc_proveedores_persona_natural"] = int(oc["proveedor_persona_natural"].sum())

    cols = ["codigoOC", "fecha", "anio", "segunda_mitad", "organismo", "organismo_nombre",
            "UnidadCompra", "UnidadCompraRUT", "ProveedorRUT", "Proveedor", "TamanoProveedor",
            "proveedor_persona_natural", "ProcedenciaOC", "EstadoOC", "monto", "NombreOC", "texto",
            "rubros", "onu_principal"]
    return oc[cols].sort_values(["organismo", "fecha"]).reset_index(drop=True)


def _moda(s: pd.Series):
    vc = s.dropna().value_counts()
    if vc.empty:
        return None
    # Desempate determinista: mayor frecuencia y luego menor código
    top = vc[vc == vc.iloc[0]].index
    return sorted(top)[0]


def procesar_licitaciones(frames: list[pd.DataFrame], modo: str, resumen: dict) -> pd.DataFrame:
    df = pd.concat(frames, ignore_index=True)
    resumen["lic_lineas_leidas"] = len(df)
    df = df[(df["Sector"].str.upper() == C.SECTOR) & (df["EstadoLicitacion"] == C.ESTADO_LICITACION_VALIDO)].copy()
    resumen["lic_lineas_sector_adjudicadas"] = len(df)

    df["ProveedorRUT"] = normalizar_rut(df["ProveedorRUT"])
    df = anonimizar_proveedores(df)
    df["familia"] = df["CodigoProductoONU"].astype("string").str.zfill(8).str[: C.NIVEL_UNSPSC]

    g = df.groupby("NroLicitacion")
    # N.º de oferentes distintos (para "% de compras con un solo oferente")
    n_oferentes = g["ProveedorRUT"].nunique()
    # Categoría UNSPSC de la licitación: la familia más frecuente entre sus ítems distintos
    items = df.drop_duplicates(["NroLicitacion", "CodigoProductoONU", "NombreItem"])
    familia = items.groupby("NroLicitacion")["familia"].agg(_moda)
    n_familias = items.groupby("NroLicitacion")["familia"].nunique()
    ganadores = (df[df["ResultadoOferta"] == "Ganadora"].drop_duplicates(["NroLicitacion", "ProveedorRUT"])
                   .groupby("NroLicitacion")["Proveedor"].agg(lambda s: " | ".join(s.astype(str))))
    # Nombre de cada familia: rubro N2 más frecuente (los rubros de Mercado Público siguen la jerarquía UNSPSC)
    nombres_fam = df.dropna(subset=["familia"]).groupby("familia")["RubroN2"].agg(_moda)

    lic = df.drop_duplicates("NroLicitacion").set_index("NroLicitacion")
    lic = lic[["NombreLicitacion", "Descripcion", "TipoLicitacion", "FechaCierre", "FechaPublicacion",
               "MontoEstimadoLicitacion", "MonedaLicitacion", "UnidadCompra", "entCode", "Institucion"]].copy()
    lic["n_oferentes"] = n_oferentes
    lic["familia"] = familia
    lic["familia_nombre"] = lic["familia"].map(nombres_fam)
    lic["n_familias"] = n_familias
    lic["proveedores_ganadores"] = ganadores
    lic = lic.reset_index()

    # Año = año de cierre (los archivos mensuales del portal se agrupan por fecha de cierre)
    fecha = _fecha(lic["FechaCierre"]).fillna(_fecha(lic["FechaPublicacion"]))
    lic["fecha_cierre"] = fecha
    lic["anio"], _ = asignar_periodo(fecha, modo)
    lic = lic[lic["anio"].isin(C.ANIOS)]
    lic["monto_estimado"] = _monto(lic["MontoEstimadoLicitacion"])
    lic["organismo"] = lic["entCode"].astype(str)
    lic["texto"] = (lic["NombreLicitacion"].fillna("").str.strip() + ". "
                    + lic["Descripcion"].fillna("").str.strip()).str.strip(". ")
    lic = lic[lic["texto"].str.len() > 0]
    resumen["lic_final"] = len(lic)
    cols = ["NroLicitacion", "anio", "fecha_cierre", "organismo", "Institucion", "UnidadCompra",
            "NombreLicitacion", "Descripcion", "texto", "TipoLicitacion", "monto_estimado",
            "n_oferentes", "familia", "familia_nombre", "n_familias", "proveedores_ganadores"]
    return lic[cols].sort_values("NroLicitacion").reset_index(drop=True)


def leer_oc(directorio: Path = C.DIR_PROCESADOS) -> pd.DataFrame:
    """Lee las órdenes de compra procesadas (un Parquet por año)."""
    partes = sorted(Path(directorio).glob("ordenes_compra_*.parquet"))
    if not partes:
        raise FileNotFoundError(Path(directorio) / "ordenes_compra_<anio>.parquet")
    return pd.concat([pd.read_parquet(p) for p in partes], ignore_index=True)


def ejecutar(origen: Path = C.DIR_RAW, destino: Path = C.DIR_PROCESADOS,
             modo: str | None = None) -> dict:
    """Corre F1 completo y escribe los Parquet procesados."""
    modo = modo or os.environ.get("OBSERVACOMPRAS_MODO", "anio")
    destino.mkdir(parents=True, exist_ok=True)
    resumen: dict = {"modo_periodo": modo}
    tmp = Path(tempfile.mkdtemp(prefix="observacompras_"))
    try:
        csvs = _expandir_comprimidos(Path(origen), tmp)
        f_oc, f_lic = [], []
        for csv in csvs:
            tipo = _tipo_archivo(csv)
            print(f"  leyendo {csv.name} ({tipo})")
            if tipo == "oc":
                f_oc.append(colapsar_oc(_leer(csv, COLS_OC), resumen))
            elif tipo == "lic":
                f_lic.append(_leer(csv, COLS_LIC))
        resumen["archivos_oc"] = len(f_oc)
        resumen["archivos_licitaciones"] = len(f_lic)
        if not f_oc or not f_lic:
            raise FileNotFoundError(f"No se encontraron CSV de OC y de licitaciones en {origen}")
        oc = procesar_oc(f_oc, modo, resumen)
        del f_oc
        lic = procesar_licitaciones(f_lic, modo, resumen)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # Un archivo por año para respetar el límite de 100 MB por archivo de GitHub
    for viejo in destino.glob("ordenes_compra*.parquet"):
        viejo.unlink()
    for anio, parte in oc.groupby("anio"):
        parte.to_parquet(destino / f"ordenes_compra_{int(anio)}.parquet", index=False, compression="zstd")
    lic.to_parquet(destino / "licitaciones.parquet", index=False, compression="zstd")
    resumen["oc_por_anio"] = {int(k): int(v) for k, v in oc["anio"].value_counts().sort_index().items()}
    resumen["lic_por_anio"] = {int(k): int(v) for k, v in lic["anio"].value_counts().sort_index().items()}
    resumen["organismos"] = int(oc["organismo"].nunique())
    resumen["periodo_fechas_oc"] = [str(oc["fecha"].min().date()), str(oc["fecha"].max().date())]
    resumen["tipos_compra_incluidos"] = sorted(oc["ProcedenciaOC"].dropna().unique().tolist())
    with open(destino / "resumen_ingesta.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)
    return resumen
