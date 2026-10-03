"""F6 - Aplicación web ObservaCompras IA (Streamlit).

Tres vistas (Tabla 2): ranking de organismos atípicos con su explicación,
clasificador UNSPSC y buscador semántico. Ejecutar con:

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from observacompras import config as C  # noqa: E402
from observacompras.ingesta import leer_oc  # noqa: E402

st.set_page_config(page_title="ObservaCompras IA", page_icon="🔎", layout="wide")

VISTAS = ["Organismos atípicos", "Clasificador UNSPSC", "Buscador semántico"]
P = C.DIR_PROCESADOS


# --------------------------------------------------------------------------
# Carga de datos y modelos
# --------------------------------------------------------------------------
@st.cache_data
def cargar_datos():
    d = {
        "puntajes": pd.read_parquet(P / "puntajes_atipicidad.parquet"),
        "oc": leer_oc(P),
        "lic": pd.read_parquet(P / "licitaciones_app.parquet"),
        "resumen": json.load(open(P / "resumen_ingesta.json", encoding="utf-8")),
        "emb_meta": json.load(open(P / "embeddings_licitaciones.json")),
        "clases": json.load(open(C.DIR_MODELOS / "clases_unspsc.json", encoding="utf-8"))["clases"],
    }
    ruta_rev = P / "oc_revision.parquet"
    d["oc_rev"] = pd.read_parquet(ruta_rev) if ruta_rev.exists() else pd.DataFrame()
    return d


@st.cache_resource
def cargar_embeddings():
    return np.load(P / "embeddings_licitaciones.npy").astype("float32")


@st.cache_resource
def cargar_clasificador():
    import keras
    return keras.models.load_model(C.DIR_MODELOS / "clasificador.keras")


@st.cache_resource(show_spinner="Cargando el modelo de lenguaje (solo la primera vez)…")
def cargar_codificador():
    """Devuelve una función texto -> embedding, o None si no está disponible."""
    try:
        from observacompras import embeddings as EMB
        # Las consultas deben codificarse con el mismo modelo que generó los embeddings guardados
        os.environ["OBSERVACOMPRAS_ENCODER"] = datos["emb_meta"]["codificador"]
        EMB.codificar(["prueba"])
        return EMB.codificar
    except Exception:  # sin conexión o sin sentence-transformers
        return None


@st.cache_resource
def buscador():
    from observacompras.busqueda import BuscadorSemantico
    return BuscadorSemantico(cargar_embeddings(), datos["lic"])


@st.cache_resource
def buscador_palabras():
    from observacompras.busqueda import BuscadorPalabrasClave
    return BuscadorPalabrasClave(datos["lic"])


def nombre_familia(codigo) -> str:
    return NOMBRES_FAM.get(codigo) or "(sin nombre)"


def etiqueta_familia(codigo) -> str:
    return f"{codigo} · {nombre_familia(codigo)}" if codigo else "—"


def clp(x) -> str:
    return "—" if pd.isna(x) else f"${x:,.0f}".replace(",", ".")


def clp_corto(x) -> str:
    """Montos grandes en millones (MM$) para las tarjetas de resumen."""
    if pd.isna(x):
        return "—"
    return f"MM$ {x / 1e6:,.0f}".replace(",", ".") if x >= 1e7 else clp(x)


def etiqueta_anio(anio: int) -> str:
    if datos["resumen"].get("modo_periodo") == "demo":
        return {C.ANIO_ENTRENAMIENTO: "Periodo 1 (ene–mar 2023)",
                C.ANIO_PRUEBA: "Periodo 2 (abr–jun 2023)"}[anio]
    return str(anio)


def advertencias():
    r = datos["resumen"]
    st.warning(f"⚠️ {C.ADVERTENCIA}")
    st.caption(
        f"Cobertura: sector {C.SECTOR.title()}, órdenes de compra entre {r['periodo_fechas_oc'][0]} y "
        f"{r['periodo_fechas_oc'][1]} y licitaciones adjudicadas. Tipos de compra incluidos: "
        f"{', '.join(r['tipos_compra_incluidos'])}. Los proveedores que son personas naturales "
        "aparecen anonimizados (Ley 19.628 y Ley 21.719)."
    )
    if r.get("modo_periodo") == "demo":
        st.info("Modo demostración: se usa solo el primer semestre de 2023. El periodo 1 (ene–mar) "
                "cumple el rol de 2023 (entrenamiento) y el periodo 2 (abr–jun), el de 2024 (evaluación).")
    if datos["emb_meta"]["codificador"] != "sbert":
        st.info("Los embeddings se generaron con un codificador de prueba (TF-IDF + SVD), no con "
                "Sentence-BERT. Ejecute el cuaderno 03 en Colab para obtener los del modelo del documento.")


# --------------------------------------------------------------------------
# Vista 1: organismos atípicos
# --------------------------------------------------------------------------
def ir_a_buscador(texto: str, organismo: str):
    st.session_state["vista"] = VISTAS[2]
    st.session_state["consulta"] = texto
    st.session_state["organismo_origen"] = organismo


def vista_atipicos():
    st.header("Organismos atípicos")
    st.write("Ranking de organismos según cuánto se aparta su perfil de compra del patrón típico "
             "aprendido por el autoencoder. El puntaje es el error de reconstrucción.")
    advertencias()
    pt = datos["puntajes"]
    anios = sorted(pt["anio"].unique(), reverse=True)
    anio = st.selectbox("Periodo", anios, format_func=etiqueta_anio)
    rk = pt[pt["anio"] == anio].sort_values("ranking")

    tabla = pd.DataFrame({
        "Ranking": rk["ranking"], "Organismo": rk["organismo_nombre"],
        "Puntaje de atipicidad": rk["puntaje"].round(4), "Percentil": rk["percentil"].round(0).astype(int),
        "Variable que más aporta": rk["variable_principal"].map(C.NOMBRES_VARIABLES),
        "N.º de compras": rk["n_compras"],
    })
    st.dataframe(tabla, hide_index=True, width="stretch", height=280)

    opciones = rk["organismo"].tolist()
    org = st.selectbox("Seleccione un organismo para ver el motivo", opciones,
                       format_func=lambda o: f"#{int(rk.loc[rk.organismo == o, 'ranking'].iloc[0])} · "
                                             f"{rk.loc[rk.organismo == o, 'organismo_nombre'].iloc[0]}")
    f = rk[rk["organismo"] == org].iloc[0]

    st.subheader(f["organismo_nombre"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Posición en el ranking", f"{int(f['ranking'])} de {len(rk)}")
    c2.metric("Percentil del puntaje", f"{f['percentil']:.0f}")
    c3.metric("Compras (OC)", f"{int(f['n_compras']):,}".replace(",", "."))
    c4.metric("Monto total", clp_corto(f["monto_total"]))
    st.markdown(f"**{f['explicacion']}**")

    # Aporte de cada variable al error
    err = pd.Series({C.NOMBRES_VARIABLES[v]: f[f"err_{v}"] for v in C.VARIABLES}).sort_values()
    colores = ["#2a6fb0" if i >= len(err) - 2 else "#b8c4d6" for i in range(len(err))]
    fig = go.Figure(go.Bar(x=err.values / err.sum() * 100, y=err.index, orientation="h", marker_color=colores,
                           hovertemplate="%{y}: %{x:.1f}% del error<extra></extra>"))
    fig.update_layout(title="Aporte de cada variable al error de reconstrucción (%)", height=380,
                      margin=dict(l=10, r=10, t=40, b=10), xaxis_title="% del error total")
    st.plotly_chart(fig, width="stretch")

    # Valores del organismo frente a la mediana del sector
    med = rk[C.VARIABLES].median()
    comp = pd.DataFrame({
        "Variable": [C.NOMBRES_VARIABLES[v] for v in C.VARIABLES],
        "Organismo": [f[v] for v in C.VARIABLES],
        "Mediana del sector": [med[v] for v in C.VARIABLES],
        "Aporte al error (%)": [f[f"err_{v}"] / err.sum() * 100 for v in C.VARIABLES],
    })
    with st.expander("Valores del organismo frente a la mediana del sector"):
        st.dataframe(comp.style.format({"Organismo": "{:,.3f}", "Mediana del sector": "{:,.3f}",
                                        "Aporte al error (%)": "{:.1f}"}),
                     hide_index=True, width="stretch")

    # Paso 3: revisar el detalle de las compras
    st.subheader("Detalle de compras por proveedor")
    oc = datos["oc"]
    oc_org = oc[(oc["organismo"] == org) & (oc["anio"] == anio)]
    prov = (oc_org.groupby(["ProveedorRUT", "Proveedor"])["monto"].agg(["sum", "count"])
                  .sort_values("sum", ascending=False).reset_index())
    prov["part"] = prov["sum"] / prov["sum"].sum()
    etiquetas = [f"{r.Proveedor} ({r.ProveedorRUT}) · {r.part:.0%} del monto · {r['count']} OC"
                 for _, r in prov.head(15).iterrows()]
    i = st.selectbox("Proveedor (ordenado por monto; el primero es el principal)", range(len(etiquetas)),
                     format_func=lambda k: etiquetas[k])
    rut = prov.iloc[i]["ProveedorRUT"]
    det = oc_org[oc_org["ProveedorRUT"] == rut].sort_values("monto", ascending=False).copy()
    det["familia_registrada"] = det["onu_principal"].astype("string").str.zfill(8).str[: C.NIVEL_UNSPSC]
    rev = datos["oc_rev"]
    if not rev.empty:
        det = det.merge(rev[["codigoOC", "sugerencia_1", "prob_1", "coincide_sugerencia"]], on="codigoOC", how="left")
    else:
        det["sugerencia_1"], det["prob_1"], det["coincide_sugerencia"] = None, np.nan, None
    vista = pd.DataFrame({
        "OC": det["codigoOC"], "Fecha": det["fecha"].dt.date, "Descripción": det["texto"],
        "Procedencia": det["ProcedenciaOC"], "Monto": det["monto"].map(clp),
        "Categoría registrada": det["familia_registrada"].map(etiqueta_familia),
        "Sugerencia del clasificador": det["sugerencia_1"].map(lambda c: etiqueta_familia(c) if isinstance(c, str) else "—"),
        "¿Coincide?": det["coincide_sugerencia"].map({True: "Sí", False: "No — revisar codificación"}).fillna("—"),
    })
    st.dataframe(vista, hide_index=True, width="stretch", height=300)
    if rev.empty or det["sugerencia_1"].isna().all():
        st.caption("Las sugerencias del clasificador se precalculan para los 30 organismos más atípicos del "
                   "periodo de evaluación. Para otras compras, use la vista Clasificador UNSPSC.")
    else:
        n_no = int((det["coincide_sugerencia"] == False).sum())  # noqa: E712
        st.caption(f"En {n_no} de {det['sugerencia_1'].notna().sum()} compras la categoría registrada no coincide "
                   "con la sugerencia más probable; podría deberse a una codificación distinta o errónea.")

    # Paso 4: comparar
    if len(det):
        texto = st.selectbox("Comparar una compra con otras similares", det["texto"].dropna().unique()[:50],
                             format_func=lambda t: t[:120])
        st.button("Buscar compras similares en otros organismos →", on_click=ir_a_buscador, args=(texto, org))


# --------------------------------------------------------------------------
# Vista 2: clasificador UNSPSC
# --------------------------------------------------------------------------
def mostrar_sugerencias(probs: np.ndarray, registrada: str | None = None):
    top = np.argsort(probs)[::-1][: C.TOP_K_SUGERENCIAS]
    filas = []
    for r, j in enumerate(top, 1):
        cod = CLASES[j]
        filas.append({"Sugerencia": r, "Familia UNSPSC": cod, "Nombre": nombre_familia(cod),
                      "Probabilidad": f"{probs[j]:.1%}",
                      "Coincide con la registrada": "Sí" if registrada == cod else ""})
    st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")
    if registrada is not None:
        if registrada == CLASES[top[0]]:
            st.success("La categoría registrada coincide con la sugerencia más probable.")
        elif registrada in [CLASES[j] for j in top]:
            st.info("La categoría registrada está entre las tres sugeridas, pero no es la más probable.")
        else:
            st.warning("La categoría registrada no está entre las tres sugeridas: podría tratarse de una "
                       "codificación distinta o errónea. Conviene revisarla.")


def vista_clasificador():
    st.header("Clasificador UNSPSC")
    st.write("Sugiere las tres familias UNSPSC más probables para la descripción de una licitación. "
             "Usa una red de lenguaje preentrenada (Sentence-BERT) congelada y un clasificador propio "
             f"entrenado con licitaciones de {etiqueta_anio(C.ANIO_ENTRENAMIENTO)}.")
    advertencias()
    modo = st.radio("Origen del texto", ["Licitación existente", "Escribir una descripción"], horizontal=True)
    lic = datos["lic"]
    if modo == "Licitación existente":
        anio = st.selectbox("Periodo", sorted(lic["anio"].unique(), reverse=True), format_func=etiqueta_anio)
        sub = lic[lic["anio"] == anio]
        solo_dif = st.checkbox("Mostrar solo licitaciones cuya categoría registrada no coincide con la sugerencia")
        if solo_dif:
            sub = sub[~sub["coincide_sugerencia"]]
        st.caption(f"{(~lic.loc[lic.anio == anio, 'coincide_sugerencia']).mean():.0%} de las licitaciones del "
                   "periodo tienen una categoría registrada distinta de la sugerencia más probable.")
        nro = st.selectbox("Licitación", sub["NroLicitacion"].tolist(),
                           format_func=lambda n: f"{n} · {sub.loc[sub.NroLicitacion == n, 'NombreLicitacion'].iloc[0][:90]}")
        if nro:
            f = sub[sub["NroLicitacion"] == nro].iloc[0]
            st.markdown(f"**{f['NombreLicitacion']}** — {f['Institucion']}")
            st.write(f["Descripcion"])
            st.markdown(f"Categoría registrada: **{etiqueta_familia(f['familia'])}**")
            probs = cargar_clasificador().predict(cargar_embeddings()[lic.index[lic.NroLicitacion == nro]], verbose=0)[0]
            mostrar_sugerencias(probs, f["familia"])
    else:
        texto = st.text_area("Descripción de la licitación", height=120,
                             placeholder="Ej.: Suministro de insumos para hemodiálisis para el hospital…")
        if st.button("Clasificar", type="primary") and texto.strip():
            cod = cargar_codificador()
            if cod is None:
                st.error("El modelo de lenguaje no está disponible en este equipo (requiere conexión la primera "
                         "vez para descargarlo). Use la opción 'Licitación existente'.")
            else:
                mostrar_sugerencias(cargar_clasificador().predict(cod([texto]), verbose=0)[0])


# --------------------------------------------------------------------------
# Vista 3: buscador semántico
# --------------------------------------------------------------------------
def tabla_resultados(res: pd.DataFrame):
    st.dataframe(pd.DataFrame({
        "Similitud": res["similitud"].round(3), "Licitación": res["NroLicitacion"],
        "Organismo": res["Institucion"], "Nombre": res["NombreLicitacion"],
        "Categoría": res["familia"].map(etiqueta_familia), "Proveedores adjudicados": res["proveedores_ganadores"],
        "Monto estimado": res["monto_estimado"].map(clp), "Periodo": res["anio"].map(etiqueta_anio),
    }), hide_index=True, width="stretch")


def vista_buscador():
    st.header("Buscador semántico")
    st.write("Encuentra licitaciones similares por su significado, aunque usen otras palabras o estén "
             "clasificadas en otra categoría. Reutiliza los vectores de 384 dimensiones del clasificador "
             "y la similitud coseno.")
    advertencias()
    k = st.slider("Número de resultados", 5, 30, 10)
    otros = st.checkbox("Solo licitaciones de otros organismos", value=True)
    modo = st.radio("Buscar a partir de", ["Texto libre", "Una licitación existente"], horizontal=True)
    if modo == "Texto libre":
        q = st.text_input("Consulta", value=st.session_state.get("consulta", ""),
                          placeholder="Ej.: servicio de mantención de equipos de rayos X")
        origen = st.session_state.get("organismo_origen") if q == st.session_state.get("consulta") else None
        if origen and otros:
            st.caption("Se excluyen las licitaciones del organismo desde el que se inició la búsqueda.")
        if q.strip():
            cod = cargar_codificador()
            if cod is not None:
                tabla_resultados(buscador().por_vector(cod([q])[0], k, excluir_organismo=origen if otros else None))
            else:
                st.caption("Modelo de lenguaje no disponible: se usa búsqueda por palabras clave (respaldo).")
                tabla_resultados(buscador_palabras().por_texto(q, k))
    else:
        lic = datos["lic"]
        nro = st.selectbox("Licitación", lic["NroLicitacion"].tolist(),
                           format_func=lambda n: f"{n} · {lic.loc[lic.NroLicitacion == n, 'NombreLicitacion'].iloc[0][:90]}")
        f = lic[lic["NroLicitacion"] == nro].iloc[0]
        st.markdown(f"**{f['NombreLicitacion']}** — {f['Institucion']} · {etiqueta_familia(f['familia'])}")
        tabla_resultados(buscador().por_licitacion(nro, k, otros_organismos=otros))


# --------------------------------------------------------------------------
# Principal
# --------------------------------------------------------------------------
try:
    datos = cargar_datos()
except FileNotFoundError as e:
    st.error(f"Faltan datos procesados o modelos ({e.filename}). Ejecute primero el pipeline: "
             "`python scripts/ejecutar_pipeline.py todo`.")
    st.stop()

CLASES = [c["codigo"] for c in datos["clases"]]
NOMBRES_FAM = {c["codigo"]: c["nombre"] for c in datos["clases"]}
NOMBRES_FAM.update(datos["lic"].dropna(subset=["familia"]).groupby("familia")["familia_nombre"].first().to_dict())

with st.sidebar:
    st.title("ObservaCompras IA")
    st.caption("MVP · Aplicaciones de IA (MTI-493) · UTFSM")
    st.radio("Vista", VISTAS, key="vista")
    st.divider()
    st.caption("Apoyo a la revisión humana: el sistema señala dónde mirar, sin calificar conductas. "
               "El juicio final corresponde al analista.")

{VISTAS[0]: vista_atipicos, VISTAS[1]: vista_clasificador, VISTAS[2]: vista_buscador}[st.session_state["vista"]]()
