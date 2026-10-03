"""Reporte de evaluación: las seis métricas de la Tabla 4 frente a sus líneas base.

Genera reports/reporte_evaluacion.md y reports/revision_explicabilidad_H3.csv.
H3 y H4 requieren juicio humano: el reporte deja las plantillas listas y lee los
resultados si los autores ya completaron las planillas.
"""
from __future__ import annotations

import json

import pandas as pd

from . import config as C


def _pct(x):
    return "—" if x is None else f"{x:.0%}"


def _miles(n) -> str:
    return f"{int(n):,}".replace(",", ".")


def _por_anio(d: dict) -> str:
    return "; ".join(f"{k}: {_miles(v)}" for k, v in d.items())


def _si_no(b):
    return "✅ Sí" if b else "❌ No"


def plantilla_h3() -> pd.DataFrame:
    punt = pd.read_parquet(C.DIR_PROCESADOS / "puntajes_atipicidad.parquet")
    top = punt[punt["anio"] == C.ANIO_PRUEBA].nsmallest(10, "ranking")
    out = top[["ranking", "organismo", "organismo_nombre", "puntaje", "percentil", "explicacion"]].copy()
    out["descriptible_en_una_frase (si/no)"] = ""
    out["frase_del_revisor"] = ""
    ruta = C.DIR_REPORTES / "revision_explicabilidad_H3.csv"
    if ruta.exists():  # no sobrescribir la revisión ya hecha por los autores
        previa = pd.read_csv(ruta, dtype=str)
        if set(previa["organismo"]) == set(out["organismo"].astype(str)):
            return previa
    out.to_csv(ruta, index=False)
    return out


def _resultado_h3(df: pd.DataFrame):
    col = df["descriptible_en_una_frase (si/no)"].fillna("").str.strip().str.lower()
    if (col == "").all():
        return None
    return int(col.isin(["si", "sí", "s"]).sum())


def _resultado_h4():
    ruta = C.DIR_REPORTES / "pruebas_usuario_H4.csv"
    if not ruta.exists():
        return None
    df = pd.read_csv(ruta)
    app = df[df["con_app"].astype(str).str.lower().isin(["si", "sí"])]
    if app.empty:
        return None
    sus = app.groupby("participante")["sus"].first()
    return {"mediana_min": float(app["tiempo_min"].median()), "sus_medio": float(sus.mean()),
            "n": int(sus.size)}


def generar() -> str:
    ae = json.load(open(C.DIR_REPORTES / "evaluacion_ae.json", encoding="utf-8"))
    clf = json.load(open(C.DIR_REPORTES / "evaluacion_clasificador.json", encoding="utf-8"))
    ing = json.load(open(C.DIR_PROCESADOS / "resumen_ingesta.json", encoding="utf-8"))
    h3 = plantilla_h3()
    n_h3 = _resultado_h3(h3)
    h4 = _resultado_h4()
    h1, est = ae["H1_recall_5pct"], ae["H1_estabilidad_top10"]
    tl, lb = clf["transfer_learning"], clf["linea_base"]

    texto_h4 = ("pendiente" if h4 is None else
                f"Mediana {h4['mediana_min']:.1f} min; SUS {h4['sus_medio']:.0f} (n = {h4['n']})")
    demo = ing.get("modo_periodo") == "demo"
    lineas = ["# Reporte de evaluación — ObservaCompras IA", ""]
    if demo:
        lineas += ["> **Modo demostración.** Resultados con el primer semestre de 2023: el periodo ene–mar "
                   "cumple el rol de 2023 (entrenamiento) y abr–jun el de 2024 (prueba). Los resultados "
                   "definitivos se obtienen al ejecutar el pipeline con 2023 y 2024 completos.", ""]
    if clf.get("advertencia_codificador"):
        lineas += [f"> **Atención:** {clf['advertencia_codificador']}", ""]

    lineas += [
        "## Datos", "",
        f"- Órdenes de compra procesadas: {_miles(ing['oc_final'])} (por periodo: {_por_anio(ing['oc_por_anio'])})",
        f"- Licitaciones adjudicadas: {_miles(ing['lic_final'])} (por periodo: {_por_anio(ing['lic_por_anio'])})",
        f"- Organismos: {ing['organismos']}; organismos-año con perfil: "
        f"{ae['n_organismos_entrenamiento']} (entrenamiento) y {ae['n_organismos_evaluacion']} (evaluación)",
        f"- Órdenes de compra a proveedores persona natural (anonimizados): {_miles(ing['oc_proveedores_persona_natural'])}",
        "",
        "## Tabla 4: métricas, líneas base y metas", "",
        "| Métrica | Resultado | Línea base | Meta | ¿Cumple? | Hip. |",
        "|---|---|---|---|---|---|",
        f"| Recall@5% de anomalías sintéticas | {_pct(h1['autoencoder'])} | {_pct(h1['linea_base_z'])} (regla z) "
        f"| ≥ 80% | {_si_no(h1['cumple_meta'])} | H1 |",
        f"| Estabilidad del ranking (top 10, media entre pares de folds) | {est['coincidencia_media_pares']:.1f} de 10 "
        f"| — | ≥ 7 de 10 | {_si_no(est['cumple_meta'])} | H1 |",
        f"| F1-macro del clasificador | {tl['f1_macro']:.3f} | {lb['f1_macro']:.3f} (TF-IDF + reg. logística) "
        f"| Línea base + 5 puntos | {_si_no(clf['H2_cumple_meta_5_puntos'])} ({clf['H2_diferencia_f1_macro_puntos']:+.1f} pts) | H2 |",
        f"| Exactitud top-3 | {_pct(tl['top3'])} | {_pct(lb['top3'])} | ≥ 90% | {_si_no(clf['top3_cumple_meta_90'])} | H2 |",
        f"| Explicabilidad (top 10) | {'pendiente' if n_h3 is None else f'{n_h3} de 10'} | — | ≥ 7 de 10 | "
        f"{'pendiente' if n_h3 is None else _si_no(n_h3 >= 7)} | H3 |",
        f"| Tiempo de tarea y SUS | {texto_h4} "
        f"| Revisión manual en planilla | Mediana < 5 min y SUS ≥ 70 | "
        f"{'pendiente' if h4 is None else _si_no(h4['mediana_min'] < 5 and h4['sus_medio'] >= 70)} | H4 |",
        "",
        "## H1 — Detector de patrones atípicos (autoencoder)", "",
        f"- Hiperparámetros elegidos por K-fold (K = 5): learning rate {ae['hiperparametros_elegidos']['learning_rate']:g}, "
        f"batch size {ae['hiperparametros_elegidos']['batch_size']}, {ae['hiperparametros_elegidos']['epochs_final']} epochs "
        f"en el modelo final (val_loss media {ae['hiperparametros_elegidos']['val_loss_media']:.5f}).",
        f"- Estabilidad: coincidencia media del top 10 entre pares de folds {est['coincidencia_media_pares']:.1f}, "
        f"mínima {est['coincidencia_minima_pares']}; organismos presentes en el top 10 de los 5 folds: "
        f"{est['interseccion_todos_los_folds']}.",
        f"- Coincidencia entre el top 10 del autoencoder y el de la regla z: {ae['coincidencia_top10_autoencoder_vs_z']} de 10.",
        "", "Recall@5% por tipo de anomalía sintética:", "",
        "| Tipo | Autoencoder | Regla z |", "|---|---|---|",
    ]
    for t, v in h1["por_tipo_autoencoder"].items():
        lineas.append(f"| {t} | {_pct(v)} | {_pct(h1['por_tipo_linea_base_z'][t])} |")
    lineas += [
        "", "![Curva de pérdida del autoencoder](figuras/ae_curva_perdida.png)", "",
        "![Búsqueda de hiperparámetros](figuras/ae_busqueda.png)", "",
        "![Recall por tipo](figuras/ae_recall_por_tipo.png)", "",
        "## H2 — Clasificador UNSPSC (transfer learning)", "",
        f"- {clf['n_clases']} familias UNSPSC ({clf['nivel_unspsc_digitos']} dígitos) con ≥ {C.MIN_EJEMPLOS_CLASE} ejemplos.",
        f"- Entrenamiento {clf['n_entrenamiento']}, validación {clf['n_validacion']}, prueba {clf['n_prueba']} "
        f"(se excluyeron {clf['prueba_excluidas_por_clase_no_vista']} licitaciones de prueba de familias no vistas).",
        f"- Hiperparámetros elegidos: learning rate {clf['hiperparametros_elegidos']['learning_rate']:g}, "
        f"batch size {clf['hiperparametros_elegidos']['batch_size']}, mejor epoch {clf['hiperparametros_elegidos']['mejor_epoch']}.",
        f"- Exactitud top-1: {_pct(tl['exactitud'])} (transfer learning) vs. {_pct(lb['exactitud'])} (línea base).",
        "", "![Curva de pérdida del clasificador](figuras/clf_curva_perdida.png)", "",
        "![Comparación](figuras/clf_comparacion.png)", "",
        "## H3 — Explicabilidad", "",
        "Los autores revisan los 10 organismos más atípicos en `reports/revision_explicabilidad_H3.csv` "
        "(columna `descriptible_en_una_frase`).", "",
        "| # | Organismo | Explicación generada |", "|---|---|---|",
    ]
    for _, r in h3.iterrows():
        lineas.append(f"| {r['ranking']} | {r['organismo_nombre']} | {r['explicacion']} |")
    lineas += [
        "", "## H4 — Utilidad", "",
        "Protocolo en `docs/guia_pruebas_usuario.md`; resultados en `reports/pruebas_usuario_H4.csv`.", "",
        "## Revisión manual de etiquetas (Tabla 5)", "",
        "Muestra de 100 licitaciones de prueba en `reports/muestra_revision_manual.csv`.", "",
        f"_{C.ADVERTENCIA}_", "",
    ]
    texto = "\n".join(lineas)
    (C.DIR_REPORTES / "reporte_evaluacion.md").write_text(texto, encoding="utf-8")
    return texto
