"""Entrenamiento y evaluación del clasificador UNSPSC (F4, H2) y preparación de F5.

Salidas:
    data/processed/embeddings_licitaciones.npy (+ .json)   vectores de 384 dim.
    data/processed/licitaciones_app.parquet               licitaciones + sugerencias top-3
    models/clasificador.keras, models/clases_unspsc.json
    reports/clf_busqueda_hiperparametros.csv, reports/clf_curva_perdida.csv
    reports/evaluacion_clasificador.json, reports/muestra_revision_manual.csv
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import clasificador as K
from . import config as C
from . import embeddings as EMB
from . import graficos as G


def _onehot(y, n):
    m = np.zeros((len(y), n), dtype="float32")
    m[np.arange(len(y)), y] = 1
    return m


def calcular_embeddings(lic: pd.DataFrame) -> np.ndarray:
    ruta = C.DIR_PROCESADOS / "embeddings_licitaciones.npy"
    meta = C.DIR_PROCESADOS / "embeddings_licitaciones.json"
    if ruta.exists() and meta.exists():
        m = json.load(open(meta))
        if m["n"] == len(lic) and m["codificador"] == EMB.nombre_codificador():
            print("  embeddings ya calculados; se reutilizan")
            return np.load(ruta).astype("float32")
    EMB.preparar_codificador(lic.loc[lic["anio"] == C.ANIO_ENTRENAMIENTO, "texto"])
    print(f"  codificando {len(lic)} licitaciones con {EMB.nombre_codificador()}")
    E = EMB.codificar(lic["texto"].tolist(), mostrar_progreso=True)
    np.save(ruta, E.astype("float16"))
    EMB.guardar_metadatos(meta, len(lic))
    return E


def ejecutar(verbose: int = 0, rapido: bool = False) -> dict:
    C.DIR_MODELOS.mkdir(exist_ok=True)
    C.DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    lic = pd.read_parquet(C.DIR_PROCESADOS / "licitaciones.parquet")
    E = calcular_embeddings(lic)

    clases = K.preparar_etiquetas(lic)
    idx_clase = {c: i for i, c in enumerate(clases)}
    nombres = lic.dropna(subset=["familia"]).groupby("familia")["familia_nombre"].first()
    with open(C.DIR_MODELOS / "clases_unspsc.json", "w", encoding="utf-8") as f:
        json.dump({"nivel_digitos": C.NIVEL_UNSPSC,
                   "clases": [{"codigo": c, "nombre": nombres.get(c)} for c in clases]},
                  f, ensure_ascii=False, indent=2)

    es_tr = (lic["anio"] == C.ANIO_ENTRENAMIENTO).values
    es_te = (lic["anio"] == C.ANIO_PRUEBA).values
    con_clase = lic["familia"].isin(clases).values
    i_tr = np.where(es_tr & con_clase)[0]
    i_te = np.where(es_te & con_clase)[0]
    # Barajar: Keras usa el 15% final como validación
    rng = np.random.default_rng(C.SEMILLA)
    i_tr = rng.permutation(i_tr)
    y_tr = lic["familia"].values[i_tr]
    y_tr = np.array([idx_clase[c] for c in y_tr])
    y_te = np.array([idx_clase[c] for c in lic["familia"].values[i_te]])
    n = len(clases)
    print(f"Clasificador: {n} clases, {len(i_tr)} licitaciones de entrenamiento, {len(i_te)} de prueba")

    # 1) Transfer learning: búsqueda de learning rate y batch size
    lrs = (1e-3,) if rapido else C.CLF_LEARNING_RATES
    bss = (128,) if rapido else C.CLF_BATCH_SIZES
    tabla, (mejor, clf, hist) = K.busqueda_hiperparametros(E[i_tr], _onehot(y_tr, n), lrs, bss, verbose)
    tabla.to_csv(C.DIR_REPORTES / "clf_busqueda_hiperparametros.csv", index=False)
    curva = pd.DataFrame({"epoch": np.arange(1, len(hist["loss"]) + 1), "loss": hist["loss"],
                          "val_loss": hist["val_loss"], "accuracy": hist["accuracy"],
                          "val_accuracy": hist["val_accuracy"]})
    curva.to_csv(C.DIR_REPORTES / "clf_curva_perdida.csv", index=False)
    G.curva_perdida(curva, f"Clasificador: pérdida (lr={mejor['learning_rate']:g}, batch={mejor['batch_size']})",
                    C.DIR_FIGURAS / "clf_curva_perdida.png", mejor["mejor_epoch"])
    G.heatmap_busqueda(tabla, "mejor_val_loss", "Clasificador: val_loss", C.DIR_FIGURAS / "clf_busqueda.png")
    clf.save(C.DIR_MODELOS / "clasificador.keras")
    with open(C.DIR_MODELOS / "config_clasificador.json", "w") as f:
        json.dump({**mejor, "capas": ["Dense 128 relu", "Dropout 0.3", f"Dense {n} softmax"],
                   "optimizador": "adam", "perdida": "categorical_crossentropy",
                   "codificador": EMB.nombre_codificador(), "modelo_base": C.MODELO_SBERT}, f, indent=2)

    # 2) Línea base TF-IDF + regresión logística, con la misma partición 85/15
    corte = int(round(len(i_tr) * (1 - C.CLF_VALIDACION)))
    t_tr, t_va = lic["texto"].values[i_tr[:corte]], lic["texto"].values[i_tr[corte:]]
    mejor_C = K.elegir_linea_base(t_tr, y_tr[:corte], t_va, y_tr[corte:], n)
    base = K.LineaBaseTfidf(mejor_C).fit(t_tr, y_tr[:corte])

    # 3) Prueba final con el año de evaluación (se usa una sola vez)
    p_nn = clf.predict(E[i_te], verbose=0)
    p_bl = base.predict_proba(lic["texto"].values[i_te], n)
    m_nn, m_bl = K.metricas(y_te, p_nn, n), K.metricas(y_te, p_bl, n)
    dif = (m_nn["f1_macro"] - m_bl["f1_macro"]) * 100
    resultado = {
        "codificador": EMB.nombre_codificador(),
        "advertencia_codificador": None if EMB.nombre_codificador() == "sbert" else
            "Resultados con codificador de PRUEBA (TF-IDF+SVD), no con Sentence-BERT. No válidos para H2.",
        "n_clases": n, "nivel_unspsc_digitos": C.NIVEL_UNSPSC,
        "n_entrenamiento": int(corte), "n_validacion": int(len(i_tr) - corte), "n_prueba": int(len(i_te)),
        "prueba_excluidas_por_clase_no_vista": int((es_te & ~con_clase).sum()),
        "hiperparametros_elegidos": mejor, "linea_base_C": mejor_C,
        "transfer_learning": m_nn, "linea_base": m_bl,
        "H2_diferencia_f1_macro_puntos": dif,
        "H2_cumple_meta_5_puntos": dif >= 5,
        "top3_cumple_meta_90": m_nn["top3"] >= 0.90,
    }
    with open(C.DIR_REPORTES / "evaluacion_clasificador.json", "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)
    G.comparacion_clasificadores(resultado, C.DIR_FIGURAS / "clf_comparacion.png")

    # 4) Sugerencias para todas las licitaciones (vistas de la aplicación)
    probs = clf.predict(E, verbose=0)
    top = K.top_k(probs, C.TOP_K_SUGERENCIAS)
    app = lic.copy()
    for r in range(C.TOP_K_SUGERENCIAS):
        app[f"sugerencia_{r + 1}"] = [clases[j] for j in top[:, r]]
        app[f"prob_{r + 1}"] = probs[np.arange(len(probs)), top[:, r]]
    app["coincide_sugerencia"] = app["familia"] == app["sugerencia_1"]
    app["en_top3"] = [f in (a, b, c) for f, a, b, c in
                      zip(app["familia"], app["sugerencia_1"], app["sugerencia_2"], app["sugerencia_3"])]
    app.to_parquet(C.DIR_PROCESADOS / "licitaciones_app.parquet", index=False)

    # 5) Muestra de 100 licitaciones de prueba para revisión manual (Tabla 5)
    muestra = app.iloc[i_te].sample(min(100, len(i_te)), random_state=C.SEMILLA)
    muestra = muestra[["NroLicitacion", "Institucion", "texto", "familia", "familia_nombre",
                       "sugerencia_1", "prob_1", "sugerencia_2", "sugerencia_3"]].copy()
    muestra["etiqueta_registrada_correcta (si/no)"] = ""
    muestra["sugerencia_1_correcta (si/no)"] = ""
    muestra["comentario"] = ""
    muestra.to_csv(C.DIR_REPORTES / "muestra_revision_manual.csv", index=False)
    return resultado
