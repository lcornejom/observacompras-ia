"""Entrenamiento y evaluación completos del autoencoder (F3, H1).

Salidas:
    models/autoencoder.keras, models/escalador_ae.json, models/config_ae.json
    data/processed/puntajes_atipicidad.parquet
    reports/ae_busqueda_hiperparametros.csv, reports/ae_curva_perdida.csv
    reports/evaluacion_ae.json, reports/figuras/ae_*.png
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import anomalias as A
from . import autoencoder as AE
from . import config as C
from . import graficos as G


def ejecutar(verbose: int = 0, rapido: bool = False) -> dict:
    """rapido=True usa una sola combinación de hiperparámetros (solo para pruebas)."""
    C.DIR_MODELOS.mkdir(exist_ok=True)
    C.DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    perfiles = pd.read_parquet(C.DIR_PROCESADOS / "perfiles.parquet")
    p_tr = perfiles[perfiles["anio"] == C.ANIO_ENTRENAMIENTO].reset_index(drop=True)
    p_te = perfiles[perfiles["anio"] == C.ANIO_PRUEBA].reset_index(drop=True)

    esc = AE.EscaladorMinMax().fit(p_tr[C.VARIABLES].values)
    esc.guardar(C.DIR_MODELOS / "escalador_ae.json")
    X_tr = esc.transform(p_tr[C.VARIABLES].values)
    X_te = esc.transform(p_te[C.VARIABLES].values)

    # 1) Búsqueda de hiperparámetros con K-fold (K = 5)
    lrs = (1e-3,) if rapido else C.AE_LEARNING_RATES
    bss = (16,) if rapido else C.AE_BATCH_SIZES
    print(f"Autoencoder: búsqueda K-fold con {len(p_tr)} organismos de {C.ANIO_ENTRENAMIENTO}")
    tabla, hists = AE.busqueda_kfold(X_tr, lrs, bss, verbose=verbose)
    tabla.to_csv(C.DIR_REPORTES / "ae_busqueda_hiperparametros.csv", index=False)
    mejor = AE.elegir_configuracion(tabla)
    lr, bs = mejor["learning_rate"], mejor["batch_size"]
    curva = AE.curva_media(hists[(lr, bs)])
    curva.to_csv(C.DIR_REPORTES / "ae_curva_perdida.csv", index=False)
    n_epochs = AE.epochs_optimos(hists[(lr, bs)])
    G.curva_perdida(curva, f"Autoencoder: pérdida media en K-fold (lr={lr:g}, batch={bs})",
                    C.DIR_FIGURAS / "ae_curva_perdida.png", n_epochs)
    G.heatmap_busqueda(tabla, "mejor_val_loss", "Autoencoder: val_loss media (K-fold)",
                       C.DIR_FIGURAS / "ae_busqueda.png")

    # 2) Modelo final con todos los organismos de 2023
    ae, _ = AE.entrenar_final(X_tr, lr, bs, n_epochs)
    ae.save(C.DIR_MODELOS / "autoencoder.keras")
    with open(C.DIR_MODELOS / "config_ae.json", "w") as f:
        json.dump({**mejor, "epochs_final": n_epochs, "variables": C.VARIABLES,
                   "capas": [16, 4, 16, len(C.VARIABLES)], "optimizador": "rmsprop", "perdida": "mse"},
                  f, indent=2)

    # 3) Puntajes de ambos años (el ranking que se muestra es el del año de prueba)
    X_all = esc.transform(perfiles[C.VARIABLES].values)
    punt = AE.tabla_puntajes(perfiles, X_all, ae)
    punt = punt.merge(perfiles, on=["organismo", "organismo_nombre", "anio"])
    punt.to_parquet(C.DIR_PROCESADOS / "puntajes_atipicidad.parquet", index=False)

    # 4) H1: anomalías sintéticas vs. línea base z
    sint = A.generar_sinteticas(p_te, p_tr)
    X_sint = esc.transform(sint[C.VARIABLES].values)
    s_real_ae, _, _ = AE.puntuar(ae, X_te)
    s_sint_ae, err_sint, Xh_sint = AE.puntuar(ae, X_sint)
    z = A.LineaBaseZ().fit(X_tr)
    s_real_z, s_sint_z = z.puntuar(X_te), z.puntuar(X_sint)
    rec_ae, umbral_ae = A.recall_en_top(s_real_ae, s_sint_ae)
    rec_z, umbral_z = A.recall_en_top(s_real_z, s_sint_z)
    sint_out = sint[["organismo", "tipo_anomalia"]].copy()
    sint_out["puntaje_ae"], sint_out["puntaje_z"] = s_sint_ae, s_sint_z
    sint_out["detectada_ae"], sint_out["detectada_z"] = s_sint_ae > umbral_ae, s_sint_z > umbral_z
    sint_out["explicacion_ae"] = [AE.frase_explicacion(err_sint[i], X_sint[i], Xh_sint[i]) for i in range(len(sint))]
    sint_out.to_csv(C.DIR_REPORTES / "ae_anomalias_sinteticas.csv", index=False)
    G.recall_por_tipo(sint_out, C.DIR_FIGURAS / "ae_recall_por_tipo.png")

    # 5) Estabilidad del ranking entre los 5 modelos del K-fold
    modelos = AE.modelos_por_fold(X_tr, lr, bs)
    estab = AE.estabilidad_top(modelos, X_te)

    # Coincidencia del top 10 entre autoencoder y regla z (informativo)
    top_ae = set(np.argsort(s_real_ae)[::-1][:10])
    top_z = set(np.argsort(s_real_z)[::-1][:10])

    resultado = {
        "anio_entrenamiento": C.ANIO_ENTRENAMIENTO, "anio_evaluacion": C.ANIO_PRUEBA,
        "n_organismos_entrenamiento": len(p_tr), "n_organismos_evaluacion": len(p_te),
        "hiperparametros_elegidos": {**mejor, "epochs_final": n_epochs},
        "H1_recall_5pct": {
            "autoencoder": rec_ae, "linea_base_z": rec_z, "meta": 0.80,
            "cumple_meta": rec_ae >= 0.80,
            "por_tipo_autoencoder": A.recall_por_tipo(sint["tipo_anomalia"], s_real_ae, s_sint_ae),
            "por_tipo_linea_base_z": A.recall_por_tipo(sint["tipo_anomalia"], s_real_z, s_sint_z),
            "n_anomalias": len(sint),
        },
        "H1_estabilidad_top10": {**estab, "meta": 7,
                                 "cumple_meta": estab["coincidencia_media_pares"] >= 7},
        "coincidencia_top10_autoencoder_vs_z": len(top_ae & top_z),
    }
    with open(C.DIR_REPORTES / "evaluacion_ae.json", "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)
    return resultado
