"""Pruebas unitarias de las funciones de cálculo (pytest)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from observacompras import anomalias as A  # noqa: E402
from observacompras import config as C  # noqa: E402
from observacompras import ingesta as I  # noqa: E402
from observacompras import perfil as P  # noqa: E402


def test_normalizar_rut_y_persona_natural():
    r = I.normalizar_rut(pd.Series(["61.608.204-3", "9.466.879-k", None]))
    assert r.tolist()[:2] == ["61608204-3", "9466879-K"]
    pn = I.es_persona_natural(r)
    assert pn.tolist()[:2] == [False, True]


def test_digito_verificador():
    assert I.digito_verificador(61608204) == "3"
    assert I.digito_verificador(76547963) == "0"


def test_anonimizacion_estable():
    df = pd.DataFrame({"ProveedorRUT": ["12345678-5", "12345678-5", "76000000-1"],
                       "Proveedor": ["Juan Pérez", "Juan Pérez", "Empresa SpA"],
                       "NombreOC": ["Honorarios Juan Pérez", "x", "y"]})
    out = I.anonimizar_proveedores(df.copy(), cols_texto=("NombreOC",))
    assert out.loc[0, "ProveedorRUT"] == out.loc[1, "ProveedorRUT"] != "12345678-5"
    assert "Juan" not in " ".join(out["Proveedor"]) + " ".join(out["NombreOC"])
    assert out.loc[2, "Proveedor"] == "Empresa SpA"


def _oc_ejemplo():
    f = pd.to_datetime
    return pd.DataFrame({
        "organismo": ["1"] * 4, "organismo_nombre": ["H"] * 4, "anio": [2023] * 4,
        "ProveedorRUT": ["A", "A", "B", "C"], "monto": [60.0, 20.0, 10.0, 10.0],
        "ProcedenciaOC": ["Trato Directo", "Compra Agil", "Compra Agil", "Licitacion Publica"],
        "fecha": [f("2023-01-10"), f("2023-02-01"), f("2023-03-01"), f("2023-09-01")],
        "segunda_mitad": [False, False, False, True], "rubros": ["X|Y", "X", "Z", None],
    })


def test_perfil_variables():
    lic = pd.DataFrame({"organismo": ["1", "1"], "anio": [2023, 2023], "n_oferentes": [1, 3]})
    p = P.calcular_perfiles(_oc_ejemplo(), lic, min_oc=1).iloc[0]
    assert p["n_compras"] == 4 and p["n_proveedores"] == 3
    assert p["cr1"] == pytest.approx(0.8)
    assert p["hhi"] == pytest.approx(80**2 + 10**2 + 10**2)
    assert p["pct_trato_directo"] == pytest.approx(0.25)
    assert p["pct_proveedores_nuevos"] == pytest.approx(1 / 3)
    assert p["n_rubros"] == 3
    assert p["pct_oferente_unico"] == pytest.approx(0.5)
    assert p["monto_total_log"] == pytest.approx(np.log1p(100))


def test_hhi_maximo_un_proveedor():
    oc = _oc_ejemplo().assign(ProveedorRUT="A")
    lic = pd.DataFrame({"organismo": ["1"], "anio": [2023], "n_oferentes": [1]})
    p = P.calcular_perfiles(oc, lic, min_oc=1).iloc[0]
    assert p["hhi"] == pytest.approx(10_000) and p["cr1"] == pytest.approx(1.0)


def test_anomalias_sinteticas_y_linea_base():
    rng = np.random.default_rng(0)
    base = pd.DataFrame(rng.uniform(0.1, 0.3, size=(40, 10)), columns=C.VARIABLES)
    base["n_compras"] = 100
    base["n_proveedores"] = 50
    base["organismo"], base["organismo_nombre"] = "x", "x"
    s = A.generar_sinteticas(base, base)
    assert len(s) == C.N_ANOMALIAS_SINTETICAS
    assert set(s["tipo_anomalia"]) == set(A.TIPOS)
    assert (s.loc[s.tipo_anomalia == "proveedor_unico", "cr1"] > 0.9).all()
    z = A.LineaBaseZ().fit(base[C.VARIABLES].values)
    rec, _ = A.recall_en_top(z.puntuar(base[C.VARIABLES].values), np.array([100.0, -1.0]))
    assert rec == 0.5


def test_autoencoder_arquitectura():
    pytest.importorskip("keras")
    from observacompras import autoencoder as AE
    m = AE.construir_autoencoder()
    assert [l.units for l in m.layers] == [16, 4, 16, 10]
    assert m.layers[-1].activation.__name__ == "sigmoid"
    assert m.loss == "mse"


def test_clasificador_arquitectura():
    pytest.importorskip("keras")
    from observacompras import clasificador as K
    m = K.construir_clasificador(7)
    assert m.input_shape == (None, 384)
    assert m.layers[0].units == 128 and m.layers[1].rate == 0.3 and m.layers[2].units == 7
    assert m.layers[2].activation.__name__ == "softmax"


def test_metricas_top3():
    from observacompras import clasificador as K
    probs = np.array([[.5, .3, .1, .1], [.1, .2, .3, .4]])
    m = K.metricas(np.array([0, 0]), probs, 4)
    assert m["exactitud"] == 0.5 and m["top3"] == 0.5


def test_curva_media_kfold_y_epochs(tmp_path):
    from observacompras import autoencoder as AE
    from observacompras import graficos as G
    hists = [{"loss": list(np.linspace(1, .1, n)), "val_loss": list(np.linspace(1, .2, n))} for n in (65, 120, 153)]
    curva = AE.curva_media(hists)
    assert len(curva) == 153
    assert curva["val_loss"].notna().sum() == 65          # solo epochs comunes a todos los folds
    tabla = pd.DataFrame({"mejor_epoch": [45, 100, 133]})
    assert AE.epochs_optimos(tabla) == 100
    G.curva_perdida_kfold(hists, curva, "t", tmp_path / "c.png", 100)
    assert (tmp_path / "c.png").exists()
