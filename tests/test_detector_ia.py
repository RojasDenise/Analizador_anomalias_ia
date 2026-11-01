"""Pruebas del módulo detector_ia.py (B2) con tablas de ventanas armadas a mano.

Uso:  python tests/test_detector_ia.py     (o: pytest tests/test_detector_ia.py)
"""
import os
import sys

import numpy as np
import pandas as pd

CARPETA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(CARPETA))

import detector_ia as b2  # noqa: E402
import ventanas  # noqa: E402

rng = np.random.default_rng(0)


def _tabla(n, **cambios):
    """n ventanas de tráfico normal; cambios pisa columnas (por ejemplo c1_rtt_medio_ms=300)."""
    indice = pd.date_range("2026-11-05 17:00:00", periods=n, freq="5s", name="inicio_ventana")
    t = pd.DataFrame({
        "duracion_s": 5,
        "c1_rtt_medio_ms": rng.normal(45, 5, n),
        "c1_rtt_max_ms": rng.normal(80, 10, n),
        "c1_pct_retrans": rng.uniform(0, 1, n),
        "c1_pct_dup_ack": rng.uniform(0, 1, n),
        "c1_ventanas_cero": 0,
        "c2_mbps": rng.normal(3, 0.4, n),
        "c2_pps": rng.normal(40, 5, n),
        "c2_tam_medio": rng.normal(900, 50, n),
        "c2_pct_tcp": rng.normal(90, 2, n),
        "c2_pct_udp": rng.normal(7, 1, n),
        "c2_pct_dns": rng.normal(3, 0.5, n),
        "ctx_max_puertos_por_ip": rng.integers(1, 4, n),
        "ctx_syn_sin_respuesta": 0,
        "ctx_consultas_dns": rng.integers(0, 3, n),
    }, index=indice)
    for col, valor in cambios.items():
        t[col] = valor
    return t


def _modelo():
    return b2.entrenar(_tabla(200))


def test_normal_casi_sin_alertas():
    modelo = _modelo()
    t = b2.puntuar(modelo, _tabla(100))
    for clave in b2.MODELOS:
        assert t[f"b2_{clave}_anomala"].mean() < 0.10, clave


def test_red_lenta_es_degradacion():
    modelo = _modelo()
    t = b2.puntuar(modelo, _tabla(10, c1_rtt_medio_ms=350.0, c1_rtt_max_ms=600.0, c1_pct_retrans=9.0))
    assert t["b2_if_anomala"].mean() >= 0.8
    assert (t["b2_causa"] == "Degradación de rendimiento").all()


def test_saturacion_es_congestion():
    modelo = _modelo()
    t = b2.puntuar(modelo, _tabla(10, c1_rtt_medio_ms=200.0, c1_pct_retrans=6.0, c2_mbps=40.0, c2_pps=900.0))
    assert t["b2_if_anomala"].mean() >= 0.8
    assert (t["b2_causa"] == "Congestión o saturación").all()


def test_uso_intenso_es_pico():
    modelo = _modelo()
    t = b2.puntuar(modelo, _tabla(10, c2_mbps=40.0, c2_pps=900.0))
    # Un pico que cambia solo 2 métricas: Isolation Forest lo detecta a veces (entre 50 y 100 %),
    # LOF y One-Class SVM siempre. Se verifica el que es estable.
    assert t["b2_lof_anomala"].all()
    assert (t["b2_causa"] == "Pico de uso normal").all()


def test_escaneo_es_inusual():
    modelo = _modelo()
    t = b2.puntuar(modelo, _tabla(10, ctx_max_puertos_por_ip=300, ctx_syn_sin_respuesta=250, c2_pps=80.0))
    # Isolation Forest no lo ve (SYN sin respuesta es siempre 0 en el normal); LOF y One-Class SVM sí
    assert t["b2_lof_anomala"].all() and t["b2_ocsvm_anomala"].all()
    assert (t["b2_causa"] == "Comportamiento inusual").all()


def test_rtt_faltante_no_rompe():
    normal = _tabla(200)
    normal.loc[normal.index[::7], "c1_rtt_medio_ms"] = np.nan   # ventanas sin TCP
    modelo = b2.entrenar(normal)
    prueba = _tabla(20)
    prueba["c1_rtt_medio_ms"] = np.nan
    t = b2.puntuar(modelo, prueba)
    assert t["b2_if_puntaje"].notna().all()


def test_columnas_que_faltan():
    # Si C2 no está, entrena solo con lo que hay
    normal = _tabla(200).drop(columns=b2.GRUPOS["volumen"])
    modelo = b2.entrenar(normal)
    assert "c2_mbps" not in modelo["columnas"]
    b2.puntuar(modelo, normal.head(10))


def test_alertas_con_formato_del_contrato():
    modelo = _modelo()
    prueba = pd.concat([_tabla(20), _tabla(5, c1_rtt_medio_ms=400.0, c1_pct_retrans=10.0)])
    prueba.index = pd.date_range("2026-11-05 17:00:00", periods=len(prueba), freq="5s", name="inicio_ventana")
    alertas = b2.generar_alertas(b2.puntuar(modelo, prueba))
    assert set(alertas) == {"Isolation Forest", "LOF", "One-Class SVM"}
    for nombre, a in alertas.items():
        ventanas.validar_alertas(a)
        assert (a["modulo"] == "b2").all()
    assert len(alertas["LOF"]) >= 5 and len(alertas["Isolation Forest"]) >= 1


def test_guardar_y_cargar(tmp_path=None):
    import tempfile
    modelo = _modelo()
    with tempfile.TemporaryDirectory() as carpeta:
        ruta = b2.guardar(modelo, os.path.join(carpeta, "b2.joblib"))
        cargado = b2.cargar(ruta)
    t1 = b2.puntuar(modelo, _tabla(5))
    t2 = b2.puntuar(cargado, t1[modelo["columnas"] + ["duracion_s"]])
    assert np.allclose(t1["b2_if_puntaje"], t2["b2_if_puntaje"])


if __name__ == "__main__":
    pruebas = [(n, f) for n, f in list(globals().items()) if n.startswith("test_")]
    for nombre, prueba in pruebas:
        prueba()
        print("OK ", nombre)
    print(f"{len(pruebas)} pruebas pasaron")
