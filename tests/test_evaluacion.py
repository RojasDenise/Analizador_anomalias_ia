"""Tests de evaluacion.py con un caso chico calculado a mano.

Uso:  python -m pytest tests/test_evaluacion.py   (o  python tests/test_evaluacion.py)
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import evaluacion as ev  # noqa: E402


def tabla_chica():
    """10 ventanas de 5 s: 0-1 normal, 2-4 red lenta (degradada), 5-6 normal, 7-8 escaneo, 9 normal."""
    indice = pd.date_range("2026-11-05 17:00:00", periods=10, freq="5s", name="inicio_ventana")
    escenario = ["normal"] * 2 + ["red lenta"] * 3 + ["normal"] * 2 + ["escaneo"] * 2 + ["normal"]
    tabla = pd.DataFrame({"escenario_real": escenario, "duracion_s": 5}, index=indice)
    tabla["degradada"] = tabla["escenario_real"] == "red lenta"
    return tabla


def alertas(indices, etiqueta, tabla, modulo="c1"):
    return pd.DataFrame({
        "inicio_ventana": tabla.index[indices],
        "modulo": modulo,
        "etiqueta": etiqueta,
        "detalle": "",
    })


def test_metricas_a_mano():
    t = tabla_chica()
    # Marca 3 y 4 (aciertos), 0 (falsa alarma); se pierde la 2
    a = alertas([0, 3, 4], "Degradación de rendimiento", t)
    glob, _ = ev.evaluar(t, {"m": a})
    fila = glob.iloc[0]
    assert (fila.VP, fila.FP, fila.FN, fila.VN) == (2, 1, 1, 6)
    assert round(fila.precision, 3) == round(2 / 3, 3)
    assert round(fila.recall, 3) == round(2 / 3, 3)
    assert round(fila.falsas_alarmas_c100, 2) == round(100 / 7, 2)
    assert fila.demora_media_s == 5.0           # evento empieza en la ventana 2, primera alerta en la 3
    assert fila.eventos_detectados == "1/1"


def test_etiquetas_que_no_son_degradacion_no_cuentan():
    t = tabla_chica()
    # El escaneo etiquetado como inusual no es un acierto ni una falsa alarma de degradación
    a = alertas([7, 8], "Comportamiento inusual", t, modulo="b2")
    glob, _ = ev.evaluar(t, {"m": a})
    assert glob.iloc[0].FP == 0 and glob.iloc[0].VP == 0


def test_alertas_repetidas_en_una_ventana_cuentan_una_vez():
    t = tabla_chica()
    a = pd.concat([alertas([2], "Degradación de rendimiento", t, "c1"),
                   alertas([2], "Congestión o saturación", t, "c2")])
    glob, _ = ev.evaluar(t, {"Reglas fijas": a})
    assert glob.iloc[0].VP == 1


def test_por_escenario_y_matriz_e4():
    t = tabla_chica()
    a = pd.concat([alertas([2, 3, 4], "Degradación de rendimiento", t),
                   alertas([7], "Degradación de rendimiento", t),
                   alertas([8], "Comportamiento inusual", t)])
    _, esc = ev.evaluar(t, {"m": a})
    esc = esc.set_index("escenario")
    assert esc.loc["red lenta", "recall"] == 1.0
    assert esc.loc["escaneo", "falsas_alarmas_c100"] == 50.0

    matriz = ev.matriz_etiquetas(t, a)
    assert matriz.loc["escaneo", "Comportamiento inusual"] == 1
    assert matriz.loc["escaneo", "Degradación de rendimiento"] == 1
    assert matriz.loc["red lenta", "Degradación de rendimiento"] == 3


def test_sin_alertas():
    t = tabla_chica()
    glob, _ = ev.evaluar(t, {"m": ev.pd.DataFrame(columns=["inicio_ventana", "modulo", "etiqueta", "detalle"])})
    assert glob.iloc[0].recall == 0 and glob.iloc[0].eventos_detectados == "0/1"


def test_hipotesis():
    glob = pd.DataFrame([
        {"metodo": "Reglas fijas", "recall": 0.5, "f1": 0.6, "falsas_alarmas_c100": 2.0},
        {"metodo": "IF", "recall": 0.8, "f1": 0.7, "falsas_alarmas_c100": 3.0},
        {"metodo": "LOF", "recall": 0.9, "f1": 0.5, "falsas_alarmas_c100": 9.0},
    ])
    h = ev.hipotesis(glob).set_index("metodo")
    assert h.loc["IF", "acepta_hipotesis"]
    assert not h.loc["LOF", "acepta_hipotesis"]


if __name__ == "__main__":
    for nombre, prueba in list(globals().items()):
        if nombre.startswith("test_"):
            prueba()
            print("ok", nombre)
