"""Pruebas del módulo rendimiento_tcp.py (C1) con paquetes armados a mano.

Uso:  python tests/test_rendimiento_tcp.py     (o: pytest tests/test_rendimiento_tcp.py)
"""
import math
import os
import sys

import pandas as pd

CARPETA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(CARPETA))

import rendimiento_tcp as c1  # noqa: E402
import ventanas  # noqa: E402

T0 = 1_793_000_000  # múltiplo de 5: el inicio de la ventana 0


def _tcp(t, rtt=None, retrans=0, dup=0, zero=0):
    return {
        "Timestamp": T0 + t, "Source IP": "192.168.0.10", "Destination IP": "1.1.1.1",
        "Transport Protocol": "TCP", "Application Protocol": "TLS", "Hostname DNS": None,
        "Length": 1514, "Src Port": 51000, "Dst Port": 443,
        "TCP SYN": 0, "TCP ACK": 1, "TCP RST": 0,
        "TCP RTT": rtt, "TCP Retransmission": retrans, "TCP Dup ACK": dup, "TCP Zero Window": zero,
    }


def _udp(t):
    return {
        "Timestamp": T0 + t, "Source IP": "192.168.0.10", "Destination IP": "8.8.8.8",
        "Transport Protocol": "UDP", "Application Protocol": "DNS", "Hostname DNS": "unne.edu.ar",
        "Length": 80, "Src Port": 53000, "Dst Port": 53,
        "TCP SYN": 0, "TCP ACK": 0, "TCP RST": 0,
        "TCP RTT": None, "TCP Retransmission": 0, "TCP Dup ACK": 0, "TCP Zero Window": 0,
    }


def _paquetes_ejemplo():
    filas = []
    # Ventana 0 (0-5 s): 30 TCP + 10 UDP.
    #   RTT en 3 paquetes: 10, 20 y 30 ms -> medio 20, máx 30.
    #   3 retransmisiones (10 % de 30 TCP), 1 ACK duplicado, 1 ventana en cero.
    for i in range(30):
        filas.append(_tcp(0.1 * i,
                          rtt={0: 0.010, 1: 0.020, 2: 0.030}.get(i),
                          retrans=int(i in (3, 4, 5)), dup=int(i == 6), zero=int(i == 7)))
    filas += [_udp(3.1 + 0.1 * i) for i in range(10)]
    # Ventana 1 (5-10 s): solo UDP -> sin TCP, RTT en NaN
    filas += [_udp(5.1 + 0.1 * i) for i in range(5)]
    # Ventana 2 (10-15 s): 5 TCP, 1 retransmisión (20 %, pero menos de 20 paquetes TCP), RTT 500 ms
    filas += [_tcp(10.1 + 0.1 * i, rtt=0.5 if i == 0 else None, retrans=int(i == 1)) for i in range(5)]
    # Ventana 3 (15-20 s): 25 TCP normales, RTT 15 ms
    filas += [_tcp(15.1 + 0.1 * i, rtt=0.015 if i == 0 else None) for i in range(25)]
    return pd.DataFrame(filas)


def _preparar():
    paquetes = _paquetes_ejemplo()
    tabla = ventanas.tabla_ventanas(paquetes)
    return paquetes, c1.calcular_metricas(paquetes, tabla)


def test_columnas_y_ventanas():
    _, t = _preparar()
    assert len(t) == 4
    for col in c1.COLUMNAS_C1:
        assert col in t.columns
    assert list(t["c1_paquetes_tcp"]) == [30, 0, 5, 25]


def test_rtt_en_ms():
    _, t = _preparar()
    w0 = t.iloc[0]
    assert abs(w0["c1_rtt_medio_ms"] - 20) < 1e-9
    assert abs(w0["c1_rtt_max_ms"] - 30) < 1e-9
    assert abs(t.iloc[2]["c1_rtt_medio_ms"] - 500) < 1e-9


def test_porcentajes_sobre_paquetes_tcp():
    _, t = _preparar()
    w0 = t.iloc[0]
    # 3 de 30 TCP = 10 % (si se dividiera por los 40 paquetes daría 7,5 %)
    assert abs(w0["c1_pct_retrans"] - 10) < 1e-9
    assert abs(w0["c1_pct_dup_ack"] - 100 / 30) < 1e-9
    assert w0["c1_ventanas_cero"] == 1


def test_ventana_sin_tcp():
    _, t = _preparar()
    w1 = t.iloc[1]
    assert math.isnan(w1["c1_rtt_medio_ms"]) and math.isnan(w1["c1_rtt_max_ms"])
    assert w1["c1_pct_retrans"] == 0 and w1["c1_pct_dup_ack"] == 0 and w1["c1_ventanas_cero"] == 0


def test_umbral_rtt():
    _, t = _preparar()
    rtt = t["c1_rtt_medio_ms"].dropna()          # 20, 500, 15 (la ventana sin RTT no cuenta)
    esperado = rtt.mean() + 3 * rtt.std()
    assert abs(c1.umbral_rtt(t) - esperado) < 1e-9
    assert abs(c1.umbral_rtt([t, t]) - (pd.concat([rtt, rtt]).mean() + 3 * pd.concat([rtt, rtt]).std())) < 1e-9
    try:
        c1.umbral_rtt(t.iloc[[1]])
        raise AssertionError("debería fallar sin muestras de RTT")
    except ValueError:
        pass


def test_reglas():
    _, t = _preparar()
    alertas = c1.reglas_c1(t, umbral_rtt_ms=100)
    ventanas.validar_alertas(alertas)
    assert (alertas["modulo"] == "c1").all()
    assert (alertas["etiqueta"] == "Degradación de rendimiento").all()
    # Ventana 0: 10 % de retransmisiones con 30 TCP. Ventana 2: RTT 500 ms.
    assert list(alertas["inicio_ventana"]) == [t.index[0], t.index[2]]
    assert "retransmisiones" in alertas["detalle"].iloc[0]
    # En la ventana 2 el 20 % no cuenta (solo 5 paquetes TCP): la alerta es solo por RTT
    assert alertas["detalle"].iloc[1].startswith("RTT medio")


def test_minimo_de_paquetes_tcp():
    _, t = _preparar()
    alertas = c1.reglas_c1(t, umbral_rtt_ms=1000, min_paquetes_tcp=1)
    assert list(alertas["inicio_ventana"]) == [t.index[0], t.index[2]]
    alertas = c1.reglas_c1(t, umbral_rtt_ms=1000)
    assert list(alertas["inicio_ventana"]) == [t.index[0]]


def test_sin_alertas():
    _, t = _preparar()
    alertas = c1.reglas_c1(t, umbral_rtt_ms=1000, max_pct_retrans=50)
    assert alertas.empty
    assert list(alertas.columns) == ventanas.COLUMNAS_ALERTAS


def test_falta_calcular_metricas():
    paquetes = _paquetes_ejemplo()
    try:
        c1.reglas_c1(ventanas.tabla_ventanas(paquetes), umbral_rtt_ms=100)
        raise AssertionError("debería pedir calcular_metricas() primero")
    except ValueError:
        pass


def test_con_datos_de_ejemplo():
    """Si existe tests/ejemplo_paquetes.csv (python tests/generar_ejemplo.py), lo usa."""
    ruta = os.path.join(CARPETA, "ejemplo_paquetes.csv")
    if not os.path.exists(ruta):
        print("  (se omite: falta ejemplo_paquetes.csv)")
        return
    paquetes = ventanas.cargar_paquetes(ruta)
    t = c1.calcular_metricas(paquetes, ventanas.tabla_ventanas(paquetes))
    intervalos = ventanas.leer_intervalos(os.path.join(CARPETA, "ejemplo_intervalos.csv"))
    t = ventanas.etiquetar_ventanas(t, intervalos, captura="ejemplo_paquetes.csv")

    normal = t[t["escenario_real"] == "normal"]
    umbral = c1.umbral_rtt(normal)
    alertas = c1.reglas_c1(t, umbral_rtt_ms=umbral)
    ventanas.validar_alertas(alertas)

    marcadas = t.index.isin(set(alertas["inicio_ventana"]))
    red_lenta = t["escenario_real"] == "red lenta"
    recall_red_lenta = marcadas[red_lenta].mean()
    falsas_normal = marcadas[t["escenario_real"] == "normal"].mean()
    print(f"  umbral RTT {umbral:.1f} ms | red lenta detectada {recall_red_lenta:.0%} "
          f"| falsas alarmas en normal {falsas_normal:.1%}")
    assert recall_red_lenta > 0.8, "C1 debería detectar la mayoría de las ventanas de red lenta"
    assert falsas_normal < 0.05, "C1 no debería marcar muchas ventanas normales"


if __name__ == "__main__":
    pruebas = [f for nombre, f in sorted(globals().items()) if nombre.startswith("test_")]
    for prueba in pruebas:
        prueba()
        print(f"OK  {prueba.__name__}")
    print(f"{len(pruebas)} pruebas pasaron")
