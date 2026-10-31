"""Pruebas del módulo C2 (uso_red.py) con paquetes armados a mano.

Uso:  python tests/test_uso_red.py     (o: pytest tests/test_uso_red.py)
"""
import os
import sys

import pandas as pd

CARPETA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(CARPETA))

import uso_red  # noqa: E402
import ventanas  # noqa: E402

T0 = 1_793_000_000  # múltiplo de 5: el inicio de la ventana 0


def _paquete(t, longitud, transporte="TCP", app="TLS", src="192.168.0.10", dst="1.1.1.1",
            sport=51000, dport=443, dns=None):
    return {
        "Timestamp": T0 + t, "Source IP": src, "Destination IP": dst,
        "Transport Protocol": transporte, "Application Protocol": app, "Hostname DNS": dns,
        "Length": longitud, "Src Port": sport, "Dst Port": dport,
        "TCP SYN": 0, "TCP ACK": 0,
    }


def _paquetes_ejemplo():
    filas = []
    # Ventana 0 (0-5 s): 10 paquetes TCP de 1000 B = 10.000 B
    filas += [_paquete(0.1 * i, 1000) for i in range(10)]
    # Ventana 1 (5-10 s): 2 DNS (100 B), 1 UDP (200 B), 2 TCP (500 B)
    filas += [_paquete(5.1, 100, "UDP", "DNS", dport=53, dns="unne.edu.ar"),
            _paquete(5.2, 100, "UDP", "DNS", dport=53, dns="google.com"),
            _paquete(5.3, 200, "UDP", "QUIC", dport=443),
            _paquete(5.4, 500), _paquete(5.5, 500)]
    # Ventana 2 (10-15 s): sin tráfico
    # Ventana 3 (15-20 s): pico, 100 paquetes de 1500 B desde otra IP
    filas += [_paquete(15 + 0.04 * i, 1500, src="8.8.4.4", dst="192.168.0.10",
                    sport=443, dport=51000) for i in range(100)]
    return pd.DataFrame(filas)


def _preparar():
    paquetes = _paquetes_ejemplo()
    tabla = ventanas.tabla_ventanas(paquetes)
    return paquetes, uso_red.calcular_metricas(paquetes, tabla)


def test_metricas_basicas():
    _, t = _preparar()
    assert len(t) == 4  # incluye la ventana vacía
    w0, w1, w2, w3 = (t.iloc[i] for i in range(4))
    assert abs(w0["c2_mbps"] - 10_000 * 8 / 5 / 1e6) < 1e-9
    assert w0["c2_pps"] == 2
    assert w0["c2_tam_medio"] == 1000
    assert w1["c2_tam_medio"] == 1400 / 5
    assert w2["c2_mbps"] == 0 and w2["c2_pps"] == 0 and w2["c2_tam_medio"] == 0
    assert abs(w3["c2_mbps"] - 150_000 * 8 / 5 / 1e6) < 1e-9


def test_reparto_protocolos():
    _, t = _preparar()
    w1 = t.iloc[1]
    assert w1["c2_pct_dns"] == 40 and w1["c2_pct_udp"] == 20
    assert w1["c2_pct_tcp"] == 40 and w1["c2_pct_otros"] == 0
    # En toda ventana con tráfico los porcentajes suman 100; en la vacía, 0
    suma = t[uso_red.COLUMNAS_PCT].sum(axis=1)
    assert list(suma.round(6)) == [100, 100, 0, 100]


def test_top_talkers():
    paquetes, _ = _preparar()
    global_ = uso_red.top_talkers(paquetes, n=2, por="origen")
    assert list(global_["ip"]) == ["8.8.4.4", "192.168.0.10"]
    assert global_["bytes"].iloc[0] == 150_000

    por_ventana = uso_red.top_talkers(paquetes, n=1, segundos=5, por="origen")
    assert len(por_ventana) == 3  # la ventana vacía no tiene talkers
    assert por_ventana["ip"].iloc[-1] == "8.8.4.4"


def test_reglas_pico():
    _, t = _preparar()
    # Referencia "normal" baja: la ventana 3 (e incluso la 0) superan 3 veces ese promedio
    alertas = uso_red.reglas_c2(t, mbps_normal=0.02)
    ventanas.validar_alertas(alertas)
    assert (alertas["modulo"] == "c2").all()
    assert (alertas["etiqueta"] == "Pico de uso normal").all()
    assert t.index[3] in set(alertas["inicio_ventana"])
    assert t.index[2] not in set(alertas["inicio_ventana"])


def test_reglas_uso_alto_y_sin_alertas():
    _, t = _preparar()
    # Capacidad 1 Mbps, 80 %: solo la ventana 3 (0,24 Mbps) NO supera 0,8 Mbps -> sin alertas
    assert uso_red.reglas_c2(t, capacidad_mbps=1.0, factor_pico=1000).empty
    # Capacidad 0,2 Mbps: la ventana 3 supera 0,16 Mbps
    alertas = uso_red.reglas_c2(t, capacidad_mbps=0.2, factor_pico=1000)
    assert list(alertas["inicio_ventana"]) == [t.index[3]]
    assert "Mbps" in alertas["detalle"].iloc[0]


def test_mbps_referencia():
    _, t = _preparar()
    assert abs(uso_red.mbps_referencia(t) - t["c2_mbps"].mean()) < 1e-12
    assert abs(uso_red.mbps_referencia([t, t]) - t["c2_mbps"].mean()) < 1e-12


def test_reglas_cruzan_con_c1():
    _, t = _preparar()
    t["c1_pct_retrans"] = [0.0, 0.0, 0.0, 8.0]
    t["c1_paquetes_tcp"] = [10, 3, 0, 100]

    def etiqueta(alertas, i):
        return alertas.set_index("inicio_ventana").loc[t.index[i], "etiqueta"]

    alertas = uso_red.reglas_c2(t, mbps_normal=0.002)
    ventanas.validar_alertas(alertas)
    assert etiqueta(alertas, 3) == "Congestión o saturación"  # volumen alto + retransmisiones
    assert etiqueta(alertas, 0) == "Pico de uso normal"       # volumen alto sin pérdida
    assert "(C1)" in alertas.set_index("inicio_ventana").loc[t.index[3], "detalle"]

    # Sin cruzar, o con pocos paquetes TCP, todo queda como pico de uso
    sin_cruce = uso_red.reglas_c2(t, mbps_normal=0.002, cruzar_con_c1=False)
    assert (sin_cruce["etiqueta"] == "Pico de uso normal").all()
    t["c1_paquetes_tcp"] = [10, 3, 0, 19]
    pocos = uso_red.reglas_c2(t, mbps_normal=0.002)
    assert (pocos["etiqueta"] == "Pico de uso normal").all()


def test_resumen_por_escenario():
    _, t = _preparar()
    t["escenario_real"] = ["normal", "normal", "normal", "saturacion"]
    r = uso_red.resumen_por_escenario(t)
    assert list(r["escenario"]) == ["normal", "saturacion"]  # normal primero
    normal = (t["c2_mbps"].iloc[0] + t["c2_mbps"].iloc[1] + t["c2_mbps"].iloc[2]) / 3
    assert abs(r["mbps_medio"].iloc[0] - normal) < 1e-12
    assert abs(r["veces_normal"].iloc[1] - t["c2_mbps"].iloc[3] / normal) < 1e-9
    frases = uso_red.interpretar(r)
    assert len(frases) == 1 and "Saturacion" in frases[0] and "pico" in frases[0]


def test_interpretar_escenario_que_c2_no_ve():
    resumen = pd.DataFrame({"escenario": ["normal", "red lenta"], "veces_normal": [1.0, 1.1]})
    frases = uso_red.interpretar(resumen)
    assert len(frases) == 1 and "no lo ve" in frases[0]


def test_con_datos_de_ejemplo():
    """Si existe tests/ejemplo_paquetes.csv (python tests/generar_ejemplo.py), lo usa."""
    ruta = os.path.join(CARPETA, "ejemplo_paquetes.csv")
    if not os.path.exists(ruta):
        print("  (se omite: falta ejemplo_paquetes.csv)")
        return
    paquetes = ventanas.cargar_paquetes(ruta)
    t = uso_red.calcular_metricas(paquetes, ventanas.tabla_ventanas(paquetes))
    import rendimiento_tcp as c1
    t = c1.calcular_metricas(paquetes, t)  # para que C2 pueda cruzar con la pérdida
    intervalos = ventanas.leer_intervalos(os.path.join(CARPETA, "ejemplo_intervalos.csv"))
    t = ventanas.etiquetar_ventanas(t, intervalos, captura="ejemplo_paquetes.csv")
    normal = t[t["escenario_real"] == "normal"]["c2_mbps"].mean()
    saturacion = t[t["escenario_real"] == "saturacion"]["c2_mbps"].mean()
    assert saturacion > 3 * normal, "la saturación debería verse como un pico de Mbps"
    alertas = uso_red.reglas_c2(t, mbps_normal=normal)
    ventanas.validar_alertas(alertas)
    marcadas = t.loc[alertas["inicio_ventana"], "escenario_real"].value_counts()
    print("  ventanas con alerta por escenario:", {k: int(v) for k, v in marcadas.items()})
    etiquetas = alertas.assign(escenario=t.loc[alertas["inicio_ventana"], "escenario_real"].values)
    print("  etiquetas por escenario:", etiquetas.groupby(["escenario", "etiqueta"]).size().to_dict())


if __name__ == "__main__":
    pruebas = [f for nombre, f in sorted(globals().items()) if nombre.startswith("test_")]
    for prueba in pruebas:
        prueba()
        print(f"OK  {prueba.__name__}")
    print(f"{len(pruebas)} pruebas pasaron")