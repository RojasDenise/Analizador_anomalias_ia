"""Genera datos sintéticos para trabajar sin capturas reales.

Crea en tests/:
  ejemplo_paquetes.csv   15 min de tráfico con el mismo formato que analizador.py
  ejemplo_intervalos.csv los tiempos "anotados" de cada escenario

Uso:  python tests/generar_ejemplo.py
"""
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

CARPETA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(CARPETA))

from analizador import COLUMNAS_PAQUETES  # noqa: E402
INICIO = datetime(2026, 11, 5, 17, 0, 0).timestamp()
DURACION = 15 * 60
rng = np.random.default_rng(42)

# Escenarios (segundos desde el inicio): nombre, desde, hasta
ESCENARIOS = [
    ("Red lenta", 180, 300),
    ("Saturacion", 420, 540),
    ("Uso intenso", 600, 660),
    ("Escaneo", 720, 760),
]


def escenario_en(t):
    for nombre, desde, hasta in ESCENARIOS:
        if desde <= t < hasta:
            return nombre
    return "normal"


def paquete(t, **campos):
    fila = {c: None for c in COLUMNAS_PAQUETES}
    fila.update({
        "Timestamp": INICIO + t,
        "Date": datetime.fromtimestamp(INICIO + t).strftime("%Y-%m-%d %H:%M:%S"),
        "TCP SYN": 0, "TCP ACK": 0, "TCP RST": 0,
        "TCP Retransmission": 0, "TCP Dup ACK": 0, "TCP Zero Window": 0,
    })
    fila.update(campos)
    return fila


def trafico_tcp(t, escenario):
    servidor = rng.choice(["142.250.78.4", "157.240.6.35", "104.18.32.7"])
    rtt = rng.normal(0.045, 0.008)
    retrans = rng.random() < 0.005
    dup = rng.random() < 0.005
    zero = False
    largo = int(rng.choice([66, 1514], p=[0.4, 0.6]))
    if escenario == "Red lenta":
        rtt += rng.normal(0.30, 0.05)
        retrans = rng.random() < 0.08
        dup = rng.random() < 0.06
    elif escenario == "Saturacion":
        rtt += rng.normal(0.12, 0.04)
        retrans = rng.random() < 0.05
        zero = rng.random() < 0.01
        largo = 1514
    elif escenario == "Uso intenso":
        largo = 1514
    entrante = rng.random() < 0.6
    return paquete(
        t,
        **{"Source IP": servidor if entrante else "192.168.0.10",
           "Destination IP": "192.168.0.10" if entrante else servidor,
           "Transport Protocol": "TCP", "Application Protocol": "TLS",
           "Length": largo,
           "Src Port": 443 if entrante else 51000, "Dst Port": 51000 if entrante else 443,
           "TCP ACK": 1, "TCP RTT": max(rtt, 0.005) if rng.random() < 0.3 else None,
           "TCP Retransmission": int(retrans), "TCP Dup ACK": int(dup), "TCP Zero Window": int(zero)},
    )


def generar():
    filas = []
    t = 0.0
    while t < DURACION:
        escenario = escenario_en(t)
        tasa = {"Saturacion": 400, "Uso intenso": 350}.get(escenario, 40)  # paquetes por segundo
        t += rng.exponential(1 / tasa)

        if rng.random() < 0.03:
            filas.append(paquete(t, **{
                "Source IP": "192.168.0.10", "Destination IP": "8.8.8.8",
                "Transport Protocol": "UDP", "Application Protocol": "DNS",
                "Hostname DNS": rng.choice(["unne.edu.ar", "google.com", "youtube.com"]),
                "Length": 80, "Src Port": 53000, "Dst Port": 53}))
        else:
            filas.append(trafico_tcp(t, escenario))

        if escenario == "Escaneo" and rng.random() < 0.5:
            filas.append(paquete(t, **{
                "Source IP": "192.168.0.25", "Destination IP": "192.168.0.50",
                "Transport Protocol": "TCP", "Application Protocol": "TCP", "Length": 58,
                "Src Port": 60000, "Dst Port": int(rng.integers(1, 1025)), "TCP SYN": 1}))

    paquetes = pd.DataFrame(filas, columns=COLUMNAS_PAQUETES)
    paquetes.to_csv(os.path.join(CARPETA, "ejemplo_paquetes.csv"), index=False)

    intervalos = pd.DataFrame([
        {"captura": "ejemplo_paquetes.csv",
         "inicio": datetime.fromtimestamp(INICIO + desde).strftime("%Y-%m-%d %H:%M:%S"),
         "fin": datetime.fromtimestamp(INICIO + hasta).strftime("%Y-%m-%d %H:%M:%S"),
         "escenario": nombre}
        for nombre, desde, hasta in ESCENARIOS
    ])
    intervalos.to_csv(os.path.join(CARPETA, "ejemplo_intervalos.csv"), index=False)
    print(f"{len(paquetes)} paquetes y {len(intervalos)} intervalos generados en {CARPETA}")


if __name__ == "__main__":
    generar()
