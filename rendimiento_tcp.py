"""Módulo C1: latencia y pérdida en conexiones TCP.

Responde si la red responde rápido y sin perder paquetes.

Parte de la tabla de ventanas de ventanas.py y le agrega columnas con prefijo c1_:
c1_paquetes_tcp   paquetes TCP en la ventana (denominador de los porcentajes)
c1_rtt_medio_ms   RTT medio (tcp.analysis.ack_rtt) en milisegundos. NaN si no hubo muestras
c1_rtt_max_ms     RTT máximo en milisegundos. NaN si no hubo muestras
c1_pct_retrans    % de paquetes TCP que son retransmisiones
c1_pct_dup_ack    % de paquetes TCP que son ACK duplicados
c1_ventanas_cero  cantidad de paquetes que anuncian ventana de recepción en cero

Además: umbral de RTT calculado con las capturas normales y reglas de línea base
que devuelven alertas con el formato del contrato (ver contrato.md).

Decisiones (ver contrato.md, sección 7):
- Los porcentajes se calculan sobre los paquetes TCP de la ventana, no sobre todos:
  retransmisiones y ACK duplicados solo existen en TCP, y el tráfico UDP/DNS los diluiría.
- El RTT queda en NaN cuando la ventana no tiene muestras: 0 ms se leería como "latencia perfecta".
- Umbral de RTT = media + 3 desvíos del RTT medio de las ventanas normales
  (regla 3-sigma, Chandola et al., 2009). Equivale a un z-score > 3, el mismo criterio
  con el que B2 explica sus alertas.
- La regla de retransmisiones solo se aplica con al menos 20 paquetes TCP en la ventana:
  así una sola retransmisión (1/20 = 5 %) nunca alcanza para superar el 5 %.
"""
import numpy as np
import pandas as pd

from ventanas import COLUMNAS_ALERTAS, SEGUNDOS_POR_DEFECTO, alertas_vacias, asignar_ventana

MODULO = "c1"
ETIQUETA = "Degradación de rendimiento"

MAX_PCT_RETRANS = 5.0     # regla de la propuesta: retransmisiones mayores al 5 %
MIN_PAQUETES_TCP = 20     # mínimo de paquetes TCP por ventana para aplicar la regla de %
K_SIGMA = 3.0             # umbral de RTT = media + K_SIGMA * desvío (captura normal)

COLUMNAS_C1 = [
    "c1_paquetes_tcp",
    "c1_rtt_medio_ms",
    "c1_rtt_max_ms",
    "c1_pct_retrans",
    "c1_pct_dup_ack",
    "c1_ventanas_cero",
]

_FLAGS = ["TCP Retransmission", "TCP Dup ACK", "TCP Zero Window"]


# ----------------------------------------------------------------------
# Métricas por ventana
# ----------------------------------------------------------------------
def _paquetes_tcp(df_paquetes, segundos):
    """Solo los paquetes TCP, con inicio_ventana y los campos numéricos limpios."""
    df = asignar_ventana(df_paquetes, segundos)
    df = df[df["Transport Protocol"] == "TCP"].copy()
    df["TCP RTT"] = pd.to_numeric(df["TCP RTT"], errors="coerce")
    for col in _FLAGS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df


def calcular_metricas(df_paquetes, tabla, segundos=None):
    """Devuelve una copia de la tabla de ventanas con las columnas c1_*.

    Ventanas sin paquetes TCP: conteos y porcentajes en 0, RTT en NaN.
    """
    if segundos is None:
        segundos = int(tabla["duracion_s"].iloc[0]) if not tabla.empty else SEGUNDOS_POR_DEFECTO

    tcp = _paquetes_tcp(df_paquetes, segundos)
    grupos = tcp.groupby("inicio_ventana")

    metricas = pd.DataFrame({
        "c1_paquetes_tcp": grupos.size(),
        "c1_rtt_medio_ms": grupos["TCP RTT"].mean() * 1000,
        "c1_rtt_max_ms": grupos["TCP RTT"].max() * 1000,
        "_retrans": grupos["TCP Retransmission"].sum(),
        "_dup_ack": grupos["TCP Dup ACK"].sum(),
        "c1_ventanas_cero": grupos["TCP Zero Window"].sum(),
    }).reindex(tabla.index)

    t = tabla.copy()
    n_tcp = metricas["c1_paquetes_tcp"].fillna(0)
    con_tcp = n_tcp.replace(0, np.nan)

    t["c1_paquetes_tcp"] = n_tcp.astype(int)
    t["c1_rtt_medio_ms"] = metricas["c1_rtt_medio_ms"]          # NaN si no hay muestras
    t["c1_rtt_max_ms"] = metricas["c1_rtt_max_ms"]              # NaN si no hay muestras
    t["c1_pct_retrans"] = (metricas["_retrans"] / con_tcp * 100).fillna(0)
    t["c1_pct_dup_ack"] = (metricas["_dup_ack"] / con_tcp * 100).fillna(0)
    t["c1_ventanas_cero"] = metricas["c1_ventanas_cero"].fillna(0).astype(int)
    return t


# ----------------------------------------------------------------------
# Umbral de RTT (línea base)
# ----------------------------------------------------------------------
def umbral_rtt(tablas_normales, k=K_SIGMA):
    """Umbral de RTT en ms: media + k desvíos del RTT medio de las ventanas normales.

    tablas_normales: una tabla o una lista de tablas ya procesadas con calcular_metricas.
    Las ventanas sin muestras de RTT (NaN) no entran en el cálculo.
    """
    if isinstance(tablas_normales, pd.DataFrame):
        tablas_normales = [tablas_normales]
    rtt = pd.concat([t["c1_rtt_medio_ms"] for t in tablas_normales]).dropna()
    if len(rtt) < 2:
        raise ValueError("Hacen falta al menos 2 ventanas con RTT en las capturas normales.")
    return float(rtt.mean() + k * rtt.std())


# ----------------------------------------------------------------------
# Reglas de línea base
# ----------------------------------------------------------------------
def reglas_c1(tabla, umbral_rtt_ms, max_pct_retrans=MAX_PCT_RETRANS, min_paquetes_tcp=MIN_PAQUETES_TCP):
    """Reglas fijas de C1 (línea base para comparar con la IA).

    Regla 1, retransmisiones: c1_pct_retrans > max_pct_retrans,
             solo en ventanas con al menos min_paquetes_tcp paquetes TCP.
    Regla 2, latencia:        c1_rtt_medio_ms > umbral_rtt_ms (las ventanas sin RTT no se marcan).

    Toda alerta sale como "Degradación de rendimiento".
    """
    faltan = [c for c in ("c1_paquetes_tcp", "c1_rtt_medio_ms", "c1_pct_retrans") if c not in tabla.columns]
    if faltan:
        raise ValueError(f"Faltan {faltan}: ejecutá calcular_metricas() primero.")

    pct = tabla["c1_pct_retrans"]
    rtt = tabla["c1_rtt_medio_ms"]
    n_tcp = tabla["c1_paquetes_tcp"]

    retrans_alto = (pct > max_pct_retrans) & (n_tcp >= min_paquetes_tcp)
    rtt_alto = rtt > umbral_rtt_ms  # NaN > x es False

    filas = []
    for ventana in tabla.index[retrans_alto | rtt_alto]:
        motivos = []
        if retrans_alto[ventana]:
            motivos.append(f"retransmisiones {pct[ventana]:.1f} % > {max_pct_retrans:g} % "
                           f"({int(n_tcp[ventana])} paquetes TCP)")
        if rtt_alto[ventana]:
            motivos.append(f"RTT medio {rtt[ventana]:.1f} ms > umbral {umbral_rtt_ms:.1f} ms")
        filas.append({
            "inicio_ventana": ventana,
            "modulo": MODULO,
            "etiqueta": ETIQUETA,
            "detalle": "; ".join(motivos),
        })

    if not filas:
        return alertas_vacias()
    return pd.DataFrame(filas, columns=COLUMNAS_ALERTAS)


# ----------------------------------------------------------------------
# Sección de Streamlit (la integración en app.py se hace después)
# ----------------------------------------------------------------------
def _grafico_con_umbral(tabla, columna, titulo_y, umbral, texto_umbral):
    import altair as alt

    datos = tabla[[columna]].reset_index()
    datos.columns = ["inicio_ventana", "valor"]
    linea = alt.Chart(datos).mark_line(point=True).encode(
        x=alt.X("inicio_ventana:T", title="Hora"),
        y=alt.Y("valor:Q", title=titulo_y),
        tooltip=[alt.Tooltip("inicio_ventana:T", title="Ventana", format="%H:%M:%S"),
                 alt.Tooltip("valor:Q", title=titulo_y, format=".1f")],
    )
    if umbral is None:
        return linea
    regla = alt.Chart(pd.DataFrame({"umbral": [umbral]})).mark_rule(
        color="red", strokeDash=[6, 4]).encode(y="umbral:Q")
    etiqueta = alt.Chart(pd.DataFrame({"umbral": [umbral], "texto": [texto_umbral]})).mark_text(
        color="red", align="left", dx=4, dy=-6).encode(y="umbral:Q", text="texto:N")
    return linea + regla + etiqueta


def seccion_streamlit(tabla, umbral_rtt_ms=None, max_pct_retrans=MAX_PCT_RETRANS):
    """Dibuja la sección "Latencia y pérdida (C1)": RTT y % de retransmisiones con su umbral."""
    import streamlit as st

    st.subheader("Latencia y pérdida (C1)")
    texto_rtt = f"umbral {umbral_rtt_ms:.1f} ms" if umbral_rtt_ms is not None else ""
    st.altair_chart(_grafico_con_umbral(tabla, "c1_rtt_medio_ms", "RTT medio (ms)",
                                        umbral_rtt_ms, texto_rtt), use_container_width=True)
    st.caption("Las ventanas sin muestras de RTT no se grafican.")
    st.altair_chart(_grafico_con_umbral(tabla, "c1_pct_retrans", "Retransmisiones (%)",
                                        max_pct_retrans, f"umbral {max_pct_retrans:g} %"),
                    use_container_width=True)
