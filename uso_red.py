"""Módulo C2: ancho de banda y uso de la red.

Parte de la tabla de ventanas de ventanas.py y le agrega columnas con prefijo c2_:
c2_mbps        megabits por segundo
c2_pps         paquetes por segundo
c2_tam_medio   tamaño medio de paquete (bytes)
c2_pct_tcp, c2_pct_udp, c2_pct_dns, c2_pct_otros   % de paquetes por protocolo (suman 100)

Además: top talkers por bytes, reparto de protocolos y reglas de línea base
que devuelven alertas con el formato del contrato (ver contrato.md).
"""
import numpy as np
import pandas as pd

from ventanas import COLUMNAS_ALERTAS, SEGUNDOS_POR_DEFECTO, alertas_vacias, asignar_ventana

MODULO = "c2"
CATEGORIAS = ("tcp", "udp", "dns", "otros")
COLUMNAS_PCT = [f"c2_pct_{c}" for c in CATEGORIAS]
COLUMNAS_C2 = ["c2_mbps", "c2_pps", "c2_tam_medio"] + COLUMNAS_PCT


# ----------------------------------------------------------------------
# metricas por ventana estandarizadas y reparto de protocolos
# ----------------------------------------------------------------------
def _categoria(df):
    """Clasifica cada paquete en una sola categoría: dns, tcp, udp u otros.

    DNS tiene prioridad (aunque viaje sobre UDP o TCP), así los porcentajes no se solapan.
    """
    app = df["Application Protocol"].astype(str).str.upper()
    es_dns = (app == "DNS") | (df["Src Port"] == 53) | (df["Dst Port"] == 53)
    transporte = df["Transport Protocol"]
    return np.select(
        [es_dns, transporte == "TCP", transporte == "UDP"],
        ["dns", "tcp", "udp"],
        default="otros",
    )


def reparto_protocolos(df_paquetes, segundos=SEGUNDOS_POR_DEFECTO):
    """% de paquetes por protocolo en cada ventana (columnas c2_pct_*)."""
    df = asignar_ventana(df_paquetes, segundos)
    df["_cat"] = _categoria(df)
    conteo = pd.crosstab(df["inicio_ventana"], df["_cat"]).reindex(columns=CATEGORIAS, fill_value=0)
    pct = conteo.div(conteo.sum(axis=1), axis=0) * 100
    pct.columns = COLUMNAS_PCT
    return pct


def calcular_metricas(df_paquetes, tabla, segundos=None):
    """Devuelve una copia de la tabla de ventanas con las columnas c2_*.

    Las ventanas sin tráfico quedan en 0.
    """
    if segundos is None:
        segundos = int(tabla["duracion_s"].iloc[0]) if not tabla.empty else SEGUNDOS_POR_DEFECTO

    t = tabla.copy()
    dur = t["duracion_s"]
    t["c2_mbps"] = t["bytes"] * 8 / dur / 1e6
    t["c2_pps"] = t["paquetes"] / dur
    t["c2_tam_medio"] = (t["bytes"] / t["paquetes"].replace(0, np.nan)).fillna(0)

    reparto = reparto_protocolos(df_paquetes, segundos).reindex(t.index).fillna(0)
    t[COLUMNAS_PCT] = reparto[COLUMNAS_PCT]
    return t


# ----------------------------------------------------------------------
#top talkers son las IPs que mas bytes mueven, como origen o destino.
# ----------------------------------------------------------------------
def top_talkers(df_paquetes, n=5, segundos=None, por="ambas"):
    """IPs que más bytes mueven.

    por: "ambas" (IP como origen o destino), "origen" o "destino".
    segundos=None -> ranking de toda la captura (columnas: ip, bytes).
    segundos=5    -> top n de cada ventana (columnas: inicio_ventana, ip, bytes).
    """
    columnas = {"ambas": ["Source IP", "Destination IP"],
                "origen": ["Source IP"],
                "destino": ["Destination IP"]}
    if por not in columnas:
        raise ValueError(f"'por' debe ser uno de {list(columnas)}")

    df = df_paquetes if segundos is None else asignar_ventana(df_paquetes, segundos)
    clave = [] if segundos is None else ["inicio_ventana"]

    partes = []
    for col in columnas[por]:
        p = df[clave + [col, "Length"]].rename(columns={col: "ip"})
        partes.append(p.dropna(subset=["ip", "Length"]))
    todo = pd.concat(partes, ignore_index=True)

    if todo.empty:
        return pd.DataFrame(columns=clave + ["ip", "bytes"])

    suma = todo.groupby(clave + ["ip"])["Length"].sum().rename("bytes")
    if segundos is None:
        return suma.nlargest(n).reset_index()
    top = suma.groupby(level="inicio_ventana", group_keys=False).nlargest(n)
    return top.reset_index()


# ----------------------------------------------------------------------
# reglas de línea base para alertas de uso de la red
# ----------------------------------------------------------------------
def mbps_referencia(tablas_normales):
    """Mbps medio de las capturas normales. Se usa como base de la regla de pico.

    tablas_normales: una tabla o una lista de tablas ya procesadas con calcular_metricas.
    """
    if isinstance(tablas_normales, pd.DataFrame):
        tablas_normales = [tablas_normales]
    return float(pd.concat([t["c2_mbps"] for t in tablas_normales]).mean())


def reglas_c2(tabla, capacidad_mbps=None, uso_max=0.8, factor_pico=3.0, mbps_normal=None):
    """Reglas fijas de C2 (línea base para comparar con la IA).

    Regla 1, uso alto: Mbps > uso_max * capacidad_mbps (se omite si no se da la capacidad).
    Regla 2, pico:     Mbps > factor_pico * mbps_normal
                    (si no se pasa mbps_normal, usa el promedio de la propia tabla).

    C2 mide volumen, no calidad: no sabe si hay degradación TCP. Por eso toda alerta sale como
    "Pico de uso normal"; la etiqueta "Congestión o saturación" se decide cruzando con C1 (B2).
    """
    if "c2_mbps" not in tabla.columns:
        raise ValueError("Falta c2_mbps: ejecutá calcular_metricas() primero.")

    mbps = tabla["c2_mbps"]
    base = mbps_normal if mbps_normal is not None else float(mbps.mean())

    uso_alto = pd.Series(False, index=tabla.index)
    if capacidad_mbps is not None:
        uso_alto = mbps > uso_max * capacidad_mbps
    pico = (mbps > factor_pico * base) if base > 0 else pd.Series(False, index=tabla.index)

    filas = []
    for ventana in tabla.index[uso_alto | pico]:
        motivos = []
        if uso_alto[ventana]:
            motivos.append(f"uso {mbps[ventana]:.2f} Mbps > {uso_max:.0%} de {capacidad_mbps} Mbps")
        if pico[ventana]:
            motivos.append(f"pico {mbps[ventana]:.2f} Mbps > {factor_pico:g} x {base:.2f} Mbps (normal)")
        filas.append({
            "inicio_ventana": ventana,
            "modulo": MODULO,
            "etiqueta": "Pico de uso normal",
            "detalle": "; ".join(motivos),
        })

    if not filas:
        return alertas_vacias()
    return pd.DataFrame(filas, columns=COLUMNAS_ALERTAS)