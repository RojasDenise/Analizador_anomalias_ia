"""Base común: agrupa el CSV de paquetes en ventanas de tiempo y etiqueta cada ventana.

Lo usan todos los módulos (C1, C2, B2 y E). Ver contrato.md.
"""
import unicodedata

import numpy as np
import pandas as pd


SEGUNDOS_POR_DEFECTO = 5

# Escenarios que cuentan como "degradada" en la verdad de referencia.
# Los demás (uso intenso, escaneo) se etiquetan pero NO son degradación.
ESCENARIOS_DEGRADADOS = ("red lenta", "saturacion")

# Formato de alertas que devuelve cada módulo
COLUMNAS_ALERTAS = ["inicio_ventana", "modulo", "etiqueta", "detalle"]

ETIQUETAS = (
    "Degradación de rendimiento",
    "Congestión o saturación",
    "Pico de uso normal",
    "Comportamiento inusual",
)


# ----------------------------------------------------------------------
# Paquetes y ventanas
# ----------------------------------------------------------------------
def cargar_paquetes(ruta_csv):
    """Lee el CSV generado por analizador.py."""
    return pd.read_csv(ruta_csv)


def asignar_ventana(df_paquetes, segundos=SEGUNDOS_POR_DEFECTO):
    """Agrega la columna inicio_ventana (datetime, hora local) a cada paquete."""
    df = df_paquetes.copy()
    inicio = np.floor(df["Timestamp"].astype(float) / segundos) * segundos
    df["inicio_ventana"] = _a_hora_local(inicio)
    return df


def tabla_ventanas(df_paquetes, segundos=SEGUNDOS_POR_DEFECTO):
    """Una fila por ventana, sin huecos (las ventanas sin tráfico quedan en 0).

    Columnas base: paquetes, bytes, duracion_s.
    Variables de contexto (prefijo ctx_):
      ctx_max_puertos_por_ip: máximo de puertos destino distintos tocados por una misma IP origen.
      ctx_syn_sin_respuesta:  SYN enviados que nunca recibieron SYN-ACK.
      ctx_consultas_dns:      paquetes DNS con nombre consultado.
    Cada módulo agrega sus propias columnas con su prefijo (c1_, c2_, b2_).
    """
    df = asignar_ventana(df_paquetes, segundos)

    grupos = df.groupby("inicio_ventana")
    tabla = pd.DataFrame({
        "paquetes": grupos.size(),
        "bytes": grupos["Length"].sum(),
    })

    # Puertos destino distintos por IP origen (el máximo en la ventana)
    con_puerto = df.dropna(subset=["Source IP", "Dst Port"])
    puertos = (con_puerto.groupby(["inicio_ventana", "Source IP"])["Dst Port"].nunique()
               .groupby(level="inicio_ventana").max())
    tabla["ctx_max_puertos_por_ip"] = puertos

    # SYN sin respuesta
    df["_syn_sin_respuesta"] = _marcar_syn_sin_respuesta(df)
    tabla["ctx_syn_sin_respuesta"] = df.groupby("inicio_ventana")["_syn_sin_respuesta"].sum()

    # Consultas DNS
    tabla["ctx_consultas_dns"] = df["Hostname DNS"].notna().groupby(df["inicio_ventana"]).sum()

    # Rellenar ventanas vacías para que la serie de tiempo sea continua
    if not tabla.empty:
        rango = pd.date_range(tabla.index.min(), tabla.index.max(), freq=f"{segundos}s")
        tabla = tabla.reindex(rango)
    tabla = tabla.fillna(0)
    tabla.index.name = "inicio_ventana"
    tabla["duracion_s"] = segundos

    enteras = ["paquetes", "bytes", "ctx_max_puertos_por_ip", "ctx_syn_sin_respuesta", "ctx_consultas_dns"]
    tabla[enteras] = tabla[enteras].astype(int)
    return tabla


def _marcar_syn_sin_respuesta(df):
    """1 para cada SYN inicial cuyo SYN-ACK nunca aparece en la captura."""
    tcp = df["Transport Protocol"] == "TCP"
    es_syn = tcp & (df["TCP SYN"] == 1) & (df["TCP ACK"] == 0)
    es_synack = tcp & (df["TCP SYN"] == 1) & (df["TCP ACK"] == 1)

    # Una conexión respondida: el SYN-ACK va del destino al origen, con los puertos invertidos
    respondidas = set(zip(
        df.loc[es_synack, "Destination IP"], df.loc[es_synack, "Source IP"],
        df.loc[es_synack, "Dst Port"], df.loc[es_synack, "Src Port"],
    ))
    claves = list(zip(df["Source IP"], df["Destination IP"], df["Src Port"], df["Dst Port"]))
    sin_respuesta = [clave not in respondidas for clave in claves]
    return (es_syn & pd.Series(sin_respuesta, index=df.index)).astype(int)


def _a_hora_local(segundos_epoch):
    """Convierte segundos desde epoch a datetime en la hora local de la compu (sin zona)."""
    return pd.to_datetime(segundos_epoch, unit="s", utc=True).dt.tz_convert(_zona_local()).dt.tz_localize(None)


def _zona_local():
    from datetime import datetime
    return datetime.now().astimezone().tzinfo


# ----------------------------------------------------------------------
# Etiquetado (verdad de referencia)
# ----------------------------------------------------------------------
def leer_intervalos(ruta_csv):
    """Lee la planilla de tiempos anotados.

    Columnas: captura, inicio, fin, escenario.
    inicio y fin en hora local: "2026-11-05 17:06:30" o solo "17:06:30"
    (en ese caso se completa con la fecha de la captura al etiquetar).
    """
    intervalos = pd.read_csv(ruta_csv, dtype=str)
    intervalos["escenario"] = intervalos["escenario"].map(_normalizar)
    return intervalos


def etiquetar_ventanas(tabla, intervalos, captura=None, escenarios_degradados=ESCENARIOS_DEGRADADOS):
    """Agrega a la tabla de ventanas:
      escenario_real: nombre del escenario anotado o "normal".
      degradada: True solo si el escenario está en escenarios_degradados.

    Una ventana pertenece a un escenario si se superpone con su intervalo.
    Si se pasa captura, solo se usan los intervalos de ese archivo.
    """
    tabla = tabla.copy()
    tabla["escenario_real"] = "normal"

    if captura is not None and "captura" in intervalos.columns:
        intervalos = intervalos[intervalos["captura"] == captura]

    if not tabla.empty:
        fecha = tabla.index.min().strftime("%Y-%m-%d")
        duracion = pd.to_timedelta(tabla["duracion_s"], unit="s")
        fin_ventana = tabla.index + duracion

        for _, fila in intervalos.iterrows():
            inicio = _a_fecha(fila["inicio"], fecha)
            fin = _a_fecha(fila["fin"], fecha)
            dentro = (tabla.index < fin) & (fin_ventana > inicio)
            tabla.loc[dentro, "escenario_real"] = fila["escenario"]

    degradados = {_normalizar(e) for e in escenarios_degradados}
    tabla["degradada"] = tabla["escenario_real"].isin(degradados)
    return tabla


def _a_fecha(valor, fecha_por_defecto):
    valor = str(valor).strip()
    if len(valor) <= 8:  # solo hora, por ejemplo "17:06:30"
        valor = f"{fecha_por_defecto} {valor}"
    return pd.Timestamp(valor)


def _normalizar(texto):
    """'Saturación' -> 'saturacion', ' Red Lenta ' -> 'red lenta'."""
    texto = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    return " ".join(texto.lower().split())


# ----------------------------------------------------------------------
# Alertas
# ----------------------------------------------------------------------
def alertas_vacias():
    return pd.DataFrame(columns=COLUMNAS_ALERTAS)


def validar_alertas(alertas):
    """Verifica que un módulo devuelva sus alertas con el formato acordado."""
    faltan = [c for c in COLUMNAS_ALERTAS if c not in alertas.columns]
    if faltan:
        raise ValueError(f"Faltan columnas en las alertas: {faltan}")
    desconocidas = set(alertas["etiqueta"]) - set(ETIQUETAS)
    if desconocidas:
        raise ValueError(f"Etiquetas fuera del contrato: {desconocidas}")
    return True
