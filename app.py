"""PySharkStream Analytica: rendimiento de red con IA (front en Streamlit).

Pantallas:
  1 · Rendimiento y Anomalías  (C1, C2, reglas fijas y modelos de IA de B2)
  2 · Gráficos                 (tráfico en el tiempo, IPs, protocolos y destinos)

Uso:  streamlit run app.py
"""
import glob
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import detector_ia as b2
import rendimiento_tcp as c1
import uso_red as c2
import ventanas as v
from analizador import AnalizadorDeRed

CARPETA = os.path.dirname(os.path.abspath(__file__))
CARPETA_CAPTURAS = os.path.join(CARPETA, "capturas")
EJEMPLO_PAQUETES = os.path.join(CARPETA, "tests", "ejemplo_paquetes.csv")
EJEMPLO_TIEMPOS = os.path.join(CARPETA, "tests", "ejemplo_intervalos.csv")

# Un color por causa, igual en toda la app
COLORES_CAUSA = {
    "Degradación de rendimiento": "#F28E2B",
    "Congestión o saturación": "#E15759",
    "Pico de uso normal": "#4E79A7",
    "Comportamiento inusual": "#EDC948",
}
ESCALA_CAUSA = alt.Scale(domain=list(COLORES_CAUSA), range=list(COLORES_CAUSA.values()))

st.set_page_config(page_title="PySharkStream Analytica", page_icon="📡", layout="wide")


# ----------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _leer_csv(origen):
    """Lee un CSV de paquetes desde una ruta o un archivo subido."""
    return pd.read_csv(origen)


@st.cache_data(show_spinner="Generando datos de ejemplo...")
def _datos_ejemplo():
    if not os.path.exists(EJEMPLO_PAQUETES):
        import sys
        sys.path.insert(0, os.path.join(CARPETA, "tests"))
        from generar_ejemplo import generar
        generar()
    return pd.read_csv(EJEMPLO_PAQUETES), v.leer_intervalos(EJEMPLO_TIEMPOS)


def _tabla(paquetes, segundos, tiempos=None, recortar_bordes=True):
    """Ventanas con métricas de C1 y C2 (igual que experimentos.procesar)."""
    tabla = v.tabla_ventanas(paquetes, segundos)
    if recortar_bordes and len(tabla) > 2:
        tabla = tabla.iloc[1:-1]   # la primera y la última ventana quedan incompletas
    if tiempos is not None:
        tabla = v.etiquetar_ventanas(tabla, tiempos)
    tabla = c1.calcular_metricas(paquetes, tabla, segundos)
    return c2.calcular_metricas(paquetes, tabla, segundos)


@st.cache_data(show_spinner="Calculando métricas, reglas y modelos de IA...")
def analizar(paquetes, normales, tiempos, segundos, contaminacion, capacidad_mbps):
    """Todo el análisis de la pantalla 1. Devuelve un diccionario con tablas y alertas."""
    tabla = _tabla(paquetes, segundos, tiempos)
    tablas_normales = [_tabla(n, segundos) for n in normales]

    umbral = c1.umbral_rtt(tablas_normales)
    mbps_normal = c2.mbps_referencia(tablas_normales)
    reglas = pd.concat([
        c1.reglas_c1(tabla, umbral),
        c2.reglas_c2(tabla, capacidad_mbps=capacidad_mbps, mbps_normal=mbps_normal),
    ], ignore_index=True)

    modelo = b2.entrenar(tablas_normales, contaminacion=contaminacion)
    puntuada = b2.puntuar(modelo, tabla)
    return {
        "tabla": puntuada,
        "umbral_rtt": umbral,
        "mbps_normal": mbps_normal,
        "reglas": reglas,
        "alertas_ia": b2.generar_alertas(puntuada),
        "ventanas_normales": sum(len(t) for t in tablas_normales),
    }


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
def sidebar():
    st.sidebar.header("Configuración")
    fuente = st.sidebar.radio("Datos", ["Datos de ejemplo", "Subir captura"], horizontal=True)

    tiempos = None
    if fuente == "Datos de ejemplo":
        paquetes, tiempos = _datos_ejemplo()
        # Referencia normal: los primeros 3 minutos, antes del primer evento
        corte = paquetes["Timestamp"].min() + 180
        normales = [paquetes[paquetes["Timestamp"] < corte]]
        nombre = "ejemplo_paquetes.csv"
        st.sidebar.caption("15 minutos simulados con red lenta, saturación, uso intenso y escaneo. "
                           "Los primeros 3 minutos se usan como tráfico normal.")
    else:
        archivo = st.sidebar.file_uploader("Captura a analizar (.csv)", type=["csv"],
                                           help="Convertí el .pcapng con: python convertir_capturas.py capturas")
        if archivo is None:
            return None
        paquetes, nombre = _leer_csv(archivo), archivo.name

        disponibles = sorted(glob.glob(os.path.join(CARPETA_CAPTURAS, "normal_*.csv")))
        elegidas = st.sidebar.multiselect("Capturas normales de referencia", disponibles,
                                          default=disponibles, format_func=os.path.basename)
        subidas = st.sidebar.file_uploader("…o subí capturas normales (.csv)", type=["csv"],
                                           accept_multiple_files=True)
        normales = [_leer_csv(r) for r in elegidas] + [_leer_csv(f) for f in subidas or []]

        archivo_tiempos = st.sidebar.file_uploader("Planilla de tiempos (opcional)", type=["csv"])
        if archivo_tiempos is not None:
            tiempos = v.leer_intervalos(archivo_tiempos)
            base = os.path.splitext(nombre)[0]
            tiempos = tiempos[tiempos["captura"].map(lambda c: os.path.splitext(str(c))[0]) == base]

    st.sidebar.divider()
    segundos = st.sidebar.select_slider("Tamaño de ventana (s)", [5, 10, 15], value=5)
    contaminacion = st.sidebar.slider("Sensibilidad de la IA (contaminación)", 0.005, 0.10, 0.01, 0.005,
                                      help="Proporción de ventanas raras que se esperan dentro del tráfico normal.")
    capacidad = st.sidebar.number_input("Capacidad del enlace (Mbps, 0 = sin dato)", min_value=0.0, value=0.0)

    return {
        "paquetes": paquetes, "nombre": nombre, "normales": normales, "tiempos": tiempos,
        "segundos": segundos, "contaminacion": contaminacion, "capacidad": capacidad or None,
    }


# ----------------------------------------------------------------------
# Pantalla 1 · Rendimiento y Anomalías
# ----------------------------------------------------------------------
def _kpis(r, modelo_ia):
    t = r["tabla"]
    alertas_ia = r["alertas_ia"][b2.MODELOS[modelo_ia]]
    col = st.columns(4)
    col[0].metric("RTT medio", f"{t['c1_rtt_medio_ms'].mean():.0f} ms",
                  help=f"Umbral de la regla: {r['umbral_rtt']:.0f} ms (media + 3 desvíos del tráfico normal)")
    col[1].metric("Retransmisiones", f"{t['c1_pct_retrans'].mean():.1f} %", help="Umbral de la regla: 5 %")
    col[2].metric("Ancho de banda medio", f"{t['c2_mbps'].mean():.2f} Mbps",
                  help=f"Tráfico normal: {r['mbps_normal']:.2f} Mbps")
    col[3].metric("Alertas", f"{len(r['reglas'])} reglas · {len(alertas_ia)} IA")


def _seccion_c2(r):
    t = r["tabla"]
    if "escenario_real" in t.columns and t["escenario_real"].nunique() > 1:
        c2.seccion_streamlit(c2.resumen_por_escenario(t))
        return
    st.subheader("Uso de la red (C2)")
    datos = t[["c2_mbps"]].reset_index()
    linea = alt.Chart(datos).mark_area(opacity=0.5, line=True).encode(
        x=alt.X("inicio_ventana:T", title="Hora"), y=alt.Y("c2_mbps:Q", title="Mbps"),
        tooltip=[alt.Tooltip("inicio_ventana:T", format="%H:%M:%S"), alt.Tooltip("c2_mbps:Q", format=".2f")])
    regla = alt.Chart(pd.DataFrame({"y": [3 * r["mbps_normal"]]})).mark_rule(
        color="red", strokeDash=[6, 4]).encode(y="y:Q")
    st.altair_chart(linea + regla, use_container_width=True)
    st.caption("Línea roja: regla de pico (3 veces el Mbps normal). "
               "Cargá la planilla de tiempos para ver la comparación por escenario.")


def _seccion_ia(r, clave):
    st.subheader("Anomalías detectadas por la IA (B2)")
    t = r["tabla"]
    datos = t[[f"b2_{clave}_puntaje", f"b2_{clave}_anomala", "b2_causa", "b2_explicacion"]].reset_index()
    datos.columns = ["inicio_ventana", "puntaje", "anomala", "causa", "explicacion"]

    linea = alt.Chart(datos).mark_line(color="#8B95A7").encode(
        x=alt.X("inicio_ventana:T", title="Hora"), y=alt.Y("puntaje:Q", title="Puntaje de anomalía"))
    puntos = alt.Chart(datos[datos["anomala"]]).mark_circle(size=80).encode(
        x="inicio_ventana:T", y="puntaje:Q",
        color=alt.Color("causa:N", scale=ESCALA_CAUSA, title="Causa probable"),
        tooltip=[alt.Tooltip("inicio_ventana:T", title="Ventana", format="%H:%M:%S"),
                 alt.Tooltip("causa:N", title="Causa"), alt.Tooltip("puntaje:Q", format=".3f"),
                 alt.Tooltip("explicacion:N", title="Métricas")])
    st.altair_chart(linea + puntos, use_container_width=True)

    alertas = r["alertas_ia"][b2.MODELOS[clave]]
    if alertas.empty:
        st.success("El modelo no encontró ventanas anómalas.")
        return

    resumen = alertas["etiqueta"].value_counts()
    cols = st.columns(len(COLORES_CAUSA))
    for col, causa in zip(cols, COLORES_CAUSA):
        col.markdown(f"<span style='color:{COLORES_CAUSA[causa]}'>●</span> **{resumen.get(causa, 0)}** "
                     f"{causa.lower()}", unsafe_allow_html=True)

    tabla = alertas.assign(hora=pd.to_datetime(alertas["inicio_ventana"]).dt.strftime("%H:%M:%S"))
    tabla = tabla[["hora", "etiqueta", "detalle"]].rename(
        columns={"hora": "Hora", "etiqueta": "Causa probable", "detalle": "Puntaje y métricas que la dispararon"})
    st.dataframe(tabla.style.map(lambda c: f"color: {COLORES_CAUSA.get(c, 'inherit')}",
                                 subset=["Causa probable"]),
                 use_container_width=True, hide_index=True)

    with st.expander("Detalle de una alerta: cuánto se alejó cada métrica de lo normal"):
        hora = st.selectbox("Ventana", tabla["Hora"])
        fila = alertas.iloc[list(tabla["Hora"]).index(hora)]
        z = r["tabla"].loc[pd.to_datetime(fila["inicio_ventana"]), "b2_explicacion"]
        st.write(f"**{fila['etiqueta']}**. {z}")


def pantalla_rendimiento(cfg):
    if not cfg["normales"]:
        st.info("Para entrenar la IA y fijar los umbrales hacen falta capturas normales: "
                "ponelas en la carpeta capturas/ (normal_1.csv, …) o subilas en la barra lateral.")
        return

    try:
        r = analizar(cfg["paquetes"], cfg["normales"], cfg["tiempos"], cfg["segundos"],
                     cfg["contaminacion"], cfg["capacidad"])
    except ValueError as error:
        st.error(f"No se pudo analizar: {error}")
        return

    nombre = st.radio("Modelo de IA", list(b2.MODELOS.values()), horizontal=True)
    modelo = {v_: k for k, v_ in b2.MODELOS.items()}[nombre]
    _kpis(r, modelo)
    st.caption(f"{cfg['nombre']} · {len(r['tabla'])} ventanas de {cfg['segundos']} s · "
               f"IA entrenada con {r['ventanas_normales']} ventanas normales")
    st.divider()

    izquierda, derecha = st.columns(2)
    with izquierda:
        c1.seccion_streamlit(r["tabla"], r["umbral_rtt"])
    with derecha:
        _seccion_c2(r)

    st.divider()
    _seccion_ia(r, modelo)

    with st.expander(f"Alertas de las reglas fijas ({len(r['reglas'])})"):
        reglas = r["reglas"].assign(Hora=pd.to_datetime(r["reglas"]["inicio_ventana"]).dt.strftime("%H:%M:%S"))
        st.dataframe(reglas[["Hora", "modulo", "etiqueta", "detalle"]], use_container_width=True, hide_index=True)

    descarga = pd.concat([r["reglas"].assign(metodo="Reglas fijas")] +
                         [a.assign(metodo=m) for m, a in r["alertas_ia"].items()], ignore_index=True)
    st.download_button("Exportar alertas a CSV", descarga.to_csv(index=False).encode("utf-8"),
                       file_name="alertas.csv", mime="text/csv")


# ----------------------------------------------------------------------
# Pantalla 2 · Gráficos
# ----------------------------------------------------------------------
def pantalla_graficos(cfg):
    paquetes = cfg["paquetes"]
    tabla = _tabla(paquetes, cfg["segundos"], recortar_bordes=False)

    st.subheader("Tráfico en el tiempo")
    serie = tabla[["c2_mbps", "c2_pps"]].reset_index().rename(columns={"c2_mbps": "Mbps", "c2_pps": "Paquetes/s"})
    largo = serie.melt("inicio_ventana", var_name="medida", value_name="valor")
    st.altair_chart(alt.Chart(largo).mark_line().encode(
        x=alt.X("inicio_ventana:T", title="Hora"), y=alt.Y("valor:Q", title=None),
        color=alt.Color("medida:N", title=None), row=alt.Row("medida:N", title=None),
        tooltip=[alt.Tooltip("inicio_ventana:T", format="%H:%M:%S"), "medida:N",
                 alt.Tooltip("valor:Q", format=".2f")],
    ).properties(height=160).resolve_scale(y="independent"), use_container_width=True)

    izquierda, derecha = st.columns(2)
    with izquierda:
        st.subheader("IPs que más tráfico mueven")
        top = c2.top_talkers(paquetes, n=10)
        top["MB"] = top["bytes"] / 1e6
        st.altair_chart(alt.Chart(top).mark_bar().encode(
            x=alt.X("MB:Q", title="MB enviados y recibidos"), y=alt.Y("ip:N", sort="-x", title=None),
            tooltip=["ip", alt.Tooltip("MB:Q", format=".2f")]), use_container_width=True)
    with derecha:
        st.subheader("Reparto por protocolo")
        reparto = c2.reparto_protocolos(paquetes, cfg["segundos"]).mean().rename("pct").reset_index()
        reparto["protocolo"] = reparto["index"].str.replace("c2_pct_", "").str.upper()
        st.altair_chart(alt.Chart(reparto).mark_arc(innerRadius=60).encode(
            theta="pct:Q", color=alt.Color("protocolo:N", title="Protocolo"),
            tooltip=["protocolo", alt.Tooltip("pct:Q", title="% de paquetes", format=".1f")]),
            use_container_width=True)

    st.subheader("Destinos de una IP de origen")
    origenes = paquetes["Source IP"].dropna().value_counts()
    ip = st.selectbox("IP de origen", origenes.index, format_func=lambda x: f"{x} ({origenes[x]} paquetes)")
    destinos = (paquetes[paquetes["Source IP"] == ip]
                .groupby(["Destination IP", "Transport Protocol"], dropna=False).size()
                .rename("paquetes").reset_index().fillna({"Transport Protocol": "otro"}))
    st.altair_chart(alt.Chart(destinos).mark_bar().encode(
        x=alt.X("paquetes:Q"), y=alt.Y("Destination IP:N", sort="-x", title="IP de destino"),
        color=alt.Color("Transport Protocol:N", title="Transporte"),
        tooltip=["Destination IP", "Transport Protocol", "paquetes"]), use_container_width=True)

    with st.expander("Nombre de dominio de una IP"):
        consulta = st.text_input("IP", value="8.8.8.8")
        if st.button("Consultar"):
            st.write(AnalizadorDeRed().get_domain_name(consulta))


# ----------------------------------------------------------------------
# Página
# ----------------------------------------------------------------------
st.title("PySharkStream Analytica")
st.caption("Rendimiento de red con IA · Redes de Datos · FaCENA-UNNE")

cfg = sidebar()
pestanas = st.tabs(["Rendimiento y Anomalías", "Gráficos"])

if cfg is None:
    for p in pestanas:
        with p:
            st.info("Subí una captura en la barra lateral o elegí «Datos de ejemplo».")
else:
    with pestanas[0]:
        pantalla_rendimiento(cfg)
    with pestanas[1]:
        pantalla_graficos(cfg)
