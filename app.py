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
import experimentos as exp
import rendimiento_tcp as c1
import uso_red as c2
import ventanas as v
from analizador import AnalizadorDeRed

CARPETA = os.path.dirname(os.path.abspath(__file__))
CARPETA_CAPTURAS = os.path.join(CARPETA, "capturas")
RUTA_TIEMPOS = os.path.join(CARPETA_CAPTURAS, "tiempos.csv")
CARPETA_RESULTADOS = os.path.join(CARPETA, "resultados")

# Un color por método en la pestaña de Evaluación
COLORES_METODO = {
    "Reglas fijas": "#8A94A6",
    "Isolation Forest": "#4FC3F7",
    "LOF": "#76B7B2",
    "One-Class SVM": "#B07AA1",
}

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
@st.cache_data(show_spinner="Leyendo la captura (las de fibra tardan un poco)...", max_entries=12)
def _leer_ruta(ruta, _marca):
    """Lee un CSV de la carpeta. _marca (fecha de modificación) renueva la caché si el archivo cambia."""
    return pd.read_csv(ruta, low_memory=False)


def _leer_csv(origen):
    """Lee un CSV de paquetes desde una ruta de la carpeta o un archivo subido."""
    if isinstance(origen, str):
        return _leer_ruta(origen, os.path.getmtime(origen))
    return pd.read_csv(origen, low_memory=False)


def _base(ruta):
    return os.path.splitext(os.path.basename(str(ruta)))[0]


def _clave(origen):
    """Identifica una captura sin tener que comparar millones de filas."""
    if isinstance(origen, str):
        return f"{origen}|{os.path.getmtime(origen)}"
    return f"subida|{origen.name}|{origen.size}|{getattr(origen, 'file_id', '')}"


def _escenarios_de_la_carpeta():
    """CSV de capturas con evento: todo menos las normales, Starlink y la planilla de tiempos."""
    rutas = sorted(glob.glob(os.path.join(CARPETA_CAPTURAS, "*.csv")))
    return [r for r in rutas if not _base(r).startswith((exp.PREFIJO_NORMAL, exp.PREFIJO_STARLINK))
            and _base(r) != "tiempos"]


def _tiempos_de(base, tiempos):
    """Filas de la planilla que corresponden a una captura (por nombre, sin extensión)."""
    if tiempos is None:
        return None
    filas = tiempos[tiempos["captura"].map(_base) == base]
    return filas if len(filas) else None


def _tabla(paquetes, segundos, tiempos=None, recortar_bordes=True):
    """Ventanas con métricas de C1 y C2 (igual que experimentos.procesar)."""
    tabla = v.tabla_ventanas(paquetes, segundos)
    if recortar_bordes and len(tabla) > 2:
        tabla = tabla.iloc[1:-1]   # la primera y la última ventana quedan incompletas
    if tiempos is not None:
        tabla = v.etiquetar_ventanas(tabla, tiempos)
    tabla = c1.calcular_metricas(paquetes, tabla, segundos)
    return c2.calcular_metricas(paquetes, tabla, segundos)


@st.cache_data(show_spinner="Calculando métricas, reglas y modelos de IA...", max_entries=8)
def analizar(clave, _paquetes, _normales, tiempos, segundos, contaminacion, capacidad_mbps):
    """Todo el análisis de la pantalla 1. Devuelve un diccionario con tablas y alertas.

    clave identifica la captura y sus normales: así la caché no tiene que comparar los
    DataFrames enteros (las capturas de fibra tienen millones de filas).
    """
    paquetes, normales = _paquetes, _normales
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
    fuente = st.sidebar.radio("Datos", ["Capturas de la carpeta", "Subir captura"], horizontal=True)

    planilla = v.leer_intervalos(RUTA_TIEMPOS) if os.path.exists(RUTA_TIEMPOS) else None
    if fuente == "Capturas de la carpeta":
        escenarios = _escenarios_de_la_carpeta()
        if not escenarios:
            st.sidebar.warning("No hay capturas en la carpeta capturas. Convertí los .pcapng con:\n\n"
                               "`python convertir_capturas.py capturas`")
            return None
        origen = st.sidebar.selectbox("Captura a analizar", escenarios, format_func=_base)
        nombre = _base(origen)
        tiempos = _tiempos_de(nombre, planilla)
        if planilla is None:
            st.sidebar.caption("No está capturas/tiempos.csv: no se marca cuándo pasó el evento.")
        elif tiempos is None:
            st.sidebar.caption(f"{nombre} no figura en tiempos.csv.")
    else:
        origen = st.sidebar.file_uploader("Captura a analizar (.csv)", type=["csv"],
                                          help="Convertí el .pcapng con: python convertir_capturas.py capturas")
        if origen is None:
            return None
        nombre = _base(origen.name)
        archivo_tiempos = st.sidebar.file_uploader("Planilla de tiempos (opcional)", type=["csv"],
                                                   help="Si no subís una, se usa capturas/tiempos.csv.")
        if archivo_tiempos is not None:
            planilla = v.leer_intervalos(archivo_tiempos)
        tiempos = _tiempos_de(nombre, planilla)

    # Normales: se proponen las del mismo entorno que la captura (ver experimentos.py)
    entorno = exp.entorno_escenario(nombre, planilla.assign(captura=planilla["captura"].map(_base))) \
        if planilla is not None else exp.ENTORNO_POR_DEFECTO
    disponibles = sorted(glob.glob(os.path.join(CARPETA_CAPTURAS, exp.PREFIJO_NORMAL + "*.csv")))
    del_entorno = [r for r in disponibles if exp.entorno_normal(_base(r)) == entorno]
    elegidas = st.sidebar.multiselect(
        "Capturas normales de referencia", disponibles, default=del_entorno or disponibles,
        format_func=_base, key=f"normales_{nombre}",
        help="Se proponen las del mismo entorno que la captura (columna entorno de tiempos.csv).")
    st.sidebar.caption(f"Entorno de la captura: **{entorno}**")
    subidas = st.sidebar.file_uploader("…o subí capturas normales (.csv)", type=["csv"],
                                       accept_multiple_files=True)
    if not elegidas and not subidas:
        st.sidebar.warning("Elegí al menos una captura normal para entrenar la IA y fijar los umbrales.")
        return None

    with st.spinner("Leyendo capturas..."):
        paquetes = _leer_csv(origen)
        normales = [_leer_csv(r) for r in elegidas] + [_leer_csv(f) for f in subidas or []]
    clave = "|".join([_clave(origen)] + [_clave(r) for r in elegidas] + [_clave(f) for f in subidas or []])

    st.sidebar.divider()
    segundos = st.sidebar.select_slider("Tamaño de ventana (s)", [5, 10, 15], value=5)
    contaminacion = st.sidebar.slider("Sensibilidad de la IA (contaminación)", 0.005, 0.10, 0.01, 0.005,
                                      help="Proporción de ventanas raras que se esperan dentro del tráfico normal.")
    capacidad = st.sidebar.number_input("Capacidad del enlace (Mbps, 0 = sin dato)", min_value=0.0, value=0.0)

    return {
        "clave": clave, "paquetes": paquetes, "nombre": nombre, "normales": normales, "tiempos": tiempos,
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
        x=alt.X("inicio_ventana:T", title="Hora", axis=alt.Axis(format="%H:%M")), y=alt.Y("c2_mbps:Q", title="Mbps"),
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
        x=alt.X("inicio_ventana:T", title="Hora", axis=alt.Axis(format="%H:%M")), y=alt.Y("puntaje:Q", title="Puntaje de anomalía"))
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
        r = analizar(cfg["clave"], cfg["paquetes"], cfg["normales"], cfg["tiempos"], cfg["segundos"],
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
        x=alt.X("inicio_ventana:T", title="Hora", axis=alt.Axis(format="%H:%M")), y=alt.Y("valor:Q", title=None),
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
# Pestaña Evaluación: lee los resultados de experimentos.py
# ----------------------------------------------------------------------
@st.cache_data
def _leer_resultado(nombre, _marca):
    """Lee resultados/<nombre>; _marca (fecha de modificación) invalida la caché al regenerarlo."""
    return pd.read_csv(os.path.join(CARPETA_RESULTADOS, nombre))


def _resultado(nombre):
    ruta = os.path.join(CARPETA_RESULTADOS, nombre)
    return _leer_resultado(nombre, os.path.getmtime(ruta)) if os.path.exists(ruta) else None


def pantalla_evaluacion():
    glob_ = _resultado("E1_global.csv")
    if glob_ is None:
        st.info("Todavía no hay resultados. Corré en la terminal, desde la carpeta del repo:\n\n"
                "`python experimentos.py --solo E1 E4`\n\ny volvé a esta pestaña.")
        return

    fecha = pd.Timestamp(os.path.getmtime(os.path.join(CARPETA_RESULTADOS, "E1_global.csv")), unit="s")
    fecha = fecha.tz_localize("UTC").tz_convert("America/Argentina/Buenos_Aires")
    st.caption(f"Resultados de experimentos.py · generados el {fecha:%d/%m/%Y %H:%M} · "
               "cada escenario se compara con las capturas normales de su entorno")

    # Respuesta a la pregunta de investigación
    hip = _resultado("E1_hipotesis.csv")
    mejor = glob_.sort_values("f1", ascending=False).iloc[0]
    base = glob_.set_index("metodo").loc["Reglas fijas"] if "Reglas fijas" in set(glob_["metodo"]) else None
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Mejor método (F1)", mejor["metodo"], f"F1 {mejor['f1']:.2f}", delta_color="off")
    if base is not None:
        k2.metric("F1 de las reglas fijas", f"{base['f1']:.2f}")
    if hip is not None:
        aceptan = hip.loc[hip["acepta_hipotesis"].astype(str) == "True", "metodo"].tolist()
        k3.metric("Modelos que superan a las reglas", f"{len(aceptan)} de {len(hip)}")
        k4.metric("Hipótesis", "Se acepta" if aceptan else "No se acepta",
                  ", ".join(aceptan) if aceptan else None, delta_color="off")

    st.subheader("Comparación de métodos (E1)")
    cols = ["metodo", "precision", "recall", "f1", "falsas_alarmas_c100", "eventos_detectados", "demora_media_s"]
    vista = glob_[[c for c in cols if c in glob_.columns]].rename(columns={
        "metodo": "Método", "precision": "Precisión", "recall": "Recall", "f1": "F1",
        "falsas_alarmas_c100": "Falsas alarmas c/100", "eventos_detectados": "Eventos detectados",
        "demora_media_s": "Demora media (s)"})
    for c in ["Precisión", "Recall", "F1", "Falsas alarmas c/100", "Demora media (s)"]:
        if c in vista:
            vista[c] = pd.to_numeric(vista[c], errors="coerce")
    st.dataframe(vista.style.format({"Precisión": "{:.2f}", "Recall": "{:.2f}", "F1": "{:.2f}",
                                     "Falsas alarmas c/100": "{:.1f}", "Demora media (s)": "{:.1f}"}, na_rep="—")
                 .highlight_max(subset=["F1"], color="#1F3A4D"),
                 use_container_width=True, hide_index=True)

    largo = glob_.melt(id_vars="metodo", value_vars=["precision", "recall", "f1"],
                       var_name="medida", value_name="valor")
    largo["medida"] = largo["medida"].map({"precision": "Precisión", "recall": "Recall", "f1": "F1"})
    escala = alt.Scale(domain=list(COLORES_METODO), range=list(COLORES_METODO.values()))
    barras = alt.Chart(largo).mark_bar().encode(
        x=alt.X("metodo:N", title=None, sort=list(COLORES_METODO), axis=alt.Axis(labels=False, ticks=False)),
        y=alt.Y("valor:Q", title=None, scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("metodo:N", scale=escala, title="Método", legend=alt.Legend(orient="bottom")),
        column=alt.Column("medida:N", title=None, sort=["Precisión", "Recall", "F1"]),
        tooltip=["metodo", "medida", alt.Tooltip("valor:Q", format=".2f")],
    ).properties(width=180, height=220)
    st.altair_chart(barras)

    if hip is not None:
        with st.expander("Regla de decisión de la hipótesis"):
            st.caption("Un modelo supera a las reglas si mejora el recall o el F1 y no sube las falsas alarmas "
                       "más de 2 cada 100 ventanas normales.")
            st.dataframe(hip.rename(columns={
                "metodo": "Método", "delta_recall": "Δ recall", "delta_f1": "Δ F1",
                "delta_falsas_alarmas_c100": "Δ falsas alarmas c/100", "acepta_hipotesis": "Supera a las reglas"})
                .style.format({"Δ recall": "{:+.2f}", "Δ F1": "{:+.2f}", "Δ falsas alarmas c/100": "{:+.1f}"}),
                use_container_width=True, hide_index=True)

    esc = _resultado("E1_por_escenario.csv")
    if esc is not None:
        st.subheader("Por escenario")
        st.caption("En red lenta y saturación importa cuánto detecta (recall); en los demás, cuánto se equivoca "
                   "(falsas alarmas cada 100 ventanas).")
        esc = esc.copy()
        esc["valor"] = np.where(esc["degradado"].astype(str) == "True",
                                esc["recall"].map(lambda x: "—" if pd.isna(x) else f"recall {x:.2f}"),
                                esc["falsas_alarmas_c100"].map(lambda x: "—" if pd.isna(x) else f"{x:.1f} falsas c/100"))
        st.dataframe(esc.pivot(index="escenario", columns="metodo", values="valor")
                     .reindex(columns=[m for m in COLORES_METODO if m in set(esc["metodo"])]),
                     use_container_width=True)

    mat = _resultado("E4_matrices.csv")
    if mat is not None:
        st.subheader("Caso de confusión (E4): qué etiqueta puso cada método")
        metodo = st.radio("Método", [m for m in COLORES_METODO if m in set(mat["metodo"])], horizontal=True,
                          key="e4_metodo")
        m = mat[mat["metodo"] == metodo].drop(columns="metodo").set_index("escenario_real")
        st.dataframe(m.dropna(axis=1, how="all").fillna(0).astype(int), use_container_width=True)

    _otros_experimentos()

    st.download_button("Exportar comparación (CSV)", glob_.to_csv(index=False).encode("utf-8"),
                       file_name="E1_global.csv", mime="text/csv")


def _grafico_experimento(datos, columna_x, titulo_x, columna_y="f1", titulo_y="F1", dominio_y=(0.4, 1)):
    """Una medida de cada método según una variable del experimento (líneas con puntos)."""
    escala = alt.Scale(domain=list(COLORES_METODO), range=list(COLORES_METODO.values()))
    base = alt.Chart(datos).encode(
        x=alt.X(f"{columna_x}:O", title=titulo_x, axis=alt.Axis(labelAngle=0)),
        y=alt.Y(f"{columna_y}:Q", title=titulo_y, scale=alt.Scale(domain=list(dominio_y))),
        color=alt.Color("metodo:N", scale=escala, title="Método", legend=alt.Legend(orient="bottom")),
        tooltip=["metodo", alt.Tooltip(f"{columna_x}:O", title=titulo_x), alt.Tooltip("f1:Q", title="F1", format=".2f"),
                 alt.Tooltip("falsas_alarmas_c100:Q", title="Falsas alarmas c/100", format=".1f")],
    )
    return (base.mark_line(strokeWidth=2.5) + base.mark_point(size=70, filled=True)).properties(height=260)


def _otros_experimentos():
    """E2 (tamaño de ventana) y E3 (cantidad de capturas normales), si ya se corrieron."""
    e2, e3 = _resultado("E2_ventanas.csv"), _resultado("E3_normales.csv")
    if e2 is None and e3 is None:
        st.caption("E2 y E3 todavía no se corrieron: `python experimentos.py --solo E2 E3`")
        return
    st.subheader("Otros experimentos")
    izquierda, derecha = st.columns(2)
    if e2 is not None:
        with izquierda:
            st.markdown("**E2 · Tamaño de ventana: falsas alarmas**")
            tope = max(15, float(e2["falsas_alarmas_c100"].max()) + 1)
            st.altair_chart(_grafico_experimento(e2, "ventana_s", "Ventana (s)", "falsas_alarmas_c100",
                                                 "Falsas alarmas", (0, tope)), use_container_width=True)
            mejor = e2.loc[e2["f1"].idxmax()]
            st.caption(f"Mejor F1: {mejor['metodo']} con ventanas de {int(mejor['ventana_s'])} s "
                       f"({mejor['f1']:.2f}). Con ventanas más grandes suben las falsas alarmas de todos los "
                       "métodos, porque una ventana mezcla tráfico normal con el evento.")
    if e3 is not None:
        with derecha:
            st.markdown("**E3 · Capturas normales para entrenar: F1**")
            st.altair_chart(_grafico_experimento(e3, "capturas_normales", "Capturas normales por entorno"),
                            use_container_width=True)
            lof = e3[e3["metodo"] == "LOF"].set_index("capturas_normales")["f1"]
            if len(lof) > 1:
                st.caption(f"LOF: F1 {lof.iloc[0]:.2f} con {lof.index[0]} captura y {lof.iloc[-1]:.2f} con "
                           f"{lof.index[-1]}: con unos 10 minutos de tráfico normal ya alcanza.")


# ----------------------------------------------------------------------
# Página
# ----------------------------------------------------------------------
st.title("PySharkStream Analytica")
st.caption("Rendimiento de red con IA · Redes de Datos · FaCENA-UNNE")

cfg = sidebar()
pestanas = st.tabs(["Rendimiento y Anomalías", "Gráficos", "Evaluación"])

if cfg is None:
    for p in pestanas[:2]:
        with p:
            st.info("Elegí una captura de la carpeta o subí un CSV en la barra lateral.")
else:
    with pestanas[0]:
        pantalla_rendimiento(cfg)
    with pestanas[1]:
        pantalla_graficos(cfg)
with pestanas[2]:
    pantalla_evaluacion()
