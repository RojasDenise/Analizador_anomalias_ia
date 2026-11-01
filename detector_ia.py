"""Módulo B2: modelos de IA para detectar degradaciones.

Responde si una ventana se sale de lo normal y por qué.

Parte de la tabla de ventanas ya procesada por C1 y C2 (columnas c1_*, c2_* y ctx_*),
entrena tres modelos no supervisados SOLO con tráfico normal y marca las ventanas anómalas:
    Isolation Forest (modelo principal), Local Outlier Factor y One-Class SVM.

Agrega a la tabla columnas con prefijo b2_:
b2_<modelo>_puntaje   puntaje de anomalía (más alto = más raro; > 0 suele ser anómalo)
b2_<modelo>_anomala   True si el modelo marca la ventana como anómala
b2_causa              causa probable según las métricas que más se alejaron de lo normal
b2_explicacion        las 3 métricas con mayor z-score y su valor
con <modelo> = if, lof u ocsvm.

Decisiones (ver contrato.md, sección 8):
- Se entrena solo con ventanas normales: el modelo aprende cómo se ve la red sana.
- Las métricas que faltan (por ejemplo, RTT en ventanas sin TCP) se completan con la
  mediana de las ventanas normales, para no inventar "latencia perfecta" ni descartar ventanas.
- La detección la hace el modelo; la etiqueta la decide una regla sobre los z-scores
  (cuánto se alejó cada métrica de su media normal, en desvíos). Así cada alerta se explica.
- Cada modelo devuelve sus propias alertas: E los evalúa por separado.

Limitación conocida (para el informe): Isolation Forest solo corta por valores que vio al entrenar.
Una métrica que en el tráfico normal es siempre igual (por ejemplo, SYN sin respuesta = 0) nunca se
usa para aislar, así que un escaneo puede pasarle desapercibido. LOF y One-Class SVM sí lo detectan.
"""
import numpy as np
import pandas as pd

from ventanas import COLUMNAS_ALERTAS, alertas_vacias

MODULO = "b2"
CONTAMINACION = 0.01   # proporción esperada de anomalías DENTRO del tráfico normal de entrenamiento
Z_MINIMO = 3.0         # desvío mínimo para considerar que una métrica "se disparó"

# Métricas que usa el modelo, agrupadas según lo que indican
GRUPOS = {
    "rendimiento": ["c1_rtt_medio_ms", "c1_rtt_max_ms", "c1_pct_retrans",
                    "c1_pct_dup_ack", "c1_ventanas_cero"],
    "volumen": ["c2_mbps", "c2_pps"],
    "inusual": ["ctx_max_puertos_por_ip", "ctx_syn_sin_respuesta", "ctx_consultas_dns"],
}
# El tamaño medio y el reparto por protocolo de C2 no se usan: describen la composición del tráfico,
# no su rendimiento, y en las pruebas agregaban ruido que hacía que Isolation Forest no viera
# degradaciones reales (bajaba de 100 % a 47 % de detección en red lenta).
CARACTERISTICAS = [c for grupo in GRUPOS.values() for c in grupo]

# Nombre que usa E para cada modelo (contrato.md, sección 6)
MODELOS = {"if": "Isolation Forest", "lof": "LOF", "ocsvm": "One-Class SVM"}

NOMBRES_LEGIBLES = {
    "c1_rtt_medio_ms": "RTT medio (ms)", "c1_rtt_max_ms": "RTT máximo (ms)",
    "c1_pct_retrans": "retransmisiones (%)", "c1_pct_dup_ack": "ACK duplicados (%)",
    "c1_ventanas_cero": "ventanas en cero", "c2_mbps": "Mbps", "c2_pps": "paquetes/s",
    "c2_tam_medio": "tamaño medio (B)", "c2_pct_tcp": "% TCP", "c2_pct_udp": "% UDP",
    "c2_pct_dns": "% DNS", "ctx_max_puertos_por_ip": "puertos distintos por IP",
    "ctx_syn_sin_respuesta": "SYN sin respuesta", "ctx_consultas_dns": "consultas DNS",
}


# ----------------------------------------------------------------------
# Preparación de los datos
# ----------------------------------------------------------------------
def _columnas_disponibles(tabla):
    columnas = [c for c in CARACTERISTICAS if c in tabla.columns]
    if not columnas:
        raise ValueError("La tabla no tiene métricas: ejecutá calcular_metricas de C1 y C2 primero.")
    return columnas


def _matriz(modelo, tabla):
    """Matriz de características con los faltantes completados con la mediana normal."""
    faltan = [c for c in modelo["columnas"] if c not in tabla.columns]
    if faltan:
        raise ValueError(f"Faltan columnas que se usaron al entrenar: {faltan}")
    x = tabla[modelo["columnas"]].apply(pd.to_numeric, errors="coerce")
    return x.fillna(modelo["medianas"])


# ----------------------------------------------------------------------
# Entrenamiento
# ----------------------------------------------------------------------
def entrenar(tablas_normales, contaminacion=CONTAMINACION, gamma_ocsvm=None, random_state=0):
    """Entrena los tres modelos con ventanas de tráfico normal.

    tablas_normales: una tabla o una lista de tablas (una por captura normal),
                     ya procesadas con calcular_metricas de C1 y C2.
    gamma_ocsvm: ancho del kernel de One-Class SVM. Por defecto 0.1 / cantidad de métricas:
                 con pocas ventanas de entrenamiento, el valor de scikit-learn ("scale")
                 encierra demasiado ajustado al tráfico normal y da muchas falsas alarmas.
    Devuelve un diccionario con todo lo necesario para puntuar (se guarda con guardar()).
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import OneClassSVM

    if isinstance(tablas_normales, pd.DataFrame):
        tablas_normales = [tablas_normales]
    normal = pd.concat(tablas_normales)
    if len(normal) < 20:
        raise ValueError("Hacen falta al menos 20 ventanas normales para entrenar.")

    columnas = _columnas_disponibles(normal)
    x = normal[columnas].apply(pd.to_numeric, errors="coerce")
    medianas = x.median().fillna(0)
    x = x.fillna(medianas)

    escala = StandardScaler().fit(x)
    x_esc = escala.transform(x)

    modelos = {
        "if": IsolationForest(n_estimators=200, contamination=contaminacion, random_state=random_state),
        "lof": LocalOutlierFactor(n_neighbors=min(20, len(x) - 1), novelty=True,
                                  contamination=contaminacion),
        "ocsvm": OneClassSVM(nu=contaminacion,
                             gamma=gamma_ocsvm if gamma_ocsvm is not None else 0.1 / len(columnas)),
    }
    for m in modelos.values():
        m.fit(x_esc)

    # Media y desvío normales para explicar las alertas (z-score). Desvío mínimo para no dividir por ~0.
    media = x.mean()
    desvio = x.std().replace(0, np.nan).fillna(1.0).clip(lower=1e-6)

    return {
        "columnas": columnas,
        "medianas": medianas,
        "media": media,
        "desvio": desvio,
        "escala": escala,
        "modelos": modelos,
        "contaminacion": contaminacion,
        "ventanas_entrenamiento": len(x),
    }


def guardar(modelo, ruta="modelos/b2_modelos.joblib"):
    import os

    import joblib
    os.makedirs(os.path.dirname(ruta) or ".", exist_ok=True)
    joblib.dump(modelo, ruta)
    return ruta


def cargar(ruta="modelos/b2_modelos.joblib"):
    import joblib
    return joblib.load(ruta)


# ----------------------------------------------------------------------
# Puntaje y explicación
# ----------------------------------------------------------------------
def zscores(modelo, tabla):
    """Cuántos desvíos se alejó cada métrica de su media normal (positivo = más alto que lo normal)."""
    x = _matriz(modelo, tabla)
    return (x - modelo["media"]) / modelo["desvio"]


def _causa(z_fila):
    """Regla de etiquetado sobre los z-scores de una ventana (ver tabla de la propuesta)."""
    def maximo(grupo):
        cols = [c for c in GRUPOS[grupo] if c in z_fila.index]
        return float(z_fila[cols].max()) if cols else 0.0

    rend, vol, inus = maximo("rendimiento"), maximo("volumen"), maximo("inusual")

    # Las consultas DNS crecen con el volumen: lo inusual tiene que superar también al volumen
    if inus >= Z_MINIMO and inus >= max(rend, vol):
        return "Comportamiento inusual"
    if rend >= Z_MINIMO and vol >= Z_MINIMO:
        return "Congestión o saturación"
    if rend >= Z_MINIMO:
        return "Degradación de rendimiento"
    if vol >= Z_MINIMO:
        return "Pico de uso normal"
    # Anómala para el modelo, pero ninguna métrica pasa el umbral: se toma el grupo que más se alejó
    mayor = max([(rend, "Degradación de rendimiento"), (vol, "Pico de uso normal"),
                 (inus, "Comportamiento inusual")])
    return mayor[1]


def _explicacion(z_fila, valores, n=3):
    top = z_fila.sort_values(ascending=False).head(n)
    return "; ".join(f"{NOMBRES_LEGIBLES.get(c, c)} {valores[c]:.1f} (z={z:.1f})" for c, z in top.items())


def puntuar(modelo, tabla):
    """Devuelve una copia de la tabla con las columnas b2_*."""
    x = _matriz(modelo, tabla)
    x_esc = modelo["escala"].transform(x)

    t = tabla.copy()
    for clave, m in modelo["modelos"].items():
        # decision_function: > 0 normal, < 0 anómalo. Se invierte para que más alto = más raro.
        t[f"b2_{clave}_puntaje"] = -m.decision_function(x_esc)
        t[f"b2_{clave}_anomala"] = m.predict(x_esc) == -1

    z = (x - modelo["media"]) / modelo["desvio"]
    t["b2_causa"] = [_causa(z.loc[i]) for i in z.index]
    t["b2_explicacion"] = [_explicacion(z.loc[i], x.loc[i]) for i in z.index]
    return t


# ----------------------------------------------------------------------
# Alertas
# ----------------------------------------------------------------------
def generar_alertas(tabla_puntuada):
    """Alertas de cada modelo con el formato del contrato.

    Devuelve {"Isolation Forest": alertas, "LOF": alertas, "One-Class SVM": alertas},
    listo para pasar a evaluacion.evaluar junto con las reglas fijas.
    """
    if "b2_causa" not in tabla_puntuada.columns:
        raise ValueError("Falta b2_causa: ejecutá puntuar() primero.")

    salida = {}
    for clave, nombre in MODELOS.items():
        col = f"b2_{clave}_anomala"
        if col not in tabla_puntuada.columns:
            continue
        anomalas = tabla_puntuada[tabla_puntuada[col]]
        if anomalas.empty:
            salida[nombre] = alertas_vacias()
            continue
        salida[nombre] = pd.DataFrame({
            "inicio_ventana": anomalas.index,
            "modulo": MODULO,
            "etiqueta": anomalas["b2_causa"].values,
            "detalle": [f"{nombre}: puntaje {p:.3f}; {e}" for p, e in
                        zip(anomalas[f"b2_{clave}_puntaje"], anomalas["b2_explicacion"])],
        }, columns=COLUMNAS_ALERTAS)
    return salida


# ----------------------------------------------------------------------
# Sección de Streamlit (la integración en app.py se hace después)
# ----------------------------------------------------------------------
def seccion_streamlit(tabla_puntuada, modelo_principal="if"):
    """Dibuja la sección "Anomalías (B2)": puntaje en el tiempo, tabla de alertas y detalle."""
    import altair as alt
    import streamlit as st

    st.subheader("Anomalías (B2)")
    clave = st.selectbox("Modelo", list(MODELOS), index=list(MODELOS).index(modelo_principal),
                         format_func=MODELOS.get)
    col_p, col_a = f"b2_{clave}_puntaje", f"b2_{clave}_anomala"

    datos = tabla_puntuada[[col_p, col_a, "b2_causa"]].reset_index()
    datos.columns = ["inicio_ventana", "puntaje", "anomala", "causa"]
    linea = alt.Chart(datos).mark_line().encode(
        x=alt.X("inicio_ventana:T", title="Hora"), y=alt.Y("puntaje:Q", title="Puntaje de anomalía"))
    puntos = alt.Chart(datos[datos["anomala"]]).mark_circle(size=70).encode(
        x="inicio_ventana:T", y="puntaje:Q", color=alt.Color("causa:N", title="Causa probable"),
        tooltip=[alt.Tooltip("inicio_ventana:T", format="%H:%M:%S"), "causa:N",
                 alt.Tooltip("puntaje:Q", format=".3f")])
    st.altair_chart(linea + puntos, use_container_width=True)

    alertas = generar_alertas(tabla_puntuada)[MODELOS[clave]]
    st.caption(f"{len(alertas)} ventanas anómalas")
    st.dataframe(alertas, use_container_width=True, hide_index=True)
