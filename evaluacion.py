"""Módulo E · Evaluación y comparación.

Responde la pregunta de investigación: ¿la IA detecta degradaciones de rendimiento
mejor que las reglas con umbrales fijos?

Recibe:
  - la tabla de ventanas etiquetada (ventanas.etiquetar_ventanas), que es la verdad de referencia;
  - las alertas de cada método, con el formato de contrato.md.
Devuelve, para cada método y cada escenario: precisión, recall, F1, falsas alarmas
cada 100 ventanas normales y demora de detección.

Uso típico:
    metodos = {
        "Reglas fijas": pd.concat([alertas_c1, alertas_c2]),
        "Isolation Forest": alertas_if,
        "LOF": alertas_lof,
        "One-Class SVM": alertas_ocsvm,
    }
    global_, por_escenario = evaluar(tabla_etiquetada, metodos)
"""
import os
from datetime import datetime

import numpy as np
import pandas as pd

from ventanas import COLUMNAS_ALERTAS


# Etiquetas que cuentan como "el método detectó una degradación".
# "Pico de uso normal" y "Comportamiento inusual" NO son degradación (ver contrato.md).
ETIQUETAS_DEGRADACION = ("Degradación de rendimiento", "Congestión o saturación")

RUTA_RESULTADOS = os.path.join("resultados", "resultados.csv")


# ----------------------------------------------------------------------
# Paso 1: pasar las alertas de un método a una marca por ventana
# ----------------------------------------------------------------------
def marcar_ventanas(tabla, alertas, etiquetas_positivas=ETIQUETAS_DEGRADACION):
    """Serie booleana alineada con la tabla: True si el método marcó degradación en esa ventana.

    Si un método tiene varias alertas en la misma ventana (por ejemplo, de C1 y de C2),
    la ventana cuenta una sola vez.
    """
    if alertas is None or len(alertas) == 0:
        return pd.Series(False, index=tabla.index)

    faltan = [c for c in COLUMNAS_ALERTAS if c not in alertas.columns]
    if faltan:
        raise ValueError(f"Faltan columnas en las alertas: {faltan}")

    positivas = alertas[alertas["etiqueta"].isin(etiquetas_positivas)]
    ventanas_marcadas = set(pd.to_datetime(positivas["inicio_ventana"]))
    return pd.Series(tabla.index.isin(ventanas_marcadas), index=tabla.index)


# ----------------------------------------------------------------------
# Paso 2: métricas
# ----------------------------------------------------------------------
def _division(a, b):
    return a / b if b else np.nan


def metricas(real, predicho):
    """Matriz de confusión y métricas a partir de dos series booleanas por ventana."""
    real = real.astype(bool)
    predicho = predicho.astype(bool)
    vp = int((real & predicho).sum())    # degradada y marcada
    fp = int((~real & predicho).sum())   # normal pero marcada (falsa alarma)
    fn = int((real & ~predicho).sum())   # degradada y no marcada
    vn = int((~real & ~predicho).sum())  # normal y no marcada

    precision = _division(vp, vp + fp)
    recall = _division(vp, vp + fn)
    f1 = _division(2 * precision * recall, precision + recall) if vp else (0.0 if (vp + fp + fn) else np.nan)
    return {
        "ventanas": len(real),
        "degradadas": vp + fn,
        "VP": vp, "FP": fp, "FN": fn, "VN": vn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "falsas_alarmas_c100": _division(100 * fp, fp + vn),
    }


def eventos(tabla):
    """Agrupa las ventanas consecutivas del mismo escenario en eventos.

    Devuelve un DataFrame con escenario, inicio, fin y degradada por evento (sin los tramos normales).
    """
    if tabla.empty:
        return pd.DataFrame(columns=["escenario", "inicio", "fin", "degradada"])
    cambio = tabla["escenario_real"].ne(tabla["escenario_real"].shift())
    grupo = cambio.cumsum()
    resumen = tabla.groupby(grupo).agg(
        escenario=("escenario_real", "first"),
        degradada=("degradada", "first"),
    )
    resumen["inicio"] = tabla.index.to_series().groupby(grupo.values).min().values
    resumen["fin"] = tabla.index.to_series().groupby(grupo.values).max().values
    resumen = resumen[resumen["escenario"] != "normal"].reset_index(drop=True)
    return resumen[["escenario", "inicio", "fin", "degradada"]]


def demoras(tabla, predicho):
    """Para cada evento degradado: segundos entre el inicio del evento y la primera alerta dentro de él.

    NaN si el método nunca lo detectó.
    """
    filas = []
    for _, ev in eventos(tabla).iterrows():
        if not ev["degradada"]:
            continue
        dentro = (tabla.index >= ev["inicio"]) & (tabla.index <= ev["fin"])
        marcadas = tabla.index[dentro & predicho.values]
        demora = (marcadas.min() - ev["inicio"]).total_seconds() if len(marcadas) else np.nan
        filas.append({"escenario": ev["escenario"], "inicio": ev["inicio"], "demora_s": demora,
                      "detectado": len(marcadas) > 0})
    return pd.DataFrame(filas, columns=["escenario", "inicio", "demora_s", "detectado"])


# ----------------------------------------------------------------------
# Paso 3: comparación de métodos (E1)
# ----------------------------------------------------------------------
def evaluar(tabla, metodos):
    """Compara varios métodos sobre la misma tabla etiquetada.

    tabla:   salida de ventanas.etiquetar_ventanas (columnas escenario_real y degradada).
    metodos: {"nombre del método": DataFrame de alertas}.

    Devuelve (resultado_global, resultado_por_escenario).
    """
    if "degradada" not in tabla.columns:
        raise ValueError("La tabla no está etiquetada: usar ventanas.etiquetar_ventanas primero")

    filas_global, filas_escenario = [], []
    real = tabla["degradada"]

    for nombre, alertas in metodos.items():
        predicho = marcar_ventanas(tabla, alertas)

        fila = {"metodo": nombre, **metricas(real, predicho)}
        d = demoras(tabla, predicho)
        fila["eventos_detectados"] = f"{int(d['detectado'].sum())}/{len(d)}"
        fila["demora_media_s"] = d["demora_s"].mean() if len(d) else np.nan
        filas_global.append(fila)

        for escenario, sub in tabla.groupby("escenario_real"):
            m = metricas(sub["degradada"], predicho[sub.index])
            degradado = bool(sub["degradada"].any())
            d_esc = d[d["escenario"] == escenario]
            filas_escenario.append({
                "metodo": nombre,
                "escenario": escenario,
                "degradado": degradado,
                "ventanas": m["ventanas"],
                "marcadas": m["VP"] + m["FP"],
                # En escenarios degradados importa cuánto detecta; en los demás, cuánto se equivoca
                "recall": m["recall"] if degradado else np.nan,
                "falsas_alarmas_c100": np.nan if degradado else m["falsas_alarmas_c100"],
                "demora_media_s": d_esc["demora_s"].mean() if len(d_esc) else np.nan,
            })

    return pd.DataFrame(filas_global), pd.DataFrame(filas_escenario)


def matriz_etiquetas(tabla, alertas):
    """E4 · Caso de confusión: qué etiqueta puso el método en las ventanas de cada escenario.

    Filas: escenario real. Columnas: etiqueta asignada (o "sin alerta"). Valores: cantidad de ventanas.
    Sirve para ver si el escaneo se etiqueta como "Comportamiento inusual" y no como degradación.
    """
    asignada = pd.Series("sin alerta", index=tabla.index)
    if alertas is not None and len(alertas):
        a = alertas.copy()
        a["inicio_ventana"] = pd.to_datetime(a["inicio_ventana"])
        # Si hay varias alertas en una ventana, gana la de degradación
        a["_prioridad"] = a["etiqueta"].isin(ETIQUETAS_DEGRADACION).astype(int)
        a = a.sort_values("_prioridad").drop_duplicates("inicio_ventana", keep="last")
        mapa = a.set_index("inicio_ventana")["etiqueta"]
        coinciden = tabla.index.isin(mapa.index)
        asignada[coinciden] = mapa.reindex(tabla.index[coinciden]).values
    return pd.crosstab(tabla["escenario_real"], asignada.rename("etiqueta asignada"))


# ----------------------------------------------------------------------
# Paso 4: guardar resultados de cada experimento (E1 a E4)
# ----------------------------------------------------------------------
def registrar_resultado(resultado, experimento, descripcion="", ruta=RUTA_RESULTADOS):
    """Agrega las filas de un resultado a la tabla común de resultados (CSV).

    experimento: "E1", "E2", "E3" o "E4". descripcion: por ejemplo "ventana 10 s".
    """
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    df = resultado.copy()
    df.insert(0, "descripcion", descripcion)
    df.insert(0, "experimento", experimento)
    df.insert(0, "fecha", datetime.now().strftime("%Y-%m-%d %H:%M"))
    df.to_csv(ruta, mode="a", header=not os.path.exists(ruta), index=False)
    return ruta


def hipotesis(resultado_global, metodo_base="Reglas fijas", tolerancia_fa=2.0):
    """Regla de decisión de la hipótesis (E1).

    Se acepta para un modelo si mejora el recall o el F1 de las reglas y no sube las
    falsas alarmas más de `tolerancia_fa` cada 100 ventanas normales.
    """
    base = resultado_global.set_index("metodo").loc[metodo_base]
    filas = []
    for _, fila in resultado_global.iterrows():
        if fila["metodo"] == metodo_base:
            continue
        mejora = fila["recall"] > base["recall"] or fila["f1"] > base["f1"]
        delta_fa = fila["falsas_alarmas_c100"] - base["falsas_alarmas_c100"]
        filas.append({
            "metodo": fila["metodo"],
            "delta_recall": fila["recall"] - base["recall"],
            "delta_f1": fila["f1"] - base["f1"],
            "delta_falsas_alarmas_c100": delta_fa,
            "acepta_hipotesis": bool(mejora and delta_fa <= tolerancia_fa),
        })
    return pd.DataFrame(filas)


# ----------------------------------------------------------------------
# Pestaña Evaluación en Streamlit
# ----------------------------------------------------------------------
def seccion_streamlit(resultado_global, resultado_escenario, matrices=None):
    """Dibuja la pestaña de Evaluación. matrices: {"método": matriz_etiquetas(...)} para E4."""
    import streamlit as st

    st.subheader("Comparación de métodos")
    formato = {"precision": "{:.2f}", "recall": "{:.2f}", "f1": "{:.2f}",
               "falsas_alarmas_c100": "{:.1f}", "demora_media_s": "{:.0f}"}
    columnas = ["metodo", "precision", "recall", "f1", "falsas_alarmas_c100",
                "eventos_detectados", "demora_media_s"]
    st.dataframe(resultado_global[columnas].style.format(formato, na_rep="—"),
                 use_container_width=True, hide_index=True)

    st.bar_chart(resultado_global.set_index("metodo")[["precision", "recall", "f1"]])

    st.subheader("Por escenario")
    st.dataframe(resultado_escenario.style.format(
        {"recall": "{:.2f}", "falsas_alarmas_c100": "{:.1f}", "demora_media_s": "{:.0f}"}, na_rep="—"),
        use_container_width=True, hide_index=True)

    if matrices:
        st.subheader("Caso de confusión (E4)")
        metodo = st.selectbox("Método", list(matrices))
        st.dataframe(matrices[metodo], use_container_width=True)

    st.download_button("Exportar resultados a CSV",
                       resultado_global.to_csv(index=False).encode("utf-8"),
                       file_name="resultados_evaluacion.csv", mime="text/csv")
