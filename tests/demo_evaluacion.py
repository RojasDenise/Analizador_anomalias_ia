"""Demo del módulo E con los datos de ejemplo.

Mientras C1, C2 y B2 no estén terminados, esta demo usa métodos PROVISORIOS
(reglas y modelos muy simples) solo para probar que la evaluación funciona.
Cuando cada módulo esté listo, se reemplazan por sus alertas reales.

Uso:
    python tests/generar_ejemplo.py
    python tests/demo_evaluacion.py
"""
import os
import sys

import pandas as pd

CARPETA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(CARPETA))

import evaluacion as ev  # noqa: E402
import ventanas as v     # noqa: E402


def metricas_provisorias(paquetes, tabla):
    """Unas pocas métricas por ventana para los métodos provisorios (C1 y C2 harán las reales)."""
    df = v.asignar_ventana(paquetes)
    tcp = df[df["Transport Protocol"] == "TCP"].groupby("inicio_ventana")
    m = pd.DataFrame({
        "rtt_ms": tcp["TCP RTT"].mean() * 1000,
        "pct_retrans": tcp["TCP Retransmission"].mean() * 100,
        "zero_window": tcp["TCP Zero Window"].sum(),
    }).reindex(tabla.index).fillna(0)
    m["mbps"] = tabla["bytes"] * 8 / tabla["duracion_s"] / 1e6
    m["puertos"] = tabla["ctx_max_puertos_por_ip"]
    m["syn_sin_resp"] = tabla["ctx_syn_sin_respuesta"]
    return m


def a_alertas(mascara, etiqueta, modulo):
    return pd.DataFrame({"inicio_ventana": mascara.index[mascara], "modulo": modulo,
                         "etiqueta": etiqueta, "detalle": "provisorio"})


def reglas_provisorias(m):
    rendimiento = (m["pct_retrans"] > 5) | (m["rtt_ms"] > 150)
    saturacion = (m["zero_window"] > 0) & (m["mbps"] > m["mbps"].median() * 3)
    return pd.concat([a_alertas(rendimiento, "Degradación de rendimiento", "c1"),
                      a_alertas(saturacion, "Congestión o saturación", "c2")])


def modelos_provisorios(m, normal):
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import OneClassSVM

    escala = StandardScaler().fit(m[normal])
    x_entreno, x = escala.transform(m[normal]), escala.transform(m)
    modelos = {
        "Isolation Forest": IsolationForest(contamination=0.02, random_state=0),
        "LOF": LocalOutlierFactor(novelty=True, contamination=0.02),
        "One-Class SVM": OneClassSVM(nu=0.02),
    }
    salida = {}
    for nombre, modelo in modelos.items():
        anomala = pd.Series(modelo.fit(x_entreno).predict(x) == -1, index=m.index)
        # Etiqueta provisoria: si suben puertos o SYN sin respuesta es inusual; si no, degradación
        inusual = anomala & ((m["puertos"] > 20) | (m["syn_sin_resp"] > 5))
        pico = anomala & ~inusual & (m["pct_retrans"] < 1) & (m["rtt_ms"] < 100)
        degradacion = anomala & ~inusual & ~pico
        salida[nombre] = pd.concat([
            a_alertas(degradacion, "Degradación de rendimiento", "b2"),
            a_alertas(inusual, "Comportamiento inusual", "b2"),
            a_alertas(pico, "Pico de uso normal", "b2"),
        ])
    return salida


def main():
    paquetes = v.cargar_paquetes(os.path.join(CARPETA, "ejemplo_paquetes.csv"))
    tabla = v.tabla_ventanas(paquetes)
    tabla = v.etiquetar_ventanas(tabla, v.leer_intervalos(os.path.join(CARPETA, "ejemplo_intervalos.csv")))

    m = metricas_provisorias(paquetes, tabla)
    # Se entrena con la primera parte normal (antes del primer evento), como si fuera la captura de referencia
    normal = (tabla["escenario_real"] == "normal") & (tabla.index < tabla.index[0] + pd.Timedelta(minutes=3))

    metodos = {"Reglas fijas": reglas_provisorias(m), **modelos_provisorios(m, normal)}
    glob, esc = ev.evaluar(tabla, metodos)

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.2f}".format)
    print("\n== E1 · Comparación de métodos ==")
    print(glob[["metodo", "precision", "recall", "f1", "falsas_alarmas_c100",
                "eventos_detectados", "demora_media_s"]].to_string(index=False))
    print("\n== Por escenario ==")
    print(esc.to_string(index=False))
    print("\n== Hipótesis ==")
    print(ev.hipotesis(glob).to_string(index=False))
    print("\n== E4 · Caso de confusión (Isolation Forest) ==")
    print(ev.matriz_etiquetas(tabla, metodos["Isolation Forest"]))

    ruta = ev.registrar_resultado(glob, "E1", "demo con datos de ejemplo (métodos provisorios)")
    print(f"\nResultados agregados a {ruta}")


if __name__ == "__main__":
    main()
