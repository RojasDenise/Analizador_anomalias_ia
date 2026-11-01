"""Demo de B2 con los datos de ejemplo, usando C1 y (si está) C2.

Uso:
    python tests/generar_ejemplo.py
    python tests/demo_detector_ia.py
"""
import os
import sys

import pandas as pd

CARPETA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(CARPETA))

import detector_ia as b2       # noqa: E402
import rendimiento_tcp as c1   # noqa: E402
import ventanas as v           # noqa: E402

try:
    import uso_red as c2       # noqa: E402  (está en la rama de C2)
except ImportError:
    c2 = None


def main():
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 10)
    paquetes = v.cargar_paquetes(os.path.join(CARPETA, "ejemplo_paquetes.csv"))
    tabla = v.tabla_ventanas(paquetes)
    tabla = v.etiquetar_ventanas(tabla, v.leer_intervalos(os.path.join(CARPETA, "ejemplo_intervalos.csv")))

    tabla = c1.calcular_metricas(paquetes, tabla)
    if c2 is not None:
        tabla = c2.calcular_metricas(paquetes, tabla)
    else:
        print("Aviso: uso_red.py (C2) no está en esta rama; se entrena solo con C1 y contexto.\n")

    # Entrenar con la parte normal del principio (en el trabajo real: las capturas normal_*.pcapng)
    normal = tabla[(tabla["escenario_real"] == "normal")
                   & (tabla.index < tabla.index[0] + pd.Timedelta(minutes=3))]
    modelo = b2.entrenar(normal)
    print(f"Entrenado con {modelo['ventanas_entrenamiento']} ventanas y {len(modelo['columnas'])} métricas.\n")

    puntuada = b2.puntuar(modelo, tabla)
    alertas = b2.generar_alertas(puntuada)

    for nombre, a in alertas.items():
        etiquetas = a.set_index("inicio_ventana")["etiqueta"].reindex(tabla.index).fillna("sin alerta")
        print(f"== {nombre}: {len(a)} ventanas anómalas")
        print(pd.crosstab(tabla["escenario_real"], etiquetas.rename("etiqueta asignada")), "\n")

    print("Ejemplo de alerta:")
    print(alertas["Isolation Forest"].iloc[0].to_dict())
    print("\nModelo guardado en", b2.guardar(modelo))


if __name__ == "__main__":
    main()
