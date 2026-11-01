# Módulo B2 · Modelos de IA

Este zip trae el módulo B2 listo, con el mismo formato que C1 y C2: métricas por ventana, alertas según `contrato.md`, sección de Streamlit y tests.

## Archivos

| Archivo | Qué es |
| --- | --- |
| `detector_ia.py` | El módulo: entrena Isolation Forest, LOF y One-Class SVM con tráfico normal, puntúa cada ventana y genera las alertas con su causa probable. |
| `tests/test_detector_ia.py` | 9 tests con tablas armadas a mano (red lenta, saturación, uso intenso, escaneo, RTT faltante, formato de alertas). |
| `tests/demo_detector_ia.py` | Demo con los datos de ejemplo, usando C1 y C2. |
| `contrato.md` | El contrato de `main` con la sección 8 (B2) agregada al final. |

## Cómo subirlo

En la carpeta del repo:

```
git checkout main
git pull origin main
git checkout -b feature/b2-modelos-ia
```

Copiá los archivos del zip encima (reemplazando `contrato.md`) y después:

```
git add -A
git commit -m "b2: modelos de IA, causa probable y tests"
git push -u origin feature/b2-modelos-ia
```

## Cómo probarlo

```
pip install -r requirements.txt
python tests/test_detector_ia.py
python tests/generar_ejemplo.py
python tests/demo_detector_ia.py
```

La demo usa C2 (`uso_red.py`) si está en la carpeta; si no, entrena solo con C1 y las variables de contexto.

## Cómo se usa con los datos reales

```python
import ventanas as v, rendimiento_tcp as c1, uso_red as c2, detector_ia as b2

def procesar(csv):
    p = v.cargar_paquetes(csv)
    t = v.tabla_ventanas(p)
    return c2.calcular_metricas(p, c1.calcular_metricas(p, t))

normales = [procesar(f"normal_{i}.csv") for i in (1, 2, 3)]
modelo = b2.entrenar(normales)          # solo tráfico normal
b2.guardar(modelo)                      # modelos/b2_modelos.joblib

escenario = procesar("red_lenta_1.csv")
alertas = b2.generar_alertas(b2.puntuar(modelo, escenario))
# alertas = {"Isolation Forest": ..., "LOF": ..., "One-Class SVM": ...} -> a evaluacion.evaluar
```

## A tener en cuenta

- **Entrenar solo con capturas normales.** Si una captura de escenario entra al entrenamiento, el modelo aprende que la degradación es normal.
- **Isolation Forest puede no ver el escaneo**: no usa métricas que en el tráfico normal son siempre iguales (SYN sin respuesta = 0). LOF y One-Class SVM sí lo detectan. Es un resultado para el informe, no un error.
- **One-Class SVM da más falsas alarmas** que los otros dos. Su parámetro `gamma_ocsvm` en `entrenar()` se puede ajustar (por defecto 0,1 dividido la cantidad de métricas).
- El tamaño medio y el reparto por protocolo de C2 no entran al modelo: en las pruebas agregaban ruido y bajaban la detección de Isolation Forest.
