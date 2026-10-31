"""Experimentos E1 a E5 con las capturas reales (módulo E).

Junta todo: procesa las capturas, calcula C1 y C2, fija los umbrales de las reglas,
entrena los modelos de B2 con las capturas normales y evalúa cada método.

Carpeta de capturas esperada (los nombres importan):
    capturas/
        normal_1.pcapng, normal_2.pcapng, normal_3.pcapng      tráfico normal (entrenamiento)
        red_lenta_1.pcapng, saturacion_1.pcapng, ...           escenarios (evaluación)
        starlink_normal_1.pcapng, ...                          opcional: E5
        tiempos.csv                                            planilla: captura, inicio, fin, escenario
También acepta los CSV ya generados por analizador.py con el mismo nombre (normal_1.csv, ...).

Uso:
    python experimentos.py                              # todos los experimentos
    python experimentos.py --solo E1 E4                 # algunos
    python experimentos.py --capacidad-mbps 300         # activa la regla de uso alto de C2

Resultados en resultados/: un CSV por experimento y la tabla común resultados.csv.
"""
import argparse
import glob
import os

import pandas as pd

import evaluacion as ev
import ventanas as v

CARPETA_CAPTURAS = "capturas"
CARPETA_RESULTADOS = "resultados"
PREFIJO_NORMAL = "normal_"
PREFIJO_STARLINK = "starlink_normal_"


# ----------------------------------------------------------------------
# Carga de capturas (con caché: PyShark tarda varios minutos por captura)
# ----------------------------------------------------------------------
def _nombre(ruta):
    return os.path.splitext(os.path.basename(ruta))[0]


def listar_capturas(carpeta):
    """{nombre sin extensión: ruta}. Si hay .pcapng y .csv con el mismo nombre, usa el .csv."""
    rutas = {}
    for patron in ("*.pcapng", "*.pcap", "*.csv"):
        for ruta in sorted(glob.glob(os.path.join(carpeta, patron))):
            nombre = _nombre(ruta)
            if nombre == "tiempos":
                continue
            rutas[nombre] = ruta
    return rutas


def cargar_paquetes(ruta):
    """Lee un CSV de paquetes; si es una captura, la convierte una vez y guarda el CSV al lado."""
    if ruta.endswith(".csv"):
        return v.cargar_paquetes(ruta)
    csv = os.path.splitext(ruta)[0] + ".csv"
    if not os.path.exists(csv):
        from analizador import AnalizadorDeRed
        print(f"  convirtiendo {os.path.basename(ruta)} con PyShark (puede tardar)...")
        AnalizadorDeRed().cargar_archivo_pcap(pcap_path=ruta, path_df=csv)
    return v.cargar_paquetes(csv)


def leer_tiempos(ruta):
    """Planilla de tiempos, con la columna captura sin extensión para comparar con los nombres."""
    tiempos = v.leer_intervalos(ruta)
    tiempos["captura"] = tiempos["captura"].map(_nombre)
    return tiempos


# ----------------------------------------------------------------------
# Procesamiento de una captura: ventanas, etiquetas, C1 y C2
# ----------------------------------------------------------------------
def procesar(nombre, paquetes, tiempos, segundos):
    """Tabla de ventanas de una captura, etiquetada y con las métricas de C1 y C2.

    Se descartan la primera y la última ventana: la grabación empieza y termina a mitad de una
    ventana, así que tienen menos paquetes de lo normal. En el entrenamiento, esas ventanas
    incompletas hacían que Isolation Forest aceptara como normal casi cualquier cosa.
    """
    import rendimiento_tcp as c1
    import uso_red as c2

    tabla = v.tabla_ventanas(paquetes, segundos)
    if len(tabla) > 2:
        tabla = tabla.iloc[1:-1]
    tabla = v.etiquetar_ventanas(tabla, tiempos[tiempos["captura"] == nombre])
    tabla = c1.calcular_metricas(paquetes, tabla, segundos)
    tabla = c2.calcular_metricas(paquetes, tabla, segundos)
    tabla["captura"] = nombre
    return tabla


# ----------------------------------------------------------------------
# Métodos: reglas fijas (C1 + C2) y modelos (B2)
# ----------------------------------------------------------------------
def ajustar_metodos(normales, capacidad_mbps=None):
    """Con las tablas normales fija los umbrales de las reglas y entrena los modelos."""
    import detector_ia as b2
    import rendimiento_tcp as c1
    import uso_red as c2

    return {
        "umbral_rtt": c1.umbral_rtt(normales),
        "mbps_normal": c2.mbps_referencia(normales),
        "capacidad_mbps": capacidad_mbps,
        "b2": b2.entrenar(normales),
    }


def aplicar_metodos(ajuste, tabla):
    """Alertas de cada método sobre una tabla: {"Reglas fijas": ..., "Isolation Forest": ..., ...}."""
    import detector_ia as b2
    import rendimiento_tcp as c1
    import uso_red as c2

    reglas = pd.concat([
        c1.reglas_c1(tabla, ajuste["umbral_rtt"]),
        c2.reglas_c2(tabla, capacidad_mbps=ajuste["capacidad_mbps"], mbps_normal=ajuste["mbps_normal"]),
    ], ignore_index=True)
    return {"Reglas fijas": reglas, **b2.generar_alertas(b2.puntuar(ajuste["b2"], tabla))}


def _aplicar_por_captura(ajuste, tablas):
    """Aplica los métodos captura por captura y une todo en una sola tabla para evaluar.

    Cada captura se corre en el tiempo (400 días por captura) para que dos capturas grabadas
    una atrás de otra no compartan ventanas. Las demoras y conteos no cambian.
    """
    unidas, alertas = [], {}
    for i, tabla in enumerate(tablas):
        corrimiento = pd.Timedelta(days=400 * i)
        t = tabla.copy()
        t.index = t.index + corrimiento
        unidas.append(t)
        for metodo, a in aplicar_metodos(ajuste, tabla).items():
            a = a.copy()
            a["inicio_ventana"] = pd.to_datetime(a["inicio_ventana"]) + corrimiento
            alertas.setdefault(metodo, []).append(a)
    alertas = {m: pd.concat(lista, ignore_index=True) for m, lista in alertas.items()}
    return pd.concat(unidas), alertas


def correr(tablas_normales, tablas_escenario, capacidad_mbps=None):
    """Ajusta con las normales, aplica sobre los escenarios y evalúa.

    Devuelve (resultado global, resultado por escenario, alertas, tabla unida).
    """
    ajuste = ajustar_metodos(tablas_normales, capacidad_mbps)
    unida, alertas = _aplicar_por_captura(ajuste, tablas_escenario)
    glob_, esc = ev.evaluar(unida, alertas)
    return glob_, esc, alertas, unida


# ----------------------------------------------------------------------
# Experimentos
# ----------------------------------------------------------------------
def _guardar(df, nombre):
    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
    ruta = os.path.join(CARPETA_RESULTADOS, nombre)
    df.to_csv(ruta, index=False)
    return ruta


def _mostrar(titulo, df, columnas=None):
    print(f"\n== {titulo} ==")
    print((df[columnas] if columnas else df).round(2).to_string(index=False))


COLS = ["metodo", "precision", "recall", "f1", "falsas_alarmas_c100", "eventos_detectados", "demora_media_s"]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Experimentos E1 a E5")
    parser.add_argument("--capturas", default=CARPETA_CAPTURAS)
    parser.add_argument("--tiempos", default=None, help="por defecto <capturas>/tiempos.csv")
    parser.add_argument("--segundos", type=int, default=v.SEGUNDOS_POR_DEFECTO)
    parser.add_argument("--ventanas-e2", type=int, nargs="+", default=[5, 10, 15])
    parser.add_argument("--capacidad-mbps", type=float, default=None)
    parser.add_argument("--solo", nargs="+", default=["E1", "E2", "E3", "E4", "E5"])
    args = parser.parse_args(argv)

    pd.set_option("display.width", 200)
    rutas = listar_capturas(args.capturas)
    tiempos = leer_tiempos(args.tiempos or os.path.join(args.capturas, "tiempos.csv"))

    normales = sorted(n for n in rutas if n.startswith(PREFIJO_NORMAL))
    starlink = sorted(n for n in rutas if n.startswith(PREFIJO_STARLINK))
    escenarios = sorted(n for n in rutas if n not in normales and n not in starlink)
    print(f"Normales: {normales}\nEscenarios: {escenarios}\nStarlink: {starlink or 'ninguna (E5 se omite)'}")
    if not normales or not escenarios:
        raise SystemExit("Faltan capturas normales o de escenario en la carpeta.")
    sin_tiempos = [n for n in escenarios if n not in set(tiempos["captura"])]
    if sin_tiempos:
        print(f"Aviso: estas capturas no tienen tiempos anotados y quedan como todo normal: {sin_tiempos}")

    print("\nCargando capturas...")
    paquetes = {n: cargar_paquetes(rutas[n]) for n in normales + escenarios + starlink}

    def tablas(nombres, segundos):
        return [procesar(n, paquetes[n], tiempos, segundos) for n in nombres]

    s = args.segundos
    tn, te = tablas(normales, s), tablas(escenarios, s)

    if "E1" in args.solo or "E4" in args.solo:
        g, esc, alertas, unida = correr(tn, te, args.capacidad_mbps)

        if "E1" in args.solo:
            _mostrar(f"E1 · Comparación de métodos (ventana {s} s)", g, COLS)
            _mostrar("E1 · Por escenario", esc)
            h = ev.hipotesis(g)
            _mostrar("E1 · Hipótesis (contra reglas fijas)", h)
            _guardar(g, "E1_global.csv"), _guardar(esc, "E1_por_escenario.csv"), _guardar(h, "E1_hipotesis.csv")
            ev.registrar_resultado(g, "E1", f"ventana {s} s, {len(normales)} normales, {len(escenarios)} escenarios")

        if "E4" in args.solo:
            print("\n== E4 · Caso de confusión: etiqueta asignada por escenario ==")
            filas = []
            for metodo, a in alertas.items():
                m = ev.matriz_etiquetas(unida, a)
                print(f"\n{metodo}\n{m}")
                filas.append(m.reset_index().assign(metodo=metodo))
            _guardar(pd.concat(filas, ignore_index=True), "E4_matrices.csv")

    if "E2" in args.solo:
        filas = []
        for seg in args.ventanas_e2:
            try:
                g2, _, _, _ = correr(tablas(normales, seg), tablas(escenarios, seg), args.capacidad_mbps)
            except ValueError as error:
                print(f"Aviso E2, ventana {seg} s omitida: {error}")
                continue
            filas.append(g2.assign(ventana_s=seg))
        e2 = pd.concat(filas, ignore_index=True) if filas else pd.DataFrame(columns=["ventana_s"] + COLS)
        _mostrar("E2 · Tamaño de ventana", e2, ["ventana_s"] + COLS)
        _guardar(e2, "E2_ventanas.csv")
        ev.registrar_resultado(e2, "E2", f"ventanas {args.ventanas_e2}")

    if "E3" in args.solo:
        filas = []
        for k in range(1, len(normales) + 1):
            try:
                g3, _, _, _ = correr(tn[:k], te, args.capacidad_mbps)
            except ValueError as error:
                print(f"Aviso E3, {k} captura(s) normal(es) omitida: {error}")
                continue
            filas.append(g3.assign(capturas_normales=k))
        e3 = pd.concat(filas, ignore_index=True) if filas else pd.DataFrame(columns=["capturas_normales"] + COLS)
        _mostrar("E3 · Cantidad de capturas normales", e3, ["capturas_normales"] + COLS)
        _guardar(e3, "E3_normales.csv")
        ev.registrar_resultado(e3, "E3", f"1 a {len(normales)} capturas normales")

    if "E5" in args.solo and starlink:
        # Métodos ajustados con fibra, aplicados a tráfico normal de Starlink: toda alerta es falsa alarma
        ajuste = ajustar_metodos(tn, args.capacidad_mbps)
        ts, alertas5 = _aplicar_por_captura(ajuste, tablas(starlink, s))
        g5, _ = ev.evaluar(ts, alertas5)
        _mostrar("E5 · Falsas alarmas en Starlink (ajustado con fibra)", g5,
                 ["metodo", "ventanas", "FP", "falsas_alarmas_c100"])
        _guardar(g5, "E5_starlink.csv")
        ev.registrar_resultado(g5, "E5", f"{len(starlink)} capturas normales de Starlink")

    print(f"\nResultados guardados en {CARPETA_RESULTADOS}/")


if __name__ == "__main__":
    main()
