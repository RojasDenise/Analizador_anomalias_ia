"""Experimentos E1 a E5 con las capturas reales (módulo E).

Junta todo: procesa las capturas, calcula C1 y C2, fija los umbrales de las reglas,
entrena los modelos de B2 con las capturas normales y evalúa cada método.

Carpeta de capturas esperada (los nombres importan):
    capturas/
        normal_1.pcapng, normal_2.pcapng, normal_3.pcapng      tráfico normal (entrenamiento)
        red_lenta_1.pcapng, saturacion_1.pcapng, ...           escenarios (evaluación)
        starlink_normal_1.pcapng, ...                          opcional: E5
        tiempos.csv                                            planilla: captura, inicio, fin, escenario[, entorno]
También acepta los CSV ya generados por analizador.py con el mismo nombre (normal_1.csv, ...).

Entornos de captura: cada escenario se compara solo con las normales de su mismo entorno
(misma red, misma máquina, mismo generador de tráfico). El entorno se lee así:
    normales:   normal_<entorno>_<n>  (normal_vm_milo_1 -> "vm_milo");  normal_<n> -> "fibra"
    escenarios: columna opcional "entorno" de tiempos.csv; si falta o está vacía -> "fibra"
Los umbrales y los modelos se ajustan por entorno y después se evalúa todo junto.

Uso:
    python experimentos.py                              # todos los experimentos
    python experimentos.py --solo E1 E4                 # algunos
    python experimentos.py --capacidad-mbps 300         # activa la regla de uso alto de C2

Resultados en resultados/: un CSV por experimento y la tabla común resultados.csv.
"""
import argparse
import glob
import os
import re

import pandas as pd

import evaluacion as ev
import ventanas as v

CARPETA_CAPTURAS = "capturas"
CARPETA_RESULTADOS = "resultados"
PREFIJO_NORMAL = "normal_"
PREFIJO_STARLINK = "starlink_normal_"
ENTORNO_POR_DEFECTO = "fibra"


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


def entorno_normal(nombre):
    """Entorno de una captura normal según su nombre: normal_vm_milo_2 -> "vm_milo", normal_3 -> "fibra"."""
    resto = re.sub(r"_?\d+$", "", nombre[len(PREFIJO_NORMAL):])
    return resto or ENTORNO_POR_DEFECTO


def entorno_escenario(nombre, tiempos):
    """Entorno de una captura de escenario, de la columna "entorno" de tiempos.csv (o "fibra")."""
    if "entorno" not in tiempos.columns:
        return ENTORNO_POR_DEFECTO
    valores = tiempos.loc[tiempos["captura"] == nombre, "entorno"].dropna().str.strip()
    valores = valores[valores != ""]
    return valores.iloc[0] if len(valores) else ENTORNO_POR_DEFECTO


def agrupar(nombres, entorno_de):
    """{entorno: [nombres]} manteniendo el orden."""
    grupos = {}
    for n in nombres:
        grupos.setdefault(entorno_de(n), []).append(n)
    return grupos


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


def _aplicar_por_captura(ajuste, tablas, desde=0):
    """Aplica los métodos captura por captura y une todo en una sola tabla para evaluar.

    Cada captura se corre en el tiempo (400 días por captura) para que dos capturas grabadas
    una atrás de otra no compartan ventanas. Las demoras y conteos no cambian.
    desde: cuántas capturas ya se corrieron antes (para seguir la numeración entre entornos).
    """
    unidas, alertas = [], {}
    for i, tabla in enumerate(tablas, start=desde):
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

    Acepta listas (un solo entorno) o diccionarios {entorno: [tablas]}: en ese caso los umbrales
    y los modelos se ajustan con las normales de cada entorno y se aplican solo a sus escenarios.
    Devuelve (resultado global, resultado por escenario, alertas, tabla unida).
    """
    if not isinstance(tablas_normales, dict):
        tablas_normales = {ENTORNO_POR_DEFECTO: tablas_normales}
        tablas_escenario = {ENTORNO_POR_DEFECTO: tablas_escenario}

    unidas, alertas, n = [], {}, 0
    for entorno, escenarios in tablas_escenario.items():
        if not escenarios:
            continue
        if not tablas_normales.get(entorno):
            raise ValueError(f"No hay capturas normales del entorno '{entorno}'.")
        ajuste = ajustar_metodos(tablas_normales[entorno], capacidad_mbps)
        unida, a = _aplicar_por_captura(ajuste, escenarios, desde=n)
        n += len(escenarios)
        unidas.append(unida)
        for metodo, lista in a.items():
            alertas.setdefault(metodo, []).append(lista)
    if not unidas:
        raise ValueError("No hay capturas de escenario para evaluar.")
    alertas = {m: pd.concat(lista, ignore_index=True) for m, lista in alertas.items()}
    unida = pd.concat(unidas)
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
    grupo_n = agrupar(normales, entorno_normal)
    grupo_e = agrupar(escenarios, lambda n: entorno_escenario(n, tiempos))
    print(f"Starlink: {starlink or 'ninguna (E5 se omite)'}")
    for entorno in sorted(set(grupo_n) | set(grupo_e)):
        print(f"Entorno {entorno}: normales {grupo_n.get(entorno, [])} · escenarios {grupo_e.get(entorno, [])}")
    if not normales or not escenarios:
        raise SystemExit("Faltan capturas normales o de escenario en la carpeta.")
    sin_normales = [e for e in grupo_e if e not in grupo_n]
    if sin_normales:
        raise SystemExit(f"Estos entornos tienen escenarios pero ninguna captura normal: {sin_normales}")
    sin_tiempos = [n for n in escenarios if n not in set(tiempos["captura"])]
    if sin_tiempos:
        print(f"Aviso: estas capturas no tienen tiempos anotados y quedan como todo normal: {sin_tiempos}")

    print("\nCargando capturas...")
    paquetes = {n: cargar_paquetes(rutas[n]) for n in normales + escenarios + starlink}

    def tablas(nombres, segundos):
        return [procesar(n, paquetes[n], tiempos, segundos) for n in nombres]

    def por_entorno(grupos, segundos):
        return {e: tablas(nombres, segundos) for e, nombres in grupos.items()}

    s = args.segundos
    tn, te = por_entorno(grupo_n, s), por_entorno(grupo_e, s)

    # Resumen de C2 por escenario (normales + escenarios), para la sección de C2 del front
    import uso_red as c2
    _guardar(c2.resumen_por_escenario(pd.concat(sum(tn.values(), []) + sum(te.values(), []))), "C2_resumen.csv")

    if "E1" in args.solo or "E4" in args.solo:
        g, esc, alertas, unida = correr(tn, te, args.capacidad_mbps)

        if "E1" in args.solo:
            _mostrar(f"E1 · Comparación de métodos (ventana {s} s)", g, COLS)
            _mostrar("E1 · Por escenario", esc)
            h = ev.hipotesis(g)
            _mostrar("E1 · Hipótesis (contra reglas fijas)", h)
            _guardar(g, "E1_global.csv"), _guardar(esc, "E1_por_escenario.csv"), _guardar(h, "E1_hipotesis.csv")
            ev.registrar_resultado(g, "E1", f"ventana {s} s, {len(normales)} normales, {len(escenarios)} escenarios, {len(grupo_e)} entornos")

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
                g2, _, _, _ = correr(por_entorno(grupo_n, seg), por_entorno(grupo_e, seg), args.capacidad_mbps)
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
        # Cada entorno usa sus primeras k capturas normales (o todas, si tiene menos de k)
        for k in range(1, max(len(x) for x in tn.values()) + 1):
            try:
                g3, _, _, _ = correr({e: x[:k] for e, x in tn.items()}, te, args.capacidad_mbps)
            except ValueError as error:
                print(f"Aviso E3, {k} captura(s) normal(es) omitida: {error}")
                continue
            filas.append(g3.assign(capturas_normales=k))
        e3 = pd.concat(filas, ignore_index=True) if filas else pd.DataFrame(columns=["capturas_normales"] + COLS)
        _mostrar("E3 · Cantidad de capturas normales", e3, ["capturas_normales"] + COLS)
        _guardar(e3, "E3_normales.csv")
        ev.registrar_resultado(e3, "E3", f"1 a {k} capturas normales por entorno")

    if "E5" in args.solo and starlink and ENTORNO_POR_DEFECTO not in tn:
        print("Aviso: E5 se omite, necesita capturas normales de fibra (normal_1, normal_2, ...).")
    elif "E5" in args.solo and starlink:
        # Métodos ajustados con fibra, aplicados a tráfico normal de Starlink: toda alerta es falsa alarma
        ajuste = ajustar_metodos(tn[ENTORNO_POR_DEFECTO], args.capacidad_mbps)
        ts, alertas5 = _aplicar_por_captura(ajuste, tablas(starlink, s))
        g5, _ = ev.evaluar(ts, alertas5)
        _mostrar("E5 · Falsas alarmas en Starlink (ajustado con fibra)", g5,
                 ["metodo", "ventanas", "FP", "falsas_alarmas_c100"])
        _guardar(g5, "E5_starlink.csv")
        ev.registrar_resultado(g5, "E5", f"{len(starlink)} capturas normales de Starlink")

    print(f"\nResultados guardados en {CARPETA_RESULTADOS}/")


if __name__ == "__main__":
    main()
