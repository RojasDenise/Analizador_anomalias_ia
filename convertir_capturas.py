"""Convierte capturas .pcap/.pcapng al CSV del contrato (ver contrato.md) usando tshark directo.

Genera el mismo formato que AnalizadorDeRed.cargar_archivo_pcap, pero mucho más rápido:
PyShark procesa unos 350 paquetes/s; tshark -T fields procesa decenas de miles por segundo.

experimentos.py usa el .csv si existe al lado de la captura, así que alcanza con convertir
antes de correrlo; no hace falta tocar nada más.

Uso:
    python convertir_capturas.py C:\\capturas_tp                  # convierte todas las capturas sin CSV
    python convertir_capturas.py C:\\capturas_tp --forzar         # reconvierte aunque ya exista el CSV
    python convertir_capturas.py C:\\capturas_tp\\prueba.pcapng    # una sola captura
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime

import pandas as pd

from analizador import COLUMNAS_PAQUETES

# Dónde busca tshark si no está en el PATH (instalación normal de Wireshark)
RUTAS_TSHARK = [
    r"C:\Program Files\Wireshark\tshark.exe",
    r"C:\Program Files (x86)\Wireshark\tshark.exe",
]

# Campos que se piden a tshark, en este orden
CAMPOS = [
    "frame.time_epoch",
    "frame.len",
    "frame.protocols",
    "ip.src", "ipv6.src", "ip.dst", "ipv6.dst",
    "tcp.srcport", "tcp.dstport", "udp.srcport", "udp.dstport",
    "tcp.flags.syn", "tcp.flags.ack", "tcp.flags.reset",
    "tcp.analysis.ack_rtt",
    "tcp.analysis.retransmission", "tcp.analysis.fast_retransmission",
    "tcp.analysis.duplicate_ack",
    "tcp.analysis.zero_window",
    "dns.qry.name",
]


def buscar_tshark():
    ruta = shutil.which("tshark")
    if ruta:
        return ruta
    for candidata in RUTAS_TSHARK:
        if os.path.exists(candidata):
            return candidata
    raise SystemExit("No se encontró tshark. Instalá Wireshark o agregá su carpeta al PATH.")


def _flag(serie):
    """Flags de tshark ('1', 'True', '0', 'False', vacío) -> 0/1."""
    return serie.str.strip().str.lower().isin(["1", "true"]).astype(int)


def _presente(serie):
    """Campos tipo evento (retransmisión, ACK duplicado...): 1 si tshark los informa en el paquete."""
    return (serie.str.strip() != "").astype(int)


def _primero(a, b):
    """Valor de a; si está vacío, el de b (IPv4 o IPv6, TCP o UDP)."""
    return a.where(a.str.strip() != "", b)


def _limpiar_primero(serie):
    """Si un campo trae varias ocurrencias separadas por coma, se queda con la primera."""
    return serie.str.split(",").str[0].str.strip()


TAMANO_BLOQUE = 200_000  # filas por bloque: mantiene la memoria acotada aunque la captura sea enorme


def _limpiar(df):
    """Si un campo trae varias ocurrencias separadas por coma, se queda con la primera."""
    for col in CAMPOS:
        if col not in ("frame.protocols", "dns.qry.name"):
            df[col] = _limpiar_primero(df[col])
    return df


def _leer_bloques(tsv, tamano=None):
    """Lee la salida de tshark de a bloques, sin cargar todo el archivo en memoria."""
    lector = pd.read_csv(tsv, sep="\t", header=None, names=CAMPOS, dtype=str,
                         keep_default_na=False, quoting=3, on_bad_lines="skip",
                         chunksize=tamano or TAMANO_BLOQUE)
    for bloque in lector:
        yield _limpiar(bloque)


def _construir(df):
    """Arma las columnas del contrato a partir de los campos crudos de tshark."""
    ts = pd.to_numeric(df["frame.time_epoch"], errors="coerce")
    es_tcp = df["tcp.srcport"].str.strip() != ""
    es_udp = df["udp.srcport"].str.strip() != ""

    transporte = pd.Series(None, index=df.index, dtype="object")
    transporte[es_udp] = "UDP"
    transporte[es_tcp] = "TCP"

    out = pd.DataFrame(index=df.index)
    out["Timestamp"] = ts
    local = pd.to_datetime(ts, unit="s", utc=True).dt.tz_convert(
        datetime.now().astimezone().tzinfo).dt.tz_localize(None)
    out["Date"] = local.dt.strftime("%Y-%m-%d %H:%M:%S")
    out["Source IP"] = _primero(df["ip.src"], df["ipv6.src"]).replace("", None)
    out["Destination IP"] = _primero(df["ip.dst"], df["ipv6.dst"]).replace("", None)
    out["Transport Protocol"] = transporte
    out["Application Protocol"] = df["frame.protocols"].str.split(":").str[-1].str.upper()
    out["Hostname DNS"] = _limpiar_primero(df["dns.qry.name"]).replace("", None)
    out["Length"] = pd.to_numeric(df["frame.len"], errors="coerce").astype("Int64")
    out["Src Port"] = pd.to_numeric(_primero(df["tcp.srcport"], df["udp.srcport"]), errors="coerce").astype("Int64")
    out["Dst Port"] = pd.to_numeric(_primero(df["tcp.dstport"], df["udp.dstport"]), errors="coerce").astype("Int64")
    out["TCP SYN"] = _flag(df["tcp.flags.syn"])
    out["TCP ACK"] = _flag(df["tcp.flags.ack"])
    out["TCP RST"] = _flag(df["tcp.flags.reset"])
    out["TCP RTT"] = pd.to_numeric(df["tcp.analysis.ack_rtt"], errors="coerce")
    out["TCP Retransmission"] = ((_presente(df["tcp.analysis.retransmission"])
                                  + _presente(df["tcp.analysis.fast_retransmission"])) > 0).astype(int)
    out["TCP Dup ACK"] = _presente(df["tcp.analysis.duplicate_ack"])
    out["TCP Zero Window"] = _presente(df["tcp.analysis.zero_window"])
    return out[COLUMNAS_PAQUETES]


def convertir(pcap, csv=None, tshark=None, verbose=False):
    """Convierte una captura a CSV. Devuelve (ruta del csv, cantidad de paquetes, segundos).

    El CSV se escribe en un archivo temporal y recién al terminar toma su nombre definitivo:
    si algo falla a mitad de camino no queda un CSV incompleto que experimentos.py tome por bueno.
    """
    tshark = tshark or buscar_tshark()
    csv = csv or os.path.splitext(pcap)[0] + ".csv"
    inicio = time.time()

    comando = [tshark, "-r", pcap, "-n", "-T", "fields",
               "-E", "separator=/t", "-E", "occurrence=f", "-E", "quote=n"]
    for campo in CAMPOS:
        comando += ["-e", campo]

    # tshark escribe a un archivo temporal para no cargar toda la salida en memoria
    with tempfile.NamedTemporaryFile("w+", suffix=".tsv", delete=False, encoding="utf-8") as tmp:
        ruta_tmp = tmp.name
    csv_parcial = csv + ".parcial"
    filas = 0
    try:
        if verbose:
            print("  ejecutando tshark (es lo que más tarda)...", flush=True)
        with open(ruta_tmp, "w", encoding="utf-8") as salida:
            resultado = subprocess.run(comando, stdout=salida, stderr=subprocess.PIPE, text=True)
        if resultado.returncode != 0:
            raise RuntimeError(f"tshark falló en {os.path.basename(pcap)}: {resultado.stderr.strip()[:300]}")
        if verbose:
            print(f"  tshark terminó en {time.time() - inicio:.0f} s; armando el CSV...", flush=True)

        for bloque in _leer_bloques(ruta_tmp):
            tabla = _construir(bloque)
            tabla.to_csv(csv_parcial, mode="w" if filas == 0 else "a", header=(filas == 0), index=False)
            filas += len(tabla)
            if verbose:
                print(f"  {filas} paquetes procesados", flush=True)
        if filas == 0:
            raise RuntimeError(f"{os.path.basename(pcap)} no tiene paquetes.")
        os.replace(csv_parcial, csv)
    finally:
        os.remove(ruta_tmp)
        if os.path.exists(csv_parcial):
            os.remove(csv_parcial)
    return csv, filas, time.time() - inicio


def listar(ruta):
    if os.path.isfile(ruta):
        return [ruta]
    capturas = []
    for patron in ("*.pcapng", "*.pcap"):
        capturas += sorted(glob.glob(os.path.join(ruta, patron)))
    return capturas


def main(argv=None):
    parser = argparse.ArgumentParser(description="Convierte capturas a CSV con tshark")
    parser.add_argument("ruta", help="una captura o una carpeta con capturas")
    parser.add_argument("--forzar", action="store_true", help="reconvertir aunque el CSV ya exista")
    args = parser.parse_args(argv)

    capturas = listar(args.ruta)
    if not capturas:
        raise SystemExit("No se encontraron capturas .pcap/.pcapng.")
    tshark = buscar_tshark()

    for pcap in capturas:
        csv = os.path.splitext(pcap)[0] + ".csv"
        if os.path.exists(csv) and not args.forzar:
            print(f"- {os.path.basename(pcap)}: ya tiene CSV (usar --forzar para rehacerlo)")
            continue
        try:
            _, paquetes, segundos = convertir(pcap, csv, tshark, verbose=True)
        except RuntimeError as error:
            print(f"- {os.path.basename(pcap)}: ERROR {error}")
            continue
        print(f"- {os.path.basename(pcap)}: {paquetes} paquetes en {segundos:.1f} s -> {os.path.basename(csv)}")


if __name__ == "__main__":
    main(sys.argv[1:])