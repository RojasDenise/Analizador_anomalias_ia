import os
import socket
from datetime import datetime

import pandas as pd
import psutil
import pyshark


# Carpeta donde se guardan los CSV generados (funciona en Windows, Linux y macOS)
CARPETA_TMP = "tmp"

# Columnas del CSV de paquetes: una fila por paquete (ver contrato.md)
COLUMNAS_PAQUETES = [
    "Timestamp",            # segundos desde epoch (float), es la referencia de tiempo
    "Date",                 # fecha y hora local, legible
    "Source IP",
    "Destination IP",
    "Transport Protocol",   # TCP, UDP o vacío
    "Application Protocol", # capa más alta que reconoce Wireshark
    "Hostname DNS",         # nombre consultado, si es un paquete DNS
    "Length",               # frame.len en bytes
    "Src Port",
    "Dst Port",
    "TCP SYN",              # 0/1
    "TCP ACK",              # 0/1
    "TCP RST",              # 0/1
    "TCP RTT",              # tcp.analysis.ack_rtt en segundos
    "TCP Retransmission",   # 0/1
    "TCP Dup ACK",          # 0/1
    "TCP Zero Window",      # 0/1
]


def _ruta_por_defecto(nombre_archivo):
    os.makedirs(CARPETA_TMP, exist_ok=True)
    return os.path.join(CARPETA_TMP, nombre_archivo)


def _flag(valor):
    """Convierte los flags de tshark ('1', 'True', '0', 'False') a 0/1."""
    if valor is None:
        return 0
    return 1 if str(valor).strip().lower() in ("1", "true") else 0


def _numero(valor):
    try:
        return float(valor)
    except (TypeError, ValueError):
        return None


class AnalizadorDeRed:
    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Extracción de campos de un paquete
    # ------------------------------------------------------------------
    def obtener_info_ip(self, packet):
        """Devuelve (origen, destino) para IPv4 o IPv6; (None, None) si no es IP."""
        ip_layer = getattr(packet, "ip", None) or getattr(packet, "ipv6", None)
        if ip_layer:
            return ip_layer.src, ip_layer.dst
        return None, None

    def obtener_hostname_dns(self, packet):
        dns_layer = getattr(packet, "dns", None)
        if dns_layer is not None:
            return getattr(dns_layer, "qry_name", None)
        return None

    def obtener_info_transporte(self, packet):
        """Puertos e indicadores TCP. Los campos que no aplican quedan en None/0."""
        datos = {
            "Src Port": None,
            "Dst Port": None,
            "TCP SYN": 0,
            "TCP ACK": 0,
            "TCP RST": 0,
            "TCP RTT": None,
            "TCP Retransmission": 0,
            "TCP Dup ACK": 0,
            "TCP Zero Window": 0,
        }

        tcp = getattr(packet, "tcp", None)
        if tcp is not None:
            datos["Src Port"] = _numero(getattr(tcp, "srcport", None))
            datos["Dst Port"] = _numero(getattr(tcp, "dstport", None))
            datos["TCP SYN"] = _flag(getattr(tcp, "flags_syn", None))
            datos["TCP ACK"] = _flag(getattr(tcp, "flags_ack", None))
            datos["TCP RST"] = _flag(getattr(tcp, "flags_reset", None))
            datos["TCP RTT"] = _numero(getattr(tcp, "analysis_ack_rtt", None))
            # Los campos tcp.analysis.* solo existen cuando Wireshark detecta el evento
            datos["TCP Retransmission"] = int(
                tcp.get_field("analysis_retransmission") is not None
                or tcp.get_field("analysis_fast_retransmission") is not None
            )
            datos["TCP Dup ACK"] = int(tcp.get_field("analysis_duplicate_ack") is not None)
            datos["TCP Zero Window"] = int(tcp.get_field("analysis_zero_window") is not None)
            return datos

        udp = getattr(packet, "udp", None)
        if udp is not None:
            datos["Src Port"] = _numero(getattr(udp, "srcport", None))
            datos["Dst Port"] = _numero(getattr(udp, "dstport", None))
        return datos

    def extraer_campos(self, packet):
        """Arma la fila del CSV para un paquete. La usan la captura en vivo y la de archivo."""
        timestamp = float(packet.sniff_timestamp)
        source_ip, destination_ip = self.obtener_info_ip(packet)

        fila = {
            "Timestamp": timestamp,
            "Date": datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S"),
            "Source IP": source_ip,
            "Destination IP": destination_ip,
            "Transport Protocol": packet.transport_layer,
            "Application Protocol": packet.highest_layer,
            "Hostname DNS": self.obtener_hostname_dns(packet),
            "Length": int(packet.length),
        }
        fila.update(self.obtener_info_transporte(packet))
        return fila

    # ------------------------------------------------------------------
    # Captura
    # ------------------------------------------------------------------
    def capturar_trafico(self, interfaz="Wi-Fi", filtro=None, cantidad_paquetes=300, path_df=None):
        path_df = path_df or _ruta_por_defecto("df_trafico_stream.csv")
        cap = pyshark.LiveCapture(interface=interfaz, display_filter=filtro)

        captured_data = [self.extraer_campos(packet)
                         for packet in cap.sniff_continuously(packet_count=cantidad_paquetes)]

        pd.DataFrame(captured_data, columns=COLUMNAS_PAQUETES).to_csv(path_df, index=False)
        return captured_data

    def cargar_archivo_pcap(self, pcap_path: str = "", path_df=None):
        path_df = path_df or _ruta_por_defecto("df_trafico.csv")
        cap = pyshark.FileCapture(input_file=pcap_path, tshark_path="tshark")

        try:
            captured_data = [self.extraer_campos(packet) for packet in cap]
        finally:
            cap.close()

        pd.DataFrame(captured_data, columns=COLUMNAS_PAQUETES).to_csv(path_df, index=False)
        return captured_data

    # ------------------------------------------------------------------
    # Utilidades usadas por la app
    # ------------------------------------------------------------------
    def obtener_interfaces_disponibles(self):
        interfaces_info = psutil.net_if_addrs()

        interfaces = []
        for interfaz, direcciones in interfaces_info.items():
            for direccion in direcciones:
                interfaces.append({
                    "Nombre de la interfaz": interfaz,
                    "Familia": direccion.family.name,
                    "Dirección": direccion.address,
                    "Máscara de Red": direccion.netmask,
                })
        return interfaces

    def get_domain_name(self, ip_address):
        try:
            domain_name, _, _ = socket.gethostbyaddr(ip_address)
            return domain_name
        except (socket.herror, socket.gaierror, OSError):
            return "No se encontró el nombre de dominio"


if __name__ == "__main__":
    analizador = AnalizadorDeRed()

    opcion = input("¿Desea capturar tráfico en vivo (L) o leer un archivo pcap (F)? ").upper()

    if opcion == "F":
        pcap_path = input("Ingrese la ruta del archivo pcap (ejemplo: capturas/normal_1.pcapng): ")
        datos_capturados = analizador.cargar_archivo_pcap(pcap_path=pcap_path)
    elif opcion == "L":
        datos_capturados = analizador.capturar_trafico()
    else:
        print("Opción no válida. Por favor, elija 'L' o 'F'.")
        datos_capturados = []

    print(pd.DataFrame(datos_capturados, columns=COLUMNAS_PAQUETES))
