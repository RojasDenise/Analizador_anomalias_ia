import pyshark
import pandas as pd
import matplotlib.pyplot as plt
import psutil
from datetime import datetime
import re
import socket


class AnalizadorDeRed:
    def __init__(self):
        pass

    def obtener_hostname_dns(self, packet):
        # Verificar si es un paquete DNS
        if 'DNS' in packet:
            dns_layer = packet.dns
            if dns_layer.qry_name:
                hostname = dns_layer.qry_name
                return hostname
        return None

    def capturar_trafico(self, interfaz='Wi-Fi', filtro=None, cantidad_paquetes=300, path_df="tmp\df_trafico_stream.csv"):
        captured_data = []

        cap = pyshark.LiveCapture(interface=interfaz, display_filter=filtro)

        for packet in cap.sniff_continuously(packet_count=cantidad_paquetes):
            timestamp = float(packet.sniff_timestamp)
            formatted_time = datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

            # Obtener el hostname del paquete DNS
            hostname_dns = self.obtener_hostname_dns(packet)
            # Obtener credenciales HTTP
            http_dict = self.obtener_credenciales_http(packet)

            captured_data.append({
                'Timestamp': packet.sniff_timestamp,
                'Date': formatted_time,
                'Source IP': packet.ip.src,
                'Destination IP': packet.ip.dst,
                'Transport Protocol': packet.transport_layer,
                'Application Protocol': packet.highest_layer,
                'Hostname DNS': hostname_dns,
                'HTTP_Protocol_text_plain': http_dict['HTTP_Protocol_text_plain'], 
                'HTTP_Protocol_Username': http_dict['HTTP_Protocol_Username'], 
                'HTTP_Protocol_Password': http_dict['HTTP_Protocol_Password']
            })

        df = pd.DataFrame(captured_data)
        df.to_csv(path_df, index=False)
        return captured_data


    def obtener_hostname_dns_pcap(self, packet):
        # Verificar si es un paquete DNS
        if hasattr(packet, 'dns') and 'DNS' in packet:
            dns_layer = packet.dns
            if dns_layer.qry_name:
                hostname = dns_layer.qry_name
                return hostname
        return None

    def obtener_info_ip(self, packet):
        # Obtener información de capa IP de manera genérica
        ip_layer = getattr(packet, 'ip', None) or getattr(packet, 'ipv6', None)
        if ip_layer:
            return ip_layer.src, ip_layer.dst
        return None, None
    
    def cargar_archivo_pcap(self, pcap_path:str="",path_df="tmp\df_trafico.csv"):
        captured_data = []

        # Crear la instancia de FileCapture
        cap = pyshark.FileCapture(input_file=pcap_path, tshark_path="tshark")
  

        for packet in cap:
            timestamp = float(packet.sniff_timestamp)
            formatted_time = datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

            # Obtener el hostname del paquete DNS
            hostname_dns = self.obtener_hostname_dns_pcap(packet)

            # Obtener credenciales HTTP
            http_dict = self.obtener_credenciales_http(packet)

            # Obtener información de capa IP
            source_ip, destination_ip = self.obtener_info_ip(packet)

            captured_data.append({
                'Timestamp': packet.sniff_timestamp,
                'Date': formatted_time,
                'Source IP': source_ip,
                'Destination IP': destination_ip,
                'Transport Protocol': packet.transport_layer,
                'Application Protocol': packet.highest_layer,
                'Hostname DNS': hostname_dns,
                'HTTP_Protocol_text_plain': http_dict['HTTP_Protocol_text_plain'], 
                'HTTP_Protocol_Username': http_dict['HTTP_Protocol_Username'], 
                'HTTP_Protocol_Password': http_dict['HTTP_Protocol_Password']
            })
        

        df = pd.DataFrame(captured_data)
        df.to_csv(path_df, index=False)
        return captured_data
    

    def obtener_interfaces_disponibles(self):
        interfaces_info = psutil.net_if_addrs()
       
        interfaces = []
        for interfaz, direcciones in interfaces_info.items():
            for direccion in direcciones:
                dict_interfaz = {}
                dict_interfaz["Nombre de la interfaz"] = interfaz
                dict_interfaz["Familia"] = direccion.family.name
                dict_interfaz["Dirección"] = direccion.address
                dict_interfaz["Máscara de Red"] = direccion.netmask
                interfaces.append(dict_interfaz)
        return interfaces

    def obtener_credenciales_http(self, packet):
        http_protocol = None
        username = None
        password = None
        if packet[4].startswith('POST'):
            print("Paquete HTTP POST capturado:")
            print(packet)

            # Extraer username y password del formulario HTTP
            load = packet[10].load
            username = re.search(r'username=(.*?)&', load)
            password = re.search(r'password=(.*?)&', load)

            if username and password:
                print(f"Username: {username.group(1)}")
                print(f"Password: {password.group(1)}")
        return {"HTTP_Protocol_text_plain": http_protocol, "HTTP_Protocol_Username": username, "HTTP_Protocol_Password": password}


    def get_domain_name(self, ip_address):
        try:
            domain_name, _, _ = socket.gethostbyaddr(ip_address)
            return domain_name
        except socket.herror:
            return "No se encontró el nombre de dominio"
    
    
    
if __name__ == '__main__':
    analizador = AnalizadorDeRed()

    # Preguntar al usuario si desea capturar en vivo o leer un archivo pcap
    opcion = input("¿Desea capturar trafico en vivo (L) o leer un archivo pcap (F)? Ingrese la letra que corresponda: ").upper()

    if opcion == 'F':
        # Leer un archivo pcap
        pcap_path = input("Ingrese la ruta del archivo pcap (ejemplo: capturas\captura_7_11_full.pcapng): ")
        datos_capturados = analizador.cargar_archivo_pcap(pcap_path=pcap_path)
        usuarios_contrasena: analizador.obtener_credenciales_http()
    elif opcion == 'L':
        # Capturar en vivo
        datos_capturados = analizador.capturar_trafico()
    else:
        print("Opción no válida. Por favor, elija 'L' o 'F'.")

    # Resto del código
    print(pd.DataFrame(datos_capturados))
    print("*++++++"*20)

