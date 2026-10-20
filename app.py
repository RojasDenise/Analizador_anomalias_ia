import streamlit as st
import pandas as pd
from analizador import AnalizadorDeRed
import matplotlib.pyplot as plt

analizador_red = AnalizadorDeRed()

# Lista para almacenar los datos capturados
path_df="tmp\df_trafico.csv"
colums_df = ["Source IP", "Destination IP", "Transport Protocol", "Date", "Application Protocol", "Hostname DNS", "HTTP_Protocol_text_plain", "HTTP_Protocol_Username", "HTTP_Protocol_Password"]

datos_capturados = pd.DataFrame(columns=colums_df)

@st.cache_data
def get_UN_data(set_index:str, archivo_df):
    df = archivo_df.set_index(set_index)
    # df = pd.read_csv(path_df).set_index(set_index)
    return df

@st.cache_data
def get_dataframe(df, id_loc:str, etiqueta:str, num_filas:int):
    if not id:
            st.error("por favor seleccione al menos una opcion.")
    else:
        data = df.loc[id_loc]
        st.write(f"### {etiqueta}", data[:num_filas].sort_index())

@st.cache_data
def plot_dataframe(df, id_loc:str, dest_ip:str="Todas", con_protocolo_transporte:bool=False, con_protocolo_aplicacion:bool=False):
    if not id:
            st.error("por favor seleccione al menos una opcion.")
    else:
        data = df[df["Source IP"] == id_loc]
        if dest_ip != "Todas":
            data = data[data['Destination IP'] == dest_ip]

        if con_protocolo_transporte:
             ip_destination_counts = data.groupby(['Destination IP', 'Transport Protocol']).size().unstack().fillna(0)
        elif con_protocolo_aplicacion:
             ip_destination_counts = data.groupby(['Destination IP', 'Application Protocol']).size().unstack().fillna(0)
        else:
             ip_destination_counts = data['Destination IP'].value_counts()

        st.write(f"##### Grafico del trafico. IP Origen: {id_loc} a IP Destino {dest_ip} ")
        fig, ax = plt.subplots(figsize=(12, 6))
        ip_destination_counts.plot(kind='bar', color='skyblue', ax=ax)
        plt.xlabel('IP Destino')
        plt.ylabel('Frecuencia')

        # Rotar las etiquetas del eje x
        plt.xticks(rotation=45, ha="right")

        # Ajustar la posición de las etiquetas
        plt.tight_layout()

        st.pyplot(fig)

@st.cache_data
def plot_time_series(df):
    # Convertir la columna 'timestamp' a tipo datetime
    df['timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')

    # Crear una nueva columna 'seconds' que representa el tiempo en segundos
    df['seconds'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()

    # Agrupar por segundos y contar la cantidad de paquetes en cada segundo
    counts_per_second = df.groupby('seconds').size()

    # Crear el gráfico de serie temporal
    fig, ax = plt.subplots()
    counts_per_second.plot(kind='line', marker='o', color='skyblue', ax=ax)
    plt.xlabel('Tiempo (segundos)')
    plt.ylabel('Número de Paquetes')
    plt.title('Serie Temporal del Tráfico')
    st.pyplot(fig)


st.header('PySharkStream Analytica')
st.header('Analizador de REDES  :sunglasses:', divider='rainbow')

st.sidebar.header("Configuración")
# Añadir widget para cargar archivo CSV en lugar de pcap
archivo_csv = st.sidebar.file_uploader("Cargar archivo CSV", type=["csv"])

# Verifico que se subio el archivo a analizar.
if archivo_csv is not None:
    datos_capturados = pd.read_csv(archivo_csv)
    df_origen = get_UN_data(set_index=colums_df[0],archivo_df=datos_capturados)
    df_destino = get_UN_data(set_index=colums_df[1],archivo_df=datos_capturados)
    df_transport_protocol = get_UN_data(set_index=colums_df[2],archivo_df=datos_capturados)
    df_date = get_UN_data(set_index=colums_df[3],archivo_df=datos_capturados)
    df_aplication_protocol = get_UN_data(set_index=colums_df[4],archivo_df=datos_capturados)


    interfaz = st.sidebar.selectbox("Nombre de la interfaz de red", ["Wi-Fi", "Ethernet"])
    ip_origen = st.sidebar.selectbox("Filtro de paquetes por Direccion IP Origen", list(df_origen.index))
    ip_destino = st.sidebar.selectbox("Filtro de paquetes por Direccion IP Destino", list(df_destino.index.unique()))
    protocolo_transporte = st.sidebar.selectbox("Filtro de paquetes por protocolo de capa de transporte", list(df_transport_protocol.index.unique()))
    fecha_hora = st.sidebar.selectbox("Filtro de paquetes por Fecha y Hora", list(df_date.index.unique()))
    protocolo_aplicacion = st.sidebar.selectbox("Filtro de paquetes por protocolo de capa de apicacion", list(df_aplication_protocol.index.unique()))
    cantidad_paquetes = st.sidebar.slider("Número de paquetes a capturar", min_value=1, max_value=len(datos_capturados.index), value=50)

tab1, tab2 , tab3, tab4= st.tabs(["Interfaces", "Análisis de Tráfico de Red", "Graficos", "Seguridad - HTTP"])

with tab1:
    st.title("Interfaces")
    intfaces = analizador_red.obtener_interfaces_disponibles()
    # Imprimir los elementos de la lista
    for interfaz in intfaces:
        for key, value in interfaz.items():
            st.write(f"{key}:  {value}")


        st.write("-" * 30)


with tab2:
    try:
        st.title("Análisis de Tráfico de Red")
        if archivo_csv == None:
            st.warning("Debe subir un archivo")
        else:
            if st.sidebar.button("Capturar"):
                # Visualiza los datos capturados
                st.subheader("Datos Capturados - General")
                st.dataframe(datos_capturados, use_container_width=True)

                st.subheader("Datos Capturados - IP Origen")
                get_dataframe(df_origen, ip_origen, colums_df[0], cantidad_paquetes)

                st.subheader("Datos Capturados - IP Destino")
                get_dataframe(df_destino, ip_destino, colums_df[1], cantidad_paquetes)

                st.subheader("Datos Capturados - Protocolo de Capa de Transporte")
                get_dataframe(df_transport_protocol, protocolo_transporte, colums_df[2], cantidad_paquetes)

                st.subheader("Datos Capturados - Fecha y Hora")
                get_dataframe(df_date, fecha_hora, colums_df[3], cantidad_paquetes)

                st.subheader("Datos Capturados - Protocolo de Capa de Transporte")
                get_dataframe(df_aplication_protocol, protocolo_aplicacion, colums_df[4], cantidad_paquetes)

    except Exception as e:
        st.error(f"Error: {e}")

with tab3:
    st.title("Análisis del Tráfico de Red por IP")
    if archivo_csv == None:
            st.warning("Debe subir un archivo")
    else:
         
        st.subheader("Datos Capturados - IP Origen", divider="rainbow")
        plot_dataframe(df=datos_capturados, id_loc=ip_origen)

        st.subheader("Nombre de Dominio de la IP")
        ip_dominio = st.text_input("Ingrese la IP que desea consultar: ", value="8.8.8.8")
        st.write(f"La dirección IP {ip_dominio} corresponde al dominio: {analizador_red.get_domain_name(ip_dominio)}")
        st.divider()

        # st.subheader("Datos Capturados - IP Origen por Protocolo de Transporte", divider="rainbow")
        # plot_dataframe(df=datos_capturados, id_loc=ip_origen, dest_ip=ip_destino, con_protocolo_transporte=True)
        # st.divider()
        
        st.subheader("Uso de la Red", divider="rainbow")
        plot_time_series(df=datos_capturados)

# with tab4:
#     st.title("Análisis del Tráfico de Red por IP")
#     if archivo_csv == None:
#         st.warning("Debe subir un archivo")
#     else:
#         st.write(f"La dirección IP {ip_dominio} corresponde al dominio: {analizador_red.get_domain_name(ip_dominio)}")
          


