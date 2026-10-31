# Contrato de datos

Formato que respetan todos los módulos (C1, C2, B2 y E). Si algo de acá cambia, se acuerda entre todos.

## 1. CSV de paquetes (`analizador.py`)

Una fila por paquete. Se genera con `cargar_archivo_pcap` (archivo) o `capturar_trafico` (en vivo).

| Columna | Contenido |
| --- | --- |
| `Timestamp` | Segundos desde epoch. Es la referencia de tiempo. |
| `Date` | Fecha y hora **local** de la compu, legible. |
| `Source IP`, `Destination IP` | IPv4 o IPv6. Vacío si el paquete no es IP. |
| `Transport Protocol` | `TCP`, `UDP` o vacío. |
| `Application Protocol` | Capa más alta que reconoce Wireshark. |
| `Hostname DNS` | Nombre consultado, solo en paquetes DNS. |
| `Length` | Tamaño del paquete en bytes (`frame.len`). |
| `Src Port`, `Dst Port` | Puertos TCP o UDP. |
| `TCP SYN`, `TCP ACK`, `TCP RST` | Flags TCP (0/1). |
| `TCP RTT` | `tcp.analysis.ack_rtt` en segundos. Solo en los ACK que Wireshark puede medir. |
| `TCP Retransmission` | 1 si es retransmisión (normal o rápida). |
| `TCP Dup ACK` | 1 si es ACK duplicado. |
| `TCP Zero Window` | 1 si el receptor anunció ventana en cero. |

## 2. Tabla de ventanas (`ventanas.py`)

`tabla_ventanas(df_paquetes, segundos=5)` devuelve **una fila por ventana**, sin huecos, indexada por `inicio_ventana` (hora local).

Columnas base: `paquetes`, `bytes`, `duracion_s`.

Variables de contexto (prefijo `ctx_`):

- `ctx_max_puertos_por_ip`: máximo de puertos destino distintos tocados por una misma IP origen.
- `ctx_syn_sin_respuesta`: SYN que nunca recibieron SYN-ACK.
- `ctx_consultas_dns`: paquetes DNS con nombre consultado.

Cada módulo agrega sus columnas con su prefijo: `c1_` (latencia y pérdida), `c2_` (ancho de banda y uso), `b2_` (modelos de IA). Ejemplos: `c1_rtt_medio_ms`, `c2_mbps`, `b2_puntaje`.

## 3. Alertas

Cada módulo devuelve un DataFrame con estas columnas (`ventanas.COLUMNAS_ALERTAS`):

| Columna | Contenido |
| --- | --- |
| `inicio_ventana` | Ventana donde ocurre la alerta. |
| `modulo` | `c1`, `c2` o `b2`. |
| `etiqueta` | Una de las cuatro de abajo. |
| `detalle` | Texto con los valores que la dispararon. |

Etiquetas permitidas (`ventanas.ETIQUETAS`):

- Degradación de rendimiento
- Congestión o saturación
- Pico de uso normal
- Comportamiento inusual

`ventanas.validar_alertas(alertas)` chequea el formato.

## 4. Tiempos anotados y regla de etiquetado

Quien graba un escenario completa la planilla y la exporta a CSV con estas columnas:

| captura | inicio | fin | escenario |
| --- | --- | --- | --- |
| red_lenta_1.pcapng | 17:06:30 | 17:09:30 | Red lenta |

- `inicio` y `fin` van en **hora local** de la compu que capturó: `17:06:30` o `2026-11-05 17:06:30`.
- Escenarios: `Red lenta`, `Saturacion`, `Uso intenso`, `Escaneo` (no importan mayúsculas ni tildes).

`etiquetar_ventanas(tabla, intervalos, captura=...)` agrega:

- `escenario_real`: el escenario anotado que se superpone con la ventana, o `normal`.
- `degradada`: `True` **solo** para `Red lenta` y `Saturacion`. Uso intenso y Escaneo se etiquetan, pero no cuentan como degradación.

Esa es la verdad de referencia con la que E evalúa a todos los métodos.

## 5. Datos de ejemplo

`python tests/generar_ejemplo.py` genera `tests/ejemplo_paquetes.csv` (15 min con los cuatro escenarios) y `tests/ejemplo_intervalos.csv`. Sirve para programar sin capturas reales.

## 7. Latencia y pérdida (`rendimiento_tcp.py`, módulo C1)

`calcular_metricas(df_paquetes, tabla)` agrega a la tabla de ventanas:

| Columna | Contenido |
| --- | --- |
| `c1_paquetes_tcp` | Paquetes TCP en la ventana (denominador de los porcentajes). |
| `c1_rtt_medio_ms`, `c1_rtt_max_ms` | RTT medio y máximo (`tcp.analysis.ack_rtt`) en ms. **NaN si la ventana no tiene muestras de RTT** (0 ms se leería como "latencia perfecta"). |
| `c1_pct_retrans` | % de paquetes TCP que son retransmisiones. |
| `c1_pct_dup_ack` | % de paquetes TCP que son ACK duplicados. |
| `c1_ventanas_cero` | Paquetes que anuncian ventana de recepción en cero. |

- Los porcentajes se calculan sobre los **paquetes TCP** de la ventana (no sobre todos). Sin TCP quedan en 0.
- **B2:** las columnas de RTT pueden tener NaN; hay que contemplarlo antes de entrenar los modelos.
- `umbral_rtt(tablas_normales)`: media + 3 desvíos del `c1_rtt_medio_ms` de las ventanas normales (regla 3-sigma, Chandola et al., 2009).
- `reglas_c1(tabla, umbral_rtt_ms)`: alerta `Degradación de rendimiento` si `c1_pct_retrans` > 5 % (solo con al menos 20 paquetes TCP en la ventana) o si `c1_rtt_medio_ms` > umbral.
