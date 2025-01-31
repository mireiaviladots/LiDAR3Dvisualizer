import laspy
import numpy as np
import csv
from pyproj import Proj
from scipy.spatial import cKDTree
from pathlib import Path

# Cargar el archivo LAS
las = laspy.read("cloud1retorno.las")

# Extraer las coordenadas x, y, y z
x = las.x
y = las.y
z = las.z

nube_puntos = np.vstack((x,y,z)).T    # Matriz de tamaño (N,3), cada fila representa x,y,z

print(nube_puntos)

# Define la proyección UTM 31N
# utm_proj = Proj(proj='utm', zone=31, datum='WGS84')

# Lista para almacenar las coordenadas UTM y el valor extra
utm_coords = []

# Rutas de los archivos CSV
input_file = 'sensorDataPrueba.csv'
input_file = Path(input_file)
output_file = input_file.with_name('sensorDataPruebaConverted.csv')

# Abre el archivo CSV de entrada y el archivo CSV de salida
with open(input_file, mode='r', newline='', encoding='utf-8') as infile, \
        open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
    # Crea un lector y un escritor CSV
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    writer.writerow(['Easting', 'Northing', 'Dosis'])  # Encabezados

    # Itera sobre cada fila en el archivo de entrada
    for row in reader:
        easting = float(row[0])  # Primer elemento: easting
        northing = float(row[1])  # Segundo elemento: northing
        valorextra = float(row[2])  # Tercer elemento: valor extra
        writer.writerow([easting, northing, valorextra])
        utm_coords.append([easting, northing, valorextra])

        # Convierte a UTM
        # latitude = float(row[0])  # Primer elemento: latitud
        # longitude = float(row[1])  # Segundo elemento: longitud
        # valorextra = float(row[2])  # Tercer elemento: valor extra
        # easting, northing = utm_proj(longitude, latitude)
        # writer.writerow([easting, northing, valorextra])
        # utm_coords.append([easting, northing, valorextra])  # Va agregando los elementos al final de la lista

utm_coords = np.array(utm_coords)  # Matriz de tamaño (N,3), cada fila representa easting, northing, y dosis
print(utm_coords)
utm_points = utm_coords[:, :2]  # Sólo coordenadas [easting, northing]
dosis = utm_coords[:, 2]  # Dosis correspondiente

# Construir el KD-Tree para los puntos UTM del CSV (BUSQUEDA EFICIENTE)
tree = cKDTree(utm_points)

# Determinar los límites del área del CSV con dosis
x_min, y_min = np.min(utm_points, axis=0)   # Mínimo de cada columna (lat, long)
x_max, y_max = np.max(utm_points, axis=0)   # Máximo de cada columna (lat, long)

# Filtrar puntos de la nube dentro del área de dosis
dentro_area = (
    (nube_puntos[:, 0] >= x_min) & (nube_puntos[:, 0] <= x_max) &
    (nube_puntos[:, 1] >= y_min) & (nube_puntos[:, 1] <= y_max)
)

# Solo los puntos dentro del área
puntos_dentro = nube_puntos[dentro_area]

# Definir la función de interpolación inversa de la distancia (IDW)
def idw_interpolation(point, neighbors, doses, p=2):    # point: de la nube / neighbors: (k,2) vecinos cercanos del CSV al point de la nube (k filas 2 columnas x,y)/ doses: k dosis del CSV
    distancias = np.linalg.norm(neighbors - point, axis=1) # np.linalg.norm: calcula el modulo del vector
    if np.any(distancias == 0):  # Si el punto de la nube coincide con algun cercano del CSV
        return doses[distancias == 0][0]  # Devuelve dosis
    pesos = 1 / (distancias ** p)  # Puntos más cercanos (distancia inferior), más peso
    dosis_interpolada = np.sum(pesos * doses) / np.sum(pesos) # Formula
    return dosis_interpolada

# Parámetros de IDW
k = 2  # Número de vecinos
p = 2  # Exponente de ponderación

# Interpolación para cada punto del área
dosis_nube = np.full(len(puntos_dentro), np.nan)    # Vector de tamaño puntos_dentro
for i, punto in enumerate(puntos_dentro[:, :2]):    # Coordenada x,y
    distancias, indices = tree.query(punto, k=k)    # Busca los k puntos mas cercanos a punto (punto de la nube)
    dosis_nube[i] = idw_interpolation(punto, utm_points[indices], dosis[indices], p=p)  # Punto nube, los k puntos del csv más cercanos y sus dosis

# Guardar resultados en un nuevo archivo LAS
las_filtrada = laspy.create()

las_filtrada.header.scale = las.header.scale
las_filtrada.header.offset = las.header.offset

las_filtrada.x = puntos_dentro[:, 0]
las_filtrada.y = puntos_dentro[:, 1]
las_filtrada.z = puntos_dentro[:, 2]

# Agregar columna de dosis
las_filtrada.add_extra_dim(laspy.ExtraBytesParams(name="Dosis", type=np.float32))
las_filtrada["Dosis"] = dosis_nube

output_file = Path("cloud2retorno_con_dosis_idwK.las")
las_filtrada.write(output_file)