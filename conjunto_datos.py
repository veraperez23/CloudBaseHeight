import os
import pandas as pd
from datetime import datetime

# Rutas de archivo
img_dir = "./semana_valladolid/imagenes_camara" #Directorio donde se encuentran las imágenes a tomar en el conjunto de datos
txt_dir = "./semana_valladolid/txt_ceilometros" #Directorio donde se encuentran los archivos .txt del ceilómetro
output_file = "semana_valladolid.txt" # Localización del archivo dataset

# Lectura de los archivos del ceilómetro
print("Cargando datos del ceilómetro...")
df_list = []
for filename in os.listdir(txt_dir):
    if filename.endswith(".txt"): #dentro de la carpeta txt_dir se queda solo con los archivos .txt, que son los del ceilómetro
        archivo = os.path.join(txt_dir, filename)
        df_temp = pd.read_csv(archivo, sep=';')
        df_list.append(df_temp)


df_ceilometro = pd.concat(df_list, ignore_index=True) #junta todos los datos de la semana
df_ceilometro['time'] = pd.to_datetime(df_ceilometro['time']) #transforma la primera columna en una variable temporal para python
df_ceilometro = df_ceilometro.sort_values('time').set_index('time') #ordena cronológicamente y pone como índice el tiempo

# Lactura de imágenes
print("Procesando imágenes y buscando los datos asociados...")
resultados = []

for root, dirs, files in os.walk(img_dir): # os.walk recorre automáticamente todas las subcarpetas
    for file in files:
        if file.endswith(".jpg"):
            try:
                # Extraemos fecha y hora
                partes = file.split('_')
                fecha_str = partes[1]     
                hora_str = partes[2][:4]  
                
                # Convertir a variable temporal
                img_time = datetime.strptime(f"{fecha_str}{hora_str}", "%Y%m%d%H%M")
                
                # Buscamos el valor de la CBH más cercana
                indice_cercano = df_ceilometro.index.get_indexer([img_time], method='nearest')[0]
                fila_cercana = df_ceilometro.iloc[indice_cercano]
                
                # Tomamos las variables decisivas de esa fila y aplicamos las condiciones de limpiado de datos
                cbh_0 = fila_cercana['cbh_0']
                state_optics = fila_cercana['state_optics']
                sci = fila_cercana['sci']

                if cbh_0 != -1 and state_optics >= 90 and sci == 0:
                    resultados.append(f"{file};{cbh_0}")
                    
            except Exception as e:
                print(f"Error procesando la imagen {file}: {e}")

# .txt final
print("Guardando resultados en la carpeta de salida...")
with open(output_file, 'w') as f:
    for linea in resultados:
        f.write(linea + "\n")

print(f"Emparejamiento realizado, se han obtenido {len(resultados)} imágenes válidas.")