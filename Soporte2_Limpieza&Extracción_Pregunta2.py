#Tarea 2
#Pregunta 2

#¿En qué medida el acceso a computador e internet en el hogar explica las diferencias 
#en el puntaje global de las pruebas Saber 11, una vez considerado el estrato socioeconómico, en Cundinamarca? 


#librerias necesarias para exploración y limpieza
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#subir archivo con los datos
df = pd.read_csv("Datos_Cundinamarca.csv")
df.head()

#valores faltantes en todas las columnas
print(df.isna().sum())

#columnas de interes
columnas_interes = [
    "PUNT_GLOBAL",
    "FAMI_TIENECOMPUTADOR",
    "FAMI_TIENEINTERNET",
    "FAMI_ESTRATOVIVIENDA",
    "COLE_MCPIO_UBICACION"
]

df = df[columnas_interes]

#verificando que este bien
print(df)

#analizando como se ven los nulos de los datos para luego quitarlos
print("Filas antes de limpiar:", len(df))

print("Valores únicos computador:")
print(df["FAMI_TIENECOMPUTADOR"].unique())

print("Valores únicos internet:")
print(df["FAMI_TIENEINTERNET"].unique())

print("Valores únicos estrato:")
print(df["FAMI_ESTRATOVIVIENDA"].unique())


#reemplazar strings problemáticos por NaN reales
df.replace(
    ["nan", "NaN", "NAN", "Sin estrato", ""],
    np.nan,
    inplace=True
)

#puntaje global debe ser numerico
df["PUNT_GLOBAL"] = pd.to_numeric(df["PUNT_GLOBAL"], errors="coerce")

#eliminar los vacios
df_limpio = df.dropna(subset=[
    "PUNT_GLOBAL",
    "FAMI_TIENECOMPUTADOR",
    "FAMI_TIENEINTERNET",
    "FAMI_ESTRATOVIVIENDA"
])

#cuantas borre y verificar que este bien
print("Filas originales:", df.shape[0])
print("Filas después de limpieza:", df_limpio.shape[0])
print("Filas eliminadas:", df.shape[0] - df_limpio.shape[0])

print("\nNulos restantes:")
print(df_limpio[[
    "PUNT_GLOBAL",
    "FAMI_TIENECOMPUTADOR",
    "FAMI_TIENEINTERNET",
    "FAMI_ESTRATOVIVIENDA"
]].isnull().sum())

#Asignamos 1 cuando es "Si" 
#Asignamos 0 cuando es "No"
#Para los estratos cambiamos de "estrato 1" a 1 con todos los estratos. 


#verificando el tipo de datos que hay
print(df_limpio.dtypes)

#verificar que datos hay en la base de datos
print(df_limpio["FAMI_ESTRATOVIVIENDA"].unique())
print(df_limpio["FAMI_TIENEINTERNET"].unique())
print(df_limpio["FAMI_TIENECOMPUTADOR"].unique())

import numpy as np

#eliminar "sin estrato"
df_limpio["FAMI_ESTRATOVIVIENDA"] = df_limpio["FAMI_ESTRATOVIVIENDA"].replace(
    "Sin Estrato", np.nan
)

df_limpio = df_limpio.dropna(subset=["FAMI_ESTRATOVIVIENDA"])

#binarias
df_limpio["FAMI_TIENECOMPUTADOR"] = df_limpio["FAMI_TIENECOMPUTADOR"].map({
    "Si": 1,
    "No": 0
})

df_limpio["FAMI_TIENEINTERNET"] = df_limpio["FAMI_TIENEINTERNET"].map({
    "Si": 1,
    "No": 0
})

#estrato a numero
df_limpio["FAMI_ESTRATOVIVIENDA"] = (
    df_limpio["FAMI_ESTRATOVIVIENDA"]
    .str.replace("Estrato ", "")
    .astype(int)
)

#verificar que este bien
print(df_limpio.dtypes)

print("\nValores únicos finales:")
print(df_limpio["FAMI_TIENECOMPUTADOR"].unique())
print(df_limpio["FAMI_TIENEINTERNET"].unique())
print(df_limpio["FAMI_ESTRATOVIVIENDA"].unique())


print(df_limpio)


#quitar tildes de los datos
df_limpio['COLE_MCPIO_UBICACION'] = (
    df_limpio['COLE_MCPIO_UBICACION']
    .str.strip()
    .str.upper()
    .str.replace('Á','A', regex=False)
    .str.replace('É','E', regex=False)
    .str.replace('Í','I', regex=False)
    .str.replace('Ó','O', regex=False)
    .str.replace('Ú','U', regex=False)
)

print(df_limpio['COLE_MCPIO_UBICACION'].value_counts())

print(df_limpio[["FAMI_TIENECOMPUTADOR", 
                 "FAMI_ESTRATOVIVIENDA", 
                 "FAMI_TIENEINTERNET"]])
