import numpy as np
import pandas as pd 

#subir archivo con los datos
df = pd.read_csv("Datos_Cundinamarca.csv")
df.head()

columnas_interes = [
    "PUNT_GLOBAL",
    "FAMI_TIENECOMPUTADOR",
    "FAMI_TIENEINTERNET",
    "FAMI_ESTRATOVIVIENDA",
    "COLE_MCPIO_UBICACION"
]

df = df[columnas_interes]

#verificando que este bien
df

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