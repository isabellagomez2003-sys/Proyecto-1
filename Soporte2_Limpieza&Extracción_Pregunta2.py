#librerias necesarias para exploración y limpieza
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#subir archivo con los datos
df = pd.read_csv("Datos_Cundinamarca.csv")
df.head()

#valores faltantes en todas las columnas
df.isna().sum()