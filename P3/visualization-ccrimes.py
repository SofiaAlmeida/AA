"""
  Sofía Almeida Bruno
  Análisis exploratorio de datos
  http://archive.ics.uci.edu/ml/datasets/Communities+and+Crime "
"""

import pandas as pd
import numpy as np

# Leemos los nombres de los atributos
attributes = np.genfromtxt('./datos/communities.names', dtype = "|U50", skip_header = 75, max_rows = 128, delimiter = " ")[:,1]

# Leemos el conjunto de datos
df = pd.read_csv('./datos/communities.data', sep=",", na_values ='?', names = attributes)

print(df.shape)
print(df.info())

# Eliminamos las variables no predictivas
df.drop(labels=['state', 'county', 'community', 'communityname', 'fold'], axis=1, inplace = True)

print(df.shape)
print(df.isnull().any())
