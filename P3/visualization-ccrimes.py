"""
  Sofía Almeida Bruno
  Análisis exploratorio de datos
  http://archive.ics.uci.edu/ml/datasets/Communities+and+Crime "
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
plt.style.use('seaborn')

#-----------------------------------------------------------------------
# Correlation matrix
def plotCorrelationMatrix(df):
    filename = df.name
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()
    #plt.savefig("../fig/corr_matrix" + filename + ".pdf")
    plt.clf()
#-----------------------------------------------------------------------
# Leemos los nombres de los atributos
attributes = np.genfromtxt('./datos/communities.names', dtype = "|U50", skip_header = 75, max_rows = 128, delimiter = " ")[:,1]

# Leemos el conjunto de datos
df = pd.read_csv('./datos/communities.data', sep=",", na_values ='?', names = attributes)
df.name = 'Communities and Crimes'

print("Tamaño: ", df.shape)
print("Información: ", df.info())

# Eliminamos las variables no predictivas
df.drop(labels=['state', 'county', 'community', 'communityname', 'fold'], axis=1, inplace = True)

print("Tamaño tras eliminar not predictive: ", df.shape)

# Dividimos en training y test
df, df_test = train_test_split(df, test_size=0.25, random_state=123456)


print("Tamaño training: ", df.shape)
print("Tamaño test: ", df_test.shape)

# Estudio de la variable a predecir ------------------------------------
# Graficamos la distribución de la variable a predecir 
df['ViolentCrimesPerPop'].hist(edgecolor='white', linewidth=0.6, alpha = 0.8)
plt.title("ViolentCrimesPerPop")
plt.xlabel("Valor")
plt.ylabel("Frecuencia")

plt.savefig("./fig/HistViolentCrimesPerPop.pdf")
#plt.show()
#print(df['ViolentCrimesPerPop'].value_counts(bins=10))
plt.clf()

# Variables con valores perdidos
names_mv = df.columns[df.isnull().any()].tolist()
print("Variables con valores perdidos: ", names_mv)
print("Nº Variables con valores perdidos: ", len(names_mv))

print(df[names_mv])

percent_missing = df[names_mv].isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'column_name': names_mv,
                                 'percent_missing': percent_missing})
print("Porcentaje de valores perdidos: ", missing_value_df)

# Correlaciones 
#plotCorrelationMatrix(df) #Son muchas variables y no se lee nada
