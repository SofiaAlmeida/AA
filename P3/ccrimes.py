"""
  Sofía Almeida Bruno
  Análisis exploratorio de datos
  http://archive.ics.uci.edu/ml/datasets/Communities+and+Crime "
"""

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import VarianceThreshold
#plt.style.use('seaborn')

# Leemos los nombres de los atributos
attributes = np.genfromtxt('./datos/communities.names', dtype = "|U50", skip_header = 75, max_rows = 128, delimiter = " ")[:,1]

# Leemos el conjunto de datos
df = pd.read_csv('./datos/communities.data', sep=",", na_values ='?', names = attributes)
df.name = 'Communities and Crimes'

print("Tamaño: ", df.shape)

# Dividimos en training y test
X =  df.drop(labels = ['ViolentCrimesPerPop'], axis = 1)
y = df['ViolentCrimesPerPop']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123456)

print("Tamaño training: ", X_train.shape, y_train.shape)
print("Tamaño test: ", X_test.shape, y_test.shape)


# Preprocesado----------------------------------------------------------

# Eliminamos las variables no predictivas
to_drop = np.array(['state', 'county', 'community', 'communityname', 'fold'])
X_train = X_train.drop(labels = to_drop, axis=1)
print("Tamaño train tras drop (5): ", X_train.shape)

mv_sum = X_train.isnull().sum()
mv = mv_sum * 100 / len(X_train)

# Eliminamos variables con más de un 30% de valores perdidos
for column in X_train:
    if mv[column] > 30.0:
        X_train.drop(labels=[column], axis=1, inplace = True)
        to_drop = np.append(to_drop, column)
print("Tamaño train tras drop (variables con mv)", X_train.shape)

# Imputamos los valores perdidos de aquellas variables que tengan menos de un 30% de valores perdidos
imputer = KNNImputer()

X = imputer.fit_transform(X_train)
# Eliminamos variables con varianza muy baja
var = VarianceThreshold(threshold=(.97 * (1 - .97)))


# Añadimos complejidad al modelo
poly = PolynomialFeatures(2)
X = poly.fit_transform(X)
print("Tras poli", X.shape)
X = var.fit_transform(X)
print("Tras var", X.shape)
# Reducimos mediante regularización lasso
#lasso = Lasso(max_iter = 100000, alpha = 0.01)
lasso = LassoCV(n_jobs = -1, max_iter = 50000, verbose = True, cv = 3)

preprocessing = Pipeline(steps=[
    ('imputer', imputer),
#    ('scale',StandardScaler()),
    ('poly', poly),
    ('Variance', var),
    ('lasso', SelectFromModel(lasso))])

preprocessing.fit_transform(X_train, y_train)
