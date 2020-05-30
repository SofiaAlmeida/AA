"""
  Sofía Almeida Bruno
  Análisis exploratorio de datos
  http://archive.ics.uci.edu/ml/datasets/Communities+and+Crime "
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error
import time

#-------------------------------------------------------------------------------

# Ajuste de parámetros
def adjust_params(X, y, model, params):
    print("------ Grid Search...")
    grid = GridSearchCV(model, params, cv=5, n_jobs=2, verbose=1, scoring='neg_mean_squared_error')
    grid.fit(X, y)
    print("Mejores parámetros:")
    print(grid.best_params_)
    print("Error CV")
    print(-grid.best_score_)

    print("Grid scores:")
    print()
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    
    return grid.best_estimator_
#------------------------------------------------------------------

#-------------------------------------------------------------------------------

# Leemos los nombres de los atributos
attributes = np.genfromtxt('./datos/communities.names', dtype = "|U50", skip_header = 75, max_rows = 128, delimiter = " ")[:,1]

# Leemos el conjunto de datos
df = pd.read_csv('./datos/communities.data', sep=",", na_values ='?', names = attributes)
df.name = 'Communities and Crimes'

# Dividimos en training y test
X =  df.drop(labels = ['ViolentCrimesPerPop'], axis = 1)
y = df['ViolentCrimesPerPop']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123456)

# Preprocesado----------------------------------------------------------

# Eliminamos las variables no predictivas
to_drop = np.array(['state', 'county', 'community', 'communityname', 'fold'])
X_train = X_train.drop(labels = to_drop, axis=1)

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

# Añadimos complejidad al modelo
poly = PolynomialFeatures(2)
# Reducimos mediante regularización lasso
lasso = LassoCV(n_jobs = -1, max_iter = 50000, verbose = True, cv = 4)

preprocessing = Pipeline(steps=[
    ('imputer', imputer),
    ('poly', poly),
    ('lasso', SelectFromModel(lasso))])

X_train = preprocessing.fit_transform(X_train, y_train)

reg =  SGDRegressor(tol=1e-4)
reg.fit(X_train, y_train)
params_reg =  {'alpha':[1/(10.0**i) for i in range(1,5)], 'learning_rate':['constant', 'optimal','invscaling', 'adaptive'], 'max_iter':[5000,10000,15000]}

best_reg = adjust_params(X_train, y_train, reg, params_reg)

# Preprocesamos el conjunto de test 
X_test = X_test.drop(labels = to_drop, axis=1)
X_test = preprocessing.transform(X_test)

# Utilizando el mejor modelo predecimos el valor de X_test
y_pred = best_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Error cuadrático medio en test: {:.4f}".format(mse))
