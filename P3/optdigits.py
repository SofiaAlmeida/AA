"""
  Sofía Almeida Bruno
  https://archive.ics.uci.edu/ml/datasets/optical+recognition+of+handwritten+digits
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

#-------------------------------------------------------------------------------
# Ajuste de parámetros
def adjust_params(X, y, model, params):
    print("------ Grid Search...")
    grid = GridSearchCV(model, params, cv=5, n_jobs=-1, verbose=1, scoring='accuracy')
    grid.fit(X, y)
    print("Mejores parámetros:")
    print(grid.best_params_)
    print("Error CV")
    print(grid.best_score_)
    
    print("Grid scores:")
    print()
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
    
    return grid.best_estimator_
    
#------------------------------------------------------------------------------
# Leemos los datos de los ficheros
train = np.genfromtxt('./datos/optdigits.tra',  delimiter = ",")
test = np.genfromtxt('./datos/optdigits.tes',  delimiter = ",")

# Separamos la variable a predecir
X_train = train[:,0:-1]
y_train = train[:,-1]
X_test= test[:,0:-1]
y_test = test[:,-1]

# Preprocesado----------------------------------------------------------

# Añadimos complejidad al modelo
poly = PolynomialFeatures(2)

# Seleccionamos variables utilizando lasso
lasso = LassoCV(n_jobs = -1, max_iter = 55000, verbose = True, cv = 5)

preprocessing = Pipeline(steps=[
    ('poly', poly),
    ('scale',StandardScaler()),
    ('lasso', SelectFromModel(lasso)),
])

X_train = preprocessing.fit_transform(X_train, y_train)

lreg = LogisticRegression(penalty = 'l2', tol=0.0001, random_state=123456, solver='saga', max_iter=100, multi_class='multinomial', n_jobs=-1)
params_lreg =  {'C':[0.5, 1.0, 2.0], 'max_iter':[250, 500, 1000]}

best_lreg = adjust_params(X_train, y_train, lreg, params_lreg)

# Preprocesamos el conjunto de test 
X_test = preprocessing.transform(X_test)

# Utilizando el mejor modelo predecimos el valor de X_test
y_pred = best_lreg.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy obtenido en test: {:.4f}".format(acc))
