# Autora: Sofía Almeida Bruno
# Asignatra: Aprendizaje Automático
# Práctica 0 - Introducción a Python

from sklearn import datasets
import seaborn as sns; sns.set()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
import math

def ej1():
    # Leemos la base de datos iris de scikit-learn
    iris = datasets.load_iris()
    print("Cargada la base de datos iris")
    print(type(iris))

    # Obtenemos las características (X) y la clase (y)
    X = iris.data
    y = iris.target
    print("Número de elementos en X: " + str(len(X)))
    print("Número de elementos en y: " + str(len(y)))
    
    # Nos quedamos con las dos últimas características
    print(type(X))
    X_last = X[:, -2:]
    print("Número de características de X: " + str(len(X[0])))
    print("Número de características al quedarnos con las 2 últimas: " + str(len(X_last[0])))

    # Visualizamos los datos
    df = pd.DataFrame({"Longitud de pétalo" : X_last[:,0],
                       "Anchura de pétalo" : X_last[:,1],
                       "Especie" : y[:]})
    ax = sns.scatterplot(x="Longitud de pétalo", y="Anchura de pétalo", data=df, hue="Especie", palette=sns.color_palette("Set1", n_colors=3, desat=.5))
    plt.show()

def ej2():
    # Leemos la base de datos iris de scikit-learn
    iris = datasets.load_iris()
    print("Cargada la base de datos iris")
    X = iris.data
    y = iris.target

    
    # Separamos en training (80% de los datos) y test (20%) aleatoriamente y conservando la proporción de elementos en cada clase
    sss = StratifiedShuffleSplit(n_splits=1, random_state=12345, test_size=0.2)
  
    for train_index, test_index in sss.split(X, y):
        print("índices de los elementos de cada conjunto:")
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    
    print("Porcentaje clase 0 en train: ", np.count_nonzero(y_train == 0) / y_train.shape[0] * 100, "%")
    print("Porcentaje clase 0 en test: ", np.count_nonzero(y_test == 0) / y_test.shape[0] * 100, "%", "\n")
    print("Porcentaje clase 1 en train: ", np.count_nonzero(y_train == 1) / y_train.shape[0] * 100, "%")
    print("Porcentaje clase 1 en test: ", np.count_nonzero(y_test == 1) / y_test.shape[0] * 100, "%", "\n")
    print("Porcentaje clase 2 en train: ", np.count_nonzero(y_train == 2) / y_train.shape[0] * 100, "%")
    print("Porcentaje clase 2 en test: ", np.count_nonzero(y_test == 2) / y_test.shape[0] * 100, "%", "\n")

def ej3():
    # Obtener 100 valores equiespaciados entre 0 y 2pi
    values = np.linspace(0, 2*math.pi, 100)
    print(values)
    print("Número de elementos: ", values.shape[0])

    # Obtener el valor de sin(x)

    # Obtener el valor de cos(x)

    # Obtener el valor de sin(x) + cos(x)

    # Visualizar las tres curvas simultáneamente en el mismo plot (líneas discontinuas en negro, azul y rojo
    
#ej1()
#ej2()
ej3()
