# Autora: Sofía Almeida Bruno
# Asignatra: Aprendizaje Automático
# Práctica 0 - Introducción a Python

from sklearn import datasets
import seaborn as sns; sns.set()
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

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
    
    # Separamos en training (80% de los datos) y test (20%) aleatoriamente y conservando la proporción de elementos en cada clase
    sss = StratifiedShuffleSplit(n_splits=1, random_state=12345, test_size=0.2)
    index = sss.split(iris.data, iris.target)
    print(index[0], index[1])
    print("Training: ", index.train, "Test: ", index.test)
    
    
#ej1()
ej2()
