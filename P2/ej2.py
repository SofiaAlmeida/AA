# -*- coding: utf-8 -*-
#-------------------------------------------------------------------
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import seaborn as sns
import math
from numpy import linalg as LA
from scipy.special import expit

#-------------------------------------------------------------------
# Fijamos la semilla
np.random.seed(1)
save = False

#-------------------------------------------------------------------
# Calcula una lista de N vectores de dimensión dim
# Cada vector contiene dim números aleatorios uniformes en el intervalo rango
def simula_unif(N, dim, rango):
    return np.random.uniform(rango[0],rango[1],(N,dim))

#-------------------------------------------------------------------
# Simula de forma aleatoria los parámetros a y b de una recta
# y = ax+b que corta al cuadrado intervalo x intervalo
def simula_recta(intervalo):
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))

    x1 = points[0,0]
    x2 = points[1,0]
    y1 = points[0,1]
    y2 = points[1,1]
    
    # y = a*x + b
    a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.
    
    return a, b

#-------------------------------------------------------------------
# Devolvemos el valor signo(f) para un conjunto de puntos Nx2
# f(x,y) = y - ax -b
def f(points, a, b):
    vector_b =  b*np.array(np.ones((points.shape[0], 1)))
    f = points[:,1] - a * points[:,0] - vector_b[:,0]
    return np.sign(f)

#-------------------------------------------------------------------
# Cambia el 10% de las etiquetas positivas y el 10% de las etiquetas negativas
def modify_class(y):
    y_mod = np.copy(y)
    # Almacenamos los índices de los 1 y -1 originales
    index1 = np.where(y_mod == 1)[0]
    index_1 = np.where(y_mod == -1)[0]

    # Seleccionamos los índices a modificar
    mod1 = np.random.choice(index1, round(0.1*index1.shape[0]))
    mod_1 = np.random.choice(index_1, round(0.1*index_1.shape[0]))

    # Alteramos el 10% de los valores de cada clase
    y_mod[mod1] = -1
    y_mod[mod_1] = 1

    return y_mod

# Gráfico de los puntos (x), coloreando su clase (y) y pintando la recta dada por a y b
def plot_line(x, y, w, save = False, title = "", name = "", xlim = [-1, 1], ylim = [-1, 1]):
    sns.set()
    fig = plt.figure()
    ax = fig.add_subplot()
    
    # Plot points
    colors = ['tab:pink', 'tab:blue']
    scatter = ax.scatter(x[:,0], x[:,1], c = y, alpha = 0.6, edgecolors='none', cmap = matplotlib.colors.ListedColormap(colors))

    # Plot line
    delta = 0.3
    A = np.arange(xlim[0] - 5, xlim[1] + 5, delta)
    B = np.arange(ylim[0] - 5, ylim[1] + 5, delta)
    X, Y = np.meshgrid(A, B)
    zs = np.array([w[0] + x1*w[1] + y1*w[2] for x1,y1 in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    CS = ax.contour(X, Y, Z, levels = 0, alpha = 0.6, colors = ['black'])
    CS.collections[1].set_label("w*X")

    # Leyenda
    legend1 = ax.legend(*scatter.legend_elements(), loc=(1.02,0.1), title="Clases")
    plt.legend(loc = (1.02, 0))
    ax.add_artist(legend1)
    ax.set_xlabel('')
    ax.set_ylabel('')	
    ax.set_title(title)		

    if save:
        plt.savefig("./fig/" + name + ".png", bbox_inches="tight")
    else:
        plt.show()



#-------------------------------------------------------------------
# Calcula el hiperplano solución a un problema de clasificación binaria
# Usando el algoritmo PLA
# datos: matriz de características
# label: etiqueta de la fila de la matriz correspondiente (1 o -1)
# max_iter: número máximo de iteraciones
# vini: valor inicial del vecto
def ajusta_PLA(datos, label, max_iter, vini):
    w = vini
    it = 0
    changes = True

    datos = np.hstack((np.ones((datos.shape[0],1)), datos))	
    
    while it < max_iter and changes:
        #datos, label = shuffle(datos, label)
        changes = False
        for x, y in zip(datos, label):
            # Si sign(x*w) != y
            if np.sign(np.dot(x, w)) != y:
                changes = True
                w = (w.T + x*y).T
        it += 1

    return w, it

#-------------------------------------------------------------------
def ej1_zeros(points, y, save, name, rango):
    
    # Ejercicio 1 a ----------------------------------------------------
    wini = np.zeros((3, 1))
    max_iter = 2000
    it = []
    acc = []
    
    for i in range(0,10):
        w, it_ = ajusta_PLA(points, y, max_iter, wini)
        it = np.append(it, it_)
        y_pred = np.sign(w[0] + points[:,0]*w[1] + points[:,1]*w[2])
        acc = np.append(acc, accuracy_score(y, y_pred))

    print("Punto inicial (0,0,0)^T, resultados medios-----------------")
    print("it: ", it)
    print("Número de iteraciones medio: ", np.mean(it))
    print("Desviación típica: ", np.std(it))
    print("acc: ", acc)
    print("Accuracy  medio: ", np.mean(acc))
    print("Desviación típica: ", np.std(acc)) 
    plot_line(points, y, w, save, name = name,  xlim = rango, ylim = rango)
    input("\n--- Pulsar tecla para continuar ---\n")

#-------------------------------------------------------------------
def ej1_rand(points, y):
    
    # Ejercicio 1 a ----------------------------------------------------
    max_iter = 2000
    it = []
    acc = []

    for i in range(0,10):
        wini = np.random.rand(3, 1)
        
        w, it_ = ajusta_PLA(points, y, max_iter, wini)
        it = np.append(it, it_)
        y_pred = np.sign(w[0] + points[:,0]*w[1] + points[:,1]*w[2])
        acc = np.append(acc, accuracy_score(y, y_pred))

    print("Punto inicial aleatorio, resultados medios-----------------")
    print("it: ", it)
    print("Número de iteraciones medio: ", np.mean(it))
    print("Desviación típica: ", np.std(it))
    print("acc: ", acc)
    print("Accuracy  medio: ", np.mean(acc))
    print("Desviación típica: ", np.std(acc))
    input("\n--- Pulsar tecla para continuar ---\n")

#-------------------------------------------------------------------
def  grad_err(x, y, w):
    den = expit(- y * np.dot(x, w))
    print(y.shape)
    print((y * np.dot(x, w)).shape)
    frac = y * den
    print(frac.shape)
    vector = frac * x
    return - (1 / x.shape[0]) * np.sum(vector, axis = 1)
    
#-------------------------------------------------------------------
# Gradiente Descendente Estocastico Regresión logística 
# x:  matriz de datos
# y: valor correspondiente a cada elemento
# lr: tasa de aprendizaje
# max_iters: número máximo de iteraciones
# tam_minbatch: tamaño de los minibatchs
def sgdRL(x, y, lr, tam_minibatch = 1, w = np.zeros((3,1))):
    # Inicializamos número de minibatchs según el total de elementos y el tamaño
    num_minibatch = x.shape[0] / tam_minibatch
    iterate = True

    x = np.hstack((np.ones((x.shape[0],1)), x))	
    
    
    while(iterate):
	# Desordenamos los conjuntos
        x, y = shuffle(x, y)
	# Creamos los minibatchs
        minibatchs_x = np.array_split(x, num_minibatch)
        minibatchs_y = np.array_split(y, num_minibatch)
        
        w_ant = w
	        
        for xj, yj in zip(minibatchs_x, minibatchs_y):
            # Actualizamos w
            w = w - lr * grad_err(xj, yj, w)

        # Comprobamos la condición de parada
        print(LA.norm(w - w_ant))
        iterate = LA.norm(w - w_ant) >= 0.01
	
    return w
    
    
#-------------------------------------------------------------------
# Ejercicio 1 a ---------------------------------------------------
# Generamos los puntos y los etiquetamos
np.random.seed(123456)
N = 100
dim = 2
rango = [-50, 50]
intervalo = [-50, 50]
points = simula_unif(N, dim, rango)
a, b = simula_recta(intervalo)
y = f(points, a, b)
y_err = modify_class(y)

print("Ejercicio 1------------------------------")
print("Etiquetas sin ruido------------------------------")
#ej1_zeros(points, y, save, "21", rango)
#ej1_rand(points, y)
print("Etiquetas con ruido------------------------------")
#ej1_zeros(points, y_err, save, "21ruido", rango)
#ej1_rand(points, y_err)

sgdRL(points, y, 0.01, tam_minibatch = 1)
