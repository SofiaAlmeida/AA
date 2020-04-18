# -*- coding: utf-8 -*-
#-------------------------------------------------------------------
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score

#-------------------------------------------------------------------
# Fijamos la semilla
np.random.seed(1)

# Guardamos o no imágenes
save = True

#-------------------------------------------------------------------
# Devuelve clase 1 si el elemento es >= 0, -1 en otro caso
def class_(x):
    return np.array([1.0 if xx >= 0 else -1.0 for xx in x])

#-------------------------------------------------------------------
# Calcula una lista de N vectores de dimensión dim
# Cada vector contiene dim números aleatorios uniformes en el intervalo rango
def simula_unif(N, dim, rango):
    return np.random.uniform(rango[0],rango[1],(N,dim))

#-------------------------------------------------------------------
# Calcula una lista de longitud N de vectores de longitud dim
# Cada posición del vector contiene un número aleatorio extraído
# de una distribución gaussiana de media 0 y varianza dada
def simula_gaus(N, dim, sigma):
    media = 0    
    out = np.zeros((N,dim),np.float64)        
    for i in range(N):
            # Para cada columna dim se emplea un sigma determinado. Es decir, para 
            # la primera columna se usará una N(0,sqrt(5)) y para la segunda N(0,sqrt(7))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)
        
    return out

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
# Dibuja el conjunto de puntos points
# points: array Nx2
def draw_points(points, title = "", save = False, name = " "):
    sns.set()
    fig = plt.figure()
    ax = fig.add_subplot()
    # Plot outputs
    scatter = ax.scatter(points[:,0], points[:,1], alpha = 0.6, color = 'tab:pink', edgecolors='none')

    # Leyenda	
    ax.set_title(title)	

    # Eliminamos eje superior y derecho
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
	
    # Añadimos cuadrícula
    plt.grid(True, color="white", linewidth = 1)

    if save:
        plt.savefig("./fig/" + name + ".png")
    else:
        plt.show()


#-------------------------------------------------------------------
# Gráfico de los puntos (x), coloreando su clase (y) y pintando la función dada por fun
def plot(x, y, fun, save = False, title = "", name = "", legend = "", xlim = [-1, 1], ylim = [-1, 1]):
    sns.set()
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_facecolor('white')
    colors = ['tab:pink', 'tab:blue']
    
    # Plot function 
    delta = 0.25
    A = np.arange(xlim[0] - 5, xlim[1] + 5, delta)
    B = np.arange(ylim[0] - 5, ylim[1] + 5, delta)
    X, Y = np.meshgrid(A, B)
    zs = np.array([fun(a, b) for a,b in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    # Filled contour
    CS = ax.contourf(X, Y, Z, levels = 0, alpha = 0.15)
    
    proxy1 = plt.Rectangle((0,0),1,1,fc = CS.collections[1].get_facecolor()[0], label = 'f(x,y) < 0') 
    
    proxy2 = plt.Rectangle((0,0),1,1,fc = CS.collections[0].get_facecolor()[0], label = 'f(x,y) > 0')
    
    #CS.collections[1].set_label([', 'f(x,y)<0'])
    ax.patches += [proxy1, proxy2]
    # Contour
    CS2 = ax.contour(X, Y, Z, levels = 0, alpha = 0.3, colors = ['black'])
    CS2.collections[1].set_label(legend)
    
    # Plot points
    scatter = ax.scatter(x[:,0], x[:,1], c = y, alpha = 0.6, edgecolors='none', cmap =matplotlib.colors.ListedColormap(colors))

    # Legend
    legend1 = ax.legend(*scatter.legend_elements(), loc=(1.02,0.22), title="Clases")
    plt.legend(loc = (1.02, 0))
    ax.add_artist(legend1)
    ax.set_xlabel('')
    ax.set_ylabel('')	
    ax.set_title(title)	

    # Add grid
    plt.grid(True, color="gray", linewidth = 1, alpha = 0.3)
    plt.subplots_adjust(right=0.8)		

    if save:
        plt.savefig("./fig/" + name + ".png", bbox_inches="tight")
    else:
        plt.show()

# Gráfico de los puntos (x), coloreando su clase (y) y pintando la recta dada por a y b
def plot_line(x, y, a, b, save = False, title = "", name = "", xlim = [-1, 1], ylim = [-1, 1]):
    sns.set()
    fig = plt.figure()
    ax = fig.add_subplot()
    
    # Plot points
    colors = ['tab:pink', 'tab:blue']
    scatter = ax.scatter(x[:,0], x[:,1], c = y, alpha = 0.6, edgecolors='none', cmap =matplotlib.colors.ListedColormap(colors))

    # Plot line
    delta = 0.1
    A = np.arange(xlim[0] - 5, xlim[1] + 5, delta)
    B = np.arange(ylim[0] - 5, ylim[1] + 5, delta)
    X, Y = np.meshgrid(A, B)
    zs = np.array([line(x1, y1, a, b) for x1,y1 in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    CS = ax.contour(X, Y, Z, levels = 0, alpha = 0.6, colors = ['black'])
    CS.collections[1].set_label("y - (" + f"{a:.4f}" + ")x - (" + f"{b:.4f}" + ")")

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
# Devolvemos el valor signo(f) para un conjunto de puntos Nx2
# f(x,y) = y - ax -b
def f(points, a, b):
    vector_b =  b*np.array(np.ones((points.shape[0], 1)))
    f = points[:,1] - a * points[:,0] - vector_b[:,0]
    return class_(f)

#-------------------------------------------------------------------
# f(x,y) = y - ax - b
def line(x, y, a, b):
    return y - a * x - b


#-------------------------------------------------------------------
# Cambia el 10% de las etiquetas positivas y el 10% de las etiquetas negativas
def modify_class(y):
    # Almacenamos los índices de los 1 y -1 originales
    index1 = np.where(y == 1)[0]
    index_1 = np.where(y == -1)[0]

    # Seleccionamos los índices a modificar
    mod1 = np.random.choice(index1, round(0.1*index1.shape[0]))
    mod_1 = np.random.choice(index_1, round(0.1*index_1.shape[0]))

    # Alteramos el 10% de los valores de cada clase
    y[mod1] = -1
    y[mod_1] = 1

    return y

#-------------------------------------------------------------------
#f1(x,y) = (x-10)² + (y-20)²-400
def f1(x,y):
    return (x - 10)**2 + (y-20)**2 - 400

#-------------------------------------------------------------------
# f2(x,y) = 0.5(x+10)² + (y-20)²- 400
def f2(x,y):
    return 0.5 * (x+10)**2 + (y-20)**2 - 400


#-------------------------------------------------------------------
# f3(x,y) = 0.5(x-10)² + (y-20)²- 400
def f3(x,y):
    return 0.5 * (x-10)**2 - (y+20)**2 - 400

#-------------------------------------------------------------------
# f4(x,y) = y - 20x² - 5x + 3
def f4(x,y):
    return y - 20*x**2 - 5*x + 3


#-------------------------------------------------------------------
# Devuelve el accuracy y balanced accuracy que da la función fun
# si las etiquetas correctas son las de y
def metrics(x, y, fun):
    y_pred = class_(fun(x[:,0], x[:,1]))
    acc = accuracy_score(y, y_pred)
    bal_acc = balanced_accuracy_score(y, y_pred)
    return acc, bal_acc
    

#-------------------------------------------------------------------
# Ejercicio 1 a ----------------------------------------------------
# Dibujamos una nube de puntos
N = 50
dim = 2
rango = [-50, 50]
points1a = simula_unif(N, dim, rango)
draw_points(points1a, "", save, "points1a")


input("\n--- Pulsar tecla para continuar ---\n")

# Ejercicio 1 b ----------------------------------------------------
sigma = [5, 7]
points1b = simula_gaus(N, dim, sigma)
draw_points(points1b, "", save, "points1b")


input("\n--- Pulsar tecla para continuar ---\n")

#-------------------------------------------------------------------
# Ejercicio 2 a ----------------------------------------------------
np.random.seed(123456)
N = 100
dim = 2
rango = [-50, 50]
intervalo = [-50, 50]
points2 = simula_unif(N, dim, rango)
a, b = simula_recta(intervalo)
y = f(points2, a, b)

plot_line(points2, y, a, b, save, name = "recta", xlim = rango, ylim = rango)

input("\n--- Pulsar tecla para continuar ---\n")

# Ejercicio 2 b ----------------------------------------------------
y = modify_class(y)

plot_line(points2, y, a, b, save, name = "recta+ruido", xlim = rango, ylim = rango)

y_pred = f(points2, a, b)
acc = accuracy_score(y, y_pred)
bal_acc = balanced_accuracy_score(y, y_pred)
print("Accuracy de f : ", acc)
print("Balanced accuracy de f: ", bal_acc)

input("\n--- Pulsar tecla para continuar ---\n")

# Ejercicio 3 ----------------------------------------------------
plot(points2, y, f1, save, "", name = "f1", xlim = rango, ylim = rango, legend = "(x-10)² + (y-20)²- 400")
acc, bal_acc = metrics(points2, y, f1)
print("Accuracy de f1: ", acc)
print("Balanced accuracy de f1: ", bal_acc)

input("\n--- Pulsar tecla para continuar ---\n")

plot(points2, y, f2, save, "", name = "f2", xlim = rango, ylim = rango, legend = "0.5(x+10)² + (y-20)²- 400")
acc, bal_acc = metrics(points2, y, f2)
print("Accuracy de f2: ", acc)
print("Balanced accuracy de f2: ", bal_acc)

input("\n--- Pulsar tecla para continuar ---\n")

plot(points2, y, f3, save, "", name = "f3", xlim = rango, ylim = rango, legend = "0.5(x-10)² + (y-20)²- 400")
acc, bal_acc = metrics(points2, y, f3)
print("Accuracy de f3: ", acc)
print("Balanced accuracy de f3: ", bal_acc)

input("\n--- Pulsar tecla para continuar ---\n")

plot(points2, y, f4, save, "", name = "f4", xlim = rango, ylim = rango, legend = "y - 20x² - 5x + 3")
acc, bal_acc = metrics(points2, y, f4)
print("Accuracy de f4: ", acc)
print("Balanced accuracy de f4: ", bal_acc)
input("\n--- Pulsar tecla para continuar ---\n")
