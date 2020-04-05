# -*- coding: utf-8 -*-
#-------------------------------------------------------------------
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

#-------------------------------------------------------------------
# Fijamos la semilla
np.random.seed(1)
# Guardamos o no imágenes
save = True

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
def plot(x, y, fun, save = False, title = "", name = "", xlim = [-1, 1], ylim = [-1, 1]):
    sns.set()
    fig = plt.figure()
    ax = fig.add_subplot()
    
    # Plot points
    colors = ['tab:pink', 'tab:blue']
    scatter = ax.scatter(x[:,0], x[:,1], c = y, alpha = 0.6, edgecolors='none', cmap =matplotlib.colors.ListedColormap(colors))

    # Plot function 
    plt.plot(x[:,1], (-w[0] - w[1]*x[:,1]) / w[2], alpha = 0.6, color = 'black', linewidth=1.3, label = 'SGD')

    delta = 0.025
    a = np.arange(xlim[0], xlim[1], delta)
    b = np.arange(ylim[0], ylim[1], delta)
    X, Y = np.meshgrid(a, b)
    zs = np.array([fun(a, b) for a,b in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    CS = ax.contour(X, Y, Z, levels = 0, alpha = 0.6, colors = ['black'])

        
    #ax.set_xlim(-1.1, 1.1)
    #ax.set_ylim(-1.1, 1.1)

    # Leyenda
    legend1 = ax.legend(*scatter.legend_elements(), loc=(1.04,0), title="Clases")
    ax.add_artist(legend1)
    ax.set_xlabel('')
    ax.set_ylabel('')	
    ax.set_title(title)	

    # Añadimos cuadrícula
    #	plt.grid(True, color="white", linewidth = 1)
    #	plt.subplots_adjust(right=0.8)		

    if save:
        plt.savefig("./fig/" + name + ".png")
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
    CS.collections[1].set_label("y - " + f"{a:.4f}" + "x - (" + f"{b:.4f}" + ")")

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
    return np.sign(f)

#-------------------------------------------------------------------
# f(x,y) = y - ax - b
def line(x, y, a, b):
    return y - a * x - b

#-------------------------------------------------------------------
# Ejercicio 1 a ----------------------------------------------------
# Dibujamos una nube de puntos
N = 50
dim = 2
rango = [-50, 50]
points1a = simula_unif(N, dim, rango)
draw_points(points1a, "", save, "points1a")


# Ejercicio 1 b ----------------------------------------------------
sigma = [5, 7]
points1b = simula_gaus(N, dim, sigma)
draw_points(points1b, "", save, "points1b")

#-------------------------------------------------------------------
# Ejercicio 2 a ----------------------------------------------------
N = 500
dim = 2
rango = [-50, 50]
intervalo = [-50, 50]
points2 = simula_unif(N, dim, rango)
a, b = simula_recta(intervalo)
y = f(points2, a, b)

plot_line(points2, y, a, b, save, name = "recta", xlim = rango, ylim = rango)
