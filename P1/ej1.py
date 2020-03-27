# -*- coding: utf-8 -*-

#############################
#####    BIBLIOTECAS    #####
#############################
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import seaborn as sns
sns.set()

#-------------------------------------------------------------------------------#
#------------- Ejercicio sobre la búsqueda iterativa de óptimos ----------------#
#-------------------------------------------------------------------------------#


#------------------------------Ejercicio 1 -------------------------------------#

# Fijamos la semilla

def E(w):
   u = w[0]
   v = w[1]
   return (u * math.exp(v) - 2 * v * math.exp(-u))**2

# Derivada parcial de E respecto de u
def Eu(w):
    u = w[0]
    v = w[1]
    return 2 * (u * math.exp(v) - 2 * v * math.exp(-u)) * (math.exp(v) + 2 * v * math.exp(-u))

# Derivada parcial de E respecto de v
def Ev(w):
    u = w[0]
    v = w[1]
    return 2 * (u * math.exp(v) - 2 * v * math.exp(-u)) * (u * math.exp(v) - 2 * math.exp(-u))
	
# Gradiente de E
def gradE(w):
    return np.array([Eu(w), Ev(w)])

# Gradiente desdendente
def gd(w, lr, grad_fun, fun, epsilon, max_iters):
    it = 0
    while (fun(w) > epsilon and it < max_iters):
        w = w - lr * grad_fun(w)
        it += 1
    return w, it

print ('\nGRADIENTE DESCENDENTE')
print ('\nEjercicio 1\n')
w_0 = (1, 1)
lr = 0.1
epsilon = 10**-14
max_iters = 10**10
w, num_ite = gd(w_0, lr, gradE, E, epsilon, max_iters)
print ('Numero de iteraciones: ', num_ite)
input("\n--- Pulsar tecla para continuar ---\n")
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')

input("\n--- Pulsar tecla para continuar ---\n")

#------------------------------Ejercicio 2 -------------------------------------#

def f(w):
    x = w[0]
    y = w[1]

    return (x - 2)**2 + 2 * (y + 2)**2 + 2 * math.sin(2 * math.pi * x) * math.sin(2 * math.pi * y)
	
# Derivada parcial de f respecto de x
def fx(w):
    x = w[0]
    y = w[1]

    return 2 * (x-2) + 4  * math.pi * math.sin(2 * math.pi * y) * math.cos(2 * math.pi * x)

# Derivada parcial de f respecto de y
def fy(w):
    x = w[0]
    y = w[1]
    
    return 4 * (y + 2) + 4 * math.pi * math.sin(2 * math.pi * x) * math.cos(2 * math.pi * y)
	
# Gradiente de f
def gradf(w):
    return np.array([fx(w), fy(w)])
	
# a) Usar gradiente descendente para minimizar la función f, con punto inicial (1,-1)
# tasa de aprendizaje 0.01 y max 50 iteraciones. Repetir con tasa de aprend. 0.1
def gd_grafica(w, lr, grad_fun, fun, max_iters = 10):
    it = 0
    graf = np.array([])

    while (it < max_iters):
        w = w - lr * grad_fun(w)
        graf = np.append(graf, fun(w))
        it += 1
    
    plt.plot(range(0, it), graf, '--', marker = mpath.Path.unit_circle())
    plt.xlabel('Iteraciones')
    plt.ylabel('f(x,y)')
    plt.title('Valor f(x,y), lr = ' + str(lr))
    plt.savefig('fig/1lr' + str(lr) + '.png')
    plt.clf()

print ('Resultados ejercicio 2\n')
print ('\nGrafica con learning rate igual a 0.01')
w_0 = (1, -1)
lr = 0.01
max_iters = 50
gd_grafica(w_0, lr, gradf, f, max_iters)

print ('\nGrafica con learning rate igual a 0.1')
lr = 0.1
gd_grafica(w_0, lr, gradf, f, max_iters)


input("\n--- Pulsar tecla para continuar ---\n")


# b) Obtener el valor minimo y los valores de (x,y) con los
# puntos de inicio siguientes:
lr = 0.01
max_iters = 100

def gd(w, lr, grad_fun, fun, max_iters = 10):	
    it = 0
    while (it < max_iters):
        w = w - lr * grad_fun(w)
        it += 1
    return w

print ('Punto de inicio: (2.1, -2.1)\n')

w_0 = (2.1, -2.1)
w = gd(w_0, lr, gradf, f, max_iters)

print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor minimo: ',f(w))

input("\n--- Pulsar tecla para continuar ---\n")

w_0 = (3.0, -3.0)
w = gd(w_0, lr, gradf, f, max_iters)

print ('Punto de inicio: (3.0, -3.0)\n')
print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor minimo: ',f(w))

input("\n--- Pulsar tecla para continuar ---\n")

print ('Punto de inicio: (1.5, 1.5)\n')

w_0 = (1.5, 1.5)
w = gd(w_0, lr, gradf, f, max_iters)

print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor minimo: ',f(w))

input("\n--- Pulsar tecla para continuar ---\n")

print ('Punto de inicio: (1.0, -1.0)')

w_0 = (1.9, -1.0)
w = gd(w_0, lr, gradf, f, max_iters)

print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor mínimo: ',f(w))

input("\n--- Pulsar tecla para continuar ---\n")
