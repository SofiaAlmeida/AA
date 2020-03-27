# -*- coding: utf-8 -*-

#############################
#####     LIBRERIAS     #####
#############################
import numpy as np
from sklearn.utils import shuffle
from numpy import linalg as LA


#-------------------------------------------------------------------------------#
#---------------------- Ejercicio sobre regresión lineal -----------------------#
#-------------------------------------------------------------------------------#

#------------------------------Ejercicio 1 -------------------------------------#


# Función para leer los datos
def readData(file_x, file_y):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []
	
	# Solo guardamos los datos cuya clase sea la 1 o la 5
	for i in range(0,datay.size):
		if datay[i] == 5 or datay[i] == 1:
			if datay[i] == 5:
				y.append(1)
			else:
				y.append(-1)
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y
	
# Función para calcular el error
def Err(x,y,w):
	return  (1 / x.shape[0]) * (LA.norm(np.dot(x, w) - y))**2

# Función para calcular el gradiente de la función de error
def grad_err(x, y, w):
	h = np.dot(x, w) 
	return (2 / x.shape[0]) * sum(np.dot(x.T, (h - y)))
	
	
# Gradiente Descendente Estocastico
def sgd(x, y, lr, max_iters, tam_minibatch):
	num_minibatch = x.shape[0] / tam_minibatch
	it = 0
	w = np.array([0.0, 0.0, 0.0])
	
	while(it < max_iters):
		# Desordenamos los conjuntos
		x, y = shuffle(x, y)
		minibatchs_x = np.array_split(x, num_minibatch)
		minibatchs_y = np.array_split(y, num_minibatch)
	
		for xj, yj in zip(minibatchs_x, minibatchs_y):
			w = w - lr * grad_err(xj, yj, w)

		it += 1	
	
	return w
	
# Algoritmo pseudoinversa	
def pseudoinverse(x, y):

	return w
	
# Lectura de los datos de entrenamiento
x, y = readData('./datos/X_train.npy', './datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('./datos/X_test.npy', './datos/y_test.npy')

print ('EJERCICIO SOBRE REGRESION LINEAL\n')
print ('Ejercicio 1\n')
# Gradiente descendente estocastico

w = sgd(x, y, 0.1, 50, 30)

print ('Bondad del resultado para grad. descendente estocástico:\n')
print ("Ein: ", Err(x, y, w))
print ("Eout: ", Err(x_test, y_test, w))

input("\n--- Pulsar tecla para continuar ---\n")

# Algoritmo Pseudoinversa

w = pseudoinverse(x, y)

print ('\nBondad del resultado para el algoritmo de la pseudoinversa:\n')
print ("Ein: ", Err())
print ("Eout: ", Err())


#------------------------------Ejercicio 2 -------------------------------------#

# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	return 
	
# EXPERIMENTO	
# a) Muestra de entrenamiento N = 1000, cuadrado [-1,1]x[-1,1]	

print ('Ejercicio 2\n')
print ('Muestra N = 1000, cuadrado [-1,1]x[-1,1]')




# -------------------------------------------------------------------

# d) Ejecutar el experimento 1000 veces

print ('Errores Ein y Eout medios tras 1000reps del experimento:\n')
print ("Ein media: ", Ein_media)
print ("Eout media: ", Eout_media)

input("\n--- Pulsar tecla para continuar ---\n")
