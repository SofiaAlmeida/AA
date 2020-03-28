# -*- coding: utf-8 -*-

#############################
#####     LIBRERIAS     #####
#############################
import numpy as np
from sklearn.utils import shuffle
from numpy import linalg as LA
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
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
# Error = 1/N * || x*w -y||^2
def Err(x,y,w):
	return  (1 / x.shape[0]) * (LA.norm(np.dot(x, w) - y))**2

# Función para calcular el gradiente de la función de error
def grad_err(x, y, w):
	h = np.dot(x, w) 
	return (2 / x.shape[0]) * np.dot(x.T, (h - y))
	
	
# Gradiente Descendente Estocastico
# x:  matriz de datos
# y: valor correspondiente a cada elemento
# lr: tasa de aprendizaje
# max_iters: número máximo de iteraciones
# tam_minbatch: tamaño de los minibatchs
def sgd(x, y, lr, max_iters, tam_minibatch, w = np.array([0.0, 0.0, 0.0])):
	# Inicializamos número de minibatchs según el total de elementos y el tamaño
	num_minibatch = x.shape[0] / tam_minibatch
	it = 0
	
	while(it < max_iters):
		# Desordenamos los conjuntos
		x, y = shuffle(x, y)
		# Creamos los minibatchs
		minibatchs_x = np.array_split(x, num_minibatch)
		minibatchs_y = np.array_split(y, num_minibatch)
	
		for xj, yj in zip(minibatchs_x, minibatchs_y):
			# Actualizamos w
			w = w - lr * grad_err(xj, yj, w)

		it += 1	
	
	return w
	
# Algoritmo pseudoinversa	
def pseudoinverse(x, y):
	# Pseudoinversa de x = (xT*x)^-1 * xT
	pseudoinv = np.dot(LA.inv(np.dot(x.T, x)), x.T)
	return np.dot(pseudoinv, y)
	
# Gráfica de resultados del hiperplano de regresión
def graph3d(x, y, w, title):
	fig = plt.figure()
	ax = fig.add_subplot(111,projection='3d')
	a = np.arange(-1.0, 1.0, 0.025)
	b = np.arange(-1.0, 1.0, 0.025)
	A, B = np.meshgrid(a, b)
	zs = np.array([np.dot([1, a, b], w) for a,b in zip(np.ravel(A), np.ravel(B))])
	Z = zs.reshape(A.shape)
	surf = ax.plot_surface(A, B, Z, color = 'cyan', label= "Hiperplano") # Dibujamos el hiperplano
	surf._facecolors2d=surf._facecolors3d
	surf._edgecolors2d=surf._edgecolors3d
	
	ax.scatter(x[:, 1], x[:,2], y, color = 'purple', label = "Conjunto de entrenamiento") # pintamos los puntos

	ax.set_xlabel('Intensidad promedio')
	ax.set_ylabel('Simetría')
	ax.set_zlabel('Etiquetas\n 1 si es un 5,\n -1 si es un 1')
	ax.set_title(title, loc = 'left')
	ax.legend(loc='upper left')

	ax.view_init(elev=10., azim=312)
	plt.show()
	#plt.savefig("./prueba/" + title +"%312.png")
	plt.clf()

# Gráfica de resultados - clasificación con 2 resultados, uno de sgd y otro de pseudoinversa
def graph(x, y, w, w1):
	sns.set()
	fig = plt.figure()
	ax = fig.add_subplot()
	# Plot outputs
	colors = ['tab:pink', 'tab:blue']

	scatter = ax.scatter(x[:,1], x[:,2], c = y, alpha = 0.6, edgecolors='none', cmap =matplotlib.colors.ListedColormap(colors))
	plt.plot(x[:,1], (-w[0] - w[1]*x[:,1]) / w[2], alpha = 0.5, color = 'black', linewidth=1.3, label = 'SGD')
	plt.plot(x[:,1], (-w1[0] - w1[1]*x[:,1]) / w1[2], color = 'gray', alpha = 0.5, linewidth=1.3, label = 'Pesudoinversa')

	# Leyenda
	legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Clases")
	ax.legend(loc = 'upper right')
	ax.add_artist(legend1)
	ax.set_xlabel('Intensidad promedio')
	ax.set_ylabel('Simetría')	
	ax.set_title("")	

	# Eliminamos eje superior y derecho
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	
	# Añadimos cuadrícula
	plt.grid(True, color="white", linewidth = 1)
	plt.show()
	#plt.savefig("./fig/results2.png")


# Lectura de los datos de entrenamiento
x, y = readData('./datos/X_train.npy', './datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('./datos/X_test.npy', './datos/y_test.npy')

print ('EJERCICIO SOBRE REGRESION LINEAL\n')
print ('Ejercicio 1\n')
# Gradiente descendente estocastico

w_sgd = sgd(x, y, 0.1, 75, 75)

print ('Bondad del resultado para grad. descendente estocástico:\n')
print ("Ein: ", Err(x, y, w_sgd))
print ("Eout: ", Err(x_test, y_test, w_sgd))

input("\n--- Pulsar tecla para continuar ---\n")

# Algoritmo Pseudoinversa

w = pseudoinverse(x, y)

graph(x, y, w_sgd, w)

print ('\nBondad del resultado para el algoritmo de la pseudoinversa:\n')
print ("Ein: ", Err(x, y, w))
print ("Eout: ", Err(x_test, y_test, w))
input("\n--- Pulsar tecla para continuar ---\n")


#------------------------------Ejercicio 2 -------------------------------------#

# Simula datos en un cuadrado [-size,size]x[-size,size]
# Devuelve N coordenadas 2D de puntos uniformemente muestreados dentro del cuadrado 
# [-size, size] x [-size, size]
def simula_unif(N, d, size):
	return np.random.uniform(-size, size, (N, d))
	
# Dibuja el conjunto de puntos points
# points: array Nx2
def draw_points(points):
	sns.set()
	fig = plt.figure()
	ax = fig.add_subplot()
	# Plot outputs
	scatter = ax.scatter(points[:,0], points[:,1], alpha = 0.6, color = 'tab:pink', edgecolors='none')

	# Leyenda	
	ax.set_title("Muestra de entrenamiento")	

	# Eliminamos eje superior y derecho
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	
	# Añadimos cuadrícula
	plt.grid(True, color="white", linewidth = 1)
	plt.show()
	#plt.savefig("./fig/muestra2d.png")

# Calcula el valor f de una matriz x, Nx2
def f(x1, x2):
	return np.sign((x1 - 0.2)**2 + x2**2 - 0.6)

# Devuelve el vector de clases del conjunto x(Nx2)
# La clase viene determinada por la función f
# Añade ruido cambiando el 10% de las etiquetas
def classes(x):
	# Calculamos la clase determinada por f
	y = np.array([f(x[i,0], x[i,1]) for i in range(0, x.shape[0])])
	# Alteramos el 10% de los valores
	y[np.random.rand(*y.shape)>=0.9] *= -1 	
	return y
	
# Devuelve el vector de clases del conjunto x(Nx2)
# La clase viene determinada por la función f
# Añade ruido cambiando el 10% de las etiquetas
def classes_nonoise(x):
	# Calculamos la clase determinada por f
	y = np.array([f(x[i,0], x[i,1]) for i in range(0, x.shape[0])])
	return y

# Devuelve una muestra x y un vector y con las clases correspondientes
def get_set(N, size):
	sample = simula_unif(N, 2, size)
	y = classes(sample)

	# Añadimos una columna de unos a la muestra
	x = np.hstack((np.ones((N,1)), sample))	
	return x, y

# Devuelve una muestra x y un vector y con las clases correspondientes
# x = 1, x_1, x_2, x_1*x_2, x_1^2, x_2^2
def get_set_nolineal(N, size):
	sample = simula_unif(N, 2, size)
	y = classes(sample)

	# Añadimos una columna de unos a la muestra
	x = np.array([[1, x1, x2, x1*x2, x1**2, x2**2] for x1, x2 in sample])
 	
	return x, y

# Devuelve una muestra x y un vector y con las clases correspondientes (sin errores)
# x = 1, x_1, x_2, x_1*x_2, x_1^2, x_2^2
def get_set_nolineal_nonoise(N, size):
	sample = simula_unif(N, 2, size)
	y = classes_nonoise(sample)

	# Añadimos una columna de unos a la muestra
	x = np.array([[1, x1, x2, x1*x2, x1**2, x2**2] for x1, x2 in sample])
 	
	return x, y


# Dibuja el conjunto de puntos points y los colorea según su clase y
# points: array Nx2
def draw_classes(points, y):
	sns.set()
	fig = plt.figure()
	ax = fig.add_subplot()

	# Plot outputs
	colors = ['tab:pink', 'tab:blue']

	scatter = ax.scatter(points[:,0], points[:,1], c = y, alpha = 0.6, edgecolors='none', cmap =matplotlib.colors.ListedColormap(colors))

	# Leyenda	
	# Shrink current axis by 20%
	box = ax.get_position()
	#ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

	# Put a legend to the right of the current axis
	ax.set_title("Muestra de entrenamiento")	
	legend1 = ax.legend(*scatter.legend_elements(), title="Clases", loc=(1.04,0))
	ax.add_artist(legend1)


	# Añadimos cuadrícula
	plt.grid(True, color="white", linewidth = 1)

	plt.subplots_adjust(right=0.8)	
	plt.show()
	#plt.savefig("./fig/clases2d.png")

# Gráfico de los puntos, coloreando su clase y pintando la recta dada por w
# Caso lineal
def plot(x, y, w):
	sns.set()
	fig = plt.figure()
	ax = fig.add_subplot()
	# Plot outputs
	colors = ['tab:pink', 'tab:blue']

	scatter = ax.scatter(x[:,1], x[:,2], c = y, alpha = 0.6, edgecolors='none', cmap =matplotlib.colors.ListedColormap(colors))
	plt.plot(x[:,1], (-w[0] - w[1]*x[:,1]) / w[2], alpha = 0.6, color = 'black', linewidth=1.3, label = 'SGD')

	ax.set_xlim(-1.1, 1.1)
	ax.set_ylim(-1.1, 1.1)

	# Leyenda
	legend1 = ax.legend(*scatter.legend_elements(), loc=(1.04,0), title="Clases")
	ax.add_artist(legend1)
	ax.set_xlabel('')
	ax.set_ylabel('')	
	ax.set_title("Ajuste mediante SGD")	

	# Añadimos cuadrícula
	plt.grid(True, color="white", linewidth = 1)
	plt.subplots_adjust(right=0.8)		

	plt.show()
	#plt.savefig("./fig/results22.png")

# Gráfico de los puntos, coloreando su clase y pintando la recta dada por w
# caso no lineal
def plot_nolineal(x, y, w, title = 'nolineal'):
	sns.set()
	fig = plt.figure()
	ax = fig.add_subplot()
	# Plot outputs
	colors = ['tab:pink', 'tab:blue']

	scatter = ax.scatter(x[:,1], x[:,2], c = y, alpha = 0.6, edgecolors='none', cmap =matplotlib.colors.ListedColormap(colors))

	delta = 0.025
	a = np.arange(-1.0, 1.0, delta)
	b = np.arange(-1.0, 1.0, delta)
	X, Y = np.meshgrid(a, b)
	zs = np.array([np.dot([1, a, b, a*b, a*a, b*b], w) for a,b in zip(np.ravel(X), np.ravel(Y))])
	Z = zs.reshape(X.shape)
	CS = ax.contour(X, Y, Z, levels = 0, alpha = 0.6, colors = ['black'])
	
	ax.set_xlim(-1.1, 1.1)
	ax.set_ylim(-1.1, 1.1)

	# Leyenda
	legend1 = ax.legend(*scatter.legend_elements(), loc=(1.04,0), title="Clases")
	ax.add_artist(legend1)
	ax.set_xlabel('')
	ax.set_ylabel('')	
	ax.set_title("Ajuste no lineal")	

	# Añadimos cuadrícula
	plt.grid(True, color="white", linewidth = 1)
	plt.subplots_adjust(right=0.8)		
	plt.show()
	#plt.savefig("./fig/" + title + ".png")

# Generamos 2 conjuntos de datos, llamamos a sgd y calculamos los errores
# p = True genera las gráficas asociadas al conjunto x
def experiment(N, size, lr, max_iters, tam_minibatch, p = False):
	# Creamos conjunto de training y test
	x, y = get_set(N, size)
	x_test, y_test = get_set(N, size)

	w = sgd(x, y, lr, max_iters, tam_minibatch)
	E_in = Err(x, y, w)
	E_out = Err(x_test, y_test, w)
	
	if(p):
		draw_points(x[:,1:])
		draw_classes(x[:,1:], y)
		plot(x, y, w)

	return w, E_in, E_out
	
# Generamos 2 conjuntos de datos, llamamos a sgd y calculamos los errores
# p = True genera las gráficas asociadas al conjunto x
def experiment_nolineal(N, size, lr, max_iters, tam_minibatch, p = False):
	# Creamos conjunto de training y test
	x, y = get_set_nolineal(N, size)
	x_test, y_test = get_set_nolineal(N, size)

	w = sgd(x, y, lr, max_iters, tam_minibatch, np.zeros(6))
	E_in = Err(x, y, w)
	E_out = Err(x_test, y_test, w)
	
	if(p):
		plot_nolineal(x, y, w)

	return w, E_in, E_out
	
# Generamos 2 conjuntos de datos, llamamos a sgd y calculamos los errores
# Datos sin error en el 10%
# p = True genera las gráficas asociadas al conjunto x
def experiment_nonoise(N, size, lr, max_iters, tam_minibatch, p = False):
	# Creamos conjunto de training y test
	x, y = get_set_nolineal_nonoise(N, size)
	x_test, y_test = get_set_nolineal_nonoise(N, size)

	w = sgd(x, y, lr, max_iters, tam_minibatch, np.zeros(6))
	E_in = Err(x, y, w)
	E_out = Err(x_test, y_test, w)
	
	if(p):
		draw_points(x[:,1:])
		draw_classes(x[:,1:], y)
		plot_nolineal(x, y, w, 'Sin ruido')

	return w, E_in, E_out


# EXPERIMENTO	
# a) Muestra de entrenamiento N = 1000, cuadrado [-1,1]x[-1,1]	

print ('Ejercicio 2\n')
print ('Muestra N = 1000, cuadrado [-1,1]x[-1,1]')
N = 1000
size = 1

# c) Usando como vector de características (1, x_1 , x_2 ) ajustar un modelo de regresion
#  lineal al conjunto de datos generado y estimar los pesos w.

w, ein, eout = experiment(N, size, 0.1, 75, 75, True)

print ('Bondad del resultado para grad. descendente estocástico:\n')
print ("Ein: ", ein)
print ("Eout: ", eout)
input("\n--- Pulsar tecla para continuar ---\n")

# -------------------------------------------------------------------

# d) Ejecutar el experimento 1000 veces
Ein = np.array([])
Eout = np.array([])
for i in range(0, 1000):
	w, ein, eout = experiment(N, size, 0.1, 75, 75)
	Ein = np.append(Ein, ein)
	Eout = np.append(Eout, eout)
	
Ein_media = np.mean(Ein)
Eout_media = np.mean(Eout)

print ('Errores Ein y Eout medios tras 1000reps del experimento:\n')
print ("Ein media: ", Ein_media)
print ("Eout media: ", Eout_media)

input("\n--- Pulsar tecla para continuar ---\n")

# -------------------------------------------------------------------

# Repetir para phi
experiment_nolineal(N, size, 0.1, 75, 75, True)
Ein = np.array([])
Eout = np.array([])
for i in range(0, 1000):
	w, ein, eout = experiment_nolineal(N, size, 0.1, 75, 75)
	Ein = np.append(Ein, ein)
	Eout = np.append(Eout, eout)

Ein_media = np.mean(Ein)
Eout_media = np.mean(Eout)
print ('Bondad del resultado para características no lineales (tras 1000 repeticiones):\n')
print ("Ein: ", Ein_media)
print ("Eout: ", Eout_media)
input("\n--- Pulsar tecla para continuar ---\n")

#-----------------
# probamos el caso no lineal sin datos erróneos
w, ein, eout = experiment_nonoise(N, size, 0.1, 75, 75, True)
print("Resultado ajuste no lineal sin ruido: ")
print ("Ein: ", ein)
print ("Eout: ", eout)


