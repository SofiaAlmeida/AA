# -*- coding: utf-8 -*-

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
import math
import seaborn as sns

def fun(u, v):
   return (u * math.exp(v) - 2 * v * math.exp(-u))**2

sns.set()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = y = np.arange(-2.5, 2.5, 0.025)
X, Y = np.meshgrid(x, y)
zs = np.array([fun(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

ax.plot_surface(X, Y, Z)

ax.set_xlabel('Eje u')
ax.set_ylabel('Eje v')
ax.set_zlabel('Valor E(u,v)')
ax.set_title('Representación E(u,v)')

plt.show()
#plt.savefig('./fig/E1.png')
plt.clf()

def fun2(x,y):
   return (x - 2)**2 + 2 * (y + 2)**2 + 2 * math.sin(2 * math.pi * x) * math.sin(2 * math.pi * y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = y = np.arange(-2.9, 2.9, 0.025)
X, Y = np.meshgrid(x, y)
zs = np.array([fun2(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

ax.plot_surface(X, Y, Z)

ax.set_xlabel('Eje x')
ax.set_ylabel('Eje y')
ax.set_zlabel('Valor f(x,y)')
ax.set_title('Representación f(x,y)')
plt.show()
#plt.savefig('./fig/f1.png')
plt.clf()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = y = np.arange(-2.9, 2.9, 0.025)
X, Y = np.meshgrid(x, y)
zs = np.array([fun2(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

surf = ax.plot_surface(X, Y, Z, label='f(x,y)')
surf._facecolors2d=surf._facecolors3d
surf._edgecolors2d=surf._edgecolors3d

ptos_ini = np.array([[2.1,-2.1], [3.0,-3.0], [1.5,1.5], [1.0,-1.0]])
ax.scatter(ptos_ini[:,0], ptos_ini[:,1], [fun2(x,y) for x,y in ptos_ini], label = "Puntos iniciales", color='black', alpha = 1) # pintamos los puntos

ax.view_init(elev=10., azim=330)
ax.set_xlabel('Eje x')
ax.set_ylabel('Eje y')
ax.set_zlabel('Valor f(x,y)')
ax.legend(loc='upper left')
plt.show()
#plt.savefig('./fig/f1+pts.png')

