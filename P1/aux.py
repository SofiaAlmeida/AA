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

plt.savefig('./fig/E1.png')
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
plt.savefig('./fig/f1.png')
plt.clf()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(*([2.1,-2.1].reshape(-1,1)), fun2(2.1,-2.1), 'r*', markersize=10)
x = y = np.arange(-2.9, 2.9, 0.025)
X, Y = np.meshgrid(x, y)
zs = np.array([fun2(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

ax.plot_surface(X, Y, Z)

ax.set_xlabel('Eje x')
ax.set_ylabel('Eje y')
ax.set_zlabel('Valor f(x,y)')
ax.set_title('Representación f(x,y)')
plt.savefig('./fig/f1+pts.png')

