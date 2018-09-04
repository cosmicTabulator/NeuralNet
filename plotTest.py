#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 18:59:28 2018

@author: grahamcooke
"""

import colorsys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

"""
x = np.arange(-10, 10, 0.1) + np.zeros((200, 1))

y = x.T[::-1]

x = x.ravel()
y = y.ravel()

z = x**2 - y**2

z = z.reshape(200,200)
"""

def blend(a, b, factor):
    return (factor*a + (1-factor)*b)


z = np.arange(-5, 5, 0.05)

z = z.reshape((200, 1))[::-1]*1j + z

z = z

grid = ((np.real(z)%5 <= 0.1) + (np.imag(z)%5 <= 0.1))*1

arg = np.angle(z)/(2*np.pi) + 0.5
arg += (arg > 0.5)*-0.5 + (arg < 0.5)*0.5
mag = np.sqrt(np.real(z)**2 + np.imag(z)**2)
mag = np.log2(mag)%1
r = np.zeros(arg.shape)
g = np.copy(r)
b = np.copy(g)


f = 0.8
r = blend(arg, mag, f)
b = blend((1-arg), mag, f)
g = blend(0, mag, f)


'''
for i in range(arg.shape[0]):
    for j in range(arg.shape[1]):
        rgb = colorsys.hls_to_rgb(arg[i][j], mag[i][j], 1)
        #print(rgb[0])
        r[i][j] = rgb[0]
        g[i][j] = rgb[1]
        b[i][j] = rgb[2]
'''




r.resize((r.shape[0], r.shape[1], 1))
g.resize((g.shape[0], g.shape[1], 1))
b.resize((b.shape[0], b.shape[1], 1))

im = np.concatenate((r,g,b), axis=2)

fig, ax = plt.subplots()
#ax.imshow(z)
#ax.imshow(arg)
ax.imshow(im)

plt.show()