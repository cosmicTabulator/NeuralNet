#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 18:59:28 2018

@author: grahamcooke
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


x = np.arange(-10, 10, 0.1) + np.zeros((200, 1))

y = x.T[::-1]

x = x.ravel()
y = y.ravel()

z = x**2 - y**2

z = z.reshape(200,200)

print(z.shape)

fig, ax = plt.subplots()
ax.imshow(z)

plt.show()