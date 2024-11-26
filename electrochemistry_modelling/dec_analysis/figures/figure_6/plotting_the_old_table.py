import numpy as np
import copy
import math
import os
from electrochemistry_modelling import harmonics_and_fourier_transform


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})
plt.rcParams.update({'lines.linewidth': 2})

x = [25,50,75,100,125,150,
     175,200,225,250,275,300]

y_experimental = [157.5, 155.7, 159.2, 165.5, 176.2, 160.3,
                  205.9, 224.7, 247.2, 269.2, 290.5, 323.3]

y_simualted = [72.2, 85.3, 105.1, 129.7, 157.9, 188.5,
               220.9, 254.4, 288.9, 324.0, 359.5, 395.3]

y_dip_70_simualted = [164.80, 168.87, 175.76, 185.77,
                      198.98, 215.56, 235.38, 258.17,
                      283.57, 311.10, 340.41, 371.30]

plt.figure(figsize=(8.4/2.5, 2))
plt.ylabel('Separation / mV')
plt.xlabel('Amplitude ($\mathrm{\Delta}$E) / mV')
plt.plot(x, y_experimental, color = '#000000', label = 'Experiment')
plt.scatter(x, y_experimental, marker= "+", color = '#000000')
plt.plot(x, y_simualted, color = '#E69F00', label = r'$E^0_\sigma = 0$ mV')
plt.scatter(x, y_simualted, color = '#E69F00')
# plt.plot(x, y_dip_60_simualted, color = '#56B4E9', label = '60')
# plt.scatter(x, y_dip_60_simualted, color = '#56B4E9')
plt.plot(x, y_dip_70_simualted, color = '#009E73', label = r'$E^0_\sigma = 70$ mV')
plt.scatter(x, y_dip_70_simualted, color = '#009E73')
plt.legend(loc='best', frameon = False, columnspacing = 0.3, handlelength = 1.0)

plt.tight_layout()

plt.savefig('figure_6.png', dpi=500, transparent = True)

