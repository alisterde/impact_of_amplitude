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

x = ['0.25','0.5','1','2','4','8','16','32','64','128','256']

y = [1,2,2,7,15,12,13,17,17,11,6]

plt.figure(figsize=(8.4/2.5, 2))
# plt.bar(x, y, color ='maroon', width = 0.4)
plt.bar(x, y)
# plt.title('(a)')
plt.xlabel('k$^0$ / s$^{-1}$')
plt.ylabel('Frequency of Occurrence')
plt.ylim(0,17)
plt.yticks([4,8,12,16])
# plt.ylabel('Separation / mV')
# plt.xlabel('Amplitude ($\mathrm{\Delta}$E) / mV')
# plt.plot(x, y_experimental, color = '#000000', label = 'Experiment')
# plt.scatter(x, y_experimental, marker= "+", color = '#000000')
# plt.plot(x, y_simualted, color = '#E69F00', label = 'Simulation')
# plt.scatter(x, y_simualted, color = '#E69F00')
# plt.legend(loc='best', frameon = False, columnspacing = 0.3, handlelength = 1.0)

plt.tight_layout()

plt.savefig('figure_1_a.png', dpi=500)
