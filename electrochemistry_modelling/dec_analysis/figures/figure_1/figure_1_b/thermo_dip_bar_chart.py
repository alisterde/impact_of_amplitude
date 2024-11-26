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

x = [12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 57.5]

y = [2, 8, 22, 40, 25, 9, 2, 2]

plt.figure(figsize=(8.4/2.5, 2))
# plt.bar(x, y, color ='maroon', width = 0.4)
plt.bar(x, y, width = 4.9)
# plt.title('(b)')
plt.xlabel('E$^0$ / mV')
plt.ylabel('Frequency of Occurrence')
plt.xlim(10,60)
plt.ylim(0,40)

# plt.ylabel('Separation / mV')
# plt.xlabel('Amplitude ($\mathrm{\Delta}$E) / mV')
# plt.plot(x, y_experimental, color = '#000000', label = 'Experiment')
# plt.scatter(x, y_experimental, marker= "+", color = '#000000')
# plt.plot(x, y_simualted, color = '#E69F00', label = 'Simulation')
# plt.scatter(x, y_simualted, color = '#E69F00')
# plt.legend(loc='best', frameon = False, columnspacing = 0.3, handlelength = 1.0)

plt.tight_layout()

plt.savefig('figure_1_b.png', dpi=500)
