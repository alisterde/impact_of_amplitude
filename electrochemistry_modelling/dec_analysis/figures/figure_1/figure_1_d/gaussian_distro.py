import math
import os
import sys
import copy
from turtle import color
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})
plt.rcParams.update({'lines.linewidth': 2})
from  scipy.stats import norm
# import seaborn as sns

# #defining constnats

# mu = -250
# sigma = 20
# x1 = -270
# x2 = -230

# #calcuatle the z-transfrom

# z1 = (x1 - mu)/sigma
# z2 = (x2 - mu)/sigma

# x = np.arange(z1,z2,0.001) # range of x in spec
# x_all = np.arange(-10,10,0.001) # entire range of x, both in and out of spec

# # mean = 0, stddev = 1, since Z-transform was calculated
# y = norm.pdf(x,0,1)
# y2 = norm.pdf(x_all,0,1)

# # build the plot
# fig,ax = plt.subplots(figsize=(9,6))
# plt.style.use('fivethirtyeight')
# ax.plot(x_all,y2)

# ax.fill_between(x,y,0,alpha=0.3,color = 'b')
# ax.fill_between(x_all,y2,0,alpha=0.1)

# sigma = 50
# x1 = -300
# x2 = -200

# #calcuatle the z-transfrom

# z1 = (x1 - mu)/sigma
# z2 = (x2 - mu)/sigma

# x = np.arange(z1,z2,0.001) # range of x in spec
# x_all = np.arange(-10,10,0.001) # entire range of x, both in and out of spec

# # mean = 0, stddev = 1, since Z-transform was calculated
# y = norm.pdf(x,0,1)
# y2 = norm.pdf(x_all,0,1)

# ax.fill_between(x,y,0,alpha=0.3,color = 'r')
# ax.fill_between(x_all,y2,0,alpha=0.1)

# ax.set_xlim([-4,4])
# ax.set_xlabel('# of standard devisations oustide th mean')
# # ax.set_yticklabels([])
# ax.set_title('Nomral Gaussian Curve')

# plt.show()

# narrow_values = np.random.normal(loc=-250,scale=20, size = 20000000)
# wide_values = np.random.normal(loc=-250,scale=50, size = 20000000)

x = np.arange(-250,250,0.001)

mean = 0.0
std = 20

narrow_values = norm.pdf(x,loc=mean ,scale=std)
narrow = np.where(x  < mean - std)
narrow_lower_index = narrow[0][-1]
print('narrow_lower_index: ', narrow_lower_index)
narrow = np.where(x < mean + std)
narrow_upper_index = narrow[0][-1]
print('narrow_upper_index: ', narrow_upper_index)

mean = 0.0
std = 50
wide_values = norm.pdf(x,loc=mean ,scale=std)
wide = np.where(x  < mean - std)
wide_lower_index = wide[0][-1]
wide = np.where(x < mean + std)
wide_upper_index = wide[0][-1]

# sns.displot(wide_values)

plt.figure(figsize=(8.4/2.5, 2))
# plt.title("(c)")
plt.ylabel("Normalised Curve")
plt.xlabel('E$^{0}$ / mV')
plt.plot(x,narrow_values,'r', label='$E^\mathrm{0}_\sigma$ = 20 mV')
plt.fill_between(x[narrow_lower_index:narrow_upper_index],narrow_values[narrow_lower_index:narrow_upper_index],0,alpha=0.3,color = 'r')
plt.plot(x,wide_values,'b', label='$E^\mathrm{0}_\sigma$ = 50 mV')
plt.fill_between(x[wide_lower_index:wide_upper_index],wide_values[wide_lower_index:wide_upper_index],0,alpha=0.3,color = 'b')
plt.yticks(ticks = [])
plt.xlim(-175,175)
# sns.displot(wide_values)
# sns.displot(narrow_values)
plt.legend(loc='best', frameon = False, columnspacing = 0.3, handlelength = 1.0)
plt.tight_layout()
plt.savefig('figure_1_d.png', dpi = 500)

