import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
plt.style.use('seaborn-ticks')
mpl.rcParams['mathtext.default'] = 'regular'
font = {'family': 'Times New Roman',
         'style': 'normal',
        'weight': 'normal',
         'color': 'black',
          'size': 18,
        }
my_path = os.path.abspath('')
x = np.linspace(0,2*np.pi,1000)
y1 = np.sin(x)
y2 = np.cos(x)
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.plot(x,y1,label=r'$sin(x)$')
ax1.set_xlabel(r'$t/\tau_a$',fontdict = font)
ax1.set_ylabel(r'value for sin(x)',fontdict = font)
ax1.tick_params(direction='in',labelsize=13)
ax1.legend(fontsize=13)
ax2 = ax1.twinx()
ax2.plot(x,y2,'r',label=r'$cos(x)$')
ax2.legend(fontsize=13)
ax2.set_ylabel(r'value for cos(x)',fontdict = font)
ax2.tick_params(direction='in',labelsize=13)
plt.show()
