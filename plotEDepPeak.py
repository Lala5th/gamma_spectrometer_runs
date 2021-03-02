#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

args = sys.argv
cmap = cm.get_cmap('gnuplot')

plt.ion()
plt.figure()

for dir in args[1:]:

    E = np.load(dir + '/InitialEnergy.npy')['0']
    y = np.load(dir + '/y.npy')['0']
    z = np.load(dir + '/z.npy')['0']
    data = np.reshape(np.load(dir + '/DepositedEnergy.npy'),(len(E),len(y),len(z)))

    norm = colors.Normalize(vmin=0,vmax=max(E))
    peaks = [np.argmax(np.sum(data['1'][i,:,:],axis = 0)) for i in range(len(E))]

    plt.plot(E,peaks,label=dir)

plt.legend()
plt.ylabel('Peak Crystal ID')
plt.xlabel('Initial Energy [MeV]')
plt.show()
