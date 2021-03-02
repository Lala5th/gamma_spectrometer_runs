#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib.widgets import Slider

args = sys.argv
cmap = cm.get_cmap('gnuplot')

dirRun = args[1]
dirRef = args[2]
id = int(args[3])

ERef = np.load(dirRef + '/InitialEnergy.npy')['0']
y = np.load(dirRef + '/y.npy')['0']
z = np.load(dirRef + '/z.npy')['0']
ref = np.reshape(np.load(dirRef + '/DepositedEnergy.npy'),(len(ERef),len(y),len(z)))

ECrits = np.load(dirRun + '/ECrit.npy')['0']
EDist = np.load(dirRun + '/Energy.npy')['1']
EDist = np.reshape(EDist,(len(ECrits),int(len(EDist)/len(ECrits))))

data = np.reshape(np.load(dirRun + '/DepositedEnergy.npy'),(len(ECrits),len(y),len(z)))['1']

layers = np.array([np.sum(ref['1'][i,:,:],axis = 0) for i in range(len(ERef))])

plt.ion()
plt.figure()
new_ERef = (ERef[1:] + ERef[:-1])/2
new_ERef = np.append(new_ERef,ERef[-1]+10)
new_ERef = np.insert(new_ERef,0,0)
l = np.histogram(EDist[id],bins=new_ERef)
l_err = np.sqrt(l[0])
EDep = np.zeros(len(z))
EDep = np.sum((l[0])*layers.T,axis=1)
EDep_d = np.sum((1.96*l_err)*layers.T,axis=1)/len(ECrits)
EDep_n = EDep - EDep_d
EDep_p = EDep + EDep_d
plincomb, = plt.plot(z,EDep/np.max(EDep),label='Linear combination')
psim, = plt.plot(z,np.sum(data[id],axis=0)/np.max(np.sum(data[id],axis=0)),label='Simulated')
plt.fill_between(z,EDep_n/np.max(EDep),EDep_p/np.max(EDep),label='95% CI',alpha=0.2)
plt.legend()
plt.ylabel('E$_{norm}$ [a.u.]')
plt.xlabel('Longitudinal crystal ID')
plt.show()
