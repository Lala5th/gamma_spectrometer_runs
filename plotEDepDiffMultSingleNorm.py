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
dirErr = args[3]
id = int(args[4])

ERef = np.load(dirRef + '/InitialEnergy.npy')['0']
y = np.load(dirRef + '/y.npy')['0']
z = np.load(dirRef + '/z.npy')['0']
ref = np.reshape(np.load(dirRef + '/DepositedEnergy.npy'),(len(ERef),len(y),len(z)))

ECrits = np.load(dirRun + '/ECrit.npy')['0']
EDist = np.load(dirRun + '/Energy.npy')['1']
EDist = np.reshape(EDist,(len(ECrits),int(len(EDist)/len(ECrits))))

data = np.reshape(np.load(dirRun + '/DepositedEnergy.npy'),(len(ECrits),len(y),len(z)))['1']

layers = np.array([np.sum(ref['1'][i,:,:],axis = 0) for i in range(len(ERef))])


Err = np.reshape(np.load(dirErr + '/DepositedEnergy.npy'),(len(ERef),len(y),len(z)))['1']

Err = np.sum(Err,axis=1)
Err[Err < 0] = 0
Err = np.sqrt(Err)

plt.ion()
plt.figure()
plt.plot(z,layers[id])

fig, (ax1, ax2) = plt.subplots(2,1,sharex=True)
new_ERef = (ERef[1:] + ERef[:-1])/2
new_ERef = np.append(new_ERef,ERef[-1]+10)
new_ERef = np.insert(new_ERef,0,0)
l = np.histogram(EDist[id],bins=new_ERef)
l_err = np.sqrt(l[0])
geantE = np.sqrt(np.sum((Err.T*np.sqrt(l[0]))**2,axis=1))

EDep = np.sum((l[0])*layers.T,axis=1)

EDep_d = 1.96*np.sqrt(np.max(l_err*layers.T,axis=1)**2 + geantE**2)/np.max(EDep)
EDep /= np.max(EDep)
d = np.sum(data[id],axis=0)
d /= np.max(d)
ax1.plot(z,EDep,label='Linear combination')
ax1.plot(z,d,label='Simulated')
ax1.fill_between(z,EDep - EDep_d,EDep + EDep_d,label='95% CI',alpha=0.2)
ax1.legend()
ax1.set_ylabel('E [MeV]')
ax2.set_ylabel('E [MeV]')
ax2.set_xlabel('Longitudinal crystal ID')
ax2.plot(z,d - EDep,label='Residual')
ax2.fill_between(z,-EDep_d,EDep_d,label='95% CI',alpha=0.2)
ax2.legend()
plt.show()
