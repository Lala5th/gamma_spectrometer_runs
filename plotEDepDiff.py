#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

args = sys.argv
cmap = cm.get_cmap('gnuplot')

dirRun = args[1]
dirRef = args[2]

ERef = np.load(dirRef + '/InitialEnergy.npy')['0']
y = np.load(dirRef + '/y.npy')['0']
z = np.load(dirRef + '/z.npy')['0']
ref = np.reshape(np.load(dirRef + '/DepositedEnergy.npy'),(len(ERef),len(y),len(z)))

EDist = np.load(dirRun + '/Energy.npy')['0']
data = np.reshape(np.load(dirRun + '/DepositedEnergy.npy'),(len(y),len(z)))['1']

layers = np.array([np.sum(ref['1'][i,:,:],axis = 0) for i in range(len(ERef))])

plt.ion()
plt.figure()
new_ERef = (ERef[1:] + ERef[:-1])/2
new_ERef = np.append(new_ERef,ERef[-1]+10)
new_ERef = np.insert(new_ERef,0,0)
l = plt.hist(EDist,bins=new_ERef)
l_err = np.sqrt(l[0])
plt.figure()
EDep = np.zeros(len(z))
EDep = np.sum((l[0])*layers.T,axis=1)
EDep_d = np.sum((1.96*l_err)*layers.T,axis=1)/len(ECrits)
EDep_n = EDep - EDep_d
EDep_p = EDep + EDep_d
plt.plot(z,EDep,label='Linear combination')
plt.plot(z,np.sum(data,axis=0),label='Simulated')
plt.fill_between(z,EDep_n,EDep_p,label='95% CI',alpha=0.2)
plt.legend()
plt.ylabel('E [MeV]')
plt.xlabel('Longitudinal crystal ID')
plt.show()
