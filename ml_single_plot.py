#!/usr/bin/python3

import sys
import numpy as np
from scipy.optimize import minimize, leastsq
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

cmap = cm.get_cmap('bwr')

plt.ion()

def get_chi2(E, x, s):
    return np.sum(((E-x)/s)**2)

def getBins(func,bins):
    centres = (bins[1:] + bins[:-1])/2
    return func(centres)*(bins[1:] - bins[:-1])

def log_prior(theta,Ec0):
    p00, p01, p02 = theta
    if Ec0*0.8 < p00 < Ec0*1.2 and 1 < p01 < 1e5 and 9e-10 < p02 < 4e-8:
        return 0.0
    return -np.inf

args = sys.argv

id = int(args[5])
dir = args[2]
dirErr = args[4]
y = np.load(dir + '/y.npy')['0']
z = np.load(dir + '/z.npy')['0']
raw_data = np.load(dir + '/DepositedEnergy.npy')['1']
data = raw_data.reshape([-1,10,70])
#Energy = np.load(dir + '/Energy.npy')['0']

Err = np.reshape(np.load(dirErr + '/DepositedEnergy.npy'),(-1,len(y),len(z)))['1']

Err = np.sum(Err,axis=1)
#Err[Err < 0] = 0
Err = np.sqrt(Err)

datas = []
for d in data:
    datas.append(d)

# Energy reference data
dir = args[1]
y = np.load(dir + '/y.npy')['0']
z = np.load(dir + '/z.npy')['0']
ERef = np.load(dir + '/InitialEnergy.npy')['0']
new_ERef = (ERef[1:] + ERef[:-1])/2
new_ERef = np.append(new_ERef,ERef[-1]+10)
new_ERef = np.insert(new_ERef,0,0)
#new_ERef = ERef
data = np.reshape(np.load(dir + '/DepositedEnergy.npy'),(-1,len(y),len(z)))['1']

d = np.sum(data,axis=1)

ECrits = np.load(args[3] + '/ECrit.npy')['0']

def EDepPred(Ec, A):
    return np.sum(getBins(lambda x : A*x**(-2/3)*np.exp(-x/Ec),bins=new_ERef)*d.T,axis=1)

def EDepVar(Ec,A):
    l = getBins(lambda x : A*x**(-2/3)*np.exp(-x/Ec),bins=new_ERef)
    l_err = np.sqrt(l)
    geantE = np.sqrt(np.sum((Err.T*l_err)**2,axis=1))
    EDep = np.sum((l[0])*d.T,axis=1)
    EDep_d = np.sqrt(np.sum((l_err*d.T)**2,axis=1) + geantE**2)
    return EDep_d

def getllfunc(data):
    def log_likelihood(p0):
        #lp = log_prior(p0,E0)
        #if not np.isfinite(lp):
        #    return -np.inf
        Ec0, A0 = p0
        pred = EDepPred(Ec0, A0)
        Var0 = EDepVar(Ec0, A0)**2
        diff2 = (data - pred) ** 2
        ll = -0.5 * np.sum(diff2/ Var0 + np.log(2 * np.pi * Var0))
        return ll
    return log_likelihood

measured = np.sum(datas[id],axis=0)
def func(params):
    return np.sum(getBins(lambda x : params[0]*x**(-2/3)*np.exp(-x/params[1]),bins=new_ERef)*d.T,axis=1)
def diff(params):
    return func(params) - measured
min,a = leastsq(diff,[1e6,50])
print("------------------------------------")
print("Ecrit\t",ECrits[id])
print("lsq\t",min[1],"\t",min[0])
ll = lambda x : -getllfunc(measured)(x)
max = minimize(ll,[min[1],min[0]], method='L-BFGS-B')
print("ml\t",max['x'][0],"\t",max['x'][1],"\t",get_chi2(EDepPred(*max['x']),measured,EDepVar(*max['x'])))
print("------------------------------------")

fig, (ax1, ax2) = plt.subplots(2,1,sharex=True)
ax1.plot(z,EDepPred(*max['x']))
ax1.plot(z,measured)
ax1.fill_between(z,EDepPred(*max['x'])-1.96*EDepVar(*max['x']),EDepPred(*max['x'])+1.96*EDepVar(*max['x']),alpha=0.2)
ax2.plot(z, EDepPred(*max['x']) - measured)
ax2.fill_between(z,-1.96*EDepVar(*max['x']),1.96*EDepVar(*max['x']),alpha=0.2)
plt.show()
