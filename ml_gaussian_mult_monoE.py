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
    return func(centres)

args = sys.argv

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

def getFunc(Ec0):
    Es = (new_ERef[1:] + new_ERef[:-1])/2
    prev = 0
    next = 0
    for E in ERef:
        prev = next
        next = E
        if E > Ec0:
            break
    prevfrac = (1 - ((Ec0 - prev)/(next-prev)))
    nextfrac = (1 - prevfrac)
    prev = 0
    next = 0
    for E in Es:
        prev = next
        next = E
        if E > Ec0:
            break
    def ret(x):
        retval = []
        for a in x:
            if a == prev:
                retval.append(prevfrac)
                continue
            if a == next:
                retval.append(nextfrac)
                continue
            retval.append(0)
        return np.array(retval)
    return ret

def EDepPred(Ec, A):
    return np.sum(getBins(lambda x : A*getFunc(Ec)(x),bins=new_ERef)*d.T,axis=1)

def EDepVar(Ec,A):
    l = getBins(lambda x : A*getFunc(Ec)(x),bins=new_ERef)
    l_err = np.sqrt(l)
    geantE = np.sqrt(np.sum((Err.T*l_err)**2,axis=1))
    EDep = np.sum((l[0])*d.T,axis=1)
    EDep_d = np.sqrt(np.sum((l_err*d.T)**2,axis=1) + geantE**2)
    return EDep_d

def getllfunc(data):
    def log_likelihood(p0):
        Ec0, A0 = p0
        pred = EDepPred(Ec0, A0)
        Var0 = EDepVar(Ec0, A0)**2
        diff2 = (data - pred) ** 2
        ll = -0.5 * np.sum(diff2/ Var0 + np.log(2 * np.pi * Var0))
        return ll
    return log_likelihood

pars = []
for E in range(10,300,20):
    for std in range(0,200,20):
        pars.append((E,std))

Es = []
stds = []
for E in range(10,300,20):
    Es.append(E)
Es.append(E + 20)
Es = np.array(Es)
Es -= 10
for std in range(0,200,20):
    stds.append(std)
stds.append(std + 20)
stds = np.array(stds,dtype=np.float64)
stds -= 10

perc = []
perc2 = []
for i in range(len(datas)):
    str = ""
    measured = np.sum(datas[i],axis=0)
    def func(params):
        return np.sum(getBins(lambda x : params[1]*getFunc(params[0])(x),bins=new_ERef)*d.T,axis=1)
    def diff(params):
        return func(params) - measured
    min,a = leastsq(diff,[10,1e5])
    str += "------------------------------------\n"
    str += f"{i}\t{pars[i][0]}\t{pars[i][1]}\n"
    str += f"Ecrit\t{ECrits[i]}\n"
    str += f"lsq\t{min[0]}\t{min[1]}\n"
    ll = lambda x : -getllfunc(measured)(x)
    max = minimize(ll,[10,1e5], method='L-BFGS-B')
    str += f"ml\t{max['x'][0]}\t{max['x'][1]}\t{get_chi2(EDepPred(*max['x']),measured,EDepVar(*max['x']))}\n"
    perc.append(max['x'][0]/ECrits[i])
    perc2.append(min[0]/ECrits[i])
    str += "------------------------------------\n"
    print(str,end="")

perc = np.reshape(np.array(perc),(len(Es)-1,len(stds)-1))
#diff = np.max([-np.log(np.min(perc)),np.log(np.max(perc)),-np.log(np.min(perc2)),np.log(np.max(perc2))])
diff = 0.2
norm = colors.Normalize(vmin=1-diff,vmax=1+diff)
plt.figure()
plt.pcolormesh(stds, Es, perc, norm=norm, cmap=cmap, shading='flat')
plt.xlabel("Convolution $\sigma$ [MeV]")
plt.ylabel("Dataset Energy [MeV]")
plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap)).set_label('Ratio')
plt.show()

perc2 = np.reshape(np.array(perc2),(len(Es)-1,len(stds)-1))
plt.figure()
plt.pcolormesh(stds, Es, perc2, norm=norm, cmap=cmap, shading='flat')
plt.xlabel("Convolution $\sigma$ [MeV]")
plt.ylabel("Dataset Energy [MeV]")
plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap)).set_label('Ratio')
plt.show()
