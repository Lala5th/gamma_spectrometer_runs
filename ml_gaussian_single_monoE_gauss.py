#!/usr/bin/python3

import sys
import numpy as np
from scipy.optimize import minimize, leastsq
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from scipy.stats import norm

gaussian = lambda x, A, mu, sig : A*norm.pdf((x-mu)/sig)

#cmap = cm.get_cmap('bwr')
cmap = colors.ListedColormap(np.array([[0,0,0.2],[0,0,0.3],[0.15,0.15,0.6],[0.3,0.3,0.6],[0.4,0.4,0.7],[0.5,0.5,0.8],[0.9,0.9,1],[0.95,0.95,1],
                                       [1,0.95,0.95],[1,0.9,0.9],[0.8,0.5,0.5],[0.7,0.4,0.4],[0.6,0.3,0.3],[0.6,0.15,0.15],[0.3,0,0],[0.2,0,0]]));

plt.ion()

def get_chi2(E, x, s):
    return np.sum(((E-x)/s)**2)

def getBins(func,bins):
    centres = (bins[1:] + bins[:-1])/2
    return func(centres)*(bins[1:] - bins[:-1])

args = sys.argv

dir = args[2]
dirErr = args[4]
id = int(args[5])
y = np.load(dir + '/y.npy')['0']
z = np.load(dir + '/z.npy')['0']
raw_data = np.load(dir + '/DepositedEnergy.npy')['1']
data = raw_data.reshape([-1,10,70])

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
Energy = np.reshape(np.load(args[3] + '/Energy.npy')['1'],(len(ECrits),-1))

def getFunc(Ec0):
    Es = (new_ERef[1:] + new_ERef[:-1])/2
    prev = 0
    next = 0
    for E in Es:
        prev = next
        next = E
        if E > Ec0:
            break
    prevfrac = 1 - ((Ec0 - prev)/(next-prev))
    nextfrac = 1 - prevfrac
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

def EDepPred(Ec, A, Sig):
    return np.sum(getBins(lambda x : gaussian(x,A,Ec,Sig),bins=new_ERef)*d.T,axis=1)

def EDepVar(Ec,A,Sig):
    l = getBins(lambda x : gaussian(x,A,Ec,Sig),bins=new_ERef)
    l_err = np.sqrt(l)
    geantE = np.sqrt(np.sum((Err.T*l_err)**2,axis=1))
    EDep = np.sum((l[0])*d.T,axis=1)
    EDep_d = np.sqrt(np.sum((l_err*d.T)**2,axis=1) + geantE**2)
    return EDep_d

def getllfunc(data):
    def log_likelihood(p0):
        Ec0, A0, Sig0 = p0
        pred = EDepPred(Ec0, A0, Sig0)
        Var0 = EDepVar(Ec0, A0, Sig0)**2
        diff2 = (data - pred) ** 2
        ll = -0.5 * np.sum(diff2/ Var0 + np.log(2 * np.pi * Var0))
        return ll
    return log_likelihood

pars = []
for E in np.arange(1,10,0.2):
    for std in np.arange(0,1,0.1):
        pars.append((E,std))

Es = []
stds = []
for E in np.arange(1,10,0.2):
    Es.append(E)
Es.append(E + 0.2)
Es = np.array(Es)
Es -= 0.1
for std in np.arange(0,1,0.1):
    stds.append(std)
stds.append(std + 0.1)
stds = np.array(stds,dtype=np.float64)
stds -= 0.05

i = id
str = ""
measured = np.sum(datas[i],axis=0)
def func(params):
    return np.sum(getBins(lambda x : params[1]*getFunc(params[0])(x),bins=new_ERef)*d.T,axis=1)
def diff(params):
    return func(params) - measured
min,a = leastsq(diff,[5,1e5,0.1])
str += "------------------------------------\n"
str += f"{i}\t{pars[i][0]}\t{pars[i][1]}\n"
str += f"Ecrit\t{ECrits[i]}\n"
str += f"lsq\t{min[0]}\t{min[1]}\t{min[2]}\n"
ll = lambda x : -getllfunc(measured)(x)
max = minimize(ll,[5,1e5,0.1], method ='L-BFGS-B')
str += f"ml\t{max['x'][0]}\t{max['x'][1]}\t{max['x'][2]}\t{get_chi2(EDepPred(*max['x']),measured,EDepVar(*max['x']))}\n"
str += "------------------------------------\n"
print(str,end="")

plt.figure()
l = plt.hist(Energy[i],bins=50)
plt.plot((l[1][1:] + l[1][:-1])/2, getBins(lambda x : gaussian(x,max['x'][1],max['x'][0],max['x'][2]),bins=l[1]))
plt.show()
