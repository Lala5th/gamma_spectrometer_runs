#!/usr/bin/python3

import sys
import numpy as np
from scipy.optimize import minimize, leastsq
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

cmap = colors.ListedColormap(np.array([[0,0,0.2],[0,0,0.3],[0.15,0.15,0.6],[0.3,0.3,0.6],[0.4,0.4,0.7],[0.5,0.5,0.8],[0.9,0.9,1],[0.95,0.95,1],
                                       [1,0.95,0.95],[1,0.9,0.9],[0.8,0.5,0.5],[0.7,0.4,0.4],[0.6,0.3,0.3],[0.6,0.15,0.15],[0.3,0,0],[0.2,0,0]]));

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

pars = []
for EC in np.arange(-0.4,0.4,0.1):
    for std in range(0,50,5):
        pars.append((EC,std))

ECs = []
stds = []
for EC in np.arange(-0.4,0.4,0.1):
    ECs.append(EC)
ECs.append(EC + 0.1)
ECs = np.array(ECs)
ECs -= 0.05
for std in range(0,10,1):
    stds.append(std)
stds.append(std + 1)
stds = np.array(stds,dtype=np.float64)
stds -= 0.5

perc = []
perc2 = []
chi2 = []
for i in range(len(datas)):
    measured = np.sum(datas[i],axis=0)
    def func(params):
        return np.sum(getBins(lambda x : params[0]*x**(-2/3)*np.exp(-x/params[1]),bins=new_ERef)*d.T,axis=1)
    def diff(params):
        return func(params) - measured
    min,a = leastsq(diff,[1e5,250])
    print("------------------------------------")
    print(f"{i}\t{pars[i][0]}\t{pars[i][1]}")
    print("Ecrit\t",ECrits[i])
    print("lsq\t",min[1],"\t",min[0])
    ll = lambda x : -getllfunc(measured)(x)
    max = minimize(ll,[min[1],min[0]], method='L-BFGS-B')
    print("ml\t",max['x'][0],"\t",max['x'][1],"\t",get_chi2(EDepPred(*max['x']),measured,EDepVar(*max['x'])))
    perc.append(max['x'][0]/ECrits[i])
    perc2.append(min[1]/ECrits[i])
    chi2.append(get_chi2(EDepPred(*max['x']),measured,EDepVar(*max['x']))/68)
    print("------------------------------------")

perc = np.reshape(np.array(perc),(len(ECs)-1,len(stds)-1))
chi2 = np.reshape(np.array(chi2),(len(ECs)-1,len(stds)-1))
diff = np.max([1-np.min(perc),np.max(perc)-1,1-np.min(perc2),np.max(perc2)-1])
diff = diff*100
norm = colors.Normalize(vmin=0.8,vmax=1.2)

plt.figure()
plt.pcolormesh(stds, ECs, perc, norm=norm, cmap=cmap, shading='flat')
plt.xlabel("$\sigma_e$ [MeV]")
plt.ylabel("$\delta_e$ [1]")
plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap)).set_label('$\mu_{fit}$/$\mu_{E}$')
plt.show()

perc2 = np.reshape(np.array(perc2),(len(ECs)-1,len(stds)-1))
plt.figure()
plt.pcolormesh(stds, ECs, perc2, norm=norm, cmap=cmap, shading='flat')
plt.xlabel("$\sigma_e$ [MeV]")
plt.ylabel("$\delta_e$ [1]")
plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap)).set_label('$\mu_{fit}$/$\mu_{E}$')
plt.show()

norm = colors.Normalize(vmin=np.min(chi2),vmax=np.max(chi2))
gnucmap = cm.get_cmap('gnuplot')
plt.figure()
plt.pcolormesh(stds, ECs, chi2, cmap=gnucmap,norm=norm, shading='flat')
plt.xlabel("$\sigma_e$ [MeV]")
plt.ylabel("$\delta_e$ [1]")
plt.colorbar(cm.ScalarMappable(cmap=gnucmap,norm=norm)).set_label('$\chi_\\nu^2$')
plt.show()
