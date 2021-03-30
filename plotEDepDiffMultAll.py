#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib.widgets import Slider
from scipy.optimize import curve_fit

def get_chi2(E, x, s):
    return np.sum(((E-x)/s)**2)

args = sys.argv
cmap = cm.get_cmap('gnuplot')

dirRun = args[1]
dirRef = args[2]
dirErr = args[3]
id = 0;

ERef = np.load(dirRef + '/InitialEnergy.npy')['0']
y = np.load(dirRef + '/y.npy')['0']
z = np.load(dirRef + '/z.npy')['0']
ref = np.reshape(np.load(dirRef + '/DepositedEnergy.npy'),(len(ERef),len(y),len(z)))['1']
#ref = np.reshape(np.load(dirErr + '/DepositedEnergy.npy'),(len(ERef),len(y),len(z)))['2']

ECrits = np.load(dirRun + '/ECrit.npy')['0']
EDist = np.load(dirRun + '/Energy.npy')['1']
EDist = np.reshape(EDist,(len(ECrits),int(len(EDist)/len(ECrits))))

data = np.reshape(np.load(dirRun + '/DepositedEnergy.npy'),(len(ECrits),len(y),len(z)))['1']

layers = np.array([np.sum(ref[i,:,:],axis = 0) for i in range(len(ERef))])


Err = np.reshape(np.load(dirErr + '/DepositedEnergy.npy'),(len(ERef),len(y),len(z)))['1']

Err = np.sum(Err,axis=1)
#Err[Err < 0] = 0
Err = np.sqrt(Err)
fig = plt.figure()
ax0 = fig.add_subplot(3,1,1,yscale="log")
ax2 = fig.add_subplot(3,1,3)
ax1 = fig.add_subplot(3,1,2,sharex=ax2)
plt.setp(ax1.get_xticklabels(),visible=False)
new_ERef = (ERef[1:] + ERef[:-1])/2
new_ERef = np.append(new_ERef,ERef[-1]+10)
new_ERef = np.insert(new_ERef,0,0)
ERef_diff = np.insert(ERef,0,0)
ERef_diff = (new_ERef[1:] - new_ERef[:-1])
l = np.histogram(EDist[id],bins=new_ERef)
l_err = np.sqrt(l[0])
ax0.plot(ERef,l[0]/ERef_diff)
func = lambda x, A : A*x**(-2/3)*np.exp(-x/ECrits[id])
fit,cov = curve_fit(func,ERef,l[0]/ERef_diff)
#ax0.plot(ERef,func(ERef,*fit))
ax0.set_ylabel("Counts [MeV$^{-1}$]")
ax0.set_xlabel("E [MeV]")
ax0.set_ylim([1e-4,5*max(np.max(func(ERef,*fit)),np.max(l[0]/ERef_diff))])

geantE = np.sqrt(np.sum((Err.T*l_err)**2,axis=1))
EDep = np.sum((l[0])*layers.T,axis=1)
EDep_d = np.sqrt(np.sum((l_err*layers.T)**2,axis=1) + geantE**2)

ax1.plot(z,EDep,label='Linear combination')
ax1.plot(z,np.sum(data[id],axis=0),label='Simulated')
ax1.fill_between(z,EDep - 1.96*EDep_d,EDep + 1.96*EDep_d,label='95% CI',alpha=0.2)
ax1.legend()
ax1.set_ylabel('E [MeV]')
ax2.set_ylabel('E [MeV]')
ax2.set_xlabel('Longitudinal crystal ID')
ax2.plot(z,np.sum(data[id],axis=0) - EDep,label='Residual')
ax2.fill_between(z,-1.96*EDep_d,1.96*EDep_d,label='95% CI',alpha=0.2)
ax2.legend()
plt.subplots_adjust(left=0.2, bottom=0.2,hspace=0.5,top=0.9)
axE = plt.axes([0.25, 0.05, 0.65, 0.03])
sE = Slider(axE,'E$_{init}$ [MeV]',ECrits[0],ECrits[-1],valstep=ECrits[1]-ECrits[0],valinit=ECrits[0])

def print_wrapped(func):
    def wrapper(*args, **kwargs):
        print("----------------------------------------------------")
        func(*args, **kwargs)
        print("----------------------------------------------------")
    return wrapper

@print_wrapped
def update(val):
    global EDep, EDep_d, l, l_err, geantE
    Eid = np.where(np.abs(sE.val-ECrits) < 1e-5)[0][0]
    l = np.histogram(EDist[Eid],bins=new_ERef)
    l_err = np.sqrt(l[0])
    geantE = np.sqrt(np.sum((Err.T*l_err)**2,axis=1))
    EDep = np.sum((l[0])*layers.T,axis=1)
    EDep_d = np.sqrt(np.sum((l_err*layers.T)**2,axis=1) + geantE**2)
    ax0.clear()
    ax0.set_yscale("log")
    ax0.plot(ERef,l[0]/ERef_diff)
    ax0.set_ylabel("Counts [MeV$^{-1}$]")
    ax0.set_xlabel("E [MeV]")
    func = lambda x, A : A*x**(-2/3)*np.exp(-x/ECrits[Eid])
    fit,cov = curve_fit(func,ERef,l[0]/ERef_diff)
    #ax0.plot(ERef,func(ERef,*fit))
    ax0.set_ylim([1e-4,5*max(np.max(func(ERef,*fit)),np.max(l[0]/ERef_diff))])
    ax1.clear()
    ax2.clear()
    ax1.plot(z,EDep,label='Linear combination')
    ax1.plot(z,np.sum(data[Eid],axis=0),label='Simulated')
    ax1.fill_between(z,EDep - 1.96*EDep_d,EDep + 1.96*EDep_d,label='95% CI',alpha=0.2)
    ax1.legend()
    ax1.set_ylabel('E [MeV]')
    ax2.set_ylabel('E [MeV]')
    ax2.set_xlabel('Longitudinal crystal ID')
    ax2.plot(z,np.sum(data[Eid],axis=0) - EDep,label='Residual')
    ax2.fill_between(z,-1.96*EDep_d,1.96*EDep_d,label='95% CI',alpha=0.2)
    ax2.legend()
    print('Energy deposited:', np.sum(data[Eid]), 'MeV')
    print('Expected deposit:', np.sum(EDep), 'MeV +-', 100*np.sum(EDep_d)/np.sum(EDep),'%')
    print('Diff:', 100*(np.sum(data[Eid])-np.sum(EDep))/(np.sum(data[Eid])),'%')
    print('Chi2:', get_chi2(EDep,np.sum(data[Eid],axis=0),EDep_d))
    print('Ndof:', len(z))
    fig.canvas.draw_idle()

sE.on_changed(update)
plt.show()
