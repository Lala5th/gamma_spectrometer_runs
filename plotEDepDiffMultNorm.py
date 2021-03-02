
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
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
new_ERef = (ERef[1:] + ERef[:-1])/2
new_ERef = np.append(new_ERef,ERef[-1]+10)
new_ERef = np.insert(new_ERef,0,0)
l = np.histogram(EDist[0],bins=new_ERef)
l_err = np.sqrt(l[0])
EDep = np.zeros(len(z))
EDep_p = np.zeros(len(z))
EDep_n = np.zeros(len(z))

for i, Ed in enumerate(layers):
    EDep += l[0][i]*Ed
    EDep_p += (l[0][i]+1.96*l_err[i])*Ed
    EDep_n += (l[0][i]-1.96*l_err[i])*Ed
plincomb, = plt.plot(z,EDep/np.max(EDep),label='Linear combination')
psim, = plt.plot(z,np.sum(data[0],axis=0)/np.max(np.sum(data[0],axis=0)),label='Simulated')
#coll = plt.fill_between(z,EDep_n,EDep_p,label='95% CI',alpha=0.2)
plt.legend()
plt.ylabel('E$_{norm}$ [a.u.]')
plt.xlabel('Longitudinal crystal ID')
axE = plt.axes([0.25, 0.1, 0.65, 0.03])

sE = Slider(axE, 'ECrit', ECrits[0], ECrits[-1], valinit=ECrits[0], valstep=ECrits[1]-ECrits[0])

def update(val):
    global coll
    Eci = np.where(np.abs(ECrits - sE.val) <1e-6)[0][0]
    l = np.histogram(EDist[Eci],bins=new_ERef)
    l_err = np.sqrt(l[0])
    EDep = np.zeros(len(z))
    EDep_p = np.zeros(len(z))
    EDep_n = np.zeros(len(z))

    for i, Ed in enumerate(layers):
        EDep += l[0][i]*Ed
        EDep_p += (l[0][i]+1.96*l_err[i])*Ed
        EDep_n += (l[0][i]-1.96*l_err[i])*Ed
    psim.set_ydata(np.sum(data[Eci],axis=0)/np.max(np.sum(data[Eci],axis=0)))
    plincomb.set_ydata(EDep/np.max(EDep))
    #ax.collections.clear()
    #coll = ax.fill_between(z,EDep_n,EDep_p,label='95% CI',alpha=0.2)
    fig.canvas.draw_idle()

sE.on_changed(update)
plt.show()
