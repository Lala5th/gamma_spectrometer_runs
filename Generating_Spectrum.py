import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

### Formulas ####
def rate_photon(E, Ec, A = 1):
    """
    get dN/dE, rate of number of photon 
    generated in response to energy
    
    variables:
    E = Energy, Ec = critical energy, 
    A = amplitude (doesn't matter as will be normalised)
    """    
    dNdE = np.zeros(len(E))
    for i in range(len(E)):
        if E[i] != 0:
            dNdE[i] = A *(E[i]**(-2/3)) * np.exp(-E[i]/Ec)
        else:
            dNdE[i] = 0
    return dNdE

def Hat(x, xmin, xmax, ymax, x_ymax):
    ratios = [10,2]
    gradient1 = ymax/(x_ymax - xmin) /ratios[0]
    gradient2 = ymax/(x_ymax - xmax) /ratios[1]
    
    if type(x) == np.float64:
        if x<x_ymax:
            # results[i] = (x[i] - xmin) * gradient1
            # results[i] = results[i] + (ratios[0] - 1) * ymax/ratios[0]
            results = ymax
        elif x>x_ymax:
            results = (x - xmax) * gradient2
            results = results + (float(ratios[1]) - 1) * float(ymax/ratios[1])
        else:
            results = ymax
    else:
        results = np.zeros(len(x))
        for i in range(len(x)):

            if x[i]<x_ymax:
                # results[i] = (x[i] - xmin) * gradient1
                # results[i] = results[i] + (ratios[0] - 1) * ymax/ratios[0]
                results[i] = ymax
            elif x[i]>x_ymax:
                results[i] = (x[i] - xmax) * gradient2
                results[i] = results[i] + (ratios[1] - 1) * ymax/ratios[1]
            else:
                results[i] = ymax
    
    return results


class Sampler:
    def __init__(self, sample_size, Ec, xmax, xmin = 0, steps = 20000):
        self.sample_size = sample_size
        self.Ec = Ec
        self.xmax = xmax
        self.xmin = xmin
        self.steps = steps      # doesn't matter as long as it is large enough >5e3
        self.samples = np.zeros(self.sample_size)
        self.energy_list = np.linspace(xmin, xmax, steps) #list of energy up to 1000 Mev
        self.dNdE_list = rate_photon(self.energy_list, Ec = self.Ec)
        self.number_photon = self.dNdE_list * self.energy_list
        self.normalised_number_photon = self.number_photon/ self.number_photon.sum()/ ((xmax - xmin)/steps)       
        self.rejection_called = False       
        
        #### Function for the normalised spectrum ####
        self.func_norm_spec = interp1d(self.energy_list, self.normalised_number_photon)
        
        
    def rejection(self, run_times = 20):
        self.rejection_called = True
        self.rejection_finished = False
        self.samples = np.zeros(int(self.sample_size))
        random_array = np.random.rand(run_times*self.sample_size, 2)
        sampled_size = 0 # Number of successful sampling
        self.C = self.normalised_number_photon.max() 
        self.Eprob = self.Ec/3      # Most probable energy around Ec/3 as in paper
        print(len(random_array))
        for j in range(len(random_array)):        
            
            if sampled_size == self.sample_size:
                self.rejection_finished = True
                break
            # a = Hat(random_array[j,0] * (self.xmax - self.xmin), self.xmin, self.xmax, self.normalised_number_photon.max(), self.Ec/3)
            a = self.C
            b = self.func_norm_spec((self.xmax -self.xmin)*random_array[j,0])
            # print (j,"a-b", a-b,self.C)
            # print("compare_value", random_array[j,1] * self.C)
            # print("value added", random_array[j,0] * (self.xmax - self.xmin))
            
            if abs(a - b) < random_array[j,1] * self.C:
                self.samples[int(sampled_size)] = random_array[j,0] * (self.xmax - self.xmin)
                sampled_size += 1
            else:
                pass
            
        if self.rejection_finished == False:
            print("something went wrong, maybe change the hat function or \
                  adjust ratios coz running is not finished")
            raise Exception 
        print("Rejection accuracy:", sampled_size/j)
        return self.samples
    
    def plot_rej(self):
        if self.rejection_called == True:
            plt.figure()
            plt.plot(self.energy_list, self.normalised_number_photon, 'x')
            plt.plot(self.energy_list, self.func_norm_spec(self.energy_list))
            # plt.plot(self.Ec/3, self.normalised_number_photon.max(), 'x')
            plt.plot(self.energy_list, Hat(self.energy_list, self.xmin, self.xmax, self.normalised_number_photon.max(), self.Ec/3))
            print(self.Ec/3),self.C
            plt.show()
    
    def set_sample_size(self, sizes):
        self.sample_size = sizes
        
        
## Example
xmin = 0
xmax = 1000
energy_critical = 150
sample_size = 10000
sampler = Sampler(sample_size, energy_critical,xmax, xmin) 
listla = sampler.rejection(20)
sampler.plot_rej()
plt.figure()
# plt.hist(listla)
plt.xlabel('Energy/MeV')
plt.ylabel('Number of photons')
to_plot = np.histogram(listla, bins = 200)
plt.plot(to_plot[1][1:], to_plot[0])
np.save('Generated_Spectrum', listla, allow_pickle=True, fix_imports=True)

I1 = 0
for i in range(len(listla)):
    if listla[i] < 1:
        I1 += 1
print(I1)