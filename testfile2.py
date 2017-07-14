
from __future__ import division
import matplotlib.pyplot as plt

import scipy.io
import numpy

from brian import *
# Neuron model parameters
defaultclock.t = 0*second

print "change 1 with rob"

Vr = -70 * mV
Vt = -55 * mV
taum = 20 * ms
taumI = 10 * ms

def func(x, alpha, beta):
#  return alpha * x[0] * 10.0 - beta * x[1] * 35.0
  return alpha * x[0] * 10.0 - beta * x[1] * 35.0

weight = 5 * mV #good

omegaIEspike = .6*mV 
omegaIIspike = .5*mV

omegaEErange = arange(.31,.33,.01)#*mV
omegaEIrange = arange(.23,.25,.01)#*mV

# Neuron model
sigmaval = 5 * mV #good

eqs = Equations('''dV/dt=(-(V-Vr)  )*(1./taum) + (sigmaval*(1./taum)**0.5)*xi : volt ''')
    
eqsI = Equations('''dV/dt=(-(V-Vr)  )*(1./taumI) + (sigmaval*(1./taumI)**0.5)*xi : volt ''')
    

k = 20
#k=100
# Neuron groups --  29 areas. 
E = NeuronGroup(N=k*4, model=eqs, threshold=Vt, reset=Vr, refractory=2 * ms)
I = NeuronGroup(N=k, model=eqsI, threshold=Vt, reset=Vr, refractory=2 * ms)

Pinput = PulsePacket(t=50 * ms, n=4*k, sigma= .3 * ms)#good

# The network structure
Exc = [ E.subgroup(k*4) for i in range(1)]
Inh = [ I.subgroup(k) for i in range(1)]

Exc_C = Connection(E, E, 'V')
Inh_C = Connection(I, I, 'V')
EtoI_C = Connection(E, I, 'V')
ItoE_C = Connection(I, E, 'V')


for kee in arange(len(omegaEErange)):
#vary omegaEE omegaEI fix omegaIE omegaII check rate write linear function of omegaEE omegaEI 
  for kei in arange(len(omegaEIrange)):   
    defaultclock.t = 0*second
    
    omegaEEspike = omegaEErange[kee]*mV
    omegaEIspike = omegaEIrange[kei]*mV
#first local. 
    Exc_C.connect_full(Exc[0], Exc[0], omegaEEspike)
    ItoE_C.connect_full(Inh[0], Exc[0], -omegaEIspike)
    EtoI_C.connect_full(Exc[0], Inh[0], omegaIEspike)
    Inh_C.connect_full(Inh[0], Inh[0], -omegaIIspike)
    
Cinput = Connection(Pinput, Exc[0], 'V')
Cinput.connect_full(weight=weight)

# Record the spikes
Mgp = [SpikeMonitor(p) for p in Exc]
Minput = SpikeMonitor(Pinput)
monitors = [Minput] + Mgp

MgpI = [SpikeMonitor(pI) for pI in Inh]
monitorsI = MgpI

# Setup the network, and run it
E.V = Vr + rand(len(E)) * (Vt - Vr)
I.V = Vr + rand(len(I)) * (Vt - Vr)

S = [PopulationRateMonitor((p), bin=5*ms) for p in Exc]
SI = [PopulationRateMonitor((pi), bin=5*ms) for pi in Inh]

run(100*ms)



plt.figure()
raster_plot(showgrouplines=True, *monitors)
plt.title('for exc.')
xlim(10,100)

plt.figure()
raster_plot(showgrouplines=True, *monitorsI)
plt.title('for Inh. ')
xlim(10,100)

#plot(S.times/ms, S.rate/Hz)
#show()
"""plt.figure()
f, axarr = subplots(2, sharex=True)
axarr[0].set_title('Change in firing rate (Hz) vs time (s)')
axarr[0].plot(S[0].times/ms,S[0].rate/Hz)
axarr[0].set_ylabel('V1')
axarr[1].plot(S[1].times/ms,S[1].rate/Hz)
axarr[1].set_ylabel('V2')
plt.show()"""

maxrate = numpy.empty([29,1])
meanrate = numpy.empty([29,1])
meanrateI = numpy.empty([29,1])

for k in range(1):
    maxrate[k,0] = max(S[k].rate[len(S[0].rate)/3:])
    meanrate[k,0] = mean(S[k].rate)
    meanrateI[k,0] = mean(SI[k].rate)    
#print (meanrate, meanrateI)
print (maxrate)
"""
plt.figure()
#plt.plot(1+arange(29),log10(maxrate)/log10(maxrate[0,0]))
plt.plot(1+arange(1),log10(maxrate/maxrate[0,0]))
#plt.title('log10 of maxrate')
plt.ylabel('log10 of Attenuation ratio of max. firing rate')
plt.xlabel('Areas')
plt.show()

plt.figure()
plt.plot(1+arange(29),(maxrate))
#plt.title('maxrate')
plt.xlabel('Areas')
plt.ylabel('Max. firing rate')
plt.show()
"""
