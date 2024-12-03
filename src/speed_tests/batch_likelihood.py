# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 17:13:40 2023

@author: alexa
"""
import numpy as np
import numexpr as ne
from utils.running import RelativeTimer
from copy import deepcopy
# ne.set_num_cores(4)

wNs = 100
Npart = 1000
Nmeas = int(1e3)
thetas = np.random.uniform(size = Npart)
ctrls = np.random.randint(0, 1000, Nmeas)
outcomes =  np.append(np.random.randint(0,100), np.random.randint(0, 1, Nmeas-1))
Nsshots = np.append(wNs, np.repeat(1, Nmeas-1))

# Assume only warmup has >1 shots. Transform int outcome into arrays of 0s and 1s.
aux = np.zeros(Nsshots[0])
aux[:outcomes[0]] = 1
w = np.random.permutation(aux)

outcomesb = np.concatenate((w, outcomes[1:]))
ctrlsb = np.concatenate((np.repeat(ctrls[0], Nsshots[0]), ctrls[1:]))

def calc_singleshot(silent = True):
    args = np.outer(thetas, 2*ctrlsb+1)
    L1 = np.sin(args)**2
    L = L1**outcomesb*(1-L1)**(1-outcomesb)
    ls = np.prod(L, axis=1)
    if not silent:
        print("> First batch likelihood obtained by calc_singleshot: ", ls[0])

def calc_multishot(silent = True):
    args = np.outer(thetas, 2*ctrls+1)
    L1 = np.sin(args)**2
    L = L1**outcomes*(1-L1)**(Nsshots-outcomes)
    ls = np.prod(L, axis=1)
    if not silent:
        print("> First weight obtained by calc_multishot: ", ls[0])
    
first_time_fun = True
def time_fun(f, timer, reps = 10):
    global first_time_fun
    if first_time_fun:
        print(f"> Will test {reps} runs. [time_fun]")
        first_time_fun = False
        
    timer.new()
    for i in range(reps):
        f(silent = False if i==0 else True)
    timer.stop()
        
rtimer = RelativeTimer()
fs = [calc_singleshot, calc_multishot]

for f in fs:
    time_fun(f, rtimer)
