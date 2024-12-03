# -*- coding: utf-8 -*-
"""
Created on Fri May 12 14:31:48 2023

@author: alexa
"""

import numpy as np
import scipy.optimize as opt
from utils.misc import RelativeTimer

def objective_function(theta):
    return -np.cos(theta)*theta

def scipy_brute(objective_function, interval, evals, silent = False):
    result = opt.brute(objective_function, [interval], Ns=evals, finish = None)
    if not silent:
        print("> Minimum obtained by scipy_brute: ", result)

def numpy_brute(objective_function, interval, evals, silent = False):
    args = np.linspace(*interval, num = int(evals))
    ims = objective_function(args)
    result = args[np.argmin(ims)]
    if not silent:
        print("> Minimum obtained by numpy_brute: ", result)

def python_brute(objective_function, interval, evals, silent = False):
    args = np.linspace(*interval, num = int(evals))
    ims = objective_function(args)
    result = args[min(range(len(ims)), key=ims.__getitem__)]
    if not silent:
        print("> Minimum obtained by python_brute: ", result)

first_time_fun = True
def time_fun(f, timer, reps = 100):
    global first_time_fun
    if first_time_fun:
        print(f"> Will test {reps} runs. [time_fun]")
        first_time_fun = False
        
    interval = (0.1, np.pi/2-0.1)
    evals = 1e3
    timer.new()
    for i in range(reps):
        f(objective_function, interval, evals, silent = False if i==0 else True)
    timer.stop()
    
rtimer = RelativeTimer()
fs = [scipy_brute, numpy_brute, python_brute]

for f in fs:
    time_fun(f, rtimer)


