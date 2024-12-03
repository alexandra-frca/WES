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
Nevals = int(1e4)
cmin = 0 
cmax = 1e6

sampler = None

def objective_function(x, sampler):
    return np.sin(x)**2

def discrete_optimization(grid, sampler):
    ims = objective_function(grid, sampler)
    ctrl_opt = grid[np.argmin(ims)]
    return ctrl_opt

def calc_unif(silent = True):
    np.round(np.linspace(self.cmin, self.cmax, num = int(Nevals)))
    discrete_optimization(grid, sampler)
    if not silent:
        print("> First batch likelihood obtained by calc_singleshot: ", ls[0])

def calc_rand(silent = True):
    np.round(np.random.uniform(self.cmin, self.cmax, int(Nevals)))
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
fs = [calc_rand, calc_unif]

for f in fs:
    time_fun(f, rtimer)
