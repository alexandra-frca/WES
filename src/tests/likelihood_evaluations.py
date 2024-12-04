# -*- coding: utf-8 -*-
'''
Testing the speed of different techniques to evaluate likelihoods.
'''
import numpy as np
import numexpr as ne
from utils.misc import RelativeTimer
from copy import deepcopy
# ne.set_num_cores(4)


def likelihood(a, m, hits, nshots):
    theta = np.arcsin(np.sqrt(a))
    arg = (2*m+1)*theta
    L = np.sin(arg)**(2*hits)*np.cos(arg)**(2*(nshots-hits))
    return L

N = 1000
locs = np.random.uniform(size = N)
weights_og = np.ones(N)/N

def calc_loop(silent = False):
    weights = deepcopy(weights_og)
    for i in range(N):
        weights[i] *= likelihood(locs[i], 1, 20, 50)
    if not silent:
        print("> First weight obtained by calc_loop: ", weights[0])

def calc_comprehension(silent = False):
    weights = deepcopy(weights_og)
    weights = [w*likelihood(locs[i], 1, 20, 50) for i,w in enumerate(weights_og)]
    if not silent:
        print("> First weight obtained by calc_compre: ", weights[0])

def calc_numpy(silent = False):
    weights = deepcopy(weights_og)
    ls = likelihood(locs, 1, 20, 50)
    weights = np.multiply(weights, ls)
    if not silent:
        print("> First weight obtained by calc_numpy: ", weights[0])
    
def calc_numexpr(silent = False):
    weights = deepcopy(weights_og)
    ls = likelihood(locs, 1, 20, 50)
    weights = ne.evaluate("weights * ls")
    if not silent:
        print("> First weight obtained by calc_numexpr: ", weights[0])
    
def calc_numexpr2(silent = False):
    weights = deepcopy(weights_og)
    m = 1
    hits = 20
    nshots = 50
    weights = ne.evaluate("weights * (sin((2*m+1)*arcsin(sqrt(locs)))**(2*hits)*cos((2*m+1)*arcsin(sqrt(locs)))**(2*(nshots-hits)))")
    if not silent:
        print("> First weight obtained by calc_numexpr2: ", weights[0])
    
first_time_fun = True
def time_fun(f, timer, reps = 100):
    global first_time_fun
    if first_time_fun:
        print(f"> Will test {reps} runs. [time_fun]")
        first_time_fun = False
        
    timer.new()
    for i in range(reps):
        f(silent = False if i==0 else True)
    timer.stop()
        
rtimer = RelativeTimer()
fs = [calc_loop, calc_comprehension, calc_numpy, calc_numexpr, calc_numexpr2]

for f in fs:
    time_fun(f, rtimer)
