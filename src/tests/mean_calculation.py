'''
Testing the speed of different techniques to calculate means.
'''

import numpy as np
import scipy.optimize as opt
from utils.misc import RelativeTimer

Nsamples = 1000
values = np.random.uniform(0, 10, Nsamples)
weights = np.random.uniform(0, 1, Nsamples)
S = np.sum(weights)

def numpy_average(values, weights, silent = False):
    result = np.average(values, weights=weights)
    if not silent:
        print("> Average obtained by numpy_average: ", result)
    
def numpy_mean(values, weights, silent = False):
    result = np.sum((values-0.5)**2*weights)/S
    if not silent:
        print("> Average obtained by numpy_mean: ", result)

first_time_fun = True
def time_fun(f, timer, reps = 1000):
    global first_time_fun
    if first_time_fun:
        print(f"> Will test {reps} runs. [time_fun]")
        first_time_fun = False
        
    interval = (0.1, np.pi/2-0.1)
    evals = 1e3
    timer.new()
    for i in range(reps):
        f(values, weights, silent = False if i==0 else True)
    timer.stop()
    
rtimer = RelativeTimer()
fs = [numpy_average, numpy_mean]

for f in fs:
    time_fun(f, rtimer)


