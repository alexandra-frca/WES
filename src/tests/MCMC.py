'''
Testing the speed of different resamplers: Liu-West vs. Metropolis.
'''

import numpy as np
import random
from scipy.stats import truncnorm
from utils.running import RelativeTimer

def mean_and_sd(locs, weights):
    wsum = np.sum(weights)
    mean = np.sum(locs*weights)/wsum
    var = np.sum((locs-mean)**2*weights)/wsum
    sd = var**0.5
    return mean, sd

Nsamples = 5000
locs = np.random.uniform(0, 1, Nsamples)
weights = np.random.uniform(0, 10, Nsamples)
wsum = np.sum(weights)
mean, sd = mean_and_sd(locs, weights)

def likelihood(a, m, hits, Nshots, log = False):
    
    theta = np.arcsin(np.sqrt(a))
    arg = (2*m+1)*theta
    L = np.sin(arg)**(2*hits)*np.cos(arg)**(2*(Nshots-hits))
    if log:
        L = np.log(L)
    
    # Attribute likelihood zero to amplitudes outside of [0,1].
    L = np.where(np.logical_or(a<0, a>1), 0, L)
    return L

def get_truncated_normal(mean, sd, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

rng = np.random.default_rng()

def resample_LW(locs, weights, mean, sd, silent = False):
    '''
    Changes only the LW filter.
    '''
    if not silent:
        print(f"> Old mean: {mean}. Old sd: {sd}. [resample_LW]")
    new_locs = rng.choice(locs, size=Nsamples, p=weights/wsum)
    # new_locs = np.array(random.choices(locs, weights=weights, k = Nsamples))
    new_locs = LW_filter(new_locs, mean, sd)

    if not silent:
        new_weights = np.ones(Nsamples)/Nsamples
        mean, sd = mean_and_sd(new_locs, new_weights)
        print(f"> New mean: {mean}. New sd: {sd}. [resample_LW]")

def LW_filter(old_locs, currmean, currsd):
    a = 0.98
    means = a*old_locs+(1-a)*currmean
    h = (1-a**2)**0.5
    sd = h*currsd
    
    Tnorm = get_truncated_normal(means, sd, low=0, upp=1)
    new_locs = Tnorm.rvs()
    
    return new_locs

def resample_MCMC_log(locs, weights, mean, sd, silent = False):
    return resample_MCMC(locs, weights, mean, sd, silent, log = True)

def resample_MCMC(locs, weights, mean, sd, silent = False, log = False):
    '''
    Changes only the LW filter.
    '''
    if not silent:
        print(f"> Old mean: {mean}. Old sd: {sd}. [resample_MCMC]")
    new_locs = rng.choice(locs, size=Nsamples, p=weights/wsum)
    # new_locs = np.array(random.choices(locs, weights=weights, k = Nsamples))

    new_locs = MCMC_kernel(new_locs, sd, log)
    print("test________")
    new_locs = MCMC_kernel(new_locs, sd, True)
    new_locs = MCMC_kernel(new_locs, sd, True)
    new_locs = MCMC_kernel(new_locs, sd, False)
    print("________")

    if not silent:
        new_weights = np.ones(Nsamples)/Nsamples
        mean, sd = mean_and_sd(new_locs, new_weights)
        print(f"> New mean: {mean}. New sd: {sd}. [resample_MCMC]")

def MCMC_kernel(old_locs, currsd, log, alpha = 0.1):
    if log:
        return MCMC_kernel_log(old_locs, currsd, alpha)
    
    old_likelihoods = likelihood(old_locs, 1, 20, 50)
    
    print("1", currsd)
    sd = alpha*currsd
    # Could propose a<0! Modify likelihood?
    proposals = np.random.normal(old_locs, scale = sd)
    new_likelihoods = likelihood(proposals, 1, 20, 50)
    acc_probs = new_likelihoods/old_likelihoods
    # Cover for NaNs induced by division by zero, and cap >1 "probabilities" at
    # 1. Note that 'np.where' only works as intended for numpy array arguments.
    acc_probs = np.where(np.logical_or(old_likelihoods==0, acc_probs>1), 
                                1, acc_probs)
    accept = np.random.binomial(1, acc_probs)
    print("> Acceptance rate: ", np.sum(accept)/len(accept))
    new_locs = np.where(accept, proposals, old_locs)
    return new_locs

def MCMC_kernel_log(old_locs, currsd, alpha = 0.1):
    
    
    old_loglikelihoods = likelihood(old_locs, 1, 20, 50, log = True)
    sd = alpha*currsd
    proposals = np.random.normal(old_locs, scale = sd)
    new_loglikelihoods = likelihood(proposals, 1, 20, 50, log = True)
    acc_diffs = new_loglikelihoods - old_loglikelihoods
    # Cover for NaNs induced by division by zero, and cap >1 "probabilities" at
    # 1. Note that 'np.where' only works as intended for numpy array arguments.
    #acc_probs = np.where(np.logical_or(old_likelihoods==0, acc_probs>1), 
    #                            1, acc_probs)

    log_u = np.log(np.random.uniform(0, 1, size = len(acc_diffs)))
    accept = log_u < acc_diffs
    print("> Acceptance rate (log): ", np.sum(accept)/len(accept))
    new_locs = np.where(accept, proposals, old_locs)
    return new_locs


first_time_fun = True
def time_fun(f, timer, reps = 10):
    global first_time_fun
    if first_time_fun:
        print(f"> Will test {reps} runs. [time_fun]")
        first_time_fun = False

    timer.new()
    for i in range(reps):
        f(locs, weights, mean, sd, silent = False if i==0 else True)
    timer.stop()
    
rtimer = RelativeTimer()
fs = [resample_LW, resample_MCMC]#, resample_MCMC_log]

for f in fs:
    time_fun(f, rtimer)

