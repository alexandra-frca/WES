# -*- coding: utf-8 -*-
"""
Created on Tue May 16 18:11:26 2023

@author: alexa
"""
import numpy as np
import random
from scipy.stats import truncnorm
from utils.misc import RelativeTimer

def mean_and_sd(locs, weights):
    wsum = np.sum(weights)
    mean = np.sum(locs*weights)/wsum
    var = np.sum((locs-mean)**2*weights)/wsum
    sd = var**0.5
    return mean, sd

Nsamples = 1000
locs = np.random.uniform(0, 1, Nsamples)
weights = np.random.uniform(0, 10, Nsamples)
wsum = np.sum(weights)
mean, sd = mean_and_sd(locs, weights)

def get_truncated_normal(mean, sd, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def resample1(locs, weights,  mean, sd, silent = False):
    if not silent:
        print(f"> Old mean: {mean}. Old sd: {sd}. [resample1]")
    new_locs = []
    for i in range(Nsamples):
        old_loc = random.choices(locs, weights=weights)[0]
        new_loc = LW_filter1(old_loc, mean, sd)
        new_locs.append(new_loc)
    
    if not silent:
        new_weights = np.ones(Nsamples)/Nsamples
        mean, sd = mean_and_sd(new_locs, new_weights)
        print(f"> New mean: {mean}. New sd: {sd}. [resample1]")
        
def LW_filter1(old_loc, currmean, currsd):
    a = 0.98
    mean = a*old_loc+(1-a)*currmean
    h = np.sqrt(1-a**2)
    sd = h*currsd
    
    new_loc = -1
    while new_loc<0 or new_loc>1:
        new_loc = np.random.normal(mean,scale=sd)
    return new_loc

def resample2(locs, weights,  mean, sd, silent = False):
    '''
    Changes only the LW filter.
    '''
    if not silent:
        print(f"> Old mean: {mean}. Old sd: {sd}. [resample2]")
    new_locs = []
    for i in range(Nsamples):
        old_loc = random.choices(locs, weights=weights)[0]
        new_loc = LW_filter1(old_loc, mean, sd)
        new_locs.append(new_loc)
        
    if not silent:
        new_weights = np.ones(Nsamples)/Nsamples
        mean, sd = mean_and_sd(new_locs, new_weights)
        print(f"> New mean: {mean}. New sd: {sd}. [resample2]")

def LW_filter2(old_loc, currmean, currsd):
    a = 0.98
    mean = a*old_loc+(1-a)*currmean
    h = np.sqrt(1-a**2)
    sd = h*currsd
    
    Tnorm = get_truncated_normal(mean, sd, low=0, upp=1)
    new_loc = Tnorm.rvs()
    return new_loc

rng = np.random.default_rng()

def resample3(locs, weights, mean, sd, silent = False):
    '''
    Changes only the LW filter.
    '''
    if not silent:
        print(f"> Old mean: {mean}. Old sd: {sd}. [resample3]")
    new_locs = rng.choice(locs, size=Nsamples, p=weights/wsum)
    # new_locs = np.array(random.choices(locs, weights=weights, k = Nsamples))
    new_locs = LW_filter3(new_locs, mean, sd)

    if not silent:
        new_weights = np.ones(Nsamples)/Nsamples
        mean, sd = mean_and_sd(new_locs, new_weights)
        print(f"> New mean: {mean}. New sd: {sd}. [resample3]")

def LW_filter3(old_locs, currmean, currsd):
    a = 0.98
    means = a*old_locs+(1-a)*currmean
    h = (1-a**2)**0.5
    sd = h*currsd
    
    Tnorm = get_truncated_normal(means, sd, low=0, upp=1)
    new_locs = Tnorm.rvs()
    
    return new_locs


first_time_fun = True
def time_fun(f, timer, reps = 100):
    global first_time_fun
    if first_time_fun:
        print(f"> Will test {reps} runs. [time_fun]")
        first_time_fun = False

    timer.new()
    for i in range(reps):
        f(locs, weights, mean, sd, silent = False if i==0 else True)
    timer.stop()
    
rtimer = RelativeTimer()
fs = [resample1, resample2, resample3]

for f in fs:
    time_fun(f, rtimer)

