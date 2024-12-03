# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 11:31:40 2024

@author: alexa
"""
from utils.misc import logsumexp

w = 1e-170
lw = np.log(w)

w1 = np.power(np.exp(lw), 2)
aux = np.exp(lw)
w2 = aux**2

ws = np.repeat(1e-200, 3)
ls = np.repeat(1e-150, 3)
lws = np.log(ws)
lls = np.log(ls)

def test1():
    print("ls*ws = ", ls*ws)
    print("lws + lls =", lws + lls)
    print("e^(lws+lls) =", np.exp(lws + lls))
    # print(np.log(ws))

def test2():
    nlws = np.exp(lws - logsumexp(ws))
    print("nlws", nlws)
    print("ws*ls", np.sum(np.exp(lws + lls)))

def ESS(ws, lse = True):
    # ws assumed to be log-weights.
    if lse:
        # Logarithm of the squared sum.
        lsqs = 2*logsumexp(ws)
        # Logarithm of the sum of squares. 
        lssq = logsumexp(ws*2)
        ESS = np.exp(lsqs - lssq)
    else:
        ws = np.exp(ws)
        wsum = np.sum(ws)
        wsum2 = np.sum(ws**2)
        ESS = wsum**2/wsum2
    return ESS

def normalize(ws, lse = True):
    if lse:
        lnorm = logsumexp(ws)
        nws = np.exp(ws-lnorm)
        norm = np.exp(lnorm)
    else:
        tws = np.exp(ws)
        norm = np.sum(tws)
        nws = tws/norm
    return nws, norm

def test_norm():
    ws0 = np.log([1, 1, 1])
    ws1 = np.log([4.1, 0.750, 1.1])
    ws2 = np.log([1e-2025, 2e-2725, 7e2825])
    ws2 = np.log([1e-200, 2e-250, 7e-250])
    ws3 = np.log([1e-200, 2e-150, 7e-250])
    
    for i, ws in enumerate([ws0, ws1, ws2, ws3]):
        for lse in [False, True]:
            print(f"{i}. Normalized" + (" (lse):" if lse else ":      "), end=" ")
            nws, norm = normalize(ws, lse = lse)
            print(nws, norm)

def test_ESS():
    ws0 = np.log([1, 1, 1])
    ws1 = np.log([4.1, 0.750, 1.1])
    ws2 = np.log([1e-200, 2e-250, 7e-250])
    ws3 = np.log([1e-200, 2e-150, 7e-250])
    
    for ws in [ws0, ws1, ws2, ws3]:
        for lse in [False, True]:
            print("ESS" + (" (lse):" if lse else ":      "), end=" ")
            r = ESS(ws, lse = lse)
            print(r)
    
# test_ESS()
test_norm()
    
    
    