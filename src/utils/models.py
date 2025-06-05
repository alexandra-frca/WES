'''
Classes for likelihood models.
'''
import math
from abc import ABC, abstractmethod
import numpy as np
import scipy.optimize as opt
from src.utils.plotting import plot_graph
from src.utils.mydataclasses import MeasurementData
from src.utils.misc import np_invalid_catch, print_1st_err

class LikelihoodModel(ABC):
 
    @abstractmethod
    def measure(self):
        pass
    
    @abstractmethod
    def likelihood(self):
        pass
    
    def create_data(self, ctrls, Nsshots):
        outcomes  = [self.measure(ctrl, nshots) for ctrl, nshots 
                     in zip(ctrls, Nsshots)]
        data = MeasurementData()
        data.append_data(ctrls, outcomes, Nsshots)
        return data
    
    def batch_likelihood(self, w: float, data: MeasurementData):
        Ls = []
        for ctrl, outcome, nshots in zip(data.ctrls, data.outcomes,
                                         data.Nsshots):
            Li = self.likelihood(w, ctrl, outcome, nshots)
            Ls.append(Li)
        L = np.product(Ls)
        return L
    
    def batch_loglikelihood(self, param: float, 
                                data: MeasurementData) -> float:
        ls = []
        for i in range(len(data.ctrls)):
            li = self.loglikelihood(param, data.ctrls[i], data.outcomes[i])
            ls.append(li)
        l = np.sum(ls)
        return l
    
    def batch_likelihoods(self, param_list: [float], data, log = False, var = "a"):
        ys = ([self.batch_loglikelihood(x, data)  for x in param_list] if log 
              else [self.batch_likelihood(x, data, var = var)  for x in param_list])
        return ys
    
    def mle(self, data, var = "a", Nevals = 1e4):
        def objective_function(param):
            return -self.batch_likelihood(param, data, var = var)
        
        param_opt = opt.brute(objective_function, [(0,1)], Ns = Nevals)[0]
        self.print_values(param_opt, var, "MLE: ")
        
    def mean_estimator(self, data, var = "a", Nevals = 1e4):
        def function(param):
            return -self.batch_likelihood(param, data, var = var)
        
        rng = self.get_range(var)
        xs = np.random.uniform(*rng, int(Nevals))
        fxs = function(xs)
        avg = np.average(xs, weights = fxs)
        self.print_values(avg, var, "Mean estimator: ")
    
    def plot_likelihood(self, data: MeasurementData, var = "a", 
                        ttl_xtra = "", log = False, atol = 1e-3):
        def is_significant(l, lmax, log):
            if log:
                return l-lmax > np.log(atol)
            else:
                return abs(l/lmax) > atol
        
        # Start with a full coverage grid.
        rng = self.get_range(var)
        xs = np.linspace(rng[0], rng[1] ,10000)
        ys = self.batch_likelihood(xs, data, var, log, info = "plot") 
            
        if all([math.isclose(y,0, abs_tol=1e-300) for y in ys]):
            print("> All the y values you want me to plot are roughly zero. "
                  + ("" if log else "Maybe try your luck with a log scale?")
                  + "[LikelihoodModel.plot_likelihood]") 
            return
        
        # Remove negligible likelihoods (considering scale) and "zoom in".
        ymax = max(ys)
        xs = [xs[k] for k in range(len(xs)) if is_significant(ys[k], ymax,log)]
        xs = np.linspace(min(xs), max(xs) ,10000)
        ys = self.batch_likelihood(xs, data, log = log, var = var) 
        ys = [y[0] for y in ys]
        
        s = ("Logl" if log else "L")
        title = s + f"ikelihood - {len(data.ctrls)} data" + ttl_xtra
        xlabel = f"Parameter ({var})" 
        ylabel = s + "ikelihood"
        
        plot_graph(xs, ys, startat0 = None if log else "y",
                        title=title, xlabel=xlabel, ylabel=ylabel)

    
first_PrecessionModel = True
first_set_Tc_est = True
class PrecessionModel(LikelihoodModel):
    def __init__(self, w: float, wmax, Tc = None, Tcrange = None):
        self.w = w
        self.wmax = wmax
        self.Tc = Tc
        self.Tc_est = None
        # For the likelihood boundaries.
        self.Tcrange = Tcrange
        global first_PrecessionModel
        if self.Tc is not None and first_PrecessionModel:
            print(f"> Using discrete coherence time Tc = {self.Tc}."
                  " [PrecessionModel.measure]")
            first_PrecessionModel = False
            
    def get_range(self, var):
        '''
        Upper bound; lower to be assumed zero except for Tc, (tuple).
        '''
        assert var in ["w", "Tc"]
        if var=="w":
            return (0, self.wmax)
        if var=="Tc":
            return self.Tcrange
    
    def value_str(self, value, var):
        return f"{var} = {value}"
    
    def set_Tc_est(self, Tc_est):
        '''
        Define constant estimate of Tc to be used in the likelihood 
        calculations.
        '''
        self.Tc_est = Tc_est
        if first_set_Tc_est:
            print(f"> Set Tc_est = {Tc_est:.0f}. This will be used for the "
                  "likelihood calculations.") 
        
    def measure(self, t: float, nshots, var = "w", prt = False) -> int:
        assert var in ["w", "Tc"]
        if var == "Tc":
            p1 = (1+np.exp(-t/self.Tc))/2
        else:
            arg = t*self.w/2
            p1 = np.sin(arg)**2
            if self.Tc is not None:
                p1 = self.damp_fun(t, p1, "real")
                
        if prt:
            print(f"> p1 = {p1}. [PrecessionModel.measure]")
        hits = np.random.binomial(nshots, p1)
        return hits
    
    def damp_fun(self, ctrls, fun, whichTc):
        '''
        Exponentially damp function evaluations according to given Tc 
        (estimated or real).
        
        The exponents differ only according to the control. Each exp is 
        associated to a control, and hence a column.
        
        'exps' will be the list of exponentials for each control.
        
        Then we can use normal (+, *, -) operations.
        
        ################ 
        
        For numpy arrays:
            A = [a1, a2, a3]
            B = [b1, b2, b3]
            C := np.outer(A, B) = [[a1*b1,  a1*b2, a1*b3],
                                   [a2*b1,  a2*b2, a2*b3],
                                   [a3*b1,  a3*b2, a3*b3]]
                                := [[c11,  c12, c13],
                                    [c21,  c22, c23],
                                    [c31,  c32, c33]]
            D = [d1, d2, d3]
            C*D = [[d1*c11,  d2*c12, d3*c13],
                   [d1*c21,  d2*c22, d3*c23],
                   [d1*c31,  d2*c32, d3*c33]]
            
            (Can replace * with +, -, etc.)
        '''
        assert whichTc in ["real", "est"]
        Tc = self.Tc if whichTc == "real" else self.Tc_est
        exps = np.exp(-ctrls/Tc)
        dfun = exps*fun + (1 - exps)/2
        return dfun
        
    def likelihood(self, param: float, t: float, hits: int, nshots: int, 
                   var = "w", prt = False):
        '''
        Likelihood based on a single datum, i.e. a (m,# 1 outcomes) tuple.
        var: wether 'param' is "w" or "Tc".
        '''
        
        if var == "Tc":
            p1 = ((1+np.exp(-t/param))/2)
            if prt:
                print(f"> p1 = {p1}. [PrecessionModel.likelihood - Tc]")
            L = self.likelihood_multishot(p1, nshots, hits)
            return L 
        
        elif var == "w":
            w = param
        else:
            raise Exception("`var` must be 'w' or 'Tc'.")
            
        L = self.likelihood_w(w, t, hits, nshots)
        return L

    def likelihood_w(self, w, t, hits, nshots):
        arg = t*w/2
        p1 = np.sin(arg)**2
        if self.Tc_est is not None:
            p1 = self.damp_fun(t, p1, "est")
        L = self.likelihood_multishot(p1, nshots, hits)
        L = self.enforce_domain(w, L, "w")
        return L
    
    @staticmethod
    def likelihood_multishot(L, nshots, hits):
        '''
        If some outcome has likelihood L, calculate the probability of getting
        that outcome 'hits' times out of 'nshots'.
        '''
        # Binomial coefficient for combinations.
        Cnh = 1 # math.comb(nshots, hits)
        return Cnh*(L**hits)*(1-L)**(nshots-hits)
    
    def batch_likelihood(self, lparam, data, var = "a", **kwargs):
        '''
        For w, log-likelihood; and return lparam, because some may
        be removed if zero likelihood.
        '''
        ctrls, outcomes = np.array(data.ctrls), np.array(data.outcomes)
        if var=="Tc":
            return self.batch_likelihood_Tc(lparam, ctrls, outcomes)
        elif var == "w":
            ws = lparam
        else:
            raise Exception(f"`var`must be 'w' or 'Tc', not {var}.")
            
        ls_joint = self.batch_likelihood_ws(ws, ctrls, outcomes, 
                                                **kwargs)
            
        return ls_joint
    
    def batch_likelihood_ws(self, ws, ctrls, outcomes, log, info):
        '''
        Loglikelihoods are returned. 
        
        'info': which likelihoods are being calculated, e.g. MCMC new or old,
        to print warning if some are zero.
        '''
        # Matrix; rows are associated with ws, columns with ctrls.
        args = np.outer(ws, ctrls/2)
        # Likelihoods conditional on outcome 1.
        L1 = self.sin2_catch(args, ws)
        
        if self.Tc_est is not None:
            L1 = self.damp_fun(ctrls, L1, "est")
            
        # Likelihood considering actual outcomes.
        L = L1**outcomes*(1-L1)**(1-outcomes)
        
        if log:
            LL = self.logl_catch(L, ws, info)
            # Loglikelihoods. List with same length as ws; each row 
            # condensed to a joint number.
            lls = np.sum(LL, axis=1)
            # Set likelihood of out of bounds parameters to 0 (log = -inf).
            lls = self.enforce_domain(ws, lls, "w", log = True)
            return lls
        else: 
            ls = np.prod(L, axis=1)
            ls = self.enforce_domain(ws, ls, "w")
            return ls
        
        # print("ls1", ls[:10])
        # print("ls2", np.exp(lls)[:10])
        # input()
    
    def logl_catch(self, L, ws, info, print_which = False):
        '''
        Ls is a matrix; element (i,j) = likelihood of w_i for experiment j.
        '''
        def cleanup(ls, print_which = False):
            # Likelihood should never be zero for old particles.
            if info == "old":
                print_which = True
                print("> Likelihood zero for old likelihood.")
            # Joint likelihood is zero if it is zero for any experiment.
            row_is_zero = np.any(L == 0, axis=1)
            zero_prob_params = np.sum(row_is_zero)
            zeros_pc = 100*zero_prob_params/len(ws)
            print(f"> {zeros_pc:.1f}% of {info} likelihoods were zero. "
                  "[PrecessionModel.logl_catch]")
            if np.isclose(zeros_pc, 100):
                print("> Quitting.")
                exit()

            if print_which:
                print("> For: ")
                # Nonzero gets indices. Returns a tuple for the dimensions.
                for i in np.nonzero(row_is_zero)[0]:
                    print(f"{ws[i] = }", end = " | ")
                # input()
        
        assert np.all(L >= 0)
        assert L.shape[0] == len(ws)
        lls = np_invalid_catch(f = np.log,
                             largs = [L], 
                             errdesc = ["divide by zero encountered in log"], 
                             cleanup = cleanup,
                             caller = "PrecessionModel.logl_catch")
        return lls
    
    def remove_zero_probs(self, L1, outcomes):
        '''
        Remove ws whose likelihood is zero for any outcome.
        
        L1: matrix where item [i,j] is likelihood of parameter i for experiment
        j, conditional on outcome 1. Outcomes is a list with the actual 
        observed outcomes.
        '''
        # Likelihood of parameters given outcome 1 is >0 for all 1 outcomes.
        condition1 = (L1 > 0) | (outcomes != 1)

        # Likelihood of parameters given outcome 0 is >0 for all 0 outcomes.
        condition2 = (L1 < 1) | (outcomes != 0)
        
        keep = np.all(condition1 & condition2, axis=1)
        
        return L1[keep], keep
    
    def sin2_catch(self, args, ws):
        '''
        Note that args is a matrix.
        
        ignore_err: whether to warn if invalid value in sin, or to pause 
        execution and require input.
        '''
        sin2 = lambda x: np.sin(x)**2
        r = np_invalid_catch(f = sin2,
                             largs = [args], 
                             errdesc = ["invalid value encountered in sin"], 
                             cleanup = lambda x: print_1st_err(sin2, x, "arg", 
                                                               [ws], ["w"]),
                             caller = "sin2_catch")
        return r
    
    def batch_likelihood_Tc(self, lparam, ctrls, outcomes):
        '''
        lparam: a list of parameters
        Ls_joint: a list of joint (all data) likelihoods, one for each parameter
        '''
        exps = np.outer(-1/lparam, ctrls)
        p1s = self.exps_catch(exps)
        
        # Parameter x data matrix (rows x cols).
        matrix = p1s**outcomes*(1-p1s)**(1-outcomes)
        Ls_joint = np.prod(matrix, axis=1)
        
        Ls_joint = self.enforce_domain(lparam, Ls_joint, "Tc")
        return Ls_joint
    
    def exps_catch(self, exps):
        def cleanup(exps):
            print("> Overflow in exps calculation.")
            f = lambda x:  ((1+np.exp(x))/2)
            # print_1st_err(f, exps, "exp")
            
        exps_calc = lambda exps: ((1+np.exp(exps))/2)
        r = np_invalid_catch(f = exps_calc,
                             largs = [exps], 
                             errdesc = ["overflow encountered in exp"], 
                             cleanup = cleanup,
                             caller = "exps_catch")
        return r

    def enforce_domain(self, lparam, ls, var, log = False):
        '''
        Correct the likelihoods 'ls' of out-of-bounds parameters in 'lparam'.
        '''
        mn, mx = self.get_range(var)
        valid = np.logical_and(lparam >= mn, lparam <= mx)
        
        zeroprob = -np.inf if log else 0
        ls = np.where(valid, ls, zeroprob)
        return ls
        
       
def get_data(M, ctrls, var):
    data = MeasurementData([], [], [])
    nshots = 1
    for ctrl in ctrls:
        outcome = M.measure(ctrl, nshots, var = var, prt = False)
        data.append_datum(ctrl, outcome, nshots)
    return data

def get_batch_data(M, ctrl, nshots, var):
    h = M.measure(ctrl, nshots, var = var, prt = True)
    data = MeasurementData([ctrl], [h], [nshots])
    return data
       
def test(var = "Tc", batch = False):
    w = 0.1
    wmax = 1
    q = PrecessionModel(w, wmax = 1, Tc = T, Tcrange = (100, 1000))
    # Give true amplitude.
    q.set_Tc_est(T)
    # Test likelihood of real parameter.
    if var == "Tc":
        param = T

    if batch:
        ctrl = 0
        nshots = 100
        data = get_batch_data(q, ctrl, nshots, var)
        q.plot_likelihood(data, var = var)
    else:
        ctrls = np.linspace(0, T, 100)
        ctrls = np.round(ctrls)
        data = get_data(q, ctrls, var)
        q.plot_likelihood(data, var = var)

if __name__ == "__main__":
    test()