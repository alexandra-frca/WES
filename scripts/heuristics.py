import numpy as np 
import os

''' 
The following is necessary to deal with KeyboardInterrupt properly for some 
reason; otherwise "forrtl: error (200): program aborting due to control-C event"
'''
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

from src.utils.misc import dict_str 
from src.algorithms.samplers import get_sampler
from src.utils.models import PrecessionModel
from src.utils.mydataclasses import MeasurementData, EstimationData, ExecutionData
from src.utils.misc import sigdecstr
from src.utils.plotting import process_and_plot
from src.utils.running import Runner, BAERunsData

NDIGITS = 3

stall_counter = 0
class AdaptiveInference():

    def __init__(self, heuristic, sampler, model, Tc = None):
        assert heuristic in ["sigma", "PGH"]
        self.heuristic = heuristic
        self.sampler = sampler
        self.model = model
        self.Tc = None
        self.data = MeasurementData([], [], [])

    def infer(self, maxPT, debug = False):
        sampler = self.sampler
        means = []
        stds = []
        cpts = [0]
        ctrls = []
        while cpts[-1] < maxPT:
            ctrl = self.choose_control()
            # print("> sampler locs:", sampler.locs)
            # print("> sampler ws:", sampler.ws)
            outcome = self.model.measure(ctrl, 1)
            self.data.append_datum(ctrl, outcome, 1)
            sampler.update_latest(self.data)
            mean, std = sampler.mean_and_std()
            means.append(mean)
            stds.append(std)
            cpts.append(cpts[-1] + ctrl)
            ctrls.append(ctrl)
            if debug: 
                print("> Chosen control: ", ctrl, "; total CPT: ", cpts[-1], "measlen", len(means))
                print("last 10 ctrls:", ctrls[-10:])

            # print(std)
            '''
            N = 100
            if len(stds)>N and (stds[-1] >= stds[-N] or all(x == ctrls[-1] for x in ctrls[-5:-1])):
                global stall_counter
                stall_counter += 1
                print("*** Learning stalled, did not reach maxPT. ***")
                break
            '''
            
        print("> Estimated ", mean)
        print("> Total number of measurements: ", len(means))
        # cpts has an initial zero that doesn't correspond to an iteration. 
        return means, stds, cpts[1:]

    def choose_control(self):
        if self.heuristic == "sigma":
            _, std = self.sampler.mean_and_std()
            t = 1/std
            return t
        else: 
            delta = 0
            counter = 0
            while delta == 0:
                counter += 1
                duo = self.sampler.random_duo()
                delta = abs(duo[0] - duo[1])
                #if counter == 10:
                #    print("> Counter reached 10!", self.sampler.locs)
            t = 1/delta
            return t
        
class Test():
    def __init__(self, heuristic, w, wmax, Tc, Tcrange, maxPT, sampler_str, sampler_kwargs,
                 silent = False, save = True, show = True):
        self.heuristic = heuristic 
        self.a = w
        self.wmax = wmax
        self.Tc = Tc
        self.Tcrange = Tcrange
        self.maxPT = maxPT
        self.sampler_str = sampler_str
        self.sampler_kwargs = sampler_kwargs
        self.silent = silent 
        self.save = save
        self.show = show

    @staticmethod
    def rand_pstr(param):
        return f"[{param[0]},{param[1]}]"

    @property
    def local_a(self):
        '''
        For each run, the real amplitude parameter will be 'local_a'.

        The 'a' attribute is always constant, and can hold:

        - A permanent value for 'a'. In that case, all runs will use it;
        'local_a' is equivalent to 'a'.

        - A tuple. In that case, each run will sample an amplitude at random
        in the interval given by the tuple.
        '''
        if isinstance(self.a, tuple):
            amin, amax = self.a
            w = np.random.uniform(amin,amax)
            print(f"> Sampled w = {w}.")
            return w
        else:
            return self.a

    @property
    def local_Tc(self):
        if isinstance(self.Tc, tuple):
            Tcmin, Tcmax = self.Tc
            Tc = np.random.uniform(Tcmin,Tcmax)
            print(f"> Sampled Tc = {Tc}.")
            return Tc
        else:
            return self.Tc

    def sqe_evolution_multiple(self, nruns, redirect = 0):
        rdata = BAERunsData()
        runner = Runner(f = self.sqe_evolution, nruns = nruns, timeout = 3,
                        process_fun = rdata.add_run_data, redirect = redirect,
                        silent = self.silent, save = self.save)

        nruns = runner.run()

        full, final = rdata.get_lists()
        _, final_dscr = rdata.get_descriptors()

        if nruns == 0:
            return 
            
        print("> Percentage of incomplete runs: ", round(stall_counter/nruns*100, 2))
        raw_estdata = self.create_estdata(*full)
        process_and_plot(raw_estdata, save = self.save, show = self.show)

        if self.save:
            exdata = self.create_execdata(raw_estdata, nruns)
            exdata.save_to_file()

    def sqe_evolution(self, i, debug = False):
        '''
        Perform a single run of Bayesian adaptive QAE, and return lists of the
        numbers of queries, errors and standard deviations (ordered by step).
        '''
        w = self.local_a; Tc = self.local_Tc

        M =  PrecessionModel(w, self.wmax, Tc = Tc, Tcrange = self.Tcrange)
        sampler = get_sampler(self.sampler_str, M, self.sampler_kwargs)
        ainf = AdaptiveInference(self.heuristic, sampler, M)

        
        means, stds, nqs = ainf.infer(maxPT = self.maxPT)
        nsqes = [(est/w - 1)**2 for est in means]
        nstds = [sqe/mean for sqe,mean in zip(stds, means)]

        print("> Final (normalized) RMSE: ", sigdecstr(nsqes[-1]**0.5, NDIGITS))
        print("> Final (normalized) std : ", sigdecstr(nstds[-1], NDIGITS))
        return nqs, nsqes, nstds
    
    def create_estdata(self, nqs, sqes, stds):
        '''
        Organize information into a EstimationData object.
        '''
        raw_estdata = EstimationData()
        raw_estdata.add_data(self.heuristic, nqs = nqs, lbs = None, errs = sqes,
                             stds = stds)
        return raw_estdata
    
    def create_execdata(self, raw_estdata, nruns):
        '''
        Organize information into a ExecutionData object.
        '''
        sampler_params = dict_str(self.sampler_kwargs)
        extra = f"{self.sampler_str}({sampler_params})"
        Ns = 1
        exdata = ExecutionData("w", raw_estdata, nruns,
                               Ns, label=self.heuristic,
                               extra_info = extra)
        return exdata


if __name__ == "__main__":
    which = 1

    sampler_str = "RWM"
    sampler_kwargs = {"Npart": 5000,
                    "thr": 0.5,
                    "var": "w",
                    "ut": "var",
                    "log": True,
                    "res_ut": False,
                    "plot": False}
    heuristic = "sigma"
    wmax = 2*np.pi
    if which == 0:
        w = 0.1
        Tc = None
        Tcrange = None
        M = PrecessionModel(w, wmax, Tc = Tc, Tcrange = Tcrange)
        sampler = get_sampler(sampler_str, M, sampler_kwargs)
        maxPT = 1e7
        ai = AdaptiveInference(heuristic, sampler, M)
        means, stds, cpts = ai.infer(maxPT)
        print(means[-1], stds[-1], cpts[-1])

    if which == 1:
        w =  (0, wmax)
        Tc = None
        Tcrange = None
        maxPT = 1e7
        nruns = 1
        t = Test(heuristic, w, wmax, Tc, Tcrange, maxPT, sampler_str, 
                 sampler_kwargs, save = True, show = True)
        t.sqe_evolution_multiple(nruns, redirect = 0)
