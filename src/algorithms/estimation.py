# -*- coding: utf-8 -*-
'''
   ============================================================================

    Offline and adaptive parameter estimation using approximate Bayesian
    inference.

    Two testing tools (a function and a class) are included:
    - 'test', which tests single runs of inference with different sampling
    strategies.
    - 'TestBAE', which tests multiple runs to get data on the evolution of
    the MSE with the number of queries (for plotting).

   ============================================================================
'''

import sys

try:
    from google.colab import files
    sys.path.append('/content/drive/Othercomputers/Laptop/Phd/3.Scripts')
    using_colab = True
except ModuleNotFoundError:
     using_colab = False

import numpy as np
import scipy.optimize as opt
from copy import deepcopy

# When running on Google Colab, modules must be explicitly reloaded to get the
# latest version (or restart the runtime). The 2 first functions take care of that.
from utils.misc import initialize_modules, reload_custom_modules
# This will take effect in the 1st execution (register default modules).
initialize_modules()
# This will take effect in the 2nd-Nth executions (reload non default modules).
reload = True
if reload and using_colab:
    reload_custom_modules()

from algorithms.samplers import get_sampler
from utils.models import PrecessionModel, QAEmodel
from utils.plotting import process_and_plot, sqe_evol_from_file
from src.utils.mydataclasses import (EstimationData, ExecutionData, MeasurementData,
                               join_estdata_files)
from utils.misc import (round_if_float, print_centered, dict_str, sigdecstr,
                        k_largest_tuples, k_smallest_tuples, kwarg_str, b10str)
from utils.running import Runner, PrintManager, BAERunsData, BAERunData


NDIGITS = 4

class ParameterEstimation():
    def __init__(self, model, Tc_precalc, Tcrange, show = False):
        global param
        self.model = model
        self.Tc_precalc = Tc_precalc
        self.Tcrange = Tcrange
        self.Tc_est = None
        self.show = show
        self.data = None
        self.cmin = None
        self.cmax = None
        self.double = False
        # Get the model string and parameter char.
        if isinstance(self.model, PrecessionModel):
            self.mstr = "precession"
            self.pchar = "w"
        if isinstance(self.model, QAEmodel):
            self.mstr = "QAE"
            self.pchar = "a"
        self.pman = PrintManager(["optimize_control",
                                  "adapt_inference",
                                  "warm_up",
                                  "off_inference",
                                  "results",
                                  "learn_Tc"])

    def adapt_inference(self, sampler,
                                  strat: dict,
                                  maxPT: int = None,
                                  print_evol = False,
                                  plot_all = False):
        '''
        Warms up to offline measurements with m = 0, then performs adaptive
        ones until a total of 'maxPT' queries has been reached.

        During the adaptive phase, we search in an expanding window that is
        doubled if chosen control has been within the 'exp_refs' highest
        for 'exp_thr' times. If a finite Tc is considered, expansion is
        capped at 'max_ctrl'.
        '''
        rd = BAERunData(sampler, self.probing_time)
        wN = strat["wnshots"] 
        if wN > 0:
            wctrl = self.warm_up(sampler, wN)
            rd.add_iteration_data(wctrl, wN)
        else:
            # Create empty dataset to append to.
            self.data = MeasurementData([], [], [])


        s = f"> Will perform 1 shot adaptive measurements up to Nq = {b10str(maxPT)}."
        self.pman.print1st(s, "adapt_inference")

        # Adaptive phase.
        astrat = deepcopy(strat); astrat.pop("wnshots")

        maxs = 0
        while rd.latest_CPT < maxPT:
            ctrl_opt, max_flag = self.choose_control(sampler, **astrat)
            if max_flag:
                maxs += 1
                if maxs > strat["exp_thr"] and not self.capped:
                    self.double = True

            nshots = 1
            outcome = self.model.measure(ctrl_opt, nshots)
            self.data.append_datum(ctrl_opt, outcome, nshots)
            sampler.update_latest(self.data)
            
            rd.add_iteration_data(ctrl_opt, 1)

        print(f"> Estimation interrupted after {len(rd)} measurements"
              f" due to Nq = {b10str(rd.latest_CPT)} >= {b10str(maxPT)}.")

        self.results(sampler, "Online")
        return rd.get_lists()

    def set_Tc_est(self, Tc_est):
        self.Tc_est = Tc_est
        self.model.set_Tc_est(Tc_est)

    def warm_up(self, sampler, wnshots):
        if not isinstance(self.Tc_precalc, bool):
            # It's a fixed value known_Tc; use it.
            Tc_est = self.Tc_precalc
            self.set_Tc_est(Tc_est)
            print(f"> Assuming given Tc = {Tc_est}. [warm_up]")
        elif self.Tc_precalc:
            Tc_est = self.learn_Tc()
            self.set_Tc_est(Tc_est)
            
        if wnshots == 0:
            return
        
        s = f"> Warming up with {wnshots} classical shots."
        self.pman.print1st(s, "warm_up")

        ctrls = self.gather_data(wnshots, how = "classical")
        assert len(ctrls) == 1; ctrl = ctrls[0]

        data = self.data
        sampler.update_latest(data)
        return ctrl

    def learn_Tc(self, Tc_nshots = 100, off_shots = 1,
                 off_rng = (0, 1)):
        '''
        off_shots: what fraction of Tc_nshots to be used in non adaptive
        measurements (initial learning phase).
        off_rng: control range for the offline measurements, as a percentage
        of Tcmax.
        '''
        def get_offline_ctrls(revert = True):
            '''
            Parameters for the linspace function.
            '''
            Tcmax = self.Tcrange[1]
            start = int(off_rng[0]*Tcmax)
            stop = int(off_rng[1]*Tcmax)
            num = int(off_shots*Tc_nshots)
            offctrls = np.linspace(start, stop, num)
            if revert:
                offctrls = offctrls[::-1]
            return num, offctrls

        print("> Learning Tc.")
        sampler_kwargs = {"Npart": 2000, "thr": 1, "factor": 1, "var": "Tc", "ut": "var"}
        Tcsampler = get_sampler("RWM", self.model, sampler_kwargs)

        offshots, offctrls = get_offline_ctrls()
        s = (f"> Fraction of off contols for Tc: {off_shots}. "
             f"In: {offctrls[0]}-{offctrls[-1]}.")
        self.pman.print1st(s, "learn_Tc")
        
        Tcdata = MeasurementData([], [], [])
        for i in range(Tc_nshots):
            if i < offshots:
                ctrl = offctrls[i]
            # Otherwise control will have been decided adaptively at the end of
            # the previous iteration.

            outcome = self.model.measure(ctrl, 1, var="Tc")
            Tcdata.append_datum(ctrl, outcome, 1)
            Tcsampler.update_seq(Tcdata)

            if i > offshots:
                Tmean, Tstd = Tcsampler.mean_and_std()
                ctrl = Tmean
        if off_shots == 1:
            Tmean, Tstd = Tcsampler.mean_and_std()

        print(f"> Estimated Tc based on {Tc_nshots} shots: "
          f"{Tmean:.0f} ± {Tstd:.0f}.")
        self.Tc_est = Tmean

        return Tmean

    def choose_control(self, *args, **kwargs):
        grid, ctrl_opt, max_flag = self.optimize_control(*args, **kwargs)
        self.print_grid_info(grid, **kwargs)
        return ctrl_opt, max_flag

    def optimize_control(self, sampler, factor = 1, Nevals = 100,
                         exp_thr = 5, exp_refs = 3, cap = True,
                         capk = 1):
        '''
        The search range is a set of integers initially spaced by 'factor' and
        going from 1 to factor*Nevals.
        '''
        if self.cmin is None:
            # First time optimizing.
            self.init_opt(factor, Nevals, cap, capk)

        if self.double:
            self.double_grid()

        grid = np.round(np.linspace(self.cmin, self.cmax, num = int(Nevals)))
        ctrl_opt = self.discrete_optimization(grid, sampler)

        max_flag = True if ctrl_opt in grid[-exp_refs:] else False
        return grid, ctrl_opt, max_flag

    def init_opt(self, factor, Nevals, cap, capk):
        self.cmin, self.cmax = 1, int(factor*Nevals)
        self.factor = factor
        self.cap = cap
        self.capk = capk

    def double_grid(self):
        print(f"> Upping search range from {self.cmin}-{self.cmax}. ",
              end = "")
        if self.capped:
            # Previously capped.
            return
        # Uncapped vs. capped limits.
        self.cmin, self.cmax = self.cmax, self.cmax*2
        if self.capped:
            # Cap now.
            self.set_capped_lims()
        self.double = False

    def discrete_optimization(self, grid, sampler):
        ims = self.objective_function(grid, sampler)
        ctrl_opt = grid[np.argmin(ims)]
        return ctrl_opt

    def objective_function(self, x, sampler):
        '''
        Function to be maximized. Considers all data gathered so far, which
        is usually intended in online estimation.
        '''
        return -sampler.expected_utilities(x, self.data)

    @property
    def capped(self):
        '''
        Window is capped if a finite Tc is considered, 'cap' is true, and
        maxctrl has been exceeded.
        '''
        is_capped = self.Tc_precalc and self.cap and self.cmax >= self.maxctrl
        return is_capped

    def set_capped_lims(self):
        print(f"Would be {self.cmin}-{self.cmax}; permanently capped"
        f" due to Tc={self.Tc_est:.0f}. ")

        self.cmin, self.cmax = int(self.maxctrl/2), self.maxctrl

        print(f"New range: {self.cmin}-{self.cmax}.")

    @property
    def maxctrl(self):
        '''
        Maximum control considered when searching for the optimal choice.
        '''
        return int(self.capk*self.Tc_est)

    def cumul_probing_times(self, ctrls, nshots_list):
        if isinstance(nshots_list, int):
            nshots_list = [nshots_list for i in range(len(ctrls))]

        PT_per_meas = [self.probing_time(ctrl)*nshots for ctrl, nshots
                       in zip(ctrls, nshots_list)]
        cumul_PTs = np.cumsum(PT_per_meas)
        return cumul_PTs

    def probing_time(self, ctrl):
        if self.mstr == "QAE":
            # The control is the number of Grover applications, calculate the
            # number of queries.
            PT = 2*ctrl+1
        if self.mstr == "precession":
            # The control is the evolution time.
            PT = ctrl
        return PT

    def gather_data(self, nshots, how = "classical"):
        if how == "classical":
            ctrls, Nsshots = [0], [nshots]
        if how == "exp":
            ctrls = self.exp_controls(nshots)
            Nsshots = [1 for i in range(nshots)]

        data = self.model.create_data(deepcopy(ctrls), Nsshots)
        self.data = data

        print(f"> Measured {nshots} shots data for {how} controls"
              f" using {type(self.model).__name__} with {self.pchar}"
              f"_real={real_param_str}. [ParameterEstimation.gather_data]")
        return ctrls

    def print_grid_info(self, grid, **kwargs):
        if self.pman.ISFIRST["optimize_control"]:
            s = ("> Optimized experimental controls on a grid over range "
                 f"[{self.cmin}, {self.cmax}].\n"
                 f"> Working with {kwarg_str(self.optimize_control)}.")
            gridstart = ", ".join([str(x) for x in grid[:3]])
            gridend = ", ".join([str(x) for x in grid[-3:]])
            s += ("\n> Grid: " + gridstart + ",..., " + gridend +  ".")
            self.pman.print1st(s, "optimize_control")

    def data_warn(f):
        def wrapper(self, *args, **kwargs):
            if self.data is None:
                print("> To perform estimation, I need data. Gather it using"
                      "the 'gather_data' method and get back to me."
                      f"[ParameterEstimation.{f.__name__}]")
                return
            return f(self, *args, **kwargs)
        return wrapper

    @data_warn
    def mle(self, Nevals = 10e3, finish = None):
        def objective_function(param):
            return -self.model.batch_likelihood(param, self.data)

        info = "without" if finish is None else "with"
        print(f"> Testing brute force MLE on the data ({info} Nelder Mead; "
              f"Nevals = {Nevals})... [test_mle]")

        if finish is None:
            param_opt = opt.brute(objective_function, [(0,1)], Ns=Nevals,
                                  finish = finish)
        else:
            param_opt = opt.brute(objective_function, [(0,1)], Ns=Nevals,
                                  finish = finish)[0]
        print("> MLE: ", param_opt)
        return param_opt

    @data_warn
    def off_inference(self, sampler, batch = False):
        '''
        Offline inference (pre-determined controls).
        '''
        data = self.data
        method = ("Batch-updating a grid " if batch else
                  f"Running SMC-{sampler.str}")
        s = f"> {method} on the data... [ParameterEstimation.off_inference]"
        self.pman.print1st(s, "off_inference")

        if batch:
            sampler.batch_update(data)
        else:
            sampler.update_seq(data)

        return self.results(sampler, "Offline")

    def exp_controls(self, Nmeas):
        if self.mstr == "precession":
            ctrls = [(9/8)**k for k in range(Nmeas)]
        if self.mstr == "QAE":
            ctrls = [k for k in range(Nmeas)]

        toprint = ", ".join(str(round_if_float(ctrl)) for ctrl in ctrls[:5])
        if len(ctrls)>1:
          toprint += ",..., " + str(round_if_float(ctrls[-1]))

        print(f"> Determined experimental controls: {toprint}."
              " [ParameterEstimation.exp_controls]")
        return ctrls

    def results(self, sampler, strat):
        s = self.get_info(sampler, strat)
        self.pman.print1st(s, "results")
        sampler.print_stats()
        mean, std = sampler.mean_and_std()
        print(f"> {strat} SMC-{sampler.str} estimate: {self.pchar} = "
              f"{sigdecstr(mean, NDIGITS)} ± {sigdecstr(std, NDIGITS)}")
        return mean, std

    def get_info(self, sampler, strat):
        data = self.data
        if data.warmup is False:
            Nmeas = len(data)
        else:
            # List warm up and adaptive data separately.
            Nmeas = (1, len(data)-1)
        nshots = data.nshots_txt()
        info = ["=============================================================="]
        info.append(f"{strat} {self.mstr} estimation with SMC-{sampler.str}")
        info.append(f"Nmeas = {Nmeas} | nshots = {nshots} | "
                    f"Npart = {sampler.Npart}")
        info.append(real_param_str)
        info.append("==============================================================")
        for i,line in enumerate(info):
            info[i] = line.center(62)
        info = "\n".join(info)
        return info


class TestBAE():

    def __init__(self, a, Tc_opts, strat, maxPT, sampler_str, sampler_kwargs):
        self.a = a
        self.Tc = Tc_opts["Tc"]
        self.Tcrange = Tc_opts["range"]
        self.strat = strat
        self.maxPT = maxPT
        self.Tc_precalc = Tc_opts["Tc_precalc"]
        self.known_Tc = Tc_opts["known_Tc"]
        self.sampler_str = sampler_str
        self.sampler_kwargs = sampler_kwargs

        # This global string is to be used by other functions for printing.
        global real_param_str
        real_param_str = self.param_str()

    def param_str(self):
        a_str = (self.rand_pstr(self.a) if isinstance(self.a,tuple)
                 else str(self.a))
        Tc_str = (self.rand_pstr(self.Tc) if isinstance(self.Tc,tuple)
                 else str(self.Tc))
        s = f"a={a_str};Tc={Tc_str}"
        return s

    @staticmethod
    def rand_pstr(param):
        return f"rand[{param[0]},{param[1]}]"

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
            a = np.random.uniform(amin,amax)
            print(f"> Sampled a = {a} (theta = {QAEmodel.theta_from_a(a)}).")
            return a
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

    def sqe_evolution_multiple(self, nruns, dummy_data = False, save = True):
        '''
        Gets the evolution of the SQUARED error with the iteration number, for
        multiple runs. All the data is joined together in a non-nested list,
        because what matters is the order. e.g.

        nqs_1 = [1, 3] | sqes_1 = [A1, B1]
        nqs_2 = [2, 5] | sqes_2 = [A2, B2]
        -> nqs_all = [1, 3, 2, 5] | sqes_all = [A1, B1, A2, B2]

        The results are then processed to get working averages (obtained by
        binning the nqs and averaging both them and the associated sqes within
        bins) and plotted. Straightforward averaging cannot be done because the
        nqs are not constant among runs, even for the same iteration.

        Also, the square root is taken when processing. The reason we don't
        use the RMSE from the outset is that we may want to do calculations
        with MSE, then take the root in the end (to mimic the usual "average
        the square -> take the root", but with curve fits/...).
        '''
        def print_info():
            info = ["Bayesian adaptive QAE"]
            info.append("- scaling of the estimation error with Nq")
            info.append(f" {real_param_str} | runs = {nruns} | "
                        f"nqs/PT ≈ 10^{round(np.log10(self.maxPT),1)}")
            print_centered(info)

        print_info()

        if dummy_data:
            # Broke this with new dataclass BAERunsData.
            sqes_w, nqs, sqes = self.create_dummy_data()
        else:
            rdata = BAERunsData()
            runner = Runner(f = self.sqe_evolution, nruns = nruns,
                            process_fun = rdata.add_run_data, redirect = 0)
                            
            nruns = runner.run()

            full, final = rdata.get_lists()
            _, final_dscr = rdata.get_descriptors()
            self.print_stats_several(final, final_dscr)

        if nruns > 0:
            raw_estdata = self.create_estdata(*full)
            process_and_plot(raw_estdata)

            if save:
                exdata = self.create_execdata(raw_estdata, nruns)
                exdata.save_to_file()

    def sqe_evolution(self):
        '''
        Perform a single run of Bayesian adaptive QAE, and return lists of the
        numbers of queries, errors and standard deviations (ordered by step).
        '''
        a = self.local_a; Tc = self.local_Tc

        M =  QAEmodel(a, Tc = Tc, Tcrange = self.Tcrange)

        Tc_precalc = self.Tc_precalc
        # Tc_precalc is False if Tc to be ignored; True if to be considered.
        # known_Tc is whether to estimate it from scratch or get an input value.
        if self.Tc_precalc and self.known_Tc:
            Tc_precalc = Tc
        # At this point Tc_precalc is False if Tc to be ignored; True if to be
        # estimated; or a float, non-Boolean, giving an estimate for Tc to use
        # directly.

        Est = ParameterEstimation(M, Tc_precalc, self.Tcrange)
        sampler = get_sampler(self.sampler_str, M, self.sampler_kwargs)

        means, stds, nqs = Est.adapt_inference(sampler, self.strat,
                                               maxPT = self.maxPT)
        nsqes = [(est/a - 1)**2 for est in means]
        nstds = [sqe/mean for sqe,mean in zip(stds, means)]
        print("> Final (normalized) RMSE: ", sigdecstr(nsqes[-1]**0.5, NDIGITS))
        print("> Final (normalized) std : ", sigdecstr(nstds[-1], NDIGITS))
        return nqs, nsqes, nstds

    def print_stats_several(self, ls, dscrs):
        for l, dscr in zip(ls, dscrs):
            self.print_stats(l, dscr)

    @staticmethod
    def print_stats(l, dscr, print_all = False):
        '''
        l: list of quantities, e.g. errors, ordered by increasing iteration #.
        dscr: descriptor for the quantities.
        print_all: whether to print every item or just summary stats.
        '''
        lenum = list(enumerate(l))
        if print_all:
            print(f"> List of {dscr}s: ", lenum)
        print(f"> Mean {dscr}:   ", sigdecstr(np.mean(l), NDIGITS))
        print(f"> Median {dscr}: ", sigdecstr(np.median(l), NDIGITS))
        if len(l)>3:
            print(f"> 3 largest {dscr}s:  ", k_largest_tuples(lenum, 3,
                                                              sortby = 1))
            print(f"> 3 smallest {dscr}s: ", k_smallest_tuples(lenum, 3,
                                                               sortby = 1))

    def create_estdata(self, nqs, sqes, stds):
        '''
        Organize information into a EstimationData object.
        '''
        # warmup_point = (Nq_w, np.mean(sqes_w)**0.5, np.mean(stds_w))
        raw_estdata = EstimationData()
        raw_estdata.add_data("BAE", nqs = nqs, lbs = None, errs = sqes,
                             stds = stds)
        return raw_estdata

    def create_execdata(self, raw_estdata, nruns):
        '''
        Organize information into a ExecutionData object.
        '''
        sampler_params = dict_str(self.sampler_kwargs)
        sampler_info = f"{self.sampler_str}({sampler_params})"
        strat_info = f"STRAT=({dict_str(self.strat)})"
        extra = f"{strat_info},{sampler_info}"
        nshots = (self.strat["wnshots"], 1)
        exdata = ExecutionData(real_param_str, raw_estdata, nruns,
                               nshots, label="BQAE",
                               extra_info = extra)
        return exdata

    def create_dummy_data(self):
        '''
        Generate quick dummy data for testing plots, save to files, etc.
        '''

        print("> USING DUMMY DATA.")
        Nq_warmup = self.nshots_tuple[0]
        nqs_all = np.linspace(Nq_warmup, self.maxPT, 100)
        sqes_all = 10/nqs_all

        # Add some noise to make the runs distinguishable.
        sqes_all = [np.random.uniform(0.5, 1.5)*sqe for sqe in sqes_all]

        sqes_warmup = [sqes_all[0]]
        return sqes_warmup, nqs_all, sqes_all

def test_evol(which):
    if which == 1:
        '''
        Test multiple runs of Bayesian adaptive QAE, and plot the RMSE as a
        function of the number of queries.
        '''
        # If a is a tuple, a float will be picked at random from the interval
        # for each run. If it is a number, it will be exused direcly.
        global Tcmax
        Tcrange = None # (2000, 5000)
        a = (0, 1)
        Tc = Tcrange # 2000
        maxPT = 10**7
        nruns = 100
        sampler_str = "RWM"
        

        Tc_opts = {"Tc": Tc,
                   "Tc_precalc": False,
                   "known_Tc": False,
                   "range": Tcrange}

        # Strategy for the adaptive optimization.
        strat = {"wnshots": 10,
                 "factor": 2,
                 "Nevals": 50,
                 "exp_refs": 3,
                 "exp_thr": 3,
                 "cap": True,
                 "capk": 1.5}
        # Sampler arguments.
        sampler_kwargs = {"Npart": 2000,
                          "thr": 0.5,
                          "var": "theta",
                          "ut": "var",
                          "plot": False}
        if sampler_str=="RWM":
            sampler_kwargs["factor"] = 1
        if sampler_str=="LW":
            sampler_kwargs["a_LW"] = 0.98

        Test = TestBAE(a, Tc_opts, strat, maxPT, sampler_str,
                            sampler_kwargs)

        Test.sqe_evolution_multiple(nruns, save = False)
    elif which == 2:
        '''
        Upload dataset from one file.
        '''
        filename = "BQAE_21_03_2024_19_55_a=rand[0,1];Tc=None,nruns=100,nshots=(100, 1),STRAT=(wnshots=100,factor=1,Nevals=50,exp_refs=3,exp_thr=3,cap=True,capk=1.5),RWM(Npart=2000,thr=0.5,var=theta,ut=var,plot=False,factor=1)#0.data"
        sqe_evol_from_file(filename)
    elif which == 3:
        '''
        Upload dataset from multiple files.
        '''
        filestart = ("adaptive_QAE_05_07_2022[LW,Nsmeas=(1,inf),Nsshots=(100,1),"
                     "Npart=1000,runs=5]")
        save = False
        select_indices = False
        indices = range(13,25) if select_indices else None

        combined_dataset = join_estdata_files(filestart, indices = indices,
                                              save = save)
        if combined_dataset:
            process_and_plot(combined_dataset)

test_evol(1)
# self.cmin, self.cmax = int(self.maxctrl/2), self.maxctrl