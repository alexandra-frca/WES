'''
Auxiliary plotting functions.
'''

import numpy as np
import scipy.optimize as opt
from statistics import mean
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from itertools import chain

from src.utils.misc import logspace
# from src.utils.mydataclasses import EstimationData
from src.utils.files import data_from_file
from src.utils.processing import safe_save_fig, process

NDIGITS = 5

class Plotter():
    
    def __init__(self, log = True):
        fig, ax = plt.subplots(1,figsize=(10,6))
        self.ax = ax
            
        if log:
            plt.xscale('log'); plt.yscale('log')
    
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        ax.spines['right'].set_color('lightgray')
        ax.spines['top'].set_color('lightgray')
        
        ax.grid(which='both')
    
    def scatter(self, xs, ys):
        print("H", len(xs), len(ys))
        self.ax.scatter(xs, ys, marker="*", color='crimson', s=400,
                        edgecolors='black', linewidth=1)
        
    def scatter_by_groups(self, grouped_points, ypower = 1):
        colors = []
        for group in grouped_points:
            # Get a new color for this group.
            new_color = ('#%06X' % np.random.randint(0, 0xFFFFFF))
            while new_color in colors: 
                new_color = ('#%06X' % np.random.randint(0, 0xFFFFFF))
            colors.append(new_color)
            
            xs = [point[0] for point in group]
            ys = [point[1]**ypower for point in group]
            self.ax.scatter(xs, ys, color = new_color, alpha = 0.4)
        
    def line(self, xs, ys):
        # Sort by order of first tuple element (x).
        sorted_pairs = sorted(zip(xs,ys))
        # zip(*l) where l = [(x0,y0), (x1,y1)] combines same-index elements   
        # among all (tuple) inputs, producing a [(x0, x1), (y0, y1)] generator.
        xs, ys = [list(tuple) for tuple in zip(*sorted_pairs)] 
        self.ax.plot(xs, ys)
        
    def curve(self, f, xrange, npoints, style="-"):
        xs = np.linspace(xrange, npoints)
        ys = f(xs)
        self.ax.plot(xs, ys, style, color = "black")
        
    def vertical_lines(self, xs):
        for x in xs:
            self.ax.axvline(x, linestyle = '--', color = 'crimson') 

    def set_labels(self, xlabel, ylabel):
        self.ax.set_xlabel(xlabel, fontsize=16, style="italic", labelpad=10)
        self.ax.set_ylabel(ylabel, fontsize=16, style="italic", labelpad=10)

def process_and_plot(raw_estdata, save = True, show = True, processing = "binning",
                     stats = ["mean", "median"], title = None):
    '''
    Up to 2 plots: one with the root mean squared error, one with the median
    error (in separate graphs).
    '''
    assert processing in ["binning", "averaging", "averaging2", "none"]
    for stat in stats:
        estdata = process(raw_estdata, stat, processing)
        plot_est_evol(estdata, save = save, show = show, stat = stat, 
                      exp_fit = False, title = title)
    return estdata    

def sqe_evol_from_file(filename):
    estdata = data_from_file(filename).estdata
    process_and_plot(estdata)

def plot_single_run(nqs, stds, errs, rl, wNs, Ns, el, accl, essl, title):
    fig, ax = get_logplot(ylabel = None, title = title, return_fig = True)
    fig.subplots_adjust(right=0.75)

    ax.scatter(nqs, stds, label="standard deviation")
    ax.scatter(nqs, errs, label="true deviation")

    rnqs, accl, essl = fix_sampler_lists(rl, nqs, wNs, Ns, accl, essl)

    ax2 = ax.twinx()
    ax2.set_ylabel("MCMC acceptance rate")
    ax2.set_ylim(0, 1)
    ax3 = ax.twinx()
    ax3.set_ylabel("ESS")
    ax3.set_ylim(0, 1)
    ax3.spines.right.set_position(("axes", 1.1))

    ax2.scatter(rnqs, accl, marker = "*", s = 120, color = "green",
                label='MCMC acc rate')
        
    ax3.scatter(nqs, essl, marker = "v", color = "gray",
                label='ESS')

    for i,rNq in enumerate(rnqs):
        label = 'resampled' if i==0 else None
        plt.axvline(x = rNq, color = 'tab:red', label = label, 
                    linestyle = 'dashed')

    for i,v in enumerate(el):
        label = 'expanded' if i==0 else None
        plt.axvline(x = nqs[int(v/Ns)], color = 'tab:purple', label = label, 
                    linestyle = 'dotted', linewidth = 2.5)

    combine_legends([ax, ax2, ax3])
    plt.tight_layout()
    plt.show()
    safe_save_fig("single_run")

def combine_legends(axs):
    handles_labels_tuples = [ax.get_legend_handles_labels() for ax in axs]

    handles, labels = zip(*handles_labels_tuples)
    handles = [x for l in handles for x in l]
    labels = [x for l in labels for x in l]

    plt.legend(handles, labels, loc="lower left", fontsize=25, framealpha=0.8)

def fix_sampler_lists(rl, nqs, wNs, Ns, accl, essl):
    '''
    Fix resampler lists according to the numbers of shots (for the warm-up and 
    others). 
    
    A "measurement" considers 1 control for N >= 1 of shots, and nqs has the 
    ordered numbers of queries per measurement. However, the shots are 
    considered in independent iterations by the sampler. We want to condense
    this information to match the measurements.

    We thus divide these iterations into groups, one for each measurement.
    We consider to have resampled for a measurement if resampling occurred at
    any shot. Additionally, the resampling statistics are averaged among groups.

    rl is the list of iterations at the end of which resampling occurred.
    nqs is the list of cumulative queries for all iterations.
    wNs is the warm-up number of shots.
    accl is the ordered list of the acceptance rates for the resampling 
    occurrences.
    essl is the list of effective sample sizes for all iterations.


    So e.g. if we have 

    wNs = 10, Ns = 1
    nqs = [Nq0 = 10, ..., Nq19] 
    rl = [2, 5, 11, 15]
    accl = [A, B, C, D]

    Return:
    rnqs = [Nq0, Nq11, Nq15]
    accl = [(A+B)/2, C, D]
    '''
    Nmeas = len(nqs)

    # groups[i]: measurement number associated with iteration i.
    groups = [0] * wNs + list(chain(*[[i+1]*Ns for i in range(Nmeas-1)]))
    # rgroups[i]: measurement number associated with resampling iteration rl[i]. 
    rgroups = [0 if x < wNs else 1 + (x - wNs) // Ns for x in rl]

    gis = sorted(set(groups))
    ris = sorted(set(rgroups))
    
    rnqs = [nqs[i] for i in ris]
    essl = [mean(val for i, val in enumerate(essl) if groups[i] == g)
             for g in gis]
    accl = [mean(val for i, val in enumerate(accl) if rgroups[i] == g)
             for g in ris]
    
    return rnqs, accl, essl

def average_first_N(l, N):
    '''
    l is a list whose Nm first item are to be averaged into a single item,
    and the following ones are to be used as is.
    '''
    # Summarize warm up statistics as average.
    m = np.mean(l[:N])
    # Statistics for other updates are used as is.
    l = [m] + l[N:]
    return l

def plot_est_evol(*args, **kwargs): 
    # expfit could be removed, just fit always. Just keeping for compatibility.
    '''
    Plot (number of queries, error) points for one or more datasets, on a 
    loglog scale. 
    
    Additionally, plot the standard quantum and Heisenberg estimation limits
    in the case of a single dataset (otherwise the different y offsets and the
    many points will make things confusing).
    
    Can also plot the CR lower bounds if intended (as long as given by the 
    'lb_dict' property of the 'estdata' objects).
    '''
    save = kwargs.pop('save', True)
    show = kwargs.pop('show', True)
    ys = ["RMSE", "std"]
    for y in ys:
        id = plot_err_evol(y, *args, **kwargs)
        if id is not None:
            if save:
                safe_save_fig(id + "_est_evol")
            if show:
                plt.show()
        else:
            plt.close()
    
LONG =  {"RMSE": "*avgtype* error",
         "std":  "*avgtype* standard deviation (normalized)"}

def plot_err_evol(which, estdatas, stat = "mean", yintercept = "fit", 
                  limits = True, CRbounds = False, plot_fit = False, 
                  iconpath = None, exp_fit = True, lims = None,
                  title = None, plotlims = True, errdisplay = "bars",
                  xrg = None): 
    '''
    Plot either the evolution of either the true error, given by the RMSE 
    (which = "RMSE"), or its estimate, given by the standard deviation 
    (which = "std"). Use either mean or median depending on "stat" arg. 
    '''
    # Support single estdata input for compatibility with earlier code.
    if type(estdatas) is not list: estdatas = [estdatas]
    
    label = LONG[which].replace("*avgtype*", stat)
    if stat == "mean" and which=="RMSE":
        label = label.replace("mean", "root mean squared")
        
    if xrg is not None and lims is not None and xrg[0] < lims[0][0]:
        xrg[0] = lims[0][0] 

    ax = get_logplot(ylabel = label.capitalize(), title = title)
    for estdata in estdatas:
        Nq_dict, lb_dict, err_dict, std_dict = estdata.unpack_data()
        dxs, dys = estdata.error_bars()
        if which=="RMSE":
            y_dict = err_dict
        elif which=="std":
            y_dict = std_dict
        if len(y_dict)==0:
            # print(f"> No {stat} data to plot for {which}. [plot_err_evol]")
            return 
        id, xrg2 = plot_error_scatter(Nq_dict, y_dict, ax, iconpath, 
                                dxs = dxs, dys = dys, errdisplay = errdisplay,
                                xrg = xrg)
    
    if len(estdatas) == 1 and plotlims:
        assert not estdata.is_empty()
        # Plot the SQL and HL, unless there are several datasets.
        plot_limits(Nq_dict, y_dict, ax, yintercept, label, xrg = xrg2)
        
    if CRbounds:
        # Plot Cramér-Rao lower bounds for the estimation error.
        plot_CR_bounds(Nq_dict, lb_dict, ax, plot_fit)
        
    arrange_legend(sort = False)
    
    if lims is not None:
        xlim, ylim = lims
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    plt.tight_layout()
    #print("> I would like to let you know I am plotting a figure."
    #      " [plot_err_vs_Nq]")
    
    return id #ax.get_xlim(), ax.get_ylim()

def arrange_legend(sort):
    if sort: 
        handles, labels = plt.gca().get_legend_handles_labels()
        print(handles, labels)
        sorted_handles_labels = sorted(zip(labels, handles), key=lambda x: x[0])
        sorted_labels, sorted_handles = zip(*sorted_handles_labels)
        plt.legend(sorted_handles, sorted_labels, loc="lower left", 
                    fontsize=12, framealpha=0.8)
    else:
        plt.legend(loc="lower left", fontsize=20, framealpha=0.8)
    
def get_logplot(ylabel, title = None, return_fig = False):
    fig, ax = plt.subplots(1,figsize=(10,6))
    
    #title = ("Scaling of the estimation error in a with the number of "
    #    "queries to A")
    xlabel = "Cumulative evolution time"
    FONTSIZE = 28
    SMALLERSIZE = 18
    
    if title is not None:
        ax.set_title(title, fontsize=FONTSIZE, pad=25)

    ax.set_xlabel(xlabel, fontsize=FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    
    plt.xscale('log'); plt.yscale('log')
    
    plt.xticks(fontsize=SMALLERSIZE)
    plt.yticks(fontsize=SMALLERSIZE)
    ax.tick_params(labelsize=SMALLERSIZE, length=15, width=2.5) 
    ax.tick_params(which = 'minor', labelsize=SMALLERSIZE, length=8, width=1.25) 

    from matplotlib.ticker import LogLocator
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=10))
        
    ax.spines['right'].set_color('lightgray')
    ax.spines['top'].set_color('lightgray')
    
    ax.grid(which='both', linestyle='--', linewidth=0.5)
    
    if return_fig:
        return fig, ax
    return ax

def plot_error_scatter(Nq_dict, err_dict, ax, iconpath = None, xrg = None,
                       dxs = None, dys = None, errdisplay = "shaded"):
    assert errdisplay in ["bars", "shaded"]
    def getImage(path):
        return OffsetImage(plt.imread(path, format="png"), 
                           zoom=0.5 if iconpath=="jface.png" else 0.05)
     
    xrg = (0, None) if xrg is None else xrg
    Nqmin, Nqmax = xrg
    # id describes the algorithm(s), for e.g. naming figures.
    id = "_".join(Nq_dict.keys())
    for key in sorted(err_dict.keys(), reverse = True):
        x, y = Nq_dict[key], err_dict[key]
        
        first_i = next(i for i,v in enumerate(x) if v >= Nqmin)
        
        if Nqmax and x[-1] > Nqmax:
            # Without the second condition, we last_i = 0 if no elements <= max.
            last_i = -next(i for i,v in enumerate(reversed(x)) if v <= Nqmax)
        else:
            last_i = len(x)

        x, y = x[first_i:last_i], y[first_i:last_i]
        
        if iconpath is not None:
            # Replace markers with a specific icon, and do not caption it.
            for xi, yi in zip(x, y):
               ab = AnnotationBbox(getImage(iconpath), (xi, yi), frameon=False)
               ax.add_artist(ab)
            continue
        
        key = fix_key(key)
        kwargs = plot_kwargs(key)
        dxs = dxs[key][first_i:last_i]
        dys = dys[key][first_i:last_i]

        if errdisplay == "bars":
            errorbar_plot(ax, x, y, dxs, dys, **kwargs)
        if errdisplay == "shaded":
            shaded_plot(ax, x, y, dys, **kwargs)
    xrg = (x[0], x[-1])
    return id, xrg

def plot_kwargs(key):
    return {"fmt": MARKER_SHAPES[key],
            #"markersize": MARKER_SIZES[key]/20,
            "color": MARKER_COLORS[key],
            "label": key}

def left_minor_tick(x0, steps=2):
    # Get minor tick steps steps "to the left of x0.
    ticks = [10**n * i for n in range(-10, 10) for i in range(1, 10)]
    ticks = sorted(t for t in ticks if t < x0)
    return ticks[-steps] if len(ticks) >= steps else None

def errorbar_plot(ax, x, y, dxs, dys, xmin = None, **kwargs):
    ax.errorbar(x, y, xerr = dxs, yerr = dys, markeredgecolor = 'black', 
                markeredgewidth = 0.75, elinewidth=0.75, capsize=3, ecolor = 'black',
                **kwargs)
    brute = False
    # ax.set_xlim(1e0, None)
    # return
    if brute: 
        # Noisy. 
        ax.set_xlim(8e2, 1.2e6)
        ax.set_ylim(1e-5, 5e-1)
        return 
    
    # To avoid empty space to the left?... Not sure why it happens
    
    xmax = ax.get_xlim()[1]
    if xmin is None: 
        # ax.set_xlim(left=max(0.1, 2*dxs[0]), right=xmax)
        ax.set_xlim(left=left_minor_tick(x[0]-dxs[0]), right=xmax)
    else: 
        ax.set_xlim(left=xmin, right=xmax)

def shaded_plot(ax, x, y, dys, **kwargs):
    fmt = kwargs.pop("fmt")
    ax.plot(x, y, fmt, **kwargs, markeredgecolor='black', markeredgewidth=0.75,
            linestyle='--', linewidth=1)

    y_lower = y - dys
    y_upper = y + dys

    ax.fill_between(
        x,
        y_lower,
        y_upper,
        color=kwargs['color'],      
        alpha=0.2,           
        edgecolor='none'    
    )

MARKER_SHAPES = {'PGH': 's',
                 'SH': 'v',
                 'WES': '8',
                 'aWES': '8',
                 'RTS': 'd',}
     
MARKER_COLORS = {'PGH': 'firebrick',
                 'SH': 'orange',
                 'WES': 'navy',
                 'aWES': '#008080',
                 'RTS': 'lightskyblue'}

MARKER_SIZES = {'PGH': 82,
                'SH': 120,
                'WES': 80,
                'aWES': 80,
                 'RTS': 110}

def fix_key(key):
    '''
    For earlier datasets where label wasn't altered to WES. 
    '''
    if key=="WES":
        return "WES"
    elif key=="aWES":
        return "aWES"
    else:
        return key
    
def plot_CR_bounds(Nq_dict, lb_dict, ax, plot_fit):
    if len(lb_dict)==0:
        print("> I don't have any Cramer Rao bound evaluations to plot! "
              "[plot_CR_bounds]")
        return
    
    for key in sorted(lb_dict.keys(), reverse = True):
        x, y = Nq_dict[key], lb_dict[key]
        color = MARKER_COLORS.get(key, "indianred")
        ax.plot(x, y, linewidth=2, color=color, linestyle="-.",
                label=f"Cramér-Rao ({key})")
        
        m, b = power_fit(x,y, seq=key)
            
        if plot_fit:
            yfit = match_intercept(x, y, m)
            ax.plot(x, yfit, linewidth=1.5, linestyle="dashed",  
                    color="black", label=f"O(Nq^{round(m,2)})")
                
def plot_limits(Nq_dict, err_dict, ax, yintercept, label, xrg = None):
    '''
    Plot the standard quantum and Heisenberg limits overposed with the data, 
    which describe the scaling of y (RMSE) wrt x (number of queries).
    
    These limits are represented by exponential decay in a linear graph / 
    straight lines in a loglog graph, but they only determine the power / slope 
    respectively. This leaves an undetermined parameter: a constant factor (B)
    / y intercept (b):
        
                 y = B*x^m <-> log(y) = m*log(x)+log(B)   (B = e^b)
    
    Thus, we must choose this parameter to fully specify a graphical 
    representation. Two options are provided:
    - yintercept == "fit": fit a log(y)=m*log(x)+b to the data, use 'b' to 
    adjust the limit lines' f(x0). 
    - yintercept == "1st": make the limit lines pass by the first datapoint 
    (x0, y0).
    '''
    bounds = ['sql','hl']
    # Draw SQL and HL as a function of x. Since they produce straight 
    # lines, the list of x coords only needs to cover the graph's width. 
    # We can take the largest Nq from the (Nq, epsilon) points; if there are 
    # several datasets, we pick the one that reaches a larger N_q (e.g. for 
    # LIS vs EIS, they're generally slightly different).    
    keys = list(err_dict.keys())
    xmaxs = [err_dict[key][-1] for key in keys]
    ref_key = keys[np.argmax(xmaxs)]
    if len(Nq_dict[ref_key]) <= 1:
        return
    
    if yintercept == "fit":
        # Determine y intercept using dataset spanning largest x-axis section.
        # print(f"> Fitting parameters for {ref_key} (to be used as reference)...")
        m, b = power_fit(Nq_dict[ref_key], err_dict[ref_key], 
                         f"{ref_key} {label}")
    
    # Do fits for other datasets if they exist, to print the fit parameters.
    for key in keys:
        if key!= ref_key:
            # print(f"> Fitting parameters for {key}...")
            power_fit(Nq_dict[key], err_dict[key], f"{key} {label}")
    
    for bound in bounds:
        power = -0.5 if bound=="sql" else -1
        xrg = Nq_dict[key] if xrg is None else xrg
        xs = logspace(xrg[0], xrg[-1], 1000)
        
        if yintercept == "fit":
            y0 = xs[0]**m*np.exp(b)
        if yintercept=="1st":
            y0 = err_dict[key][0]
        ys = match_intercept(xs, y0, power)
        
        ax.plot(xs, ys, 
                linewidth=3 if bound=="sql" else 2, 
                linestyle=":" if bound=="sql" else "--", 
                color="cadetblue" if bound=="sql" else"indianred", 
                label="Standard quantum limit" if bound=="sql" 
                    else "Heisenberg limit")
    
def power_fit(xs, ys, label = "", seq=None):
    '''
    Fit the parameters 'm' and 'b' in: 
                         y = x^m*exp(b) = B*x^m, B := e^b 
    Or equivalently:
                               log(y)=m*log(x)+b
    '''
    lower_bounds = [-1, -np.inf]  # m >= -1, b unbounded
    upper_bounds = [np.inf, np.inf]
    cf = opt.curve_fit(lambda x, m, b: m*x+b,  
                                np.log(xs),  np.log(ys),
                                bounds = (lower_bounds,
                                          upper_bounds))
    m, b = cf[0] 
    
    label = "RMSE" if len(label) == 0 else label
    # label = label.capitalize()
    print(f"> {label} = O(Nq^{round(m,2)})", end = ";")

    print(f" offset = {round(b, 2)}.")
    if seq is not None:
        m_pred = -0.75 if seq=="LIS" else -1
        print(" This should be compared with the theoretical prediction "
              f"CR=O(Nq^{m_pred}) for {seq} [power_fit]. ")
    return m, b

def match_intercept(xs, y0, m):
    # Match coordinate of a y=x^m*B function with the (xs,ys) data at y 
    # intercept by adjusting B (preserve only scaling/slope).
    b_ = np.log(y0) - m*np.log(xs[0])
    # Convert back to normal scale for plotting. The scale can be converted 
    # into loglog after if desired.
    ys_ = [np.exp(m*np.log(x) + b_) for x in xs]
    return ys_  

def plot_graph(xs, ys, startat0 = None, title="", xlabel="", ylabel=""):
    fig, ax = plt.subplots(1,figsize=(10,6))
    ax.plot(xs, ys, linewidth=1, color="black")
        
    if startat0=="x" or startat0=="both":
        ax.set_xlim(left=0)
    if startat0=="y"or startat0=="both":
        ax.set_ylim(bottom=0)
    if startat0 is None:
        ax.set_xlim(left=min(xs))
        ax.set_ylim(bottom=min(ys))
    
    ax.set_xlim(right=max(xs))
    
    ax.set_title(title, fontsize=18, pad=25)
    ax.set_xlabel(xlabel, fontsize=16, style="italic", labelpad=10)
    ax.set_ylabel(ylabel, fontsize=16, style="italic", labelpad=10)
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    if startat0=="x" or startat0=="both":
        # Avoid the ugly whitespace around the yaxis.
        fill = ax.fill_between([0]+xs, [ys[0]]+ys, alpha = 0.5)
    else:
        fill = ax.fill_between(xs, ys, y2=ax.get_ylim()[0], alpha = 0.5)
    fill.set_facecolors('darkgray')
    fill.set_edgecolors('darkgray')
    
    ax.spines['right'].set_color('lightgray')
    ax.spines['top'].set_color('lightgray')
    
    ax.xaxis.set_major_locator(MaxNLocator())
    ax.yaxis.set_major_locator(MaxNLocator())
    
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    
    ax.grid()
    ax.grid(which='minor', alpha=0.2, linestyle='--')
    ax.grid(which='major', alpha=0.8)
    plt.tight_layout()
    plt.show()
    safe_save_fig("graph")

def barplot_from_data(ddict, ax, title = None, label = None):
    '''
    Plot a bar plot with x ticks and labels given by the keys in 'ddict',
    and bar heights given by the corresponding normalized values.
    '''
    keys = list(ddict.keys())
    vals = list(ddict.values())

    Z = sum(vals)
    # Normalize the relative frequencies for the histogram.
    for key in keys:
        ddict[key] = ddict[key]/Z
        
    # If the keys are strings, they're assumed to be binary numbers.
    if isinstance(keys[0], str):
        # Decimal keys to be used for calculations. For labeling keep binary.
        dkeys = [int(key, 2) for key in keys]
        # Adapt the width to the x span of the graph, or else overthin bars.
        width = (max(dkeys)-1)/10
    else:
        width = 0.1
        dkeys = keys
        
    ax.bar(keys, ddict.values(), width = width, color = 'lightgray', label=label)
        
    ax.set_xlim((min(dkeys)-width, max(dkeys)+width))
    ax.set_xticks(dkeys)
    ax.set_xticklabels(keys)
    
    if title is not None:
        plt.title(title, fontsize=14, pad=15)
        
def plot_warn(f):
    def wrapper(*args, **kwargs):
        Ngraphs = len(args[1])
        if Ngraphs <= 10: 
            return f(*args, **kwargs)
        ans = ""; 
        while ans!="Y" and ans!="N":
            ans = input("\n> This is going to plot over 10 graphs. "
                        f"More specifically, {Ngraphs} graphs. "
                        "Are you sure you want that?"\
                        f" [{f.__qualname__ }]\n(Y/N)\n")
        if ans=="Y":
            return f(*args, **kwargs)
    return wrapper

