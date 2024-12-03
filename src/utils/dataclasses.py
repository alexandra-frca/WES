# -*- coding: utf-8 -*-
'''
   ============================================================================
   
    For storing dataclasses, namely respecting measurement data (for 
    inference), estimation data (errors,...) and execution data (numbers of 
    shots, real parameter,...).
    
   ============================================================================
'''
import numpy as np
import random
from copy import deepcopy
from datetime import datetime
import pytz
import inspect
import re
from dataclasses import dataclass

from src.utils.misc import b10str, Iterator
from src.utils.files import data_from_file, save_as
print("1707")
@dataclass
class QAEspecs:
    n: int
    Nsol: int
    m: int = None
    
    def N(self):
        return 2**self.n
    
    def M(self):
        return 2**self.m
    
    def a(self):
        return self.Nsol/2**self.n
    
    def theta(self):
        a = self.a()
        return np.arcsin(a**0.5)
    
    def x0(self):
        M = self.M()
        theta = self.theta()
        return M*theta/np.pi
    
    def closest_outcomes(self, bstr = True):
        M = self.M()
        x0 = self.x0()
        out1 = round(x0)
        out2 = round(M-x0)
        if out2 >= M:
            # There aren't 2 most likely outcomes, but only one: the
            # probability shrinks with the distance to the real value. So if 
            # the closest integer to M-x0 is out of reach for the binary 
            # representation, it will have a smaller probability than x0. So
            # consider only the outcome associated with x0.
            out2 = "-"
        if bstr:
            # Convert to binary strings, remove '0b' prefix.
            out1 = bin(out1)[2:].zfill(self.m)
            out2 = out2 if out2=="-" else bin(out2)[2:].zfill(self.m)

        return out1, out2
    
    def estimated_thetas(self):
        M = self.M()
        out1, out2 = self.closest_outcomes(bstr = False)
        # The most likely outcomes are the closest integers to x0 and to M-x0.
        th1 = np.pi*out1/M
        th2 = "-" if out2=="-" else np.pi-np.pi*out2/M
        return th1, th2
    
    def estimated_a(self):
        th1, th2 = self.estimated_thetas()
        a1 = np.sin(th1)**2
        a2 = "-" if th2=="-" else np.sin(th2)**2
        if a2!="-":
            assert np.isclose(a1, a2), \
                    "the two theta estimates do not produce the same a."
        return a1
    
    def estimated_Nsol(self):
        N = self.N()
        est_a = self.estimated_a()
        return est_a*N

class EstimationData(): 
    '''
    Example EstimationData object:
    
    estdata.Nqs = {'LIS': [1, 2, 3], 'adaptive': [1, 2, 3]}
    estdata.lbs = {'LIS': [1, 2, 3]}
    estdata.errs = {'LIS': [1, 2, 3], 'adaptive': [1, 2, 3]}
    estdata.warmup = {'adaptive': [(100, 1)]}
    
    Note that not all data categories contain all labels necessarily, 
    except Nq_dict.
    
    Also, warmup is a list with (Nq, sqe**0.5) tuples, currently a single one.
    '''
    def __init__(self):
        self.Nq_dict = {}
        self.lb_dict = {}
        self.err_dict = {}
        self.std_dict = {}
        self.warmup_dict = {}
        
    def is_empty(self):
        empty = True
        attribute_info = inspect.getmembers(self, 
                                            lambda a:not(inspect.isroutine(a)))
        attribute_info = [a for a in attribute_info 
                      if not(a[0].startswith('__') and a[0].endswith('__'))]
        for attr_str, _ in attribute_info:
            if len(getattr(self, attr_str))>0:
                empty = False
                
        return empty
        
    def add_data(self, key, Nqs = None, lbs = None, errs = None, stds = None,
                 warmup = None):
        if Nqs is not None:
            self.Nq_dict[key] = Nqs
        if lbs is not None:
            self.lb_dict[key] = lbs
        if errs is not None:
            self.err_dict[key] = errs
        if stds is not None:
            self.std_dict[key] = stds
        if warmup is not None:
            self.warmup_dict[key] = warmup
            
    def append_data(self, key, Nqs = None, lbs = None, errs = None, stds = None,
                    warmup = None):
        if Nqs is not None:
            self.Nq_dict[key] += Nqs
        if lbs is not None:
            self.lb_dict[key] += lbs
        if errs is not None:
            self.err_dict[key] += errs
        if stds is not None:
            self.std_dict[key] += stds
        if warmup is not None:
            self.warmup_dict[key] += warmup
        
    def unpack_data(self):
        return self.Nq_dict, self.lb_dict, self.err_dict, self.std_dict
    
    def get_attribute_info(self):
        attribute_info = inspect.getmembers(self, lambda a:not(inspect.isroutine(a)))
        attribute_info = [a for a in attribute_info 
                      if not(a[0].startswith('__') and a[0].endswith('__'))]
        return attribute_info
    
    @staticmethod
    def join(estdata_list, silent = True):
        '''
        Combine the estimation data from the estdata in 'estdata_list', by
        going through:
        - Each object 'estdata_i' - an ensemble of dicts.
        -- Each of its^ attributes 'estdata_i.datacat_j' - a dict containing 1 
           category of data from 'estdata_i'.
        --- Each of its^ entries 'estdata_i.datacat_j[datalabel_k]' - a 
            (key = datalabel_k, value = list_ijk) pair, with the data in  
            category 'datacat_j' for strategy 'datalabel'.
        ---> Add 'list_ijk' to joint_estdata.datacat[datalabel_k], the result.
        !! Exception: warmup[label] should be a single tuple, not a list. !!
        '''
        
        joint_estdata = EstimationData()
        for estdata in estdata_list: 
            attribute_info = estdata.get_attribute_info()
            for attr_str, attr_obj in attribute_info: 
                receptor = getattr(joint_estdata, attr_str)
                for datalabel in attr_obj:
                    try:
                        receptor[datalabel] += deepcopy(attr_obj[datalabel])
                    except KeyError:
                        # List doesn't exist yet, create.
                        receptor[datalabel] = deepcopy(attr_obj[datalabel])
                        
        # Join the warmup data together.
        joint_estdata.warmup_dict = EstimationData.condense_warmup_dict(
            joint_estdata.warmup_dict)
        
        if not silent:
            print(f"> Combined {len(estdata_list)} datasets into the following:")
            print(joint_estdata)
            # print("> Previous datasets:")
            # for estdata in estdata_list:
            #     print(estdata)
        
        return joint_estdata
    
    @staticmethod
    def condense_warmup_dict(warmup_dict):
        for datalabel in warmup_dict:
            warmup_dict[datalabel] = [EstimationData.combine_warmup_tuples(
                warmup_dict[datalabel])]
        return warmup_dict
                
    @staticmethod
    def combine_warmup_tuples(warmup_tuples):
        # The number of shots should be common. 
        warmup_Nqs = [Nq for Nq, _ in warmup_tuples]
        if warmup_Nqs.count(warmup_Nqs[0]) != len(warmup_Nqs):
            # All elements are not the same.
            print("> Nqs for different datasets don't match. Quitting.")
            return 
        
        # Each 2nd tuple element of 'warmup' is avg_sqe**0.5. We must square 
        # this to reconstruct the average SQE, average over averages to get the 
        # global average, and then retake the square root.
        warmup_errs = [err for Nqs, err, std in warmup_tuples]
        warmup_sqes = [err**2 for err in warmup_errs]
        warmup_err = np.mean(warmup_sqes)**0.5
        
        warmup_std = np.mean([std for Nqs, err, std in warmup_tuples])
        combined_tuple = (warmup_Nqs[0], warmup_err, warmup_std)
        return combined_tuple

    def __str__(self):
        attribute_info = self.get_attribute_info()
        
        s = "==============================================================\n"
        s += "EstimationData instance storing the following information:\n"
        
        # All datasets include an 'Nq_dict', so it necessarily contains keys 
        # for all datasets (say "LIS", "adaptive",...), unlike e.g. 'lb_dict'.
        for datalabel in self.Nq_dict:
            # Print information relative to dataset labeled 'key'.
            s += "\n* " + datalabel.capitalize() + ":\n"
            # Get the attributes, in this case dictionaries possibly with an
            # entry corresponding to 'key', automatically.
            for attr_str, attr_obj in attribute_info: 
                try:
                    data = attr_obj[datalabel]
                    s += f"- {attr_str[:-5]} ({type(data).__name__} "
                    s += f"of length {len(data)}). "
                    # if attr_str=="warmup_dict":
                    #     s += str(attr_obj['adaptive'][0])
                    s += "\n"
                except KeyError:
                    pass    
        s += "===========================###================================"
        return s
    
ex1 = EstimationData()
ex1.add_data("example1", Nqs = [1,1,1], lbs = [1, 1, 1], warmup = (1,1))
ex1.add_data("example2", Nqs = [1, 1, 1], lbs = [1, 1, 1])

ex2 = EstimationData()
ex2.add_data("example1", Nqs = [4,5,3], lbs = [5, 60, 6], errs = [1])
ex2.add_data("example2", Nqs = [5, 7, 4], lbs = [2, 5, 2])

#exj = EstimationData.join([ex1, ex2])
#print(exj) 

def join_estdata_files(filestart, indices = None, save = False):
    it = Iterator(indices)
    i = it.advance()
    datasets = []
    while True:
        filename = filestart + "#" + str(i) + ".data"
        dataset = data_from_file(filename)
        if dataset is None:
            print(f"> Number of uploaded datasets: {i}. [join_dataset_files]")
            break
        else:
            datasets.append(dataset)
            i = it.advance()
            if i <= 0: 
                # Get total number of uploaded datasets (length of indices).
                i = np.abs(i)
                break
    
    if i==0:
        print("> No datasets found. [join_dataset_files]")
        return
    elif i==1:
        print("> Found a single dataset, didn't do anything. [join_dataset_files]")
        return dataset

    combined_dataset = EstimationData.join(datasets, silent = False)
    
    if save:
        runs_each = re.search(r"(?<=runs=)[1-9]+", filestart)
        runs_each = int(runs_each.group())
        runs = runs_each*i
        # Considering runs is a single digit number in the individual files, fix with regex
        filename_stem = filestart[:-2] + str(runs) + "]" + "concat"
        save_as(combined_dataset, filename_stem)
    return combined_dataset

from typing import Union

@dataclass
class ExecutionData:
    a: float
    estdata: EstimationData
    Nruns: Union[int,str]
    Nshots: int
    label: str
    extra_info: str = None
            
    def add_field(self, fieldname, obj):
        setattr(self, fieldname, obj)
        
    def __str__(self):
        attribute_info = inspect.getmembers(self, 
                                            lambda a:not(inspect.isroutine(a)))
        attribute_info = [a for a in attribute_info 
                      if not(a[0].startswith('__') and a[0].endswith('__'))]
        s = "==============================================================\n"
        s += "ExecutionData instance storing the following information:\n"
        for name, obj in attribute_info:
            if isinstance(obj, EstimationData):
                # Exemplificative dict.
                d = obj.Nq_dict
                keys = d.keys()
                no = len(keys) 
                labels = ",".join([f"'{str(key)}' ({len(d[key])} Nqs)" 
                                   for key in keys])
                obj = f"EstimationData instance with {no} datasets: {labels}"
            s += f"*{name}: {obj}\n"
        s += "===========================###================================"
        return s
    
    def filename(self):
        # Print Nruns as an integer for BQAE, because it's slower usually less 
        # executions.
        if self.label=="BQAE":
            self.Nruns = str(self.Nruns)
        runstr = (f"Nruns={self.Nruns}" if isinstance(self.Nruns, str) 
                  else f"Nruns={b10str(self.Nruns)}")
        timestamp = datetime.now(pytz.timezone('Portugal')).strftime("%d%m%y_%H%M")
        fname = (f"{self.label}_{timestamp}_{self.a},"
                 + f"{runstr},")
        if self.label != "BQAE":
            # BQAE already prints shots in strat info.
            fname +=  f"Nshots={self.Nshots}"
        if self.extra_info is not None:
            fname += f",{self.extra_info}"
        fname += ".data"
        return fname
    
    def save_to_file(self):
        save_as(self, self.filename())
    
def get_label(execdata):
    '''
    Workaround because I saved classical AE with label 'classical_AE', SQAE 
    using formula 2 with label 'SQAE_f2', etc. Labels must correspond to the 
    ones used in the EstimationData dictionaries. So these should instead be 
    'classical', 'SQAE #2'. In the future, fix this. But can patch if working
    with older datasets.
    '''
    label = execdata.label
    split = label.split('_', 1)
    if label=="BQAE":
        label = "BAE"
    
    # Verify if the split created a single string, which also behaves as list.
    if len(split)==1:
        pass
    elif split[1]=='cher':
        label = f"{split[0]} - chernoff"
    elif split[1][0]=='f':
        label = f"{split[0]} #{split[1][1]}"
    else:
        label = split[0]
    return label

@dataclass
class MeasurementData:
    '''
    For keeping inference data.

    unfold: whether to unfold multishot measurements into single shots,
    or to allow Nshots > 1. The latter is faster, but more prone to 
    underflows when calculating log-likelihoods.
    
    '''
    ctrls: [int] = None # np.ndarray = None
    outcomes: [int] = None
    Nsshots: [int]  = None
    unfold: bool = True

    
    def __len__(self):
        l1 = len(self.ctrls)
        l2 = len(self.outcomes)
        # l3 = len(self.Nsshots)
        if l1!=l2:# or l2!=l3:
            print("> The length of the dataset controls/outcomes is "
                  "unmatched. [MeasurementData.__len__]")
            return -1
        else:
            return l1
        
    def non_classical_len(self):
        return len([c for c in self.ctrls if c != 0])
        
    def __str__(self):
        s = "==============================================================\n"
        s += "MeasurementData instance storing the following information:\n"
        s += (f"* Controls ({len(self.ctrls)}): " + str(self.ctrls) +
             f"\n* Outcomes ({len(self.outcomes)}): " + str(self.outcomes) +
             #f"\n* Numbers of shots ({len(self.Nsshots)}): " + str(self.Nsshots) +
             "\n===========================###================================")
        return s
    
    def partial_data(self, Ndata):
        assert Ndata <= len(self), (f"requested partial dataset {Ndata} exceeds"
                f" available data {len(self)}")
        return MeasurementData(self.ctrls[:Ndata], self.outcomes[:Ndata], self.Nsshots[:Ndata])

    def truncated_data(self, Ndata):
        copy = deepcopy(self)
        copy.ctrls = copy.ctrls[:Ndata]
        copy.outcomes = copy.outcomes[:Ndata]
        copy.Nsshots = copy.Nsshots[:Ndata]
        return copy
    
    def append_data(self, ctrls, outcomes, Nsshots):
        for ctrl, outcome, Nshots in zip(ctrls, outcomes, Nsshots):
            self.append_datum(ctrl, outcome, Nshots)
            
    def append_datum(self, ctrl, outcome, Nshots):
        # self.ctrls.append(ctrl)
        # self.outcomes.append(outcome)
        # self.Nsshots.append(Nshots)
        if self.ctrls is None:
            self.ctrls = []
            self.outcomes = []
            self.Nsshots = []
            
        if Nshots == 1 or not self.unfold:
            self.ctrls.append(ctrl)
            self.outcomes.append(outcome)
            self.Nsshots.append(Nshots)
        else:
            # If Nshots > 1, outcome is the number of 1 outcomes. 
            self.ctrls.extend([ctrl]*Nshots)
            outcome_list = self.get_outcomes_list(outcome, Nshots)
            self.outcomes.extend(outcome_list)
            self.Nsshots.extend([1]*Nshots)
        
    def get_outcomes_list(self, ones, Nshots):
        '''
        Given an integer number of 1 outcomes (e.g. ones = 3) and the number of
        shots (e.g. Nshots = 5), return a binary list of randomly ordered 
        outcomes (e.g. [1, 0, 1, 1, 0])..
        '''
        outcome_list = ([1 for i in range(ones)] 
                        + [0 for i in range(Nshots-ones)])
        outcome_list = random.sample(outcome_list, Nshots)
        return outcome_list
    
    def total_shots(self):
        return sum(self.Nsshots)


