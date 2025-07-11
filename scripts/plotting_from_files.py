"""
Plotting datasets from src/datasets/<folder> together in a graph.
"""
import os
import re
import matplotlib.pyplot as plt

from src.algorithms.WES import fix_aWES_label, change_execdata_label
from src.utils.plotting import plot_err_evol
from src.utils.processing import process, safe_save_fig
from src.utils.files import data_from_file, save_as
from src.utils.mydataclasses import get_label, EstimationData

PROCESSING = {'SH': 'binning',
              'PGH': 'binning',
              'WES': 'binning',
              'aWES': 'binning',
              'RTS': 'binning'}

def plot_from_folder(folder, separate, stats, errdisplay, xrg, axlims, 
                     silent = False, save = False):
    '''
    Plot together all the datasets in the folder 'folder' of the folder 
    'datasets'.
    '''
    print(f"> Will plot frequency estimation results from folder '{folder}'.")
    fnlist = dataset_filenames_from_folder(folder)
    shargs = (stats, errdisplay, xrg, axlims, silent, save)
    if separate: 
        for filename in fnlist: 
            plot_estdatas([filename], *shargs)
    else:
        plot_estdatas(fnlist, *shargs)

def plot_estdatas(fnlist, stats, errdisplay, xrg, axlims, silent, save):
    for stat in stats:
        estdatas = get_estdatas(fnlist, stat, silent)
        plot_err_evol("RMSE", estdatas, stat, errdisplay = errdisplay, 
                      xrg = xrg, lims = axlims)
        if save:
            safe_save_fig(f"joint_{stat}_")
        plt.show()

def dataset_filenames_from_folder(folder_name, silent = False):
    '''
    Returns a list of the filenames for all files in the subfolder 'folder_name'
    of the folder 'datasets'.
    '''
    folder_path = os.path.join(os.getcwd(), "datasets", folder_name)
    filenames = [file for file in os.listdir(folder_path)]
    full_paths = [os.path.join(folder_path, filename) for filename in filenames]
    return full_paths

def get_estdatas(filename_list, stat, silent = False, print_maxs = False):
    '''
    Get estimation data objects from the execution data objects in the files
    and process them.
    '''
    execdatas = [data_from_file(filename, silent) for filename in filename_list]
    labels = [get_label(execdata) for execdata in execdatas]

    '''
    for label, exd in zip(labels, execdatas):
        if label == "BQAE":
            change_execdata_label(exd, "BQAE", "WES")
            r = fix_aWES_label(exd)
            exd.save_to_file()
        if label == "sigma":
            change_execdata_label(exd, "sigma", "SH")
            exd.save_to_file()
        if label == "random":
            change_execdata_label(exd, "random", "RTS")
            exd.save_to_file()
        # exd.save_to_file() to save new (separately); otherwise just this run.
    '''

    labels = [get_label(execdata) for execdata in execdatas]
    estdatas = [execdata.estdata for execdata in execdatas]

    if print_maxs:
        maxctrl = 0
        for estdata in estdatas: 
            key = list(estdata.Nq_dict.keys())[0]
            Nqs = estdata.Nq_dict[key]
            imax = Nqs.index(max(Nqs))
            Nqmax = Nqs[imax]
            prev = Nqs[imax - 1] 
            print(f"> Max control for {key}: ", Nqmax - prev)

    estdatas = [process(estdata, stat, how = PROCESSING[label])
               for label, estdata in zip (labels, estdatas)]
    return estdatas

def join_datafiles_from_folder(folder_name):
    folder_path = os.path.join(os.getcwd(), folder_name)
    filenames = [file for file in os.listdir(folder_path)]
    paths = [os.path.join(folder_path, filename) for filename in filenames]

    estdatas = []
    for path in paths: 
        estdatas.append(data_from_file(path).estdata)

    combined_dataset = EstimationData.join(estdatas, silent = False)

    nruns = 0 
    for name in filenames: 
        c = re.search(r"nruns=(\d+),", name)
        nruns += int(c.group(1))

    combined_execdata = data_from_file(paths[0])
    combined_execdata.estdata = combined_dataset
    combined_execdata.nruns = nruns
    save_as(combined_execdata, "combined_" + combined_execdata.filename())
    return combined_dataset

def crop_datafiles_from_folder(foldername, Nqmin, Nqmax):
    '''
    Remove Nq, err, std data for which Nq falls outside of [Nqmin, Nqmax].
    The point is matching different datasets to the same range. 
    Saves new datasets to datasets/`previous_folder_name`_cropped
    '''
    folder_path = os.path.join(os.getcwd(), "datasets", foldername)
    filenames = [file for file in os.listdir(folder_path)]

    estdatas = []
    for filename in filenames:
        path = os.path.join(folder_path, filename)
        execdata = data_from_file(path)
        estdata = execdata.estdata
        estdata.Nq_range()
        estdata.crop_data(Nqmin, Nqmax)
        estdata.Nq_range()
        execdata.estdata = estdata 
        new_path = os.path.join(os.getcwd(), "datasets", foldername + "_cropped", filename)
        save_as(execdata, new_path)
    

def plot_estimation_from_files(noise, separate):
    # join_datafiles_from_folder("ESS")
    # crop_datafiles_from_folder("noisy", 10e2, 2e6)
    if not noise: 
        folder = "noiseless"
        xrg = (2e2, 1e7) #if not separate else None
        axlims = ((2e2, 1e7), (4e-7, 4e-1))
    else:
         folder = "noisy"
         xrg = (1e3, 1e6) #if not separate else None
         axlims = ((8e2, 1.2e6), (5e-5, 1e-1))
         
    errdisplay = "bars" if separate else "shaded"
    stats = ["mean"]
    # folder = "single"
    plot_from_folder(folder, separate, stats, errdisplay, xrg, axlims,
                         save = False, silent = True)

if __name__ == "__main__":
    plot_estimation_from_files(noise = True, separate = False)