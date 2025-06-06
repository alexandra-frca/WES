"""
Plotting datasets from src/datasets/<folder> together in a graph.
"""
import os
import matplotlib.pyplot as plt

from src.algorithms.BAE import fix_aBAE_label
from src.utils.plotting import plot_err_evol
from src.utils.processing import process, safe_save_fig
from src.utils.files import data_from_file
from src.utils.mydataclasses import get_label

PROCESSING = {'classical': 'none',
                'sigma': 'binning',
                'BAE': 'binning'}

def plot_from_folder(folder, stats, silent = False, save = False):
    '''
    Plot together all the datasets in the folder 'folder' of the folder 
    'datasets'.
    '''
    print(f"> Will plot QAE estimation results from folder '{folder}'.")
    fnlist = dataset_filenames_from_folder(folder)
    for stat in stats:
        estdatas = get_estdatas(fnlist, stat, silent)
        plot_err_evol("RMSE", estdatas, stat)
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

def get_estdatas(filename_list, stat, silent = False):
    '''
    Get estimation data objects from the execution data objects in the files
    and process them.
    '''
    execdatas = [data_from_file(filename, silent) for filename in filename_list]
    labels = [get_label(execdata) for execdata in execdatas]

    for label, exd in zip(labels, execdatas):
        if label == "BAE":
            fix_aBAE_label(exd)

    estdatas = [execdata.estdata for execdata in execdatas]
    estdatas = [process(estdata, stat, how = PROCESSING[label])
               for label, estdata in zip (labels, estdatas)]
    return estdatas

# Previously, "canonical" had "none" processing because it was pre-processed.
if __name__ == "__main__":
    # plot_from_folder("noiseless", stats = ["mean", "median"])
    # plot_from_folder("noisy", stats = ["mean", "median"])
    # plot_from_folder("noiseless_1934", stats = ["mean", "median"])
    plot_from_folder("ESS", stats = ["mean", "median"])

