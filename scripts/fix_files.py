'''
Fixing old data files containing ExecutionData instances that were defined when
the module names were different.
'''
import sys 
import pickle
import src.utils.mydataclasses
from scripts.joint_plot import dataset_filenames_from_folder
from src.utils.files import data_from_file

def fix_datafiles(folder,  old_names, new_modules):
    '''
    Fixes pickled data files in a subfolder of the 'datasets' folder when the 
    module name has changed from old_name. new_module is the new module (not 
    str) and should be imported before. The changes are saved. 
    '''
    print(f"> Will try to fix data files in folder {folder}:")
    for old_name, new_module in zip(old_names, new_modules):
        sys.modules[old_name] = new_module 
        print(f"> {old_name} -> {new_module.__name__}")

    paths = dataset_filenames_from_folder(folder)
    for path in paths:
        data = data_from_file(path)
        with open(path, 'wb') as filehandle:
            pickle.dump(data, filehandle)
    print(f"> Fixed files.")

folder = "noisy" # "noisy"
old_names = ["src.utils.dataclasses", "utils.dataclasses", "utils"] 
new_modules = [src.utils.mydataclasses, src.utils.mydataclasses, src.utils] 
fix_datafiles(folder, old_names, new_modules)
