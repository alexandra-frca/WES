from src.utils.files import data_from_file
from src.utils.misc import estimation_errors, thin_list

def process(raw_estdata, stat, how = "binning"):
    assert how in ["binning", "averaging", "averaging2", "none"], how
    if how == "binning":
        from src.utils.binning import process_raw_estdata
        try:
            estdata = process_raw_estdata(raw_estdata, stat = stat)
        except ValueError as e:
            print("> Could not bin due to error:")
            print(e)
            estdata = raw_estdata
    if how == "averaging":
        estdata = process_nonadapt_data(raw_estdata, stat = stat)
    if how == "averaging2":
        estdata = process_nonadapt_data(raw_estdata, by_step = True, stat = stat)
    if how == "none":
        estdata = raw_estdata
    return estdata

def process_nonadapt_data(raw_estdata, stat, by_step = False, every = 2):
    '''
    by_step: whether the data is ordered already by step (each element is a list
    of errors for multiple runs for a fixed step/Nq) or not (each element is a
    list of errors foor multiple steps for a fixed run).
    '''
    # Same x values across all runs. 
    keys = list(raw_estdata.Nq_dict.keys())
    estdata = deepcopy(raw_estdata)
    
    for key in keys:
        sqe_list = raw_estdata.err_dict[key]
        err_per_step = estimation_errors(sqe_list, stat = stat, 
                                         by_step = by_step)
        estdata.err_dict[key] = thin_list(err_per_step, 1, 5) # err_per_step
        estdata.Nq_dict[key] = thin_list(estdata.Nq_dict[key], 1, 5)
    
    return estdata

def safe_save_fig(filename):
    timestamp = datetime.now(pytz.timezone('Portugal')).strftime("%d_%m_%Y_%H_%M")
    filename = filename + "_" + timestamp

    i = 0
    while os.path.exists(f'{filename}({i}).png'):
        i += 1

    filename = f'{filename}({i}).png'
    plt.savefig(filename)
    print(f"> Saved figure '{filename}'.")