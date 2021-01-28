import numpy as np
import configuration as cfg
from helper_functions import sprint

def train_batch( net,
            data_filenames,
            criterion,
            optimizer,
            batch_iter):
    mse = 1

    _set_up_batch(batch_iter, data_filenames)

    return mse


def _set_up_batch(batch_iter, data_filenames):
    """
    0123
    4567
    89
    """
    first_sample = cfg.BATCH_SIZE * _iter
    # Handling also last batch
    last_sample = min(first_sample + cfg.Batch_Size -1, len(data_filenames))
    
    sprint(data_filenames,"data_filenames", exit=True)




def validate():
    mse = 1


    return mse