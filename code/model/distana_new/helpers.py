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
    first_sample = cfg.BATCH_SIZE * batch_iter
    # Handling also last batch
    last_sample_excl = min(first_sample + cfg.BATCH_SIZE, len(data_filenames)) 

    data = np.load(data_filenames[first_sample:last_sample_excl])[:cfg.SEQ_LEN + 1]

    sprint(data, "data", exit=True)




def validate():
    mse = 1


    return mse