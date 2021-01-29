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

    data = np.load(data_filenames[first_sample])[:cfg.SEQ_LEN + 1]
    # Expand Dim for batch 
    data = data[np.newaxis, :]

    for file in data_filenames[first_sample+1:last_sample_excl]:
        data_file = np.load(file)[:cfg.SEQ_LEN + 1]
        # Expand Dim for batch 
        data_file = data_file[np.newaxis, :]
        data = np.append(data, data_file, axis=0)

    
    # shape: ( cfg.BATCH_SIZE , 41, 2, 16, 16)

    # Get first and second dimension of data
    dim0, dim1, dim2 = np.shape(data)[:3]

    # Reshape the data array to have the kernels on one dimension
    data = np.reshape(data, [dim0, dim1, dim2, cfg.PK_ROWS * cfg.PK_COLS])

    # Swap the third and fourth dimension of the data
    data = np.swapaxes(data, axis1=2, axis2=3)
    # (8, 41, 256, 2)

    # Split the data into inputs (where some noise is added) and labels
    # Add noise to all timesteps except very last one
    _net_input = np.array(
        data[:,:-1] + np.random.normal(0, cfg.DATA_NOISE, np.shape(data[:,:-1])),
        dtype=np.float32
    )

    _net_label = np.array(data[:,1:, :, 0:1], dtype=np.float32)

    if cfg.TRAINING:
        # Set the dynamic inputs with a certain probability to zero to force
        # the network to use lateral connections
        _net_input *= np.array(
            np.random.binomial(n=1, p=1 - cfg.P_ZERO_INPUT,
                               size=_net_input.shape),
            dtype=np.float32
        )

    sprint(_net_input, "_net_input")
    sprint(_net_label, "_net_label", exit=True)
    return _net_input, _net_label




def validate():
    mse = 1


    return mse