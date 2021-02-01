import numpy as np
import torch as th

import configuration as cfg

from helper_functions import sprint

def train_batch( net,
            data_filenames,
            criterion,
            optimizer,
            batch_iter,
            params,
            tensors):
    mse = None
    th.autograd.set_detect_anomaly(True)

    batch_size = cfg.BATCH_SIZE
    seq_len = cfg.SEQ_LEN
    amount_pks = cfg.PK_ROWS * cfg.PK_COLS

    # Generate the training data batch for this iteration
    net_input, net_label = _set_up_batch(batch_iter, data_filenames)


    # Set up an array of zeros to store the network outputs
    net_outputs = th.zeros(size=(batch_size,
                                 seq_len,
                                 amount_pks,
                                 params.pk_dyn_out_size))

    if optimizer:
        # Set the gradients back to zero
        optimizer.zero_grad()

    # Reset the network to clear the previous sequence
    net.reset()

    # Iterate over the whole sequence of the training example and perform a
    # forward pass
    for t in range(seq_len):
        # Teacher forcing

        # Set the dynamic input for this iteration
        dyn_net_in_step = net_input[:,t, :, :params.pk_dyn_out_size]

        # Forward the input through the network
        net.forward(dyn_in=dyn_net_in_step)

        # Store the output of the network for this sequence step
        net_outputs[:,t] = tensors.pk_dyn_out

    if criterion:
        # Get the mean squared error from the evaluation list
        # (8, 40, 256, 1)

        mse = criterion(net_outputs, th.from_numpy(net_label))
        # Alternatively, the mse can be calculated 'manually'
        # mse = th.mean(th.pow(net_outputs - th.from_numpy(net_label), 2))

        if optimizer:
            mse.backward()
            optimizer.step()

    return mse, net_input, net_label, net_outputs


def validate():
    
    return 0


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



    return _net_input, _net_label

