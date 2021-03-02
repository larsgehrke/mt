import numpy as np
import torch as th

import configuration as cfg

from debug import sprint

import kernel_net
import kernel_variables



def train_batch (et,
            data_filenames,
            batch_iter,
            tensors,
            criterion = None ,
            optimizer = None)

    # Generate the training data batch for this iteration
    # cfg.BATCH_SIZE is ideal batch size
    # returning batch_size is real batch_size (depends on end of data)
    net_input, net_label, batch_size = _set_up_batch(data_filenames, batch_iter)



def validate(net,
            data_filenames,
            tensors,
            criterion):
    mse = None
    th.autograd.set_detect_anomaly(True)

    seq_len = cfg.SEQ_LEN
    amount_pks = cfg.PK_ROWS * cfg.PK_COLS

    # Generate the training data batch for this iteration
    # cfg.BATCH_SIZE is ideal batch size
    # returning batch_size is real batch_size (depends on end of data)
    net_input, net_label, batch_size = _set_up_batch(data_filenames, all_files = True)

    mse, _, _, _ = evaluate(net, data_filenames, )
    return 

def evaluate(net,
            tensors,
            net_input,
            batch_size,
            criterion = None ,
            optimizer = None
            ):
    mse = None
    th.autograd.set_detect_anomaly(True)

    seq_len = cfg.SEQ_LEN
    amount_pks = cfg.PK_ROWS * cfg.PK_COLS
    pk_dyn_out_size = cfg.PK_DYN_OUT_SIZE 



    # Set up an array of zeros to store the network outputs
    net_outputs = th.zeros(size=(batch_size,
                                 seq_len,                              
                                 amount_pks,
                                 pk_dyn_out_size))

    if optimizer:
        # Set the gradients back to zero
        optimizer.zero_grad()

    # Reset the network to clear the previous sequence
    net.reset(batch_size)

    # Iterate over the whole sequence of the training example and perform a
    # forward pass
    for t in range(seq_len):

        # Set the dynamic input for this iteration
        dyn_net_in_step = net_input[:, t, :, :pk_dyn_out_size]


        # Forward the input through the network
        net.forward(dyn_in=dyn_net_in_step)

        # (10, 256, 1, 1)
        # Swapping the second with third dimension
        # because time dimension is need
        # (10, 1, 256, 1)
        net_outputs[:,t] = th.transpose(tensors.pk_dyn_out, 1, 2)[:,0]

    if criterion:
        # Get the mean squared error from the evaluation list
        # (8, 40, 256, 1)

        #net_outputs: (140, 2, 256, 1)

        #net_label: (140, 2, 256, 1)

        mse = criterion(net_outputs, th.from_numpy(net_label))
        # Alternatively, the mse can be calculated 'manually'
        # mse = th.mean(th.pow(net_outputs - th.from_numpy(net_label), 2))

        if optimizer:
            mse.backward()
            optimizer.step()

    return mse, net_outputs

def calculate_loss(criterion, y, y_hat):
     mse = criterion(y, y_hat)

     return 




def _set_up_batch(data_filenames, batch_iter=0, all_files = False):
    """
    0123
    4567
    89
    """

    first_sample = cfg.BATCH_SIZE * batch_iter
    
    # Handling also last batch
    last_sample_excl = min(first_sample + cfg.BATCH_SIZE, len(data_filenames))


    if(all_files):
        first_sample = 0
        last_sample_excl = len(data_filenames)

     

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

    batch_size = len(_net_input)

    # sequenz/time should be first dimension
    # and batch should be second dimension
    # _net_input = np.swapaxes(_net_input, axis1=0, axis2=1) 
    # _net_label = np.swapaxes(_net_label, axis1=0, axis2=1) 


    return _net_input, _net_label, batch_size

