'''

    Supervisor of the training progress. Responsible to check the performance.

'''

import time
import matplotlib.pyplot as plt
import torch as th
import numpy as np
import os

import tools.visualize as visualize

import tools.debug as debug

class TrainSupervisor():

    def __init__(self, epochs, trainable_params, saver = None):
        self.epochs = epochs
        self.saver = saver

        self.epoch_errors_train = []
        self.epoch_errors_val = []

        self.best_train = np.infty
        self.best_val = np.infty

        self.train_sign = "(-)"
        self.val_sign = "(-)"

        self._epoch = 0

        print("Trainable model parameters:", trainable_params)

    def finished_training(self,errors):
        self._epoch += 1

        error = np.mean(errors)
        self.epoch_errors_train.append(error)

        # Create a plus or minus sign for the training error
        self.train_sign = "(-)"
        if error < self.best_train:
            self.train_sign = "(+)"
            self.best_train = error



    def finished_validation(self, error):

        self.epoch_errors_val.append(error)

        # Save the model to file (if desired)
        if self.saver is not None and error < self.best_val:
            self.saver(self._epoch,self.epoch_errors_train, self.epoch_errors_val)  

        # Create a plus or minus sign for the validation error
        self.val_sign = "(-)"
        if error < self.best_val:
            self.best_val = error
            self.val_sign = "(+)"


    def finished_epoch(self, idx, epoch_start_time):
        #
        # Print progress to the console with nice formatting
        print('Epoch ' + str(idx+1).zfill(int(np.log10(self.epochs)) + 1)
              + '/' + str(self.epochs) + ' took '
              + str(np.round(time.time() - epoch_start_time, 2)).ljust(5, '0')
              + ' seconds.\t\tAverage epoch training error: ' + self.train_sign
              + str(np.round(self.epoch_errors_train[-1], 10)).ljust(12, ' ')
              + '\t\tValidation error: ' + self.val_sign
              + str(np.round(self.epoch_errors_val[-1], 10)).ljust(12, ' '))


    def finished(self, training_start_time):
        now = time.time()
        print('\nTraining took ' + str(np.round(now - training_start_time, 2)) + ' seconds.\n\n')
        print("Done!")


'''
    ============================================
    Supervisor (View) for the testing of DISTANA
    ============================================

'''

class TestSupervisor():

    def __init__(self, params, trainable_params):

        print("Trainable model parameters:", trainable_params)

        self.params = params

        # Test statistics
        self.time_list = []
        self.accuracy_list = []

    def plot_sample(self, mode, net_outputs, net_label, net_input, idx):

        pk_rows = self.params['pk_rows']
        pk_cols = self.params['pk_cols']
        teacher_forcing_steps = self.params['teacher_forcing_steps']
        model_name = self.params['model_name']
        diagram_folder = self.params['diagram_folder']

        # forward_pass_duration = time.time() - time_start

        # print("\tForward pass took:", forward_pass_duration, "seconds.")

        net_outputs = net_outputs.detach().numpy()

        # Plot the wave activity
        fig, axes = plt.subplots(2, 2, figsize=[10, 10], sharex="all")
        for i in range(2):
            for j in range(2):
                make_legend = True if (i == 0 and j == 0) else False
                visualize.plot_kernel_activity(
                    idx = i+j,
                    ax = axes[i, j],
                    label = net_label,
                    net_out = net_outputs,
                    pk_rows = pk_rows,
                    pk_cols = pk_cols,
                    teacher_forcing_steps = teacher_forcing_steps,
                    net_in = net_input,
                    make_legend = make_legend
                )
        fig.suptitle('Model ' + model_name, fontsize=12)
        #

        if mode == "show":
            plt.show()
        else:
            file = model_name + "_" + str(idx) +  ".png"

            # Check whether the directory to save the data exists 
            #  and create it if not
            if not os.path.exists(diagram_folder):
                os.makedirs(diagram_folder)

            plt.savefig(diagram_folder + file)
            if mode == "save":
                plt.close(fig)


        # Visualize and animate the propagation of the 2d wave
        anim = visualize.animate_2d_wave(
            mode = mode,
            pk_rows=pk_rows, 
            pk_cols=pk_cols, 
            teacher_forcing_steps=teacher_forcing_steps,
            net_label= net_label, 
            net_outputs= net_outputs, 
            net_inputs=net_input)

        if mode != "show":
            file = model_name + "_" + str(idx) +  ".mp4"
            # Check whether the directory to save the data exists 
            #  and create it if not
            if not os.path.exists(diagram_folder):
                os.makedirs(diagram_folder)
            # save the animation as an mp4.  This requires ffmpeg or mencoder to be
            # installed.  The extra_args ensure that the x264 codec is used, so that
            # the video can be embedded in html5.  You may need to adjust this for
            # your system: for more information, see
            # http://matplotlib.sourceforge.net/api/animation_api.html
            anim.save(diagram_folder+file, fps=5, extra_args=['-vcodec', 'libx264'])
        
        if mode != "save":
            plt.show()

        # Append the test statistics for this sequence to the appropriate lists
        # self.time_list.append(forward_pass_duration)


    def finished(self):

        # print("Average forward pass duration:", np.mean(self.time_list),
        #   " +-", np.std(self.time_list))
           
        print('Done')


   



















