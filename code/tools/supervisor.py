'''
    This script has a supervisor class for the training and one for the testing of the model. 
    The classes work as view components that print out import information about the run.
'''

import time
import matplotlib.pyplot as plt
import numpy as np
import os

from tools.torch_tools import Saver
import tools.visualize as visualize
import tools.debug as debug

'''
    =============================================
    Supervisor (View) for the training of DISTANA
    =============================================

'''
class TrainSupervisor():
    '''
    Responsible for the further processing of the training errors, like checking the relative performance,
    printing the results to command line and commission the saving of the model.
    '''

    def __init__(self, epochs: int, trainable_params: int, saver: Saver = None):
        '''
        Intitialisation of the TrainSupervisor.
        :param epochs: number of epochs
        :param trainable_params: number of trainable model parameters
        :param saver: Saver object responsible for saving the network
        '''
        self.epochs = epochs
        self.saver = saver

        self.epoch_errors_train = []

        self.epoch_errors_val = []
        self.best_train = np.infty
        self.best_val = np.infty

        self.train_sign = "(-)"
        self.val_sign = "(-)"

        self._epoch = 0

        self.finished_training_time = None

        print("Trainable model parameters:", trainable_params)

    def finished_training(self, errors: list):
        '''
        Further processing of training results.
        :param errors: list of training errors of the curent epoch 
        '''
        self.finished_training_time = time.time()
        self._epoch += 1

        error = np.mean(errors)
        self.epoch_errors_train.append(error)

        # Create a plus or minus sign for the training error
        self.train_sign = "(-)"
        if error < self.best_train:
            self.train_sign = "(+)"
            self.best_train = error



    def finished_validation(self, errors: list):
        '''
        Further processing of validation results.
        :param errors: list of validation errors of the curent epoch 
        '''

        error = np.mean(errors)
        self.epoch_errors_val.append(error)

        # Save the model to file (if desired)
        if self.saver is not None and error < self.best_val:
            self.saver(self._epoch,self.epoch_errors_train, self.epoch_errors_val)  

        # Create a plus or minus sign for the validation error
        self.val_sign = "(-)"
        if error < self.best_val:
            self.best_val = error
            self.val_sign = "(+)"


    def finished_epoch(self, idx, epoch_start_time: float):
        '''
        Printing out summary of epoch results.
        :param idx: index of the current epoch
        :param epoch_start_time: start time of the epoch
        '''

        #
        # Print progress to the console with nice formatting
        print('Epoch ' + str(idx+1).zfill(int(np.log10(self.epochs)) + 1)
              + '/' + str(self.epochs) + ' took '
              + str(np.round(time.time() - epoch_start_time, 2)).ljust(5, '0')
              + ' seconds.\t\tAverage epoch training error: ' + self.train_sign
              + str(np.round(self.epoch_errors_train[-1], 10)).ljust(12, ' ')
              + '\t\tValidation error: ' + self.val_sign
              + str(np.round(self.epoch_errors_val[-1], 10)).ljust(12, ' '))

        # print(str(np.round(self.finished_training_time - epoch_start_time, 3))+","
        #     + str(np.round(self.epoch_errors_train[-1], 10))+","
        #     + str(np.round(self.epoch_errors_val[-1], 10)))


    def finished(self, training_start_time):
        '''
        Prinint out the end of the training run.
        :param training_start_time: start time of the training
        '''
        now = time.time()
        print('\nTraining took ' + str(np.round(now - training_start_time, 2)) + ' seconds.\n\n')
        print("Done!")


'''
    ============================================
    Supervisor (View) for the testing of DISTANA
    ============================================

'''

class TestSupervisor():
    '''
    Responsible for the further processing of the testing errors, like printing the results to command line 
    and commission the plotting of the results.
    '''

    def __init__(self, params: dict, trainable_params: int):
        '''
        Intitialisation of the TestSupervisor.
        :param params: dictionary with all parameters
        :param trainable_params: number of trainable model parameters
        '''

        print("Trainable model parameters:", trainable_params)

        self.params = params

        # Test statistics
        self.time_list = []
        self.accuracy_list = []

        self.sample_idx = 0

        #plt.style.use('dark_background')


    def finish_batch(self, time_start: float, batch_size: int, error:float):
        '''
        Further processing of the batch results.
        :param time_start: time before the forward pass of the batch
        :param batch_size: size of the batch used for the forward pass
        :param error: test error
        '''
        forward_pass_duration = time.time() - time_start
        print(str(np.round(forward_pass_duration, 10)).ljust(12, ' '))
	    print("\tForward pass for batch size ",batch_size,
            " took: ", str(np.round(forward_pass_duration, 10)).ljust(12, ' '),
            " seconds with error ", str(np.round(error, 10)).ljust(12, ' '))

    def plot_sample(self, net_outputs: np.ndarray, net_label: np.ndarray, net_input: np.ndarray):
        '''
        Plotting the network output and the ground truth as diagram or animation.
        :param net_outputs: array of the outputs of the network
        :param net_label: array of the ground truth
        :param net_input: array of the inputs of the network
        '''
        
        image_mode = self.params['image_mode']
        video_mode = self.params['video_mode']
        pk_rows = self.params['pk_rows']
        pk_cols = self.params['pk_cols']
        teacher_forcing_steps = self.params['teacher_forcing_steps']
        model_name = self.params['model_name']
        diagram_folder = self.params['diagram_folder']

        # === PLOT ===

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
        
        # Depending on the parameter values the plot will be visualized 
        # or saved to file
        if image_mode == "show":
            plt.show()
        elif image_mode != "no":
            file = model_name + "_" + str(self.sample_idx) +  ".png"

            # Check whether the directory to save the data exists 
            #  and create it if not
            if not os.path.exists(diagram_folder):
                os.makedirs(diagram_folder)

            plt.savefig(diagram_folder + file)
            if image_mode == "save":
                plt.close(fig)


        # === VIDEO ===

        # Visualize and animate the propagation of the 2d wave
        anim = visualize.animate_2d_wave(
            pk_rows=pk_rows, 
            pk_cols=pk_cols, 
            teacher_forcing_steps=teacher_forcing_steps,
            net_label= net_label, 
            net_outputs= net_outputs, 
            net_inputs=net_input)

        if video_mode != "show" and video_mode != "no":
            file = model_name + "_" + str(self.sample_idx) +  ".mp4"
            # Check whether the directory to save the data exists 
            # and create it if not
            if not os.path.exists(diagram_folder):
                os.makedirs(diagram_folder)
            print("Save diagram as video...")
            # save the animation as an mp4.  This requires ffmpeg or mencoder to be
            # installed.  The extra_args ensure that the x264 codec is used, so that
            # the video can be embedded in html5.  You may need to adjust this for
            # your system: for more information, see
            # http://matplotlib.sourceforge.net/api/animation_api.html
            # extra_args=['-vcodec', 'libx264'])
            anim.save(diagram_folder+file, fps=5) 
        
        if video_mode != "save" and video_mode != "no":
            plt.show()

        self.sample_idx += 1


    def finished(self):
        '''
        Closing the view. If necessary, summary information can be printed out.
        '''
           
        print('Done')


   



















