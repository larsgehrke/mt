'''

    Supervisor of the training progress. Responsible to check the performance.

'''

import time
import torch as th
import numpy as np

import tools.debug as debug

class TrainSupervisor():

    def __init__(self, epochs, saver = None):
        self.epochs = epochs
        self.saver = saver

        self.epoch_errors_train = []
        self.epoch_errors_val = []

        self.best_train = np.infty
        self.best_val = np.infty

        self.train_sign = "(-)"
        self.val_sign = "(-)"

        self._epoch = 0

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


        