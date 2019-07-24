#https://github.com/Bjarten/early-stopping-pytorch
import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, coin_name, patience=7, verbose=False, logger=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.valid_loss_min = np.Inf
        self.coin_name = coin_name
        self.last_save_epoch = -1
        self.last_filename = None
        self.last_valid_accuracy = -1
        self.last_state_dict = None
        self.logger = logger

    def __call__(self, valid_loss, valid_accuracy, epoch, model, valid_size, one_count_rate):
        if epoch > 0:
            if self.valid_loss_min is np.Inf:
                self.save_checkpoint(valid_loss, valid_accuracy, epoch, model, valid_size, one_count_rate)
            elif valid_loss > self.valid_loss_min:
                self.counter += 1
                if self.verbose:
                    self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience} @ Epoch {epoch}\n')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.save_checkpoint(valid_loss, valid_accuracy, epoch, model, valid_size, one_count_rate)
                self.counter = 0
        else:
            pass

    def save_checkpoint(self, valid_loss, valid_accuracy, epoch, model, valid_size, one_count_rate):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.logger.info(f'{self.coin_name}: Saving Model @ Epoch {epoch} - Validation Loss Decreased '
                             f'({self.valid_loss_min:.6f} --> {valid_loss:.6f}) - Validation Accuracy {valid_accuracy:.6f}\n')

        new_filename = "{0}_{1}_{2:.2f}_{3:.2f}_{4}_{5:.2f}.pt".format(
            self.coin_name,
            epoch,
            valid_loss,
            valid_accuracy,
            valid_size,
            one_count_rate
        )

        self.last_state_dict = model.state_dict()

        self.last_filename = new_filename
        self.valid_loss_min = valid_loss
        self.last_save_epoch = epoch
        self.last_valid_accuracy = valid_accuracy

    def save_last_model(self):
        file_name = "./models/" + self.last_filename
        torch.save(self.last_state_dict, file_name)
