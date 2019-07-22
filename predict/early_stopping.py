#https://github.com/Bjarten/early-stopping-pytorch

import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, coin_name, patience=7, verbose=False):
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
        self.val_loss_min = np.Inf
        self.coin_name = coin_name
        self.last_save_epoch = -1

    def __call__(self, val_loss, epoch, model):
        if epoch > 99:
            if self.val_loss_min is np.Inf:
                self.val_loss_min = val_loss
                self.save_checkpoint(val_loss, epoch, model)
            elif val_loss > self.val_loss_min:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience} @ Epoch {epoch}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.val_loss_min = val_loss
                self.save_checkpoint(val_loss, epoch, model)
                self.counter = 0
        else:
            pass

    def save_checkpoint(self, val_loss, epoch, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model @ Epoch {epoch}')
        torch.save(model.state_dict(), './models/{0}_{1:.4f}_{2}.pt'.format(self.coin_name, val_loss, epoch))
        self.last_save_epoch = epoch
