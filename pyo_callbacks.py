from pyo import *
from tensorflow import keras

class ListenToLoss(keras.callbacks.Callback):

    """
    Sonifies validation loss when run on epoch or training loss when run on batch.
    Args:
            osc: pyo oscillator object for sonifying the cur_loss (required)
            hear_on: after each 'batch' or 'epoch'
            min_freq, max_freq: range of frequency for sonification
    """

    def __init__(self, osc, hear_on='batch', min_freq=300, max_freq=3000):
        self.osc = osc
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.cur_step = 0
        self.max_loss = 0

        #should be 'batch' or 'epoch'
        self.hear_on = hear_on

    def _hear_loss(self, logs):

        if 'val_loss' in logs:
            cur_loss = logs['val_loss']
        else:
            cur_loss = logs['loss']
        #set first loss as maximum loss value
        if self.cur_step == 0:
            self.max_loss = cur_loss
        self.cur_step += 1

        #normalizing loss to be between 300 and 3000
        #toMin + (num - fromMin)/(fromMax - fromMin) * (toMax - toMin)
        #if loss is above the first one, set to highest frequency
        if cur_loss > self.max_loss:
            norm_loss = self.max_freq
        else:
            norm_loss = float(self.min_freq + (cur_loss - 0)/(self.max_loss - 0) * (self.max_freq-self.min_freq))

        # print(norm_loss)
        self.osc.setFreq(norm_loss)

    def on_train_batch_end(self, batch, logs=None):

        if self.hear_on == 'batch':
            self._hear_loss(logs)

    def on_epoch_end(self, epoch, logs=None):

        if self.hear_on == 'epoch':
            self._hear_loss(logs)
