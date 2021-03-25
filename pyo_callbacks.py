from pyo import *
from tensorflow import keras
import numpy as np

#Note: before starting training and using the callbacks, a pyo server must be started, like so:
# s = Server().boot()
# s.amp=0.2
# s.start

#after training it has to be stopped
# s.stop()

class ListenToLoss(keras.callbacks.Callback):

    """
    Sonifies validation loss when run on epoch or training loss when run on batch.
    Args:
            hear_on: after each 'batch' or 'epoch'
            min_freq, max_freq: range of frequency for sonification
    """

    def __init__(self, hear_on='batch', min_freq=200, max_freq=1000):

        self.osc = Sine(freq=0).out()

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



def gen_harmonies(n_streams, base_freq=200, ratio=[1,4,5,6]):

    """Generate harmonies for as many streams as needed. Obviously limited by range of reasonable frequencies.
    Used during weights sonification.
    Args:
        base_freq: Basic frequency to build on.
        ratio: List with ratios for frequencies. Should start with 1.
        n_streams: Number of streams to build the harmonies for.
    """

    mul_base = 1
    n_ratio = len(ratio)-1

    all_harms = []

    for stream in range(n_streams+1):

        #add new octave harmonies when more streams are needed
        if stream % n_ratio == 0:

            #adjust frequency to be multiple of base frequency
            freq = base_freq * mul_base
            mul_base += 1

            #generate harmonies
            harms = [freq * ratio[i] / ratio[1] for i in range(1, len(ratio))]
            #print(harms)
            all_harms += harms

    #return as many streams as needed
    return all_harms[:n_streams]


class WeightsDense(keras.callbacks.Callback):

    """Sonifies the change of weights of a dense layer with as many streams as output neurons.
    The weights are averaged for each neuron.
    Frequency shifts after each epoch depend on how much the weights have changed during the last epoch.
    Oscillations created with an LFO depend on the amount the weights deviate from 0.

    Args:
        which_layer: index of layer to sonify.
    """

    def __init__(self, which_layer=-1):

        #define layer to be sonified
        self.which_layer = which_layer

    def on_train_begin(self, logs=None):

        #get output size of model and generate harmonies respectively
        streams = self.model.layers[-1].output_shape[self.which_layer]
        harms = gen_harmonies(base_freq=200, ratio=[1,10,12,15,18], n_streams = streams)

        #get initial weights of layer
        init_weights = self.model.layers[self.which_layer].get_weights()[0]
        self.init_weights_mean = np.mean(init_weights, axis=0)

        #adjust lfo depending on deviation of mean from 0
        self.deviation = np.absolute(self.init_weights_mean) * 200
        self.lfo = LFO(freq=self.deviation.tolist(), type=1, mul=1000, add=1200)
        
        
        self.osc = Sine(freq=harms, mul=0.5)
#         self.har = Harmonizer(self.osc, transpo=-6)
        self.bp = ButBP(self.osc, freq=self.lfo)

        #initialize frequency shifter with 0 shift
        self.shift = FreqShift(self.bp, shift=0)
        #mix channels for audio output
        self.output = self.shift.mix(2).out()



    def on_epoch_end(self, epoch, logs=None):

        #get weights and mean of weights
        weights = self.model.layers[-1].get_weights()
#         print(weights[0])
        self.weights_mean = np.mean(weights[0], axis=0)
#         print(self.weights_mean)

        #adjust lfo depending on deviation of mean to 0
        self.deviation = np.absolute(self.weights_mean) * 200
        self.lfo.setFreq(self.deviation.tolist())

        #shift frequencies depending on how much weights differ from last epoch
        change = self.weights_mean - self.init_weights_mean
        shift_by = change*10000
        self.shift.setShift(shift_by.tolist())

        #remember weights for next epoch
        self.init_weights_mean = self.weights_mean


    def on_train_batch_begin(self, batch, logs=None):

        delay_by = int(logs.get('size') * 0.2)
        #leave some time until shifting back to harmonic frequencies
        if batch == delay_by:
            self.shift.setShift(0)
