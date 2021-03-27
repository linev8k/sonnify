# soNNify - Sonification of Neural Networks

This project explores ways to sonify the training process of neural networks. The sound is audible in real-time during training, but can also be recorded along the way.
It is still raw. The first demos can serve as a starting point for more ideas and approaches. TensorFlow's Keras provides the overall framework, audioprocessing is done with [PYO](https://github.com/belangeo/pyo), integrated into custom callbacks.

## Getting Started

Install dependencies while creating a conda environment with

```sh
conda env create -f sound_environment.yml
```
If this doesn't work for you, just make sure the following packages are installed:

`tensorflow` (using [conda](https://anaconda.org/anaconda/tensorflow)), `numpy` (using [conda](https://anaconda.org/conda-forge/numpy)), `pyo` (only available through [pip](https://pypi.org/project/pyo/)).

Refer to `sound_environment.yml` for versions.

See also the [conda Documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

## Content

`pyo_callbacks.py`: Callback classes for sonification which can be passed on to model.fit().
Currently includes classes for
* Mapping of the loss to a sine wave
* Creating a harmonic ensemble of sine waves, representing the weights of a dense layer. A change in weights detunes the sound. Depending on how much weight values deviate from zero, an additional overall frequency oscillation comes forward.

`load_models.py`: Helper functions for loading an exemplary convolutional neural network and some MNIST data for demonstration purposes.

`demo.ipynb`: Notebook with demos of existing sonification methods.

`audio_files`: Recordings of example sounds, mostly from the demo.

`test_notebooks`: Notebooks for development and testing. Included for now, will probably be removed later.

## Usage

See `demo.ipynb` for demonstration of sonification.

Note: Sonification has not been tested on other networks. There might be issues because of some currently hardcoded parameters or when the sonified layer consists of too many neurons.

The general setup looks like this:

After preparing the model and the data, boot a PYO server before training:
```python
s = Server().boot()
s.amp=0.2 #lower the gain
s.start()
```

If you want to record the sound:
```python
s.recstart('path_to_my_file.wav')
```

Train the model:
```python
model.fit(x_train, y_train, callbacks=[WeightsDense()])
```

If you recorded, stop the recording:
```python
s.recstop()
```

...and stop the server:
```python
s.stop()
```
