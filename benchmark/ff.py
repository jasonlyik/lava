'''
Feed-forward network
L - number of hidden layers
n - number of neurons per hidden layer

optdigits dataset, not specified how it's used so assuming that we run inference on full test set (1797 tests)
--> also could mean only doing a single inference, though seems very unlikely
the 64 input features are rate-encoded to max 10 spikes per feature

64 input neurons -> hidden layers -> 10 output neurons

Performance measures:
Total time

Number of hidden layers: [1, 2, 3, 4]
Number of neurons: 10, 150, 300

Average over 5 evaluations. Cap test at 15 minutes.
'''

from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense
from lava.proc.monitor.process import Monitor
from lava.magma.core.run_conditions import RunSteps, RunContinuous
from lava.magma.core.run_configs import Loihi1SimCfg

import numpy as np
import time

from spikegenerator import SpikeGenerator, PySpikeGeneratorModel
from multiprocessor import Multiprocessor

class FFNetworkBody(AbstractProcess):
    def __init__(self, num_layers: int, num_neurons: int):
        super().__init__()

        self.spikes_in = InPort(shape=(64,))
        self.spikes_out = OutPort(shape=(10,))
        
        self.input_dense = Var(shape=(64,64), init=???) # TODO: init to all 1's maybe?
        self.input_neurons = Var(shape=(64,), init=???) # TODO

        self.hidden_dense = [Var(shape=(num_neurons,64), init=???)] # first hidden layer takes 64 input
        self.hidden_neurons = [Var(shape=(num_neurons,), init=???)]

        for i in range(num_layers-1):
            self.hidden_dense.append(Var(shape=(num_neurons,num_neurons), init=???)) # rest of hidden layers
            self.hidden_neurons.append(Var(shape=(num_neurons,), init=???))

        self.output_dense = Var(shape=(10,num_neurons), init=???)
        self.output_neurons = Var(shape=(10,), init=???) 
        
        # TODO
        # Up-level currents and voltages of LIF Processes
        # for resetting (see at the end of the tutorial)
        '''
        self.lif1_u = Var(shape=(w0.shape[0],), init=0)
        self.lif1_v = Var(shape=(w0.shape[0],), init=0)
        self.lif2_u = Var(shape=(w1.shape[0],), init=0)
        self.lif2_v = Var(shape=(w1.shape[0],), init=0)
        self.oplif_u = Var(shape=(w2.shape[0],), init=0)
        self.oplif_v = Var(shape=(w2.shape[0],), init=0)
        '''

class Classifier(AbstractProcess):
    """Process to gather spikes from 10 output LIF neurons and interpret the
    highest spiking rate as the classifier output"""

    def __init__(self, **kwargs):
        super().__init__()
        shape = (10,)
        n_img = kwargs.pop('num_images', 25) # TODO
        self.num_images = Var(shape=(1,), init=n_img)
        self.spikes_in = InPort(shape=shape)
        self.label_in = InPort(shape=(1,))
        self.spikes_accum = Var(shape=shape)  # Accumulated spikes for classification
        self.num_steps_per_image = Var(shape=(1,), init=128)
        self.pred_labels = Var(shape=(n_img,))
        self.gt_labels = Var(shape=(n_img,))