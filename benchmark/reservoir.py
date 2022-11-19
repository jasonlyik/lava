'''
Reservoir
x, y, z - the dimensions of the reservoir

optdigits dataset, not specified how it's used so assuming that we run inference on full test set (1797 tests)
--> also could mean only doing a single inference, though seems very unlikely
the 64 input features are rate-encoded to max 10 spikes per feature

64 input neurons -> reservoir -> 10 output neurons

Performance measures:
Total time

x, y, z,    total
3, 3, 3,    27
7, 7, 7,    343
9, 9, 9,    729
10, 10, 10, 1000

P(input node connected to reservoir node) = 0.3
P(reservoir node connected to output node) = 0.3

P(reservoir node i connected to node j) = 3 * exp(-((i_x-j_x)^2 + (i_y-j_y)^2 + (i_z-j_z)^2) / 2.5)
--> where there is a 4:1 ratio excitatory:inhibitory synapses (pos:neg weights)

Average over 5 evaluations. Cap test at 15 minutes.
'''

from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense
from lava.proc.monitor.process import Monitor
from lava.magma.core.run_conditions import RunSteps, RunContinuous
from lava.magma.core.run_configs import Loihi1SimCfg

import numpy as np
import time
import math
import random

from digitspikes import SpikeInput, PySpikeInputModel
from multiprocessor import Multiprocessor

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort

# Import parent classes for ProcessModels
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.model.py.model import PyLoihiProcessModel

# Import ProcessModel ports, data-types
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType

# Import execution protocol and hardware resources
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.resources import CPU

# Import decorators
from lava.magma.core.decorator import implements, requires

# all assuming cube dimension x = y = z
def to3d(idx: int, dim: int):
    z = idx // (dim * dim)
    idx -= (z * dim * dim)
    y = idx // dim
    x = idx % dim
    return int(x), int(y), int(z)

def to1d(x: int, y: int, z: int, dim: int):
    return int((z * dim * dim) + (y * dim) + x)

# generate weights in range [-15, 15] based on static probability
def generate_weights(prob: float, input_dim: int, output_dim: int):
    weights = np.zeros((output_dim, input_dim))
    counter = 0
    num_synapses = int(output_dim * input_dim * prob)
    while counter < num_synapses:
        first = np.random.randint(output_dim)
        second = np.random.randint(input_dim)
        if weights[first][second] != 0.0:
            continue
        else:
            weights[first][second] = 30 * np.random.random_sample() - 15
            counter += 1

    return weights

class ReservoirBody(AbstractProcess):
    def __init__(self, dim: int): # currently only support x = y = z
        super().__init__()

        self.dim = Var(shape=(1,), init=dim)
        self.cube = Var(shape=(1,), init=int(dim * dim * dim))

        self.spikes_in = InPort(shape=(64,))
        self.spikes_out = OutPort(shape=(10,))
        
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

@implements(ReservoirBody)
@requires(CPU)
class PyReservoirBodyModel(AbstractSubProcessModel):
    def __init__(self, proc):
        cube = proc.cube.init
        dim = proc.dim.init

        self.input_dense = Dense(weights=np.identity(64))
        self.input_neurons = LIF(shape=(64,), vth=190) # TODO
        
        # 0.3 prob for connection from input to reservoir
        self.input_reservoir_dense = Dense(weights=generate_weights(0.3, 64, cube))

        self.reservoir_neurons = LIF(shape=(cube, ), vth=np.random.randint(16))

        # P(reservoir node i connected to node j) = 3 * exp(-((i_x-j_x)^2 + (i_y-j_y)^2 + (i_z-j_z)^2) / 2.5)
        # --> where there is a 4:1 ratio excitatory:inhibitory synapses (pos:neg weights)
        res_weights = np.zeros((cube, cube))
        for i in range(cube):
            for j in range(cube):
                i_x, i_y, i_z = to3d(i, dim)
                j_x, j_y, j_z = to3d(j, dim)
                prob = 3 * math.exp(-((i_x-j_x)**2 + (i_y-j_y)**2 + (i_z-j_z)**2) / 2.5)
                if random.random() < prob:
                    if random.random() < 0.2:
                        res_weights[i][j] = np.random.uniform(-15, 0)
                    else:
                        res_weights[i][j] = np.random.uniform(0, 15)

        self.reservoir_dense = Dense(weights=res_weights)

        self.reservoir_output_dense = Dense(weights=generate_weights(0.3, cube, 10))
        self.output_neurons = LIF(shape=(10,), vth=np.random.randint(16))         
        # # Create aliases of SubProcess variables
        # proc.lif1_u.alias(self.lif1.u)
        # proc.lif1_v.alias(self.lif1.v)
        # proc.lif2_u.alias(self.lif2.u)
        # proc.lif2_v.alias(self.lif2.v)
        # proc.oplif_u.alias(self.output_lif.u)
        # proc.oplif_v.alias(self.output_lif.v)

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