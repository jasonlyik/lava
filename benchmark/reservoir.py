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
        

@implements(ReservoirBody)
@requires(CPU)
class PyReservoirBodyModel(AbstractSubProcessModel):
    def __init__(self, proc):
        cube = proc.cube.init
        dim = proc.dim.init

        self.input_dense = Dense(weights=np.identity(64))
        self.input_neurons = LIF(shape=(64,), vth=np.random.randint(16)) # TODO
        
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
        
        proc.spikes_in.connect(self.input_dense.s_in)
        self.input_dense.a_out.connect(self.input_neurons.a_in)
        self.input_neurons.s_out.connect(self.input_reservoir_dense.s_in)
        self.input_reservoir_dense.a_out.connect(self.reservoir_neurons.a_in) # input layer to reservoir
        self.reservoir_neurons.s_out.connect(self.reservoir_dense.s_in)
        self.reservoir_dense.a_out.connect(self.reservoir_neurons.a_in) # reservoir to reservoir
        self.reservoir_neurons.s_out.connect(self.reservoir_output_dense.s_in)
        self.reservoir_output_dense.a_out.connect(self.output_neurons.a_in)
        self.output_neurons.s_out.connect(proc.spikes_out)

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

@implements(proc=Classifier, protocol=LoihiProtocol)
@requires(CPU)
class PyClassifierModel(PyLoihiProcessModel):
    label_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int, precision=32)
    spikes_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    num_images: int = LavaPyType(int, int, precision=32)
    spikes_accum: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=32)
    num_steps_per_image: int = LavaPyType(int, int, precision=32)
    pred_labels: np.ndarray = LavaPyType(np.ndarray, int, precision=32)
    gt_labels: np.ndarray = LavaPyType(np.ndarray, int, precision=32)
        
    def __init__(self, proc_params):
        super().__init__(proc_params=proc_params)
        self.current_img_id = 0

    def post_guard(self):
        """Guard function for PostManagement phase.
        """
        if self.time_step % self.num_steps_per_image == 0 and \
                self.time_step > 1:
            return True
        return False

    def run_post_mgmt(self):
        """Post-Management phase: executed only when guard function above 
        returns True.
        """
        gt_label = self.label_in.recv()
        pred_label = np.argmax(self.spikes_accum)
        self.gt_labels[self.current_img_id] = gt_label
        self.pred_labels[self.current_img_id] = pred_label
        self.current_img_id += 1
        self.spikes_accum = np.zeros_like(self.spikes_accum)

    def run_spk(self):
        """Spiking phase: executed unconditionally at every time-step
        """
        spk_in = self.spikes_in.recv()
        self.spikes_accum = self.spikes_accum + spk_in


def reservoir_network(dim: int):
    sim_time = 0

    # todo: may tweak these to get experiment running
    num_images = 25
    num_steps_per_image = 128

    spike_input = SpikeInput(vth=190,
                             num_images=num_images,
                             num_steps_per_image=num_steps_per_image) 
    reservoir = ReservoirBody(dim)
    classifier = Classifier(num_images=num_images)

    spike_input.spikes_out.connect(reservoir.spikes_in)
    reservoir.spikes_out.connect(classifier.spikes_in)
    spike_input.label_out.connect(classifier.label_in)

    print("Starting eval")
    start = time.process_time()

    for img_id in range(num_images):
        print(f"\rCurrent image: {img_id+1}", end="")

        reservoir.run(
            condition=RunSteps(num_steps=num_steps_per_image),
            run_cfg=Loihi1SimCfg(select_sub_proc_model=True,
                                 select_tag='fixed_pt'))

    # Gather ground truth and predictions before stopping exec
    ground_truth = classifier.gt_labels.get().astype(np.int32)
    predictions = classifier.pred_labels.get().astype(np.int32)

    # Stop the execution
    reservoir.stop()
    sim_time = time.process_time() - start
    
    accuracy = np.sum(ground_truth==predictions)/ground_truth.size * 100

    print(f"\nGround truth: {ground_truth}\n"
          f"Predictions : {predictions}\n"
          f"Accuracy    : {accuracy}")

    print("Finish eval")

    return sim_time

if __name__ == '__main__':
    dims = [3, 7, 9, 10]
    num_evaluations = 5

    for dim in dims:
        print("Beginning experiment dim =", dim)
        times = []

        for trial in range(num_evaluations):
            mp = Multiprocessor(timeout=900, default_ret=0)
            mp.run(reservoir_network, dim)
            sim_time = mp.wait()
            # sim_time = reservoir_network(dim)
            
            print("Evaluation", trial)
            if sim_time != 0:
                times.append(sim_time)
            else:
                print("failed to complete in time")

        if len(times) > 0:
            avg_time = sum(times) / len(times)
        else:
            avg_time = -1

        print("Completed experiment size =", size)
        print("times", times)
        print("avg_time")
        print(avg_time)