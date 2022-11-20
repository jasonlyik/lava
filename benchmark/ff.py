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

# from spikegenerator import SpikeGenerator, PySpikeGeneratorModel
from multiprocessor import Multiprocessor



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



class FFNetworkBody(AbstractProcess):
    def __init__(self, layer: int, neuron: int):
        super().__init__()

        #trying to debug:
        #layer = kwargs.pop('layer')
        #neuron = kwargs.pop('neuron')

        self.num_layers = Var(shape=(1,), init=layer)
        self.num_neurons = Var(shape=(1,), init=neuron)

        self.spikes_in = InPort(shape=(64,))
        self.spikes_out = OutPort(shape=(10,))
        

@implements(FFNetworkBody)
@requires(CPU)
class PyFFNetworkBodyModel(AbstractSubProcessModel):
    def __init__(self, proc):

        num_layers = proc.num_layers.init     
        num_neurons = proc.num_neurons.init
        
        # Define model
        self.input_dense = Dense(weights=np.identity(64))
        self.input_neurons = LIF(shape=(64,), vth = np.random.randint(16), name="lif1")

        self.hidden_dense = []
        self.hidden_neurons = []
        self.hidden_dense.append(Dense(weights=generate_weights(0.3, 64, num_neurons), name="denseHidden"))  # first hidden layer takes 64 input
        self.hidden_neurons.append(LIF(shape=(num_neurons,), vth = np.random.randint(16), name="hidden1"))


        for i in range(num_layers-1):
            self.hidden_dense.append(Dense(weights=generate_weights(0.3, num_neurons,num_neurons), name="dense"+i)) # rest of hidden layers
            self.hidden_neurons.append(LIF(shape=(num_neurons,), vth=np.random.randint(16), name="lif"+i))


        self.output_dense = Dense(weights=generate_weights(0.3, 10, num_neurons), name="denseOutput")
        self.output_neurons = LIF(shape=(10,), vth=np.random.randint(16), name="lifOutput") 
        

        # Connections:
        proc.spikes_in.connect(self.input_dense.s_in)
        self.input_dense.a_out.connect(self.input_neurons.a_in)
        self.input_neurons.s_out.connect(self.hidden_dense[0].s_in)
        counter = 0

        for i in range(len(self.hidden_dense)-1):
            self.hidden_dense[i].a_out.connect(self.hidden_neurons[i].a_in)
            self.hidden_neurons[i].s_out.connect(self.hidden_dense[i+1].s_in)
            counter = counter + 1

        self.hidden_dense[counter].a_out.connect(self.hidden_neurons[counter].a_in)
        self.hidden_neurons[counter].s_out.connect(self.output_dense.s_in)

        self.output_dense.a_out.connect(self.output_neurons.a_in)
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

def ff_network(layer: int, neuron: int):
    sim_time = 0

    # todo: may tweak these to get experiment running
    num_images = 1000
    num_steps_per_image = 128

    spike_input = SpikeInput(vth=190,
                             num_images=num_images,
                             num_steps_per_image=num_steps_per_image) 

    
    ff = FFNetworkBody(layer = layer, neuron = neuron)
    classifier = Classifier(num_images = num_images)


    spike_input.spikes_out.connect(ff.spikes_in)
    ff.spikes_out.connect(classifier.spikes_in)
    spike_input.label_out.connect(classifier.label_in)

    # print("Starting eval")
    start = time.process_time()

    for img_id in range(num_images):
        # print(f"\rCurrent image: {img_id+1}", end="")

        ff.run(
            condition=RunSteps(num_steps=num_steps_per_image),
            run_cfg=Loihi1SimCfg(select_sub_proc_model=True,
                                 select_tag='fixed_pt'))

    # Gather ground truth and predictions before stopping exec
    ground_truth = classifier.gt_labels.get().astype(np.int32)
    predictions = classifier.pred_labels.get().astype(np.int32)

    # Stop the execution
    ff.stop()
    sim_time = time.process_time() - start
    
    # accuracy = np.sum(ground_truth==predictions)/ground_truth.size * 100

    # print(f"\nGround truth: {ground_truth}\n"
    #       f"Predictions : {predictions}\n"
    #       f"Accuracy    : {accuracy}")

    # print("Finish eval")

    return sim_time

if __name__ == '__main__':

    num_layers = [1, 2, 3, 4]
    num_neurons = [10, 150, 300]
    num_evaluations = 5

    for layer in num_layers:
        for neuron in num_neurons:
            print("Beginning experiment num_layers = ", layer, " and num_neurons = ", neuron)
            times = []

            for trial in range(num_evaluations):
                mp = Multiprocessor(timeout=900, default_ret=0)
                mp.run(ff_network, layer, neuron)
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

            print("Completed experiment num_layers = ", num_layers, " and num_neurons = ", num_neurons)
            print("times", times)
            print("avg_time")
            print(avg_time)