'''
Random Network
N - number of neurons
d - synaptic density fraction, between 0 and 1
S - number of synapses = N x (N-1) x d

10 neurons are randomly selected and stimulated to cause activity, 
	network is simulated for 1000 time steps

Assesses scalability of network size and density of networks

Performance measures:

Load: time to "load" the network and associated input spikes onto the simulator,
	creation of neurons, synapses, spike generation, monitoring objects
Sim: time required to simulate for 1000 time steps
Monitor: time to retrieve spike times in the network from the simulator

Synaptic Density: [0.1, 0.3, 0.5, 1.0]
Number of neurons: 100, 200, 500, 1000

Average over 5 evaluations. Cap test at 15 minutes.
'''

from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense
from lava.proc.monitor.process import Monitor
from lava.magma.core.run_conditions import RunSteps, RunContinuous
from lava.magma.core.run_configs import Loihi1SimCfg

import numpy as np

# TODO: will python find this, do i need an __init__.py
from spikegenerator import SpikeGenerator, PySpikeGeneratorModel


# TODO: here is the baseline code, work on the evaluation framework surrounding like
# making it run on different synaptic density, num neurons, adding timing stuff, and
# avg over 5 runs

lif = LIF(shape=???, vth=, dv=, du=, bias_mant=, name="LIF_neurons")
dense = Dense(weights=???, name="dense")

# recurrent connection
lif.s_out.connect(dense.s_in)
dense.a_out.connect(lif.a_in)

monitor_lif = Monitor()
num_steps = ???

monitor_lif.probe(lif.v, num_steps)

spike_gen = SpikeGenerator(shape=???, spike_prob=???)
dense_input = Dense(weights=???)
spike_get.s_out.connect(dense_input.s_in)
dense_input.a_out.connect(lif.a_in)

run_condition = RunSteps(num_steps=num_steps)
run_cfg = Loihi1SimCfg()
lif.run(condition=run_condition, run_cfg=run_cfg)

data_lif = monitor_lif.get_data()

lif.stop()


