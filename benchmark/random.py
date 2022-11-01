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

# TODO: will python find this, do i need an __init__.py, or just dot ahead of spikegenerator
from spikegenerator import SpikeGenerator, PySpikeGeneratorModel

def random(N: int, d: float, num_steps: int):
	load_time = 0 # TODO: add timing
	sim_time = 0
	monitor_time = 0

	lif = LIF(shape=???, vth=, dv=, du=, bias_mant=, name="LIF_neurons")
	dense = Dense(weights=???, name="dense")

	# recurrent connection
	lif.s_out.connect(dense.s_in)
	dense.a_out.connect(lif.a_in)

	monitor_lif = Monitor()

	monitor_lif.probe(lif.v, num_steps)

	# TODO: this just needs spikes of 1
	spike_gen = SpikeGenerator(shape=???, spike_prob=???)
	dense_input = Dense(weights=???) # TODO: does this control which 10 neurons spike or the spikegen?
	spike_gen.s_out.connect(dense_input.s_in)
	dense_input.a_out.connect(lif.a_in)

	run_condition = RunSteps(num_steps=num_steps)
	run_cfg = Loihi1SimCfg()
	lif.run(condition=run_condition, run_cfg=run_cfg)

	data_lif = monitor_lif.get_data()

	lif.stop()

	return load_time, sim_time, monitor_time


if __name__ == '__main__':
    N_list = [100, 200, 500, 1000]
    d_list = [0.1, 0.3, 0.5, 1.0]
    num_steps = 1000

    num_evaluations = 5
    cap_time = ???

    for N in N_list:
    	for d in d_list:

    		print("Beginning experiment N =", N, "d =", d)
    		loads = []
    		sims = []
    		monitors = []

    		for i in num_evaluations:
    			# TODO: figure out how to continue if 15 min have elapsed
    			# --> might be easier to just make bash script that does this
    			load, sim, monitor = random(N, d, num_steps)
    			loads.append(load)
    			sims.append(sim)
    			monitors.append(monitor)

    			print("Evaluation", i)
    			# print("load", load)
    			# print("sim", sim)
    			# print("monitor", monitor)

    		avg_load = loads.mean() # TODO
    		avg_sim = sims.mean()
    		avg_monitor = monitors.mean()

    		print("Completed experiment N =", N, "d = ", d)
    		print("loads", loads)
			print("sims", sims)
			print("monitors", monitors)

			print("avg_load, avg_sim, avg_monitor")
			print(avg_load, avg_sim, avg_monitor)



