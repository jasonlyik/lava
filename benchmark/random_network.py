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
import time

from spikegenerator import SpikeGenerator, PySpikeGeneratorModel
from multiprocessor import Multiprocessor

def random(N: int, d: float, num_steps: int):
    load_time = 0
    sim_time = 0
    monitor_time = 0

    # Load
    load_start = time.process_time()

    lif = LIF(shape=(N, ), vth=np.random.randint(16), name="LIF_neurons") # no leaking, integrate-fire only
    
    if d == 1.0:
        weights = 30 * np.random.random_sample((N, N)) - 15
        for i in range(N):
            weights[i][i] = 0.0
    else:
        weights = np.zeros((N, N))
        # randomly make N x (N-1) x d pairs
        counter = 0
        num_synapses = N * (N-1) * d
        while counter < num_synapses:
            first = np.random.randint(N)
            second = np.random.randint(N)
            if first == second:
                continue
            elif weights[first][second] != 0.0:
                continue
            else:
                weights[first][second] = 30 * np.random.random_sample() - 15
                counter += 1

    dense = Dense(weights=weights, name="dense")

    # recurrent connection
    lif.s_out.connect(dense.s_in)
    dense.a_out.connect(lif.a_in)

    spike_gen = SpikeGenerator(shape=(lif.a_in.shape[0], )) # sends out 10 1's, rest 0's
    x = 30 * np.random.random_sample((N,)) - 15
    dense_input = Dense(weights=np.diag(x)) # determines input weight
    spike_gen.s_out.connect(dense_input.s_in)
    dense_input.a_out.connect(lif.a_in)

    monitor_lif = Monitor()
    monitor_lif.probe(lif.v, num_steps)
    run_condition = RunSteps(num_steps=num_steps)
    run_cfg = Loihi1SimCfg()
    
    load_time = time.process_time() - load_start


    # Sim
    sim_start = time.process_time()
    lif.run(condition=run_condition, run_cfg=run_cfg)
    sim_time = time.process_time() - sim_start

    # Monitor
    monitor_start = time.process_time()
    data_lif = monitor_lif.get_data() # TODO: should check that this is working
    monitor_time = time.process_time() - monitor_start

    lif.stop()

    return load_time, sim_time, monitor_time


if __name__ == '__main__':
    N_list = [100, 200, 500, 1000]
    d_list = [0.1, 0.3, 0.5, 1.0]
    num_steps = 1000

    num_evaluations = 5

    for N in N_list:
        for d in d_list:

            print("Beginning experiment N =", N, "d =", d)
            loads = []
            sims = []
            monitors = []

            for i in range(num_evaluations):
                # mp = Multiprocessor(timeout=900, default_ret=(0, 0, 0))
                # mp.run(random, N, d, num_steps)
                # load, sim, monitor = mp.wait()
                load, sim, monitor = random(N, d, num_steps)
                
                print("Evaluation", i)
                # print("load", load)
                # print("sim", sim)
                # print("monitor", monitor)

                if (load, sim, monitor) != (0, 0, 0):
                    loads.append(load)
                    sims.append(sim)
                    monitors.append(monitor)
                else:
                    print("failed to complete in time")

            if len(loads) > 0:
                avg_load = sum(loads) / len(loads)
                avg_sim = sum(sims) / len(sims)
                avg_monitor = sum(monitors) / len(monitors)
            else:
                avg_load = -1
                avg_sim = -1
                avg_monitor = -1

            print("Completed experiment N =", N, "d = ", d)
            print("loads", loads)
            print("sims", sims)
            print("monitors", monitors)

            print("avg_load, avg_sim, avg_monitor")
            print(avg_load, avg_sim, avg_monitor)



