'''
Shortest Path

Utilize different sized, approximate power law graphs
Randomly select different nodes in graph as source node
Source node is spiked, spikes propagate throughout the network until 
    all neurons have spikes or time equal to total number of edges
    has elapsed

Graph size: [500, 5000, 10000]

Performance measures:
Total time

Average over 5 evaluations. Cap test at 15 minutes.
'''
from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense
from lava.proc.monitor.process import Monitor
from lava.magma.core.run_conditions import RunSteps, RunContinuous
from lava.magma.core.run_configs import Loihi1SimCfg

import numpy as np
import time
import random

from spikegenerator import SpikeGenerator, PySpikeGeneratorModel
from multiprocessor import Multiprocessor

'''
graph generation method:
parameters n, m
add m nodes to the graph
for the m+1th node, add edges (undirected) to the m previous nodes
keep running bucket of node_ids, every outgoing edge from a node puts another
    node_id token into the bucket (start should be one token for m nodes, m
    tokens for m+1 node)
for rest of n nodes, draw m bucket tokens and add edge to each unique node_id
'''

def shortest_path(size, graph):
    sim_time = 0

    lif = LIF(shape=(size,), vth=1, name="LIF_neurons") # threshold of 1 so that instantly fires in next timestep
    dense = Dense(weights=graph, name="dense")

    # recurrent connection
    lif.s_out.connect(dense.s_in)
    dense.a_out.connect(lif.a_in)

    # TODO: this just needs spikes of 1
    spike_gen = SpikeGenerator(shape=(lif.a_in.shape[0], ), num_spikes=1)
    dense_input = Dense(weights=np.identity(lif.a_in.shape[0]))
    spike_gen.s_out.connect(dense_input.s_in)
    dense_input.a_out.connect(lif.a_in)
    
    run_condition = RunSteps(num_steps=num_edges) # too many edges, cannot finish 300,000 steps within 15 minutes. should try to figure out how to measure if every neuron has spiked
    run_cfg = Loihi1SimCfg()

    print("Starting eval")
    start = time.process_time()
    lif.run(condition=run_condition, run_cfg=run_cfg)
    sim_time = time.process_time() - start
    print("Finish eval")

    lif.stop()

    return sim_time

if __name__ == '__main__':
    # sizes = [500, 5000, 10000]
    # ms = [100, 250, 300]
    sizes = [5000, 10000]
    ms = [100, 100]
    # TODO: generated graph is too big, try again with smaller ms, probably much smaller ms
    # know that 1000*1000 network fully connected is tractable, analyze the number of edges


    num_evaluations = 5

    for i in range(len(sizes)):
        size = sizes[i]
        m = ms[i]
        print("Beginning experiment size =", size)
        times = []

        for trial in range(num_evaluations):

            # generate graph here so it doesn't get counted as sim time
            # Graph generation
            n = size
            # m = int(n / 5) # TODO: change?

            num_edges = 0

            graph = np.zeros((size, size))

            # add edges from node m to nodes [0, m-1] in the graph
            for i in range(m):
                if np.random.randint(2) == 1:
                    graph[i][m] = 1.0
                else:
                    graph[m][i] = 1.0
            num_edges += m

            bucket = [i for i in range(m)]
            bucket.extend([m for i in range(m)])

            for node in range(m+1, n):
                random.shuffle(bucket)
                edges = list(np.unique(bucket[:m]))
                for i in edges:
                    if np.random.randint(2) == 1:
                        graph[i][node] = 1.0
                    else:
                        graph[node][i] = 1.0
                num_edges += len(edges)
                bucket.extend(edges)
            # TODO: should make histogram of the graph to see if the nodes follow power law connectivity
            print("graph generated, num edges", num_edges)


            mp = Multiprocessor(timeout=900, default_ret=0)
            mp.run(shortest_path, size, graph)
            sim_time = mp.wait()
            # sim_time = shortest_path(size)
            
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