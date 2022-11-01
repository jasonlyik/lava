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


'''
graph generation method:
parameters n, m
add m nodes to the graph
for the m+1th node, add edges (undirected) to the m previous nodes
keep running bucket of node_ids, every outgoing edge from a node puts another
	node_id token into the bucket (start should be one token for m nodes, m
	tokens for m+1 node)
for rest of n nodes, draw m bucket tokens and add edge to each unique node_id

--> make undirected by following above, then randomly choose direction of each edge
'''

def shortest_path(size: int):
	time = 0

	# Graph generation
	n = size
	m = n / 5 # TODO

	graph = ??? # TODO: some numpy n-d array

	# TODO: add edges from node m to nodes [0, m-1] in the graph (both ways)

	# TODO: all this syntax might suck
	bucket = [i for i in range(m)]
	bucket.extend([m for i in range(m)])

	for node in range(m+1, n):
		# shuffle bucket
		edges = bucket[:m]
		# add edges to graph (both ways)
		bucket.extend(edges)


	lif = LIF(shape=???, vth=, dv=, du=, bias_mant=, name="LIF_neurons")
	dense = Dense(weights=???, name="dense")

	# recurrent connection
	lif.s_out.connect(dense.s_in)
	dense.a_out.connect(lif.a_in)

	# TODO: this just needs spikes of 1
	spike_gen = SpikeGenerator(shape=???, spike_prob=???)
	dense_input = Dense(weights=???) # TODO: one node should be spiked
	spike_gen.s_out.connect(dense_input.s_in)
	dense_input.a_out.connect(lif.a_in)

	# TODO: the run condition is continuous? how do we know when all neurons have been spiked?
	# --> max number of steps equal to total num edges
	
	# run_condition = RunSteps(num_steps=num_steps)
	# run_cfg = Loihi1SimCfg()
	# lif.run(condition=run_condition, run_cfg=run_cfg)


	lif.stop()

	return time


if __name__ == '__main__':
	sizes = [500, 5000, 10000]

	num_evaluations = 5

	for size in sizes:
		print("Beginning experiment size =", size)
		times = []

		for i in num_evaluations:
			time = shortest_path(size)
			times.append(time)
			print("Evaluation", i)

		avg_time = times.mean()

		print("Completed experiment size =", size)
		print("times", times)
		print("avg_time")
		print(avg_time)