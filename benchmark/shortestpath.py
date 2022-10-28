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

--> make directed by following above, then randomly choosing direction of each edge
'''