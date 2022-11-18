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

