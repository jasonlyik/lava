from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import OutPort

from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyOutPort

import numpy as np

class SpikeGenerator(AbstractProcess):
    """Spike generator process provides spikes to subsequent Processes.

    Parameters
    ----------
    shape: tuple
        defines the dimensionality of the generated spikes per timestep
    """
    def __init__(self, shape: tuple, num_spikes: int) -> None:
        super().__init__()
        self.s_out = OutPort(shape=shape)
        self.num_spikes = Var(shape=(1,), init=num_spikes)
        self.has_spiked = Var(shape=(1,), init = 0)

@implements(proc=SpikeGenerator, protocol=LoihiProtocol)
@requires(CPU)
class PySpikeGeneratorModel(PyLoihiProcessModel):
    """Spike Generator process model."""
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    num_spikes: int = LavaPyType(int, int)
    has_spiked: int = LavaPyType(int, int)

    def run_spk(self) -> None:
        # for random only want to send out 10 spikes
        spike_data = np.zeros(self.s_out.shape[0], dtype=int)
        
        if self.has_spiked == 0:
            counter = 0
            while counter < self.num_spikes:
                rand = np.random.randint(self.s_out.shape[0])
                print('spiking', rand)
                if spike_data[rand] != 0:
                    continue
                else:
                    spike_data[rand] = 1
                    counter += 1
            self.has_spiked = 1        

        # Generate random spike data
        # spike_data = np.random.choice([0, 1], p=[1 - self.spike_prob/100, self.spike_prob/100], size=self.s_out.shape[0])
        
        # Send spikes
        self.s_out.send(spike_data)