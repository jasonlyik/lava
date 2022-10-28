from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import OutPort

from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyOutPort

class SpikeGenerator(AbstractProcess):
	"""Spike generator process provides spikes to subsequent Processes.

	Parameters
	----------
	shape: tuple
		defines the dimensionality of the generated spikes per timestep
	spike_prob: int
		spike probability in percent
	"""
	def __init__(self, shape: tuple, spike_prob: int) -> None:
		super().__init__()
		self.spike_prob = Var(shape=(1, ), init=spike_prob)
		self.s_out = OutPort(shape=shape)

@implements(proc=SpikeGenerator, protocol=LoihiProtocol)
@requires(CPU)
class PySpikeGeneratorModel(PyLoihiProcessModel):
    """Spike Generator process model."""
    spike_prob: int = LavaPyType(int, int)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def run_spk(self) -> None:
        # Generate random spike data
        spike_data = np.random.choice([0, 1], p=[1 - self.spike_prob/100, self.spike_prob/100], size=self.s_out.shape[0])
        
        # Send spikes
        self.s_out.send(spike_data)