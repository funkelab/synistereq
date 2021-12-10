from abc import ABC, abstractmethod
import logging

from synistereq.transformer import Transformer

log = logging.getLogger(__name__)

class ServiceInterface(Transformer):
    def __init__(self, dataset, name, credentials=None):
        self.dataset = dataset
        self.credentials = credentials
        self.name = name
        super().__init__()

    def get_state_metadata(self):
        """
        Returns any metadata about the state or version of the service's data.
        """
        raise NotImplementedError()

    def pre_synapse_batches(self, batch_size=None, resume_offset=None):
        """
        Yields in batches all presynaptic positions from the service.

            Args:
                batch_size: approximate size of batches to yield

            Returns:

        """
        raise NotImplementedError()

    @abstractmethod
    def get_pre_synaptic_positions(self, skid):
        """
        Returns presynaptic positions from skeleton id or
        similar concept in the given service.

            Args:
                skid (int): skeleton id of the neuron/segment

            Returns:
                positions (list of tuple of int): list of presynaptic positions in skid [(z1,y1,x1), (z2,y2,x2),...]
                ids (list of ints): associated ids
        """
        pass
