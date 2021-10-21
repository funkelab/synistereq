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
