from abc import ABC, abstractmethod
import logging
import time

log = logging.getLogger(__name__)

known_datasets = ["FAFB", "HEMI"]

class ServiceInterface(ABC):
    def __init__(self, dataset, name, credentials=None):
        if not dataset in known_datasets:
            raise ValueError(f"Dataset {dataset} not known")
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

    @abstractmethod
    def transform_position(self, position):
        """
        Returns position in self.dataset space if given
        positions in service space

            Args:
                position (tuple of ints): (z,y,x) position in service space.

            Returns:
                transformed_position (tuple if ints): (z,y,x) position in dataset space.
        """
        pass

    def transform_positions(self, positions):
        """
        Transform positions [(z,y,x), (z,y,x)] from service coordinate system
        to underlying dataset.
            
            Args:
                positions (list of tuple of ints): [(z,y,x) ...] list of positions in service space.

            Returns:
                transformed_positions (list of tuple of ints): [(z,y,x), ...] list of positions in dataset space.
        """
        log.info(f"Transform {len(positions)}, assuming {self.name} coordinates...")
        start = time.time()
        transformed_positions = [self.transform_position(p) for p in positions]
        log.info(f"... took {time.time() - start} seconds.")
        return transformed_positions

