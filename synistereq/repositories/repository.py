from abc import abstractmethod
import logging
import time

from synistereq.transformer import Transformer

log = logging.getLogger(__name__)

class Repository(Transformer):
    def __init__(self, dataset, service, name=None):
        self.dataset = dataset
        self.service = service
        if name is None:
            name = "_".join([self.dataset.name, self.service.name])
        self.name = name
        super().__init__()

    @classmethod
    def for_service_constructor(cls, dataset, service_constructor, name=None, **service_kwargs):
        service = service_constructor(dataset=dataset, **service_kwargs)
        return cls(dataset, service, name=name)

    def transform_positions(self, positions):
        """
        Transform positions [(z,y,x), (z,y,x)] from service coordinate system
        to underlying dataset.

            Args:
                positions (list of tuple of ints): [(z,y,x) ...] list of positions in service space.

            Returns:
                transformed_positions (list of tuple of ints): [(z,y,x), ...] list of positions in dataset space.
        """
        return self.service.transform_positions(positions)
