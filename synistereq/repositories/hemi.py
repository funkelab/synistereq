import daisy
import numpy as np

from synistereq.datasets import Hemi
from synistereq.interfaces import Neuprint
from .repository import Repository

class HemiNeuprint(Repository):
    def __init__(self):
        dataset = Hemi()
        self.x_shape = daisy.open_ds(dataset.container, dataset.dataset).shape[0]
        super().__init__(dataset=dataset, service=Neuprint(dataset=dataset))

    def transform_positions(self, positions):
        transformed_positions = np.asarray(positions)

        # Apply rotation between exported dataset and neuprint.
        # Permute axes [z, y, x] => [x, z, y]
        transformed_positions = transformed_positions[:,[2, 0, 1]]
        # Set x = dataset_x_max - x - 1
        transformed_positions[:, 0] = self.x_shape - transformed_positions[:, 0] - 1
        return self.service.transform_positions(transformed_positions)
