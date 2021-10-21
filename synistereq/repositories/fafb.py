import numpy as np

from synistereq.datasets import Fafb
from synistereq.interfaces import Catmaid, Flywire
from .repository import Repository

class FafbCatmaid(Repository):
    def __init__(self):
        dataset = Fafb()
        service = Catmaid(dataset=dataset, api_url="https://neuropil.janelia.org/tracing/fafb/v14/")
        super().__init__(dataset=dataset, service=service)

    def transform_positions(self, positions):
        transformed_positions = np.asarray(positions)

        # Apply 1-section offset between CATMAID and dataset.
        transformed_positions[:, 0] = transformed_positions[:, 0] - self.dataset.voxel_size[0]
        return list(map(tuple, self.service.transform_positions(transformed_positions)))

class FafbFlywire(Repository):
    def __init__(self):
        dataset = Fafb()
        service = Flywire(
            dataset=dataset,
            api_url="https://spine.janelia.org/app/transform-service",
            api_dataset="flywire_v1",
        )
        super().__init__(dataset=dataset, service=service)
        self.catmaid = FafbCatmaid()

    def transform_positions(self, positions):
        transformed_positions = self.service.transform_positions(positions)

        # Also apply FAFB CATMAID tranforms.
        transformed_positions = self.catmaid.transform_positions(transformed_positions)

        return transformed_positions
