from .dataset import Dataset
import numpy as np

class Hemi(Dataset):
    def __init__(self):
        name = "HEMI"
        container = "/nrs/flyem/data/tmp/Z0115-22.export.n5"
        dataset = "22-34/s0"
        voxel_size = np.array([8,8,8])
        super().__init__(name, container, dataset, voxel_size)

