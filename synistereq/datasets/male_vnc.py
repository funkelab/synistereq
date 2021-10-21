from .dataset import Dataset
import numpy as np

class MaleVnc(Dataset):
    def __init__(self):
        name = "MALE_VNC"
        container = "/nrs/flyem/tmp/VNC-export-v3.n5"
        dataset = "2-26/s0"
        voxel_size = np.array([8,8,8])
        super().__init__(name, container, dataset, voxel_size)

