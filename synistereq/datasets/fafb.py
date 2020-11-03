from .dataset import Dataset
import numpy as np

class Fafb(Dataset):
    def __init__(self):
        name = "FAFB"
        container = "/nrs/saalfeld/FAFB00/v14_align_tps_20170818_dmg.n5"
        dataset = "volumes/raw/s0"
        voxel_size = np.array([40,4,4])
        super().__init__(name, container, dataset, voxel_size)

