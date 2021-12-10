import torch
import daisy
import numpy as np
from torch.utils.data import Dataset, DataLoader

from synistereq.datasets import Fafb

import logging

log = logging.getLogger(__name__)

def get_data_loader(positions, dataset, size, batch_size, num_workers, prefetch_factor):
    dset = PositionDataset(positions, dataset, size)
    if len(dset) < len(positions):
        log.warn(f"Filtered out {len(positions) - len(dset)} postions")
    def collate_fn(samples):
        return torch.cat(samples, dim=0)

    loader = DataLoader(dset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        prefetch_factor=prefetch_factor,
                        collate_fn=collate_fn,
                        persistent_workers=True,
                        pin_memory=True)
    return loader

class PositionDataset(Dataset):
    def __init__(self,
                 positions,
                 dataset,
                 size):

        self.dataset = dataset
        self.data = dataset.open_daisy()

        self.voxel_size = daisy.Coordinate(dataset.voxel_size)
        self.size = daisy.Coordinate(size)
        self.size_nm = self.size * self.voxel_size

        self.positions = self.filter_positions(positions)
        self.transform = None

    def filter_positions(self, positions):
        """
        Filter out those positions whose roi is not fully
        contained in data.
        """
        positions_filtered = []

        data_roi = self.data.roi
        for p in positions:
            p_daisy = daisy.Coordinate(p)
            offset_nm = p_daisy - self.size_nm/2
            p_roi = daisy.Roi(offset_nm, self.size_nm).snap_to_grid(self.voxel_size, mode='closest')
            if data_roi.contains(p_roi):
                positions_filtered.append(p)

        return positions_filtered

    def __getitem__(self, idx):
        position = daisy.Coordinate(self.positions[idx])
        offset_nm = position - self.size_nm/2
        roi = daisy.Roi(offset_nm, self.size_nm).snap_to_grid(self.voxel_size, mode='closest')
        if roi.get_shape()[0] != self.size_nm[0]:
            roi.set_shape(self.size_nm)

        array = self.data[roi]
        array.materialize()
        array_data = array.data.astype(np.float32)
        array_data = self.normalize(array_data)
        array_data = self.transform_to_tensor(array_data)
        return array_data

    def __len__(self):
        return len(self.positions)

    def transform_to_tensor(self, data_array):
        tensor_array = torch.tensor(data_array)
        # Add channel and batch dim:
        tensor_array = tensor_array.unsqueeze(0).unsqueeze(0)
        return tensor_array

    def normalize(self, data_array):
        return self.dataset.normalize(data_array)
