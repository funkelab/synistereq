from abc import ABC, abstractmethod
import daisy
import os
import numpy as np

import logging

log = logging.getLogger(__name__)

class Dataset(ABC):
    def __init__(self, name, container, dataset, voxel_size):
        if not os.path.exists(container):
            raise ValueError(f"Container {container} does not exist")
        if not os.path.exists(container + f"/{dataset}"):
            raise ValueError(f"Dataset {dataset} does not exist")
        self.name = name
        self.container = container
        self.dataset = dataset
        self.voxel_size = voxel_size
        super().__init__()

    def open_daisy(self):
        """
        Open this dataset as a daisy array.
        """
        data = daisy.open_ds(self.container, self.dataset)

        # Correct for datasets where the container does not have the voxel size
        if data.voxel_size != tuple(self.voxel_size):
            log.warn(
                "Container has different voxel size than dataset: "\
                f"{data.voxel_size} != {self.voxel_size}")
            orig_shape = data.roi.get_shape()
            data = daisy.Array(
                data.data,
                daisy.Roi(
                    data.roi.get_offset(),
                    self.voxel_size*data.data.shape[-len(self.voxel_size):]),
                self.voxel_size,
                chunk_shape=data.chunk_shape)
            log.warn(
                "Reloaded container data with dataset voxel size, changing shape: "\
                f"{orig_shape} => {data.roi.get_shape()}")

        return data

    def get_crops(self,
                  center_positions,
                  size):
        """
        Args:

            center_positions (list of tuple of ints): [(z,y,x)] in nm

            size (tuple of ints): size of the crop, in voxels
        """


        crops = []
        size = daisy.Coordinate(size)
        voxel_size = daisy.Coordinate(self.voxel_size)
        size_nm = (size*voxel_size)
        dataset = daisy.open_ds(self.container,
                                self.dataset)
    
        dataset_resolution = None
        try:
            dataset_resolution = dataset.voxel_size
        except AttributeError:
            pass

        if dataset_resolution is not None:
            if not np.all(dataset.voxel_size == self.voxel_size):
                raise ValueError(f"Dataset {dataset} resolution missmatch {dataset.resolution} vs {self.voxel_size}")

        for position in center_positions:
            position = daisy.Coordinate(tuple(position))
            offset_nm = position - ((size/2) * voxel_size)
            roi = daisy.Roi(offset_nm, size_nm).snap_to_grid(voxel_size, mode='closest')

            if roi.get_shape()[0] != size_nm[0]:
                roi.set_shape(size_nm)

            if not dataset.roi.contains(roi):
                raise Warning(f"Location {position} is not fully contained in dataset")
                return

            crops.append(dataset[roi].to_ndarray())

        crops_batched = np.stack(crops)
        crops_batched = crops_batched.astype(np.float32)

        return crops_batched

    
    def normalize(self, 
                  data):
        assert np.min(data) >= 0, "Data type not supported for normalization"
        assert np.max(data) <= 255, "Data type not supported for normalization"
        data = data/255.0
        data = data*2.0 - 1.0
        return data

