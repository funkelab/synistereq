from .service_interface import ServiceInterface
import numpy as np

class Neuprint(ServiceInterface):
    def __init__(self, credentials=None):
        dataset = "HEMI"
        name = "NEUPRINT"
        super().__init__(dataset, name, credentials)

    def transform_position(self, position):
        """
        n5 = "/nrs/flyem/data/tmp/Z0115-22.export.n5"
    	ds = daisy.open_ds(n5, "22-34/s0")
    	x_shape,z_shape,y_shape = ds.shape

        voxel_hemi = n5_vol[X-x-1, z, y]
        if x, y, z from neuprint (note this is physical)
        with daisy: Array[(X-x-1,z,y)*voxel_size]
        """
        x_shape_hemi = 34427 
        z = position[0]
        y = position[1]
        x = position[2]

        transformed_position = np.array([x_shape_hemi - x - 1, z, y])
        transformed_position *= 8 # Neuprint coords are physical
        return tuple(transformed_position)

    def get_pre_synaptic_positions(self, skid):
        raise NotImplementedError("Service API not found")

