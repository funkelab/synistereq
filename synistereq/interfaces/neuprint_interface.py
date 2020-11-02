from .service_interface import ServiceInterface

class NeuPrint(ServiceInterface):
    def __init__(self, credentials=None):
        dataset = "HEMI"
        super().__init__(dataset, credentials)

    def transform_position(self, position):
        """
        n5 = "/nrs/flyem/data/tmp/Z0115-22.export.n5"
    	ds = daisy.open_ds(n5, "22-34/s0")
    	x_shape,z_shape,y_shape = ds.shape
        """
        x_shape_hemi = 34427 
        z = position[0]
        y = position[1]
        x = position[2]

        transformed_position = (x_shape_hemi - x - 1, z, y)
        return transformed_position

    def get_pre_synaptic_positions(self, skid):
        raise NotImplementedError("Service API not found")

