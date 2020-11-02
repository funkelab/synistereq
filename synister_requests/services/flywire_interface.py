from service_interface import ServiceInterface

class FlyWire(ServiceInterface):
    def __init__(self, credentials=None):
        super().__init__(dataset, credentials)

    def get_pre_synaptic_positions(self, skid):
        raise NotImplementedError("Service API not found")

    @staticmethod
    def transform_position(x,y,z):
        pass

