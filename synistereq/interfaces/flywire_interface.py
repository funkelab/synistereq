from service_interface import ServiceInterface

class FlyWire(ServiceInterface):
    def __init__(self, credentials=None):
        dataset = "FAFB"
        super().__init__(dataset, credentials)

    def __transform_position(self, position):
        raise NotImplementedError("Transformation not available")

    def get_pre_synaptic_positions(self, skid):
        raise NotImplementedError("Service API not found")

