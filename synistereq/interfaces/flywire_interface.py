from .service_interface import ServiceInterface

class Flywire(ServiceInterface):
    def __init__(self, credentials=None):
        dataset = "FAFB"
        name = "FLYWIRE"
        super().__init__(dataset, name, credentials)

    def __transform_position(self, position):
        raise NotImplementedError("Transformation not available")

    def get_pre_synaptic_positions(self, skid):
        raise NotImplementedError("Service API not found")

