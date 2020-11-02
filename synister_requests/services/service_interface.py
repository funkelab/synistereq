from abc import ABC

known_datasets = ["FAFB", "HEMI"]

class ServiceInterface(ABC):
    def __init__(self, dataset, credentials=None):
        if not dataset in known_datasets:
            raise ValueError(f"Dataset {dataset} not known")
        self.dataset = dataset
        self.credentials = credentials
        super().__init__()

    @abstractmethod
    def get_pre_synaptic_positions(self, skid):
        """
        Get presynaptic positions from skeleton id or 
        similar concept in the given service.
        """
        pass

    @staticmethod
    @abstractmethod
    def transform_position(x,y,z):
        """
        Transform position from service coordinate system
        to underlying dataset.
        """
        pass
