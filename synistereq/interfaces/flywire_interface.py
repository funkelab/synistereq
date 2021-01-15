from .service_interface import ServiceInterface
from .catmaid_interface import Catmaid
from synistereq.datasets import Fafb
import requests
import numpy as np
import json

class Flywire(ServiceInterface):
    def __init__(self, credentials=None):
        dataset = Fafb()
        name = "FLYWIRE"
        super().__init__(dataset, name, credentials)
        self.request_base_url = "https://spine.janelia.org/app/transform-service"

    def transform_position(self, position):
        raise NotImplementedError("Transformation not supported for a single point, use transform_positions")
        return None

    def transform_positions(self, positions, batch_size=1000, scale=4):
        """
        Positions always in physical space. (z,y,x) convention.

        Returns:
            Positions in fafb space in physical coordinates, zyx convention.
        """
        # This in fact goes from flywire to catmaid
        # Coordinates have to be given in voxel space
        # For catmaid to flywire use flywire_v1_inverse
        request_url = f"/dataset/flywire_v1/s/{scale}/values_array"
        request_url = self.request_base_url + request_url

        transformed_positions = []

        for i in range(0, len(positions), batch_size):
            batched_positions = positions[i:i+batch_size]
            # Type conversion needed here to make it json serializable
            batched_positions = np.transpose(batched_positions).tolist()
            q = {"x": list(batched_positions[2]/self.dataset.voxel_size[2]),
                 "y": list(batched_positions[1]/self.dataset.voxel_size[1]),
                 "z": list(batched_positions[0]/self.dataset.voxel_size[0])}
            r = requests.post(request_url, json=q, timeout=10)
            r.raise_for_status()
            json_data = json.loads(r.text)
            x_transformed = json_data["x"]
            y_transformed = json_data["y"]
            z_transformed = json_data["z"]
            transformed = np.round(np.transpose([z_transformed, y_transformed, x_transformed])).astype(np.uint32)
            transformed = [tuple(p*self.dataset.voxel_size) for p in transformed]
            transformed_positions.extend(transformed)

        # Service converts to catmaid space (+1 offset to n5 volume)
        catmaid = Catmaid()
        transformed_positions = catmaid.transform_positions(transformed_positions)
        return transformed_positions

    def get_pre_synaptic_positions(self, skid):
        raise NotImplementedError("Service API not found")

