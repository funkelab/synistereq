from neuprint import Client, NeuronCriteria, SynapseCriteria, fetch_synapses
import numpy as np
import configparser
import os

from .service_interface import ServiceInterface

class Neuprint(ServiceInterface):
    def __init__(
        self,
        dataset,
        credentials=os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                                "../neuprint_credentials.ini"),
    ):

        name = "NEUPRINT"
        super().__init__(dataset, name, credentials)
        self.instance = self.__get_instance(self.credentials)

    def transform_positions(self, positions):
        transformed_positions = np.asarray(positions)
        transformed_positions = np.apply_along_axis(np.multiply, 1, transformed_positions, self.dataset.voxel_size)
        return list(map(tuple, transformed_positions.astype(np.uint64)))

    def get_pre_synaptic_positions(self, skid):
        # Fetch neuron with given body id
        neuron_criteria = NeuronCriteria(bodyId=skid)
        # Fetch all presynapses
        synapse_criteria = SynapseCriteria(type='pre')

        connectors = fetch_synapses(neuron_criteria, synapse_criteria)
        # Get unique id, see: https://github.com/connectome-neuprint/neuprint-python/issues/21
        coords = connectors[[*'zyx']].astype(np.int64).values
        ids = (coords[:, 0] << 42) | (coords[:, 1] << 21) | (coords[:, 2] << 0)
        connectors.index = ids
        connectors.index.name = 'connector_id'

        x = connectors["x"].to_numpy()
        y = connectors["y"].to_numpy()
        z = connectors["z"].to_numpy()
        pos_array, ids = np.vstack([z,y,x]).T, connectors.index.to_numpy()
        return [tuple(np.round(p).astype(np.uint64)) for p in pos_array], ids
 
    def __get_instance(self, credentials):
        with open(credentials) as fp:
            config = configparser.ConfigParser()
            config.readfp(fp)
            server = config.get("Credentials", "server")
            dataset = config.get("Credentials", "dataset")
            token = config.get("Credentials", "token")

            client = Client(server, dataset=dataset, token=token)

        return client

