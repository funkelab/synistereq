from neuprint import Client, NeuronCriteria, SynapseCriteria, fetch_synapses
import numpy as np
import configparser
import os
import time

from .service_interface import ServiceInterface

class Neuprint(ServiceInterface):
    def __init__(
        self,
        dataset,
        server=None,
        neuprint_dataset=None,
        token=None,
        credentials=os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                                "../neuprint_credentials.ini"),
    ):

        name = "NEUPRINT"
        super().__init__(dataset, name, credentials)
        self.server = server
        self.neuprint_dataset = neuprint_dataset
        self.token = token
        self.instance = self.__get_instance()

    def transform_positions(self, positions):
        transformed_positions = np.asarray(positions)
        transformed_positions = np.apply_along_axis(np.multiply, 1, transformed_positions, self.dataset.voxel_size)
        return list(map(tuple, transformed_positions.astype(np.uint64)))

    def get_state_metadata(self):
        metadata = self.instance.fetch_datasets()[self.instance.dataset]
        metadata = {k: metadata[k] for k in ["last-mod", "uuid"]}

        return metadata

    @staticmethod
    def _synapse_ids(synapses):
        """
        Get unique IDs for synapses in a dataframe based on position.

        See: https://github.com/connectome-neuprint/neuprint-python/issues/21
        """
        coords = synapses[[*'zyx']].astype(np.int64).values
        ids = (coords[:, 0] << 42) | (coords[:, 1] << 21) | (coords[:, 2] << 0)

        return ids

    def pre_synapse_batches(self, batch_size=None, resume_offset=None):
        if batch_size is None:
            batch_size = 100_000
        offset = resume_offset if resume_offset is not None else 0
        retry = 0
        MAX_RETRIES = 10

        while True:
            try:
                batch = self.instance.fetch_custom(f"""
                        MATCH (s: Synapse {{type: "pre"}})
                        RETURN ID(s) AS cypher_id, s.location.x AS x, s.location.y AS y, s.location.z AS z
                        ORDER BY ID(s)
                        SKIP {offset}
                        LIMIT {batch_size}
                        """)
                retry = 0
                offset += len(batch)
                if len(batch):
                    batch.index = self._synapse_ids(batch)
                    batch.index.name = 'synapse_id'
                    yield batch
                else:
                    return
            except Exception as e:
                if retry > MAX_RETRIES:
                    raise e
                print(e)
                retry += 1
                time.sleep(0.2 * 2**retry)

    def get_pre_synaptic_positions(self, skid):
        # Fetch neuron with given body id
        neuron_criteria = NeuronCriteria(bodyId=skid)
        # Fetch all presynapses
        synapse_criteria = SynapseCriteria(type='pre')

        connectors = fetch_synapses(neuron_criteria, synapse_criteria)
        connectors.index = self._synapse_ids(connectors)
        connectors.index.name = 'connector_id'

        x = connectors["x"].to_numpy()
        y = connectors["y"].to_numpy()
        z = connectors["z"].to_numpy()
        pos_array, ids = np.vstack([z,y,x]).T, connectors.index.to_numpy()
        return [tuple(np.round(p).astype(np.uint64)) for p in pos_array], ids

    def __get_instance(self):
        if None in [self.server, self.neuprint_dataset, self.token]:
            if os.path.exists(self.credentials):
                with open(self.credentials) as fp:
                    config = configparser.ConfigParser()
                    config.readfp(fp)
                    if self.server is None:
                        self.server = config.get("Credentials", "server")
                    if self.neuprint_dataset is None:
                        self.neuprint_dataset= config.get("Credentials", "dataset")
                    if self.token is None:
                        self.token = config.get("Credentials", "token")

        client = Client(self.server, dataset=self.neuprint_dataset, token=self.token)

        return client

