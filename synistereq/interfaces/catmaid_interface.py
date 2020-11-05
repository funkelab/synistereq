import pymaid
import configparser
import os
import numpy as np

from .service_interface import ServiceInterface

class Catmaid(ServiceInterface):
    def __init__(self, 
                 credentials=os.path.join(os.path.abspath(os.path.dirname(__file__)), 
                                                          "../credentials/catmaid_credentials.ini")):
        dataset = "FAFB"
        name = "CATMAID"
        super().__init__(dataset, name, credentials)
        self.instance = self.__get_instance(self.credentials)

    def transform_position(self, position):
        return(position[0] - 40, position[1], position[2])

    def get_pre_synaptic_positions(self, skid):
        pymaid.clear_cache()
        connectors = pymaid.get_connectors(skid, relation_type='presynaptic_to')
        x = connectors["x"].to_numpy()
        y = connectors["y"].to_numpy()
        z = connectors["z"].to_numpy()
        pos_array, ids = np.vstack([z,y,x]).T, connectors["connector_id"].to_numpy()
        return [tuple(np.round(p).astype(np.uint64)) for p in pos_array], ids


    def __get_instance(self, credentials):
        with open(credentials) as fp:
            config = configparser.ConfigParser()
            config.readfp(fp)
            user = config.get("Credentials", "user")
            password = config.get("Credentials", "password")
            token = config.get("Credentials", "token")

            rm = pymaid.CatmaidInstance('https://neuropil.janelia.org/tracing/fafb/v14/',
                                        token,
                                        user,
                                        password)
        return rm
