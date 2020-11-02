from synister.catmaid_interface import Catmaid
from synister.utils import init_vgg, predict, get_raw, get_raw_dense
from synister.synister_db import SynisterDb
from tqdm import tqdm
import json
import numpy as np
import os
import pandas
import sys
import time
import zarr

import argparse
import torch
parser = argparse.ArgumentParser()

#-db DATABSE -u USERNAME -p PASSWORD -size 20
parser.add_argument("--skids", help="File with skid column, pre synapses will be grabbed from catmaid", required=False)
parser.add_argument("--locs", help="File with x,y,z columns representing synaptic locations", type=str, default=None, required=False)

db_credentials = "/groups/funke/home/ecksteinn/Projects/synex/synister_experiments/fafb/01_data/db_credentials.ini"

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def init_model():
    # Preliminary model
    train_checkpoint = "/nrs/funke/ecksteinn/synister_experiments/hemibrain_v0/02_train/setup_t0/model_checkpoint_300000"
    input_shape = (80,80,80)
    fmaps = 12
    downsample_factors = [(2, 2, 2), (2, 2, 2), (2, 2, 2),(2, 2, 2)]
    neurotransmitter_list = ["gaba", "acetylcholine", "glutamate", 
                             "serotonin", "octopamine", "dopamine"]
    output_classes = len(neurotransmitter_list)

    # Initialize model
    model = init_vgg(train_checkpoint,
                     input_shape,
                     fmaps,
                     downsample_factors,
                     output_classes)

    model_config = {"train_checkpoint": train_checkpoint,
                    "input_shape": input_shape,
                    "fmaps": fmaps,
                    "downsample_factors": downsample_factors,
                    "neurotransmitter_list": neurotransmitter_list,
                    "output_classes": output_classes}

    return model, model_config

def get_neurotransmitter(positions, 
                         model, 
                         model_config, 
                         predict_id=0, 
                         save_batches=1000, 
                         output_dir=".",
                         data_array=None,
                         data_array_offset=(0,0,0)):
    """
    positions `list of array-like of ints`:
        Synaptic postions in the hemibrain [(z0,y0,x0), (z1, y1, x1), ...]

    returns:
        list of dictionaries of neurotransmitter probabilities
    """

    neurotransmitter_list = model_config["neurotransmitter_list"]
    input_shape = model_config["input_shape"]
    output_classes = model_config["output_classes"]
    batch_size = 24

    # Disable Dropout, Batch norm etc.
    model.eval()

    # Fafb v14
    raw_container = "/nrs/flyem/data/tmp/Z0115-22.export.n5"
    raw_dataset = "22-34/s0"
    voxel_size = np.array([8,8,8])

    batch_output_dir = output_dir + "/batches"
    if not os.path.exists(batch_output_dir):
        os.makedirs(batch_output_dir)

    nt_probabilities = []
    for i in tqdm(range(0, len(positions), batch_size)):
        batched_positions = positions[i:i+batch_size]
        if data_array is None:
            raw, raw_normalized = get_raw(batched_positions,
                                          input_shape,
                                          voxel_size,
                                          raw_container,
                                          raw_dataset)
        else:
            raw, raw_normalized = get_raw_dense(batched_positions,
                                                input_shape,
                                                data_array,
                                                data_array_offset,
                                                voxel_size)

        if i % save_batches == 0:
            zarr.save(batch_output_dir +\
                      '/batch_{}_{}.zarr'.format(predict_id, i), 
                      raw_normalized)

        output = predict(raw_normalized, model)

        # Iterate over batch and grab predictions
        for k in range(np.shape(output)[0]):
            out_k = output[k,:].tolist()
            nt_probability = {neurotransmitter_list[i]: out_k[i] for i in range(output_classes)}
            nt_probabilities.append(nt_probability)

    return nt_probabilities


def get_db_neurotransmitters(skids, output_dir, save_batches=100):
    db = SynisterDb(db_credentials, "synister_hemi_v0")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model, model_config = init_model()
    i = 0
    for skid in skids:
        print("Predict {} from {} skids".format(i, len(skids)))
        synapses = db.get_synapses(match_ids=[skid])
        pos = [(s["z"], s["y"], s["x"]) for s in synapses.values()]
        ids = [int(i) for i in synapses.keys()]
        #pos = pos[:1000]
        if len(pos) > 0:
            print("Predict {} positions".format(len(pos)))
            start = time.time()
            nt_probabilities  = get_neurotransmitter(pos, 
                                                     model, 
                                                     model_config,
                                                     skid,
                                                     save_batches,
                                                     output_dir)

            nt_probabilities = [[int(ids[i]), 
                                [int(k) for k in pos[i]], 
                                nt_probabilities[i]] for i in range(len(nt_probabilities))]

            print("{} seconds per synapse".format((time.time() - start)/len(pos)))
        else:
            nt_probabilities = []

        out_file = os.path.join(output_dir, 
                                "skid_{}.json".format(skid))

        with open(out_file, "w+") as f:
            json.dump(nt_probabilities, f)

        i += 1

    out_config = os.path.join(output_dir, "model_config.json")
    with open(out_config, "w+") as f:
        json.dump(model_config, f)

def read_neuron_csv(csv_path):
    """
    The csv needs to have one column called 'skid'
    """
    data = pandas.read_csv(csv_path)
    skids = data["skid"].to_list()
    skids = list(set([int(skid) for skid in skids]))
    return skids

if __name__ == "__main__":
    args = parser.parse_args()

    skid_csv_path = args.skids
    loc_csv_path = args.locs

    if skid_csv_path is None:
        if loc_csv_path is None:
            raise ValueError("Provide list of locations or list of skids")

    if skid_csv_path is not None:
        output_dir = os.path.dirname(skid_csv_path) + "/predictions"
        skids = read_neuron_csv(skid_csv_path)
        get_db_neurotransmitters(skids, output_dir)
