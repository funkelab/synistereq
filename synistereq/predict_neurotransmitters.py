import numpy as np
import torch
from tqdm import tqdm

from synistereq.datasets import Fafb, Hemi
from synistereq.models import FafbModel, HemiModel
from synistereq.interfaces import Catmaid, Neuprint

import time
import logging

log = logging.getLogger(__name__)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

known_datasets = {"FAFB": Fafb, "HEMI": Hemi}
known_models = {"FAFB_MODEL": FafbModel, "HEMI_MODEL": HemiModel}
known_services = {"CATMAID": Catmaid, "NEUPRINT": Neuprint}

def predict_neurotransmitters(dataset,
                              service=None,
                              skids=None,
                              positions=None,
                              position_ids=None,
                              position_ids_to_skids=None,
                              batch_size=24):
    """
    Args:
        positions (list of tuple of int):

        skids (list of int):

        position_ids (list of int, optional):
    """

    check_arguments(dataset,
                    service,
                    skids,
                    positions,
                    position_ids)

    model = known_models[f"{dataset}_MODEL"]()
    dataset = known_datasets[dataset]()
    service = known_services[service]()

    positions, position_ids = prepare_positions(positions, position_ids)
    positions, position_ids, position_ids_to_skids = prepare_skids(skids, service, positions,
                                                          position_ids, position_ids_to_skids)
    positions = service.transform_positions(positions)
    nt_probabilities = predict_neurotransmitters_from_positions(positions, dataset, model, batch_size)

    return nt_probabilities, positions, position_ids, position_ids_to_skids

def predict_neurotransmitters_from_positions(positions,
                                             dataset,
                                             model,
                                             batch_size=24):
    """
    positions `list of array-like of ints`:
        Synaptic postions in the given dataset [(z0,y0,x0), (z1, y1, x1), ...]

    returns:
        list of dictionaries of neurotransmitter probabilities
    """
    log.info(f"Predicting neurotransmitters for {len(positions)} locations...")
    log.info(f"Dataset {dataset.name}, Model: {model.dataset}")

    if dataset.name != model.dataset:
        raise ValueError(f"Dataset ({dataset.name}) model ({model.dataset}) missmatch")

    torch_model = model.init_model()
    torch_model.eval()

    log.info(f"Start prediction with batch size {batch_size}...")
    start = time.time()
    nt_probabilities = []
    t_normalize = 0
    t_get_crops = 0
    t_predict = 0
    for i in tqdm(range(0, len(positions), batch_size)):
        batched_positions = positions[i:i+batch_size]
        t0 = time.time()
        crops = dataset.get_crops(batched_positions,
                                  model.input_shape)
        t_get_crops += time.time() - t0

        t0 = time.time()
        crops = dataset.normalize(crops)
        t_normalize += time.time() - t0

        crops = model.prepare_batch(crops)
        
        t0 = time.time()
        prediction = torch_model(crops)
        prediction = model.softmax(prediction)
        t_predict += time.time() - t0

        # Iterate over batch and grab predictions
        for k in range(np.shape(prediction)[0]):
            out_k = prediction[k,:].tolist()
            nt_probability = {model.neurotransmitter_list[i]:
                              out_k[i] for i in range(len(model.neurotransmitter_list))}
            nt_probabilities.append(nt_probability)

    n_positions = len(positions)
    total_time = time.time() - start
    log_stats(n_positions, 
              total_time, 
              t_normalize, 
              t_get_crops, 
              t_predict)

    return nt_probabilities

def log_stats(n_positions, total_time, t_normalize, t_get_crops, t_predict):
    p_normalize = t_normalize/total_time * 100
    p_get_crops = t_get_crops/total_time * 100
    p_predict = t_predict/total_time * 100
    log.info(f"Predicition of {n_positions} took {total_time} seconds.") 
    log.info(f"{total_time/n_positions} per position.")
    log.info(f"Stats:")
    log.info(f"Predicted positions: {n_positions}")
    log.info(f"Total time: {total_time}")
    log.info(f"Time per position: {total_time/n_positions}")
    log.info(f"Normalize: {t_normalize} ({p_normalize}%)")
    log.info(f"Fetch: {t_get_crops} ({p_get_crops}%)")
    log.info(f"Predict: {t_predict} ({p_predict}%)")

def check_arguments(dataset, service, skids, positions, position_ids):
    log.info("Check arguments...")

    if skids is None and positions is None:
        raise ValueError("Provide either skids or positions")
    if service is None and not skids is None:
        raise ValueError("Provide service if querying for skids")
    if not dataset in list(known_datasets.keys()):
        raise ValueError(f"Dataset f{dataset} not known")
    if not service in list(known_services.keys()):
        raise ValueError(f"Service f{service} not known")
    if not f"{dataset}_MODEL" in list(known_models.keys()):
        raise ValueError(f"Model f{dataset}_MODEL not known")

def prepare_positions(positions, position_ids):
    log.info("Prepare positions...")

    if positions is None:
        positions = []
        position_ids = []
    else:
        if position_ids is None:
            position_ids = [i for i in range(len(positions))]

    return positions, position_ids

def prepare_skids(skids, service, positions, position_ids, position_ids_to_skids):
    log.info("Prepare skids...")

    if skids is not None:
        for skid in skids:
            positions_skid, ids_skid = service.get_pre_synaptic_positions(skid)
            positions.extend(positions_skid)
            position_ids.extend(ids_skid)
            for id_ in ids_skid:
                position_ids_to_skids[id_] = skid

    return positions, position_ids, position_ids_to_skids

