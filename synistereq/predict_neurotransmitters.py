import numpy as np
import torch
from tqdm import tqdm

from synistereq.datasets import Fafb, Hemi, MaleVnc
from synistereq.models import FafbModel, HemiModel, MaleVncModel
from synistereq.loader import get_data_loader
from synistereq.repositories import KNOWN_REPOSITORIES

import time
import logging

log = logging.getLogger(__name__)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

known_datasets = {"FAFB": Fafb, "HEMI": Hemi, "MALE_VNC": MaleVnc}
known_models = {"FAFB_MODEL": FafbModel, "HEMI_MODEL": HemiModel, "MALE_VNC_MODEL": MaleVncModel}

def predict_neurotransmitters(dataset=None,
                              neither=False,
                              repository=None,
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
                    repository,
                    skids,
                    positions,
                    position_ids)

    if repository is not None:
        repository = KNOWN_REPOSITORIES[repository]()
    if dataset is None:
        dataset = repository.dataset
    else:
        dataset = known_datasets[dataset]()

    model = known_models[f"{dataset.name.upper()}_MODEL"]()
    if(neither):
        model.neurotransmitter_list.append("neither")

    positions, position_ids = prepare_positions(positions, position_ids)
    if skids is not None:
        positions, position_ids, position_ids_to_skids = prepare_skids(skids, repository.service, positions,
                                                            position_ids, position_ids_to_skids)
    if repository is not None:
        positions = repository.transform_positions(positions)
    nt_probabilities = predict_neurotransmitters_from_positions(positions, dataset, model, batch_size)

    return nt_probabilities, positions, position_ids, position_ids_to_skids

def predict_neurotransmitters_from_positions(positions,
                                             dataset,
                                             model,
                                             batch_size=24,
                                             prefatch_factor=30,
                                             num_workers=5):
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
    loader = get_data_loader(positions,
                             dataset,
                             model.input_shape,
                             batch_size,
                             num_workers,
                             prefatch_factor)

    nt_probabilities = []
    t_predict = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for sample in tqdm(loader, desc="Batches"):
        sample = sample.to(device)
        t0 = time.time()
        prediction = torch_model(sample)
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
              t_predict)

    return nt_probabilities

def log_stats(n_positions, total_time, t_predict):
    p_predict = t_predict/total_time * 100
    log.info(f"Predicition of {n_positions} took {total_time} seconds.")
    log.info(f"{total_time/n_positions} per position.")
    log.info(f"Stats:")
    log.info(f"Predicted positions: {n_positions}")
    log.info(f"Total time: {total_time}")
    log.info(f"Time per position: {total_time/n_positions}")
    log.info(f"Predict: {t_predict} ({p_predict}%)")

def check_arguments(dataset, repository, skids, positions, position_ids):
    log.info("Check arguments...")

    if skids is None and positions is None:
        raise ValueError("Provide either skids or positions")
    if repository is None and not skids is None:
        raise ValueError("Provide repository if querying for skids")
    if dataset is not None and not dataset in list(known_datasets.keys()):
        raise ValueError(f"Dataset {dataset} not known")
    if repository is not None and not repository in list(KNOWN_REPOSITORIES.keys()):
        raise ValueError(f"Repository {repository} not known")

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
        for skid in tqdm(skids, desc="Fetching synapse positions for skids"):
            positions_skid, ids_skid = service.get_pre_synaptic_positions(skid)
            positions.extend(positions_skid)
            position_ids.extend(ids_skid)
            for id_ in ids_skid:
                position_ids_to_skids[id_] = skid

    return positions, position_ids, position_ids_to_skids

