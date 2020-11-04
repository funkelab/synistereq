import numpy as np
from tqdm import tqdm

from synistereq.datasets import Fafb, Hemi
from synistereq.models import FafbModel, HemiModel
from synistereq.interfaces import Catmaid, Neuprint

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
    if dataset.name != model.dataset:
        raise ValueError(f"Dataset ({dataset.name}) model ({model.dataset}) missmatch")

    torch_model = model.init_model()
    torch_model.eval()

    nt_probabilities = []
    for i in tqdm(range(0, len(positions), batch_size)):
        batched_positions = positions[i:i+batch_size]
        crops = dataset.get_crops(positions,
                                  model.input_shape)
        crops = dataset.normalize(crops)
        crops = model.prepare_batch(crops)
        prediction = torch_model(crops)
        prediction = model.softmax(prediction)

        # Iterate over batch and grab predictions
        for k in range(np.shape(prediction)[0]):
            out_k = prediction[k,:].tolist()
            nt_probability = {model.neurotransmitter_list[i]:
                              out_k[i] for i in range(len(model.neurotransmitter_list))}
            nt_probabilities.append(nt_probability)

    return nt_probabilities

def check_arguments(dataset, service, skids, positions, position_ids):
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
    if positions is None:
        positions = []
        position_ids = []
    else:
        if position_ids is None:
            position_ids = [i for i in range(len(positions))]

    return positions, position_ids

def prepare_skids(skids, service, positions, position_ids, position_ids_to_skids):
    if skids is not None:
        for skid in skids:
            positions_skid, ids_skid = service.get_pre_synaptic_positions(skid)
            positions.extend(positions_skid)
            position_ids.extend(ids_skid)
            for id_ in ids_skid:
                position_ids_to_skids[id_] = skid

    return positions, position_ids, position_ids_to_skids

