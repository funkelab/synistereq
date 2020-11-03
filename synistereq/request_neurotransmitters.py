from synistereq.datasets import Fafb, Hemi
from synistereq.models import FafbModel, HemiModel
from synistereq.interfaces import Catmaid, Neuprint
from synistereq import predict_neurotransmitters

known_datasets = {"FAFB": Fafb, "HEMI": Hemi}
known_models = {"FAFB_MODEL": FafbModel, "HEMI_MODEL": HemiModel}
known_services = {"CATMAID": Catmaid, "NEUPRINT": Neuprint}

def request_neurotransmitters(dataset,
                              service,
                              skids=None,
                              positions=None,
                              position_ids=None,
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
                    positions)

    dataset = known_datasets[dataset]
    model = known_models[f"{dataset}_MODEL"]
    service = known_services[service]

    positions, position_ids = prepare_positions(positions, position_ids)
    positions, position_ids, ids_to_skids = prepare_skids(skids, service, positions, position_ids)
    nt_probabilities = predict_neurotransmitters(positions, dataset, model, batch_size)

    return nt_probabilities, positions, position_ids, ids_to_skids
    
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

def prepare_skids(skids, service, positions, position_ids):
    ids_to_skid = {id_: None for id_ in position_ids}

    if skids is not None:
        for skid in skids:
            positions_skid, ids_skid = service.get_pre_synaptic_positions(skid)
            positions.extend(positions_skid)
            position_ids.extend(ids_skid)
            for id_ in ids_skid:
                ids_to_skid[id_] = skid

    return positions, position_ids, ids_to_skid

