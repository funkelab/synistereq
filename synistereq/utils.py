import pandas
import csv

def read_skids_csv(csv_path):
    """
    The csv needs to have one column called 'skid'
    """
    data = pandas.read_csv(csv_path)
    skids = data["skid"].to_list()
    skids = list(set([int(skid) for skid in skids]))
    return skids

def read_prediction_csv(csv_path):
    data = pandas.read_csv(csv_path)
    position_ids = data["position_id"]
    skids = data["skid"]
    x = data["x"]
    y = data["y"]
    z = data["z"]
    nt_list = ["gaba", "acetylcholine", "glutamate", "serotonin", "octopamine", "dopamine"]
    nts = {nt: data[nt] for nt in nt_list}
    return position_ids, skids, x, y, z, nts

def read_positions_csv(csv_path):
    """
    The csv needs to have three columns called "id", "skid", "x", "y", "z"
    """
    data = pandas.read_csv(csv_path)
    x_list = data["x"].to_list()
    y_list = data["y"].to_list()
    z_list = data["z"].to_list()
    position_ids = data["id"].to_list()
    skids = data["skid"].to_list()
    position_ids_to_skids = {id_: skid for id_, skid in zip(position_ids, skids)}


    positions = [(z,y,x) for z,y,x in zip(z_list, y_list, x_list)]
    return positions, position_ids, position_ids_to_skids

def format_predictions(nt_probabilities, positions, position_ids, position_ids_to_skids):
    # Table layout: 
    # position_id, skid, x, y, z, nt1, ..., ntN

    header = ["position_id", "skid", "x", "y", "z"]
    header.extend([v for v in nt_probabilities[0].keys()])
    table = [header]
    for nt_prob, pos, pos_id in zip(nt_probabilities, positions, position_ids):
        skid = position_ids_to_skids[pos_id]
        if skid is None:
            skid = -1
        x = pos[2]
        y = pos[1]
        z = pos[0]

        row = [int(pos_id), skid, x, y, z]
        row.extend([v for v in nt_prob.values()])
        table.append(row)

    return table

def write_predictions(formatted_predictions, output_file):
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(formatted_predictions) 
