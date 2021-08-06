import pandas
import csv
import logging

def read_skids_csv(csv_path):
    """
    The csv needs to have one column called 'skid'
    """
    data = pandas.read_csv(csv_path)
    skids = data["skid"].to_list()
    skids = list(set([int(skid) for skid in skids]))
    return skids

def read_prediction_csv(csv_path, neither):
    data = pandas.read_csv(csv_path)
    position_ids = data["position_id"]
    skids = data["skid"]
    x = data["x"]
    y = data["y"]
    z = data["z"]
    nt_list = ["gaba", "acetylcholine", "glutamate", "serotonin", "octopamine",
            "dopamine"]
    if neither: 
        nt_list.append("neither")
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

def prepare_position_csv(csv_files, output_csv, skid_column=None, 
                         id_column=None, skids_per_csv=None):
    i = 0
    all_position_ids = []
    all_skids = []
    all_x = []
    all_y = []
    all_z = []
    for csv_path in csv_files:
        position_ids = None
        skids = None
        data = pandas.read_csv(csv_path)
        # position columns are required
        x_list = data["x"].to_list()
        y_list = data["y"].to_list()
        z_list = data["z"].to_list()
        all_x.extend(x_list)
        all_y.extend(y_list)
        all_z.extend(z_list)

        if skid_column is not None:
            skids = data[skid_column].to_list()
        else:
            if skids_per_csv is not None:
                skids = [skids_per_csv[i]] * len(x_list)
            else:
                skids = [i] * len(x_list)

        all_skids.extend(skids)

        if id_column is not None:
            position_ids = data[id_column].to_list()
        else:
            id0 = len(all_position_ids)
            id_max = len(x_list) + id0
            position_ids = [k for k in range(id0, id_max)]

        all_position_ids.extend(position_ids)
        i += 1

    header = ["id", "skid", "x", "y", "z"]
    table = [header]
    for id_, skid, x, y, z in zip(all_position_ids, all_skids, all_x, all_y, all_z):
        table.append([id_, skid, x, y, z])
    write_predictions(table, output_csv)

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


def log_config(log_file):
    logging.basicConfig(
         filename=log_file,
         level=logging.INFO,
         format='%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
         datefmt='%H:%M:%S')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


