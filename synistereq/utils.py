import pandas

def read_skids_csv(csv_path):
    """
    The csv needs to have one column called 'skid'
    """
    data = pandas.read_csv(csv_path)
    skids = data["skid"].to_list()
    skids = list(set([int(skid) for skid in skids]))
    return skids

def read_positions_csv(csv_path):
    """
    The csv needs to have three columns called "id", "x", "y", "z"
    """
    data = pandas.read_csv(csv_path)
    x_list = data["x"].to_list()
    y_list = data["y"].to_list()
    z_list = data["z"].to_list()
    position_ids = data["id"].to_list()

    positions = [(z,y,x) for z,y,x in zip(z_list, y_list, x_list)]
    return positions, position_ids

def format_predictions(nt_probabilities, positions, position_ids, ids_to_skids):
    pass






