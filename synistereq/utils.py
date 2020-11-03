import pandas

def read_skid_csv(csv_path):
    """
    The csv needs to have one column called 'skid'
    """
    data = pandas.read_csv(csv_path)
    skids = data["skid"].to_list()
    skids = list(set([int(skid) for skid in skids]))
    return skids

def read_position_csv(csv_path):
    """
    The csv needs to have three columns called "x", "y", "z"
    """
    data = pandas.read_csv(csv_path)
    x_list = data["x"].to_list()
    y_list = data["y"].to_list()
    z_list = data["z"].to_list()

    positions = [(z,y,x) for z,y,x in zip(z_list, y_list, x_list)]
    return positions

