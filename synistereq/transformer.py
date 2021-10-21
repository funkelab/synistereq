from abc import ABC

class Transformer(ABC):
    def transform_positions(self, positions):
        """
        Transform positions [(z,y,x), (z,y,x)] from service coordinate system (physical!)
        to underlying dataset.

            Args:
                positions (list of tuple of ints): [(z,y,x) ...] list of positions in service space.

            Returns:
                transformed_positions (list of tuple of ints): [(z,y,x), ...] list of positions in dataset space.
        """
        return positions

