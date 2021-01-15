import unittest
import daisy
from synistereq.interfaces import Neuprint

class NeuprintInitTestCase(unittest.TestCase):
    def runTest(self):
        interface = Neuprint()
        self.assertTrue(interface.dataset.name == "HEMI")

class NeuprintGetPreSynapticPositionsTestCase(unittest.TestCase):
    def runTest(self):
        interface = Neuprint()
        presynaptic_positions, ids = interface.get_pre_synaptic_positions(1279775082)
        presynaptic_positions = interface.transform_positions(presynaptic_positions)
        test_position = (4480, 30584, 173808)
        self.assertTrue(test_position in presynaptic_positions)

class NeuprintTransformPositionCase(unittest.TestCase):
    def runTest(self):
        interface = Neuprint()
        test_position = (80,100,120)
        interface_transform = interface.transform_position(test_position)

        z = test_position[0] 
        y = test_position[1]
        x = test_position[2]
        n5 = "/nrs/flyem/data/tmp/Z0115-22.export.n5"
        ds = daisy.open_ds(n5, "22-34/s0")
        X,Z,Y = ds.shape
        manual_transform = (X-x-1, z, y) * interface.dataset.voxel_size
        self.assertTrue(interface_transform == tuple(manual_transform))

class NeuprintTransformPositionsCase(unittest.TestCase):
    def runTest(self):
        interface = Neuprint()
        test_positions = [(80,100,120), (100,200,300)]
        interface_transforms = interface.transform_positions(test_positions)
        interface_transforms_individual = [interface.transform_position(p) for p in test_positions]
        for p, p_individual in zip(interface_transforms, interface_transforms_individual):
            self.assertTrue(p==p_individual)

if __name__ == "__main__":
    unittest.main()
