import unittest
import daisy
from synistereq.repositories import HemiNeuprint

class NeuprintInitTestCase(unittest.TestCase):
    def runTest(self):
        interface = HemiNeuprint()
        self.assertTrue(interface.dataset.name == "HEMI")

class NeuprintGetPreSynapticPositionsTestCase(unittest.TestCase):
    def runTest(self):
        repo = HemiNeuprint()
        presynaptic_positions, ids = repo.service.get_pre_synaptic_positions(1279775082)
        presynaptic_positions = repo.transform_positions(presynaptic_positions)
        test_position = (4480, 30584, 173808)
        self.assertTrue(test_position in presynaptic_positions)

class NeuprintTransformPositionCase(unittest.TestCase):
    def runTest(self):
        repo = HemiNeuprint()
        test_position = (80,100,120)
        [interface_transform] = repo.transform_positions([test_position])

        z = test_position[0]
        y = test_position[1]
        x = test_position[2]
        n5 = "/nrs/flyem/data/tmp/Z0115-22.export.n5"
        ds = daisy.open_ds(n5, "22-34/s0")
        X,Z,Y = ds.shape
        manual_transform = (X-x-1, z, y) * repo.dataset.voxel_size
        self.assertEqual(interface_transform, tuple(manual_transform))

if __name__ == "__main__":
    unittest.main()
