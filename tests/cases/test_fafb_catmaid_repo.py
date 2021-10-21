import unittest
from synistereq.repositories import FafbCatmaid

class CatmaidInitTestCase(unittest.TestCase):
    def runTest(self):
        repo = FafbCatmaid()
        self.assertTrue(repo.dataset.name == "FAFB")

class CatmaidGetPreSynapticPositionsTestCase(unittest.TestCase):
    def runTest(self):
        repo = FafbCatmaid()
        presynaptic_positions = repo.service.get_pre_synaptic_positions(16)[0]
        test_pos = (217440, 164242, 438817)

        # These tests may fail if there are large annotations changes:
        self.assertTrue(test_pos in presynaptic_positions)
        self.assertTrue(len(presynaptic_positions)>1700)
        self.assertTrue(len(presynaptic_positions)<2000)

class CatmaidTransformPositionsTestCase(unittest.TestCase):
    def runTest(self):
        repo = FafbCatmaid()
        test_pos = (217440, 164242, 438817)
        target_pos = (217400, 164242, 438817)
        transformed_pos = repo.transform_positions([test_pos])
        self.assertTrue(target_pos == transformed_pos[0])

if __name__ == "__main__":
    unittest.main()
