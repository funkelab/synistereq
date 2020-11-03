import unittest
import os
import torch
import numpy as np

from synistereq.models import FafbModel

class FafbModelInitTestCase(unittest.TestCase):
    def runTest(self):
        model = FafbModel()
        self.assertTrue(model.dataset == "FAFB")
        self.assertTrue(os.path.basename(model.checkpoint) == "fafb_checkpoint")

class FafbModelInitModelTestCase(unittest.TestCase):
    def runTest(self):
        model = FafbModel()
        torch_model = model.init_model()

class FafbModelRunModelTestCase(unittest.TestCase):
    def runTest(self):
        model = FafbModel()
        torch_model = model.init_model()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_input = torch.zeros((16,160,160))
        model_input = model_input.unsqueeze(0).unsqueeze(0)
        model_input.to(device)
        prediction = torch_model(model_input)

if __name__ == "__main__":
    unittest.main()
