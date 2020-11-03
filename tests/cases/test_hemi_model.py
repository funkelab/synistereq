import unittest
import os
import torch
import numpy as np

from synistereq.models import HemiModel

class HemiModelInitTestCase(unittest.TestCase):
    def runTest(self):
        model = HemiModel()
        self.assertTrue(model.dataset == "HEMI")
        self.assertTrue(os.path.basename(model.checkpoint) == "hemi_checkpoint")

class HemiModelInitModelTestCase(unittest.TestCase):
    def runTest(self):
        model = HemiModel()
        torch_model = model.init_model()

class HemiModelRunModelTestCase(unittest.TestCase):
    def runTest(self):
        model = HemiModel()
        torch_model = model.init_model()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_input = torch.zeros((80,80,80))
        model_input = model_input.unsqueeze(0).unsqueeze(0)
        model_input.to(device)
        prediction = torch_model(model_input)

if __name__ == "__main__":
    unittest.main()
