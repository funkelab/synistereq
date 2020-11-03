import unittest
import numpy as np

from synistereq.datasets import Fafb

class FafbInitTestCase(unittest.TestCase):
    def runTest(self):
        fafb = Fafb()
        self.assertTrue(fafb.name == "FAFB")
        self.assertTrue(fafb.container == "/nrs/saalfeld/FAFB00/v14_align_tps_20170818_dmg.n5")
        self.assertTrue(fafb.dataset == "volumes/raw/s0")
        self.assertTrue(np.all(fafb.voxel_size == np.array([40,4,4])))

class FafbGetCropsTestCase(unittest.TestCase):
    def runTest(self):
        fafb = Fafb()
        p = (217400, 164242, 438817) # z,y,x convention
        crop_size = (16,160,160)
        p_crop = fafb.get_crops([p], crop_size)
        self.assertTrue(np.max(p_crop) <= 255)
        self.assertTrue(np.min(p_crop) >= 0)
        self.assertTrue(np.all(crop_size == np.shape(p_crop)[1:]))
        self.assertTrue(1 == np.shape(p_crop)[0])
        self.assertTrue(4 == len(np.shape(p_crop)))


class FafbNormalizeTestCase(unittest.TestCase):
    def runTest(self):
        fafb = Fafb()
        p = (217400, 164242, 438817) # z,y,x convention
        crop_size = (16,160,160)
        p_crop = fafb.get_crops([p], crop_size)
        normalized_p_crop = fafb.normalize(p_crop)
        self.assertTrue(np.all(np.shape(normalized_p_crop) == np.shape(p_crop)))
        self.assertTrue(np.min(normalized_p_crop) >= -1)
        self.assertTrue(np.max(normalized_p_crop) <= 1)


if __name__ == "__main__":
    unittest.main()
