import unittest
import numpy as np

from synistereq.datasets import Hemi

class HemiInitTestCase(unittest.TestCase):
    def runTest(self):
        hemi = Hemi()
        self.assertTrue(hemi.name == "HEMI")
        self.assertTrue(hemi.container == "/nrs/flyem/data/tmp/Z0115-22.export.n5")
        self.assertTrue(hemi.dataset == "22-34/s0")
        self.assertTrue(np.all(hemi.voxel_size == np.array([8,8,8])))

class HemiGetCropsTestCase(unittest.TestCase):
    def runTest(self):
        hemi = Hemi()
        p = (9424, 68736, 184056) # z,y,x convention
        crop_size = (80,80,80)
        p_crop = hemi.get_crops([p], crop_size)
        self.assertTrue(np.max(p_crop) <= 255)
        self.assertTrue(np.min(p_crop) >= 0)
        self.assertTrue(np.all(crop_size == np.shape(p_crop)[1:]))
        self.assertTrue(1 == np.shape(p_crop)[0])
        self.assertTrue(4 == len(np.shape(p_crop)))


class HemiNormalizeTestCase(unittest.TestCase):
    def runTest(self):
        hemi = Hemi()
        p = (9424, 68736, 184056) # z,y,x convention
        crop_size = (80,80,80)
        p_crop = hemi.get_crops([p], crop_size)
        normalized_p_crop = hemi.normalize(p_crop)
        self.assertTrue(np.all(np.shape(normalized_p_crop) == np.shape(p_crop)))
        self.assertTrue(np.min(normalized_p_crop) >= -1)
        self.assertTrue(np.max(normalized_p_crop) <= 1)


if __name__ == "__main__":
    unittest.main()
