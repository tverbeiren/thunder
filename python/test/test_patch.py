import shutil
import tempfile
from numpy import array, allclose
from numpy.random import randn
from thunder.util.patch import patch
from test_utils import PySparkTestCase


class PatchTestCase(PySparkTestCase):
    def setUp(self):
        super(PatchTestCase, self).setUp()
        self.outputdir = tempfile.mkdtemp()

    def tearDown(self):
        super(PatchTestCase, self).tearDown()
        shutil.rmtree(self.outputdir)

class TestPatch(PatchTestCase):
    """Test conversion of data into RDD over patches"""

    def test_patch_no_border(self):
        """ Break a 4x4 array into 4 2x2 patches """
        data_local = [((3,2),array([1])),
            ((3,3),array([2])),
            ((3,4),array([3])),
            ((3,5),array([4])),
            ((4,2),array([5])),
            ((4,3),array([6])),
            ((4,4),array([7])),
            ((4,5),array([8])),
            ((5,2),array([9])),
            ((5,3),array([10])),
            ((5,4),array([11])),
            ((5,5),array([12])),
            ((6,2),array([13])),
            ((6,3),array([14])),
            ((6,4),array([15])),
            ((6,5),array([16]))]
        data = self.sc.parallelize(data_local)
        patches = patch(data,(2,2)).collect()
        assert(allclose(patches[0][0]))

    def test_patch_with_border(self):
        """ Break a 4x4 array into 4 2x2 patches with a 1-pixel border """
        data_local = [((3,2),array([1])),
            ((3,3),array([2])),
            ((3,4),array([3])),
            ((3,5),array([4])),
            ((4,2),array([5])),
            ((4,3),array([6])),
            ((4,4),array([7])),
            ((4,5),array([8])),
            ((5,2),array([9])),
            ((5,3),array([10])),
            ((5,4),array([11])),
            ((5,5),array([12])),
            ((6,2),array([13])),
            ((6,3),array([14])),
            ((6,4),array([15])),
            ((6,5),array([16]))]
        data = self.sc.parallelize(data_local)
        patches = sorted(patch(data,(2,2),(1,1)).collect(),key=lambda x:x[1])
        assert(allclose(patches[0][1],array([[[  0.,   0.,   0.,   0.],
        [  0.,   1.,   2.,   3.],
        [  0.,   5.,   6.,   7.],
        [  0.,   9.,  10.,  11.]]])))

    def test_patch_with_border_3d(self):
        data_local = []
        for i in range(10):
            for j in range(10):
                for k in range(10):
                    data_local += [((i,j,k),randn(2))]
        data = sc.parallelize(data_local).cache()
        patches = patch(data,(2,2,2),(1,1,1))
        assert(allclose(patches.first()[1].shape),(2,4,4,4)) # trivial assertion, just to make sure the code doesn't crash