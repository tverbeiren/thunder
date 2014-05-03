import shutil
import tempfile
from numpy import array, allclose
from thunder.util.load import subtoind, indtosub, getdims
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

    def test_patch(self):
        foo = 1
