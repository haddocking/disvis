"""Tests of volume operations and MRC input and output.

Run from the repository root with python -m unittest.
"""

import gc
import os
import tempfile
import warnings
from unittest import TestCase, main

import numpy as np

from disvis import volume
from disvis.IO.mrc import MRCFile, parse_mrc
from disvis.volume import Volume


class TestErode(TestCase):
    def setUp(self):
        array = np.zeros((5, 5, 5), dtype=np.float64)
        array[1:4, 1:4, 1:4] = 1
        self.vol = Volume(array)
        # One erosion leaves only the center of the 3x3x3 cube.
        self.once = np.zeros_like(array)
        self.once[2, 2, 2] = 1

    def test_new_volume(self):
        out = volume.erode(self.vol, 1)
        self.assertTrue(np.array_equal(out.array, self.once))
        self.assertEqual(self.vol.array.sum(), 27)

    def test_out(self):
        out = volume.zeros_like(self.vol)
        result = volume.erode(self.vol, 1, out)
        self.assertIs(result, out)
        self.assertTrue(np.array_equal(out.array, self.once))

    def test_in_place(self):
        volume.erode(self.vol, 1, self.vol)
        self.assertTrue(np.array_equal(self.vol.array, self.once))

    def test_iterations(self):
        self.assertTrue(
            np.array_equal(volume.erode(self.vol, 0).array, self.vol.array)
        )
        self.assertEqual(volume.erode(self.vol, 2).array.sum(), 0)


class TestMRC(TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.fname = os.path.join(self.tmpdir.name, 'test.mrc')
        array = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        Volume(array, voxelspacing=2.0, origin=(1, 2, 3)).tofile(self.fname)
        self.array = array

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_parse_mrc(self):
        array, voxelspacing, origin = parse_mrc(self.fname)
        self.assertTrue(np.allclose(array, self.array))
        self.assertEqual(voxelspacing, 2.0)
        self.assertTrue(np.allclose(origin, (1, 2, 3)))

    def unclosed_files(self, func):
        """Call func and return the files that it left open.

        An open file object warns with a ResourceWarning when it is garbage
        collected.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ResourceWarning)
            func()
            gc.collect()
        return [w for w in caught if issubclass(w.category, ResourceWarning)]

    def test_file_name_is_closed(self):
        with MRCFile(self.fname) as mrcfile:
            fid = mrcfile.fid
            self.assertTrue(np.allclose(mrcfile.density, self.array))
        self.assertTrue(fid.closed)
        mrcfile = MRCFile(self.fname)
        mrcfile.close()
        self.assertTrue(mrcfile.fid.closed)

    def test_parse_mrc_closes_file(self):
        unclosed = self.unclosed_files(lambda: parse_mrc(self.fname))
        self.assertEqual(unclosed, [])

    def test_open_file_stays_open(self):
        with open(self.fname, 'rb') as f:
            with MRCFile(f) as mrcfile:
                self.assertTrue(np.allclose(mrcfile.density, self.array))
            self.assertFalse(f.closed)

    def test_bad_file_is_closed(self):
        bad = os.path.join(self.tmpdir.name, 'bad.mrc')
        with open(bad, 'wb') as f:
            f.write(bytes(1024))

        def read_bad_file():
            with self.assertRaises(OSError):
                MRCFile(bad)

        self.assertEqual(self.unclosed_files(read_bad_file), [])


if __name__ == '__main__':
    main()
