import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "pinn1D"))

from plotting import create_figure_directory
from testing import write_r2_to_file


class TestPlottingHelpers(unittest.TestCase):
    def test_create_figure_directory_returns_existing_directory_path(self):
        with TemporaryDirectory() as tmpdir:
            figure_directory = os.path.join(tmpdir, "figures")

            result = create_figure_directory(figure_directory)

            self.assertEqual(result, figure_directory)
            self.assertTrue(os.path.isdir(result))

    def test_write_r2_to_file_creates_path_and_writes_r2(self):
        with TemporaryDirectory() as tmpdir:
            r2_path = os.path.join(tmpdir, "r2_result.txt")

            write_r2_to_file(r2_path, 0.93)

            self.assertTrue(os.path.exists(r2_path))
            with open(r2_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            self.assertEqual(content, "0.93")


if __name__ == "__main__":
    unittest.main()
