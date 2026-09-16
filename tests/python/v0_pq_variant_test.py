"""Read-only planning and fail-closed lifecycle tests (no Faiss required)."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from v0_pq_local_test import V0PQLocalTest as Fixture

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/v0'))
import run_pq_variant as pipeline
from v0pq_format import sha256_file


class VariantTest(unittest.TestCase):
    def setUp(self):
        self.fixture = Fixture()
        self.fixture.setUp()
        self.root = self.fixture.root
        directions, manifest = self.fixture._sample()
        index = self.root / 'index.bin'
        index.write_bytes(b'test-index')
        data = json.loads(manifest.read_text())
        data['index_sha256'] = sha256_file(index)
        manifest.write_text(json.dumps(data))
        self.output = self.root / 'variant'
        self.argv = ['--pq-m', '2', '--nbits', '6', '--directions', str(directions),
            '--sample-manifest', str(manifest), '--index-path', str(index),
            '--encoder', sys.executable, '--inspector', sys.executable,
            '--min-points-per-centroid', '1', '--output-root', str(self.output)]

    def tearDown(self):
        self.fixture.tearDown()

    def test_dry_run_and_cap(self):
        with patch.object(sys, 'argv', ['run_pq_variant.py', *self.argv, '--dry-run']):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(0, pipeline.main())
        plan = json.loads(output.getvalue())
        self.assertFalse(self.output.exists())
        self.assertEqual(5, plan['resolved_max_points_per_centroid'])
        self.assertFalse(plan['faiss_training_subsampling_expected'])
        self.assertEqual(2, plan['sidecar_code_bytes'])
        self.assertEqual(512, plan['lut_bytes_float32'])
        self.assertEqual(['train', 'codebook-validation', 'reconstruction', 'encode', 'inspect'],
                         [x[0] for x in plan['commands']])

    def test_refuse_existing_or_mismatched_index(self):
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError, 'existing'):
            pipeline.make_plan(pipeline.parser().parse_args(self.argv))
        self.output.rmdir()
        (self.root / 'index.bin').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'index_sha256'):
            pipeline.make_plan(pipeline.parser().parse_args(self.argv))

    def test_stage_failure_never_completes(self):
        args = pipeline.parser().parse_args(self.argv)
        plan = pipeline.make_plan(args)
        with patch.object(sys, 'argv', ['run_pq_variant.py', *self.argv]), \
             patch.object(pipeline, 'make_plan', return_value=plan), \
             patch.object(pipeline.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, 'train')) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                pipeline.main()
        self.assertEqual(1, run.call_count)
        self.assertTrue((self.output / 'FAILED.json').exists())
        self.assertFalse((self.output / 'COMPLETE').exists())


if __name__ == '__main__':
    # Avoid rediscovering the imported fixture's tests.
    unittest.main(defaultTest='VariantTest')
