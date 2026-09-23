import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CleanupTests(unittest.TestCase):
    def fixture(self, root):
        publication = {'status': 'published_and_verified', 'collection_url': 'https://huggingface.co/collections/example/test', 'datasets': {}}
        for name in ['skywork', 'deepscaler', 'merged']:
            data = name.encode()
            path = root / 'release' / name / 'data/train.parquet'
            path.parent.mkdir(parents=True)
            path.write_bytes(data)
            publication['datasets'][name] = {'status': 'published_and_verified', 'rows': 1, 'parquet_sha256': hashlib.sha256(data).hexdigest()}
            path = root / 'data/processed' / name / 'clean.jsonl'
            path.parent.mkdir(parents=True)
            path.write_text('discard')
        for name in ['qwen3-14b', 'deepseek-r1-distill-qwen-14b']:
            path = root / 'models' / name / 'model.safetensors.index.json'
            path.parent.mkdir(parents=True)
            path.write_text('{}')
        (root / 'reports').mkdir()
        (root / 'reports/summary.json').write_text('{}')
        (root / 'reports/rejected.jsonl').write_text('discard')
        (root / 'release/publication.json').write_text(json.dumps(publication))
        return publication

    def run_cleanup(self, root):
        script = Path(__file__).resolve().parents[1] / 'scripts/cleanup_local_datasets.py'
        return subprocess.run([sys.executable, str(script), '--root', str(root), '--execute'], capture_output=True, text=True)

    def test_only_authorized_data_removed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            (root / 'data/raw').mkdir()
            (root / 'data/raw/source.json').write_text('discard')
            result = self.run_cleanup(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((root / 'data/raw').exists())
            self.assertFalse((root / 'reports/rejected.jsonl').exists())
            self.assertTrue((root / 'reports/summary.json').exists())
            self.assertTrue((root / 'data/processed/skywork/train.parquet').is_file())
            self.assertTrue((root / 'models/qwen3-14b/model.safetensors.index.json').is_file())

    def test_unverified_publication_blocks_deletion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            publication = self.fixture(root)
            publication['status'] = 'uploading'
            (root / 'release/publication.json').write_text(json.dumps(publication))
            self.assertNotEqual(self.run_cleanup(root).returncode, 0)
            self.assertTrue((root / 'data/processed/skywork/clean.jsonl').exists())

    def test_symlink_target_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as outside:
            root = Path(temporary)
            self.fixture(root)
            sentinel = Path(outside) / 'keep.txt'
            sentinel.write_text('keep')
            (root / 'data/raw').symlink_to(outside, target_is_directory=True)
            result = self.run_cleanup(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(sentinel.exists())
            self.assertFalse((root / 'data/raw').is_symlink())


if __name__ == '__main__':
    unittest.main()
