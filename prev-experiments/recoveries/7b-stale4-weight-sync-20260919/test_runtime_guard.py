import shutil
import tempfile
import unittest
from pathlib import Path

from runtime_guard import CLIENTS, verify_source


class RuntimeSourceTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.upstream = Path(self.temporary.name) / 'upstream'
        shutil.copytree(self.root / 'prime-rl-source', self.upstream, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        shutil.copyfile(self.root / 'upstream/clients.py', self.upstream / CLIENTS)

    def test_every_pristine_upstream_file_matches(self):
        result = verify_source(self.root, self.upstream)
        self.assertEqual(result['changed_files'], [CLIENTS])
        self.assertFalse(result['shared_prime_repository_modified'])

    def test_changed_shared_trainer_source_is_rejected(self):
        path = self.upstream / 'src/prime_rl/trainer/rl/train.py'
        path.write_text(path.read_text() + '\nchanged = True\n')
        with self.assertRaisesRegex(ValueError, 'Live upstream source differs.*trainer/rl/train.py'):
            verify_source(self.root, self.upstream)

    def test_missing_upstream_configuration_is_rejected(self):
        (self.upstream / 'packages/prime-rl-configs/src/prime_rl/configs/rl.py').unlink()
        with self.assertRaisesRegex(ValueError, 'Live upstream source differs.*configs/rl.py'):
            verify_source(self.root, self.upstream)


if __name__ == '__main__':
    unittest.main()
