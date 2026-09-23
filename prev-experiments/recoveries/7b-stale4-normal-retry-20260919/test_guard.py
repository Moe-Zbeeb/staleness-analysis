import importlib.util
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from guard import PAIRED_FILES, Recovery, digest


class RecoveryEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name).resolve()
        recovery = Recovery.__new__(Recovery)
        recovery.root = root
        recovery.experiment = root / 'experiment'
        recovery.experiment.mkdir()
        recovery.run = root / 'run'
        recovery.evidence = root / 'evidence'
        recovery.spec_path = root / 'spec.json'
        recovery.spec_path.write_text('{}')
        recovery.spec = {'overlay_sha256': 'overlay'}
        recovery.cap = 4
        recovery.proof = {}
        recovery.static = lambda: None
        recovery.original = SimpleNamespace(config=lambda _: {'ckpt': {'interval': 25, 'keep_last': 4, 'keep_interval': 100}})
        self.recovery = recovery
        self.checkpoint(225)
        for component in ['trainer', 'orchestrator', 'inference']:
            path = recovery.run / 'configs/resolved' / f'{component}.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{}')
        recovery.item = {
            'resume_step': 225,
            'guard_sha256': 'original-guard',
            'source_sha256': 'source',
            'data_sha256': 'data',
            'metadata_sha256': digest(recovery.run / 'checkpoints/step_225/trainer/.metadata'),
            'progress_sha256': digest(recovery.run / 'checkpoints/step_225/orchestrator/progress.pt'),
        }
        recovery.evidence.mkdir()
        sizes = {name: (recovery.run / 'checkpoints/step_225' / name).stat().st_size for name in PAIRED_FILES}
        (recovery.evidence / 'checkpoint-files.json').write_text(json.dumps(sizes))

    def checkpoint(self, step):
        for name in PAIRED_FILES:
            path = self.recovery.run / f'checkpoints/step_{step}' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f'{step}:{name}'.encode())

    def capture(self):
        with patch.dict(os.environ, {'SLURM_JOB_ID': '12345', 'SLURM_RESTART_COUNT': '0'}):
            return self.recovery.capture(225)

    def prune(self):
        shutil.rmtree(self.recovery.run / 'checkpoints/step_225')
        for step in [900, 925, 950, 1000]:
            self.checkpoint(step)
        attempt = self.recovery.run / 'logs/attempt_9'
        attempt.mkdir(parents=True)
        (attempt / 'trainer.log').write_text('Resuming from step 225 (checkpoint)\nSUCCESS Step 1000 | completed\n')
        (attempt / 'orchestrator.log').write_text('Resuming from step 225\nSaving final checkpoint at step 1000\n')

    def audit_resume(self):
        self.recovery.resume_evidence(self.recovery.experiment, self.recovery.run, 225, 1000)

    def test_captured_checkpoint_can_be_pruned_under_retention(self):
        first = self.capture()
        self.assertEqual(first, self.capture())
        self.prune()
        self.audit_resume()
        self.assertEqual(self.recovery.proof[225]['newer_paired_checkpoints'], [900, 925, 950, 1000])

    def test_missing_capture_is_rejected(self):
        self.prune()
        with self.assertRaisesRegex(ValueError, 'Missing immutable prelaunch capture'):
            self.audit_resume()

    def test_modified_capture_is_rejected(self):
        result = self.capture()
        path = Path(result['capture_path']).parent / 'checkpoint/trainer/.metadata'
        path.chmod(0o644)
        path.write_bytes(b'tampered')
        self.prune()
        with self.assertRaisesRegex(ValueError, 'Captured evidence changed'):
            self.audit_resume()

    def test_missing_successful_resume_log_is_rejected(self):
        self.capture()
        self.prune()
        (self.recovery.run / 'logs/attempt_9/trainer.log').write_text('SUCCESS Step 1000 | completed\n')
        with self.assertRaisesRegex(ValueError, 'Missing successful trainer resume'):
            self.audit_resume()

    def test_capture_from_another_startup_overlay_is_rejected(self):
        result = self.capture()
        path = Path(result['capture_path'])
        record = json.loads(path.read_text())
        record['overlay_sha256'] = 'another-overlay'
        path.chmod(0o644)
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, 'Capture provenance mismatch: overlay_sha256'):
            self.recovery.validate_capture(path, 225)

    def test_incomplete_resume_directory_is_not_treated_as_retention(self):
        self.capture()
        self.prune()
        (self.recovery.run / 'checkpoints/step_225').mkdir()
        with self.assertRaisesRegex(ValueError, 'exists but is incomplete'):
            self.audit_resume()

    def test_extra_trainer_shard_is_rejected(self):
        (self.recovery.run / 'checkpoints/step_225/trainer/__4_0.distcp').write_bytes(b'extra')
        with self.assertRaisesRegex(ValueError, 'trainer world size is not four'):
            self.capture()

    def test_earlier_checkpoint_cannot_enter_recovery_lineage(self):
        with self.assertRaisesRegex(ValueError, 'outside recovery lineage'):
            self.recovery.resume_evidence(self.recovery.experiment, self.recovery.run, 200, 1000)

    def test_original_runtime_adaptation_is_bounded(self):
        original = Path(__file__).resolve().parents[2] / 'experiments/dapo-qwen25-math7b-grpo-stale4-5gpu/scripts/staleness_guard.py'
        module_spec = importlib.util.spec_from_file_location('original_guard_test', original)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        self.recovery.original = module
        self.recovery.original_path = original
        self.recovery.overlay = Path(__file__).parent / 'overlay.toml'
        self.recovery.item['data_sha256'] = 'data'
        seen = []
        def audit(experiment, run, steps):
            seen.append(module.runtime_parity)
            self.assertIn('recovery_overlay', module.runtime_parity.__code__.co_names)
            self.assertEqual(module.runtime_parity.__globals__['recovery_overlay'], self.recovery.overlay)
            return {'status': 'passed', 'cap': 4}
        previous = module.runtime_parity
        with patch.object(module, 'audit', side_effect=audit):
            report = self.recovery.audit(self.recovery.run, 1000)
        self.assertIsNot(seen[0], previous)
        self.assertIs(module.runtime_parity, previous)
        self.assertFalse(report['frozen_experiment_files_modified'])


if __name__ == '__main__':
    unittest.main()
