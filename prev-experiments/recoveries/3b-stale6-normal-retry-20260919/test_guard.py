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
        recovery.cap = 6
        recovery.proof = {}
        recovery.static = lambda: None
        recovery.original = SimpleNamespace(config=lambda _: {'ckpt': {'interval': 25, 'keep_last': 4, 'keep_interval': 100}})
        self.recovery = recovery
        self.checkpoint(775)
        for component in ['trainer', 'orchestrator', 'inference']:
            path = recovery.run / 'configs/resolved' / f'{component}.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{}')
        recovery.item = {
            'resume_step': 775,
            'guard_sha256': 'original-guard',
            'source_sha256': 'source',
            'data_sha256': 'data',
            'metadata_sha256': digest(recovery.run / 'checkpoints/step_775/trainer/.metadata'),
            'progress_sha256': digest(recovery.run / 'checkpoints/step_775/orchestrator/progress.pt'),
        }
        recovery.evidence.mkdir()
        sizes = {name: (recovery.run / 'checkpoints/step_775' / name).stat().st_size for name in PAIRED_FILES}
        (recovery.evidence / 'checkpoint-files.json').write_text(json.dumps(sizes))

    def checkpoint(self, step):
        for name in PAIRED_FILES:
            path = self.recovery.run / f'checkpoints/step_{step}' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f'{step}:{name}'.encode())

    def capture(self):
        with patch.dict(os.environ, {'SLURM_JOB_ID': '12345', 'SLURM_RESTART_COUNT': '0'}):
            return self.recovery.capture(775)

    def prune(self):
        shutil.rmtree(self.recovery.run / 'checkpoints/step_775')
        for step in [900, 925, 950, 1000]:
            self.checkpoint(step)
        attempt = self.recovery.run / 'logs/attempt_9'
        attempt.mkdir(parents=True)
        (attempt / 'trainer.log').write_text('Resuming from step 775 (checkpoint)\nSUCCESS Step 1000 | completed\n')
        (attempt / 'orchestrator.log').write_text('Resuming from step 775\nSaving final checkpoint at step 1000\n')

    def audit_resume(self):
        self.recovery.resume_evidence(self.recovery.experiment, self.recovery.run, 775, 1000)

    def test_captured_checkpoint_can_be_pruned_under_retention(self):
        first = self.capture()
        self.assertEqual(first, self.capture())
        self.prune()
        self.audit_resume()
        self.assertEqual(self.recovery.proof[775]['newer_paired_checkpoints'], [900, 925, 950, 1000])

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

    def test_incomplete_resume_directory_is_not_treated_as_retention(self):
        self.capture()
        self.prune()
        (self.recovery.run / 'checkpoints/step_775').mkdir()
        with self.assertRaisesRegex(ValueError, 'exists but is incomplete'):
            self.audit_resume()

    def test_extra_trainer_shard_is_rejected(self):
        (self.recovery.run / 'checkpoints/step_775/trainer/__2_0.distcp').write_bytes(b'extra')
        with self.assertRaisesRegex(ValueError, 'trainer world size is not two'):
            self.capture()

    def test_earlier_checkpoint_cannot_enter_recovery_lineage(self):
        with self.assertRaisesRegex(ValueError, 'outside recovery lineage'):
            self.recovery.resume_evidence(self.recovery.experiment, self.recovery.run, 750, 1000)

    def test_original_runtime_adaptation_is_bounded(self):
        original = Path(__file__).resolve().parents[2] / 'experiments/dapo-qwen25-3b-grpo-stale6-4gpu-tp2/scripts/staleness_guard.py'
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
            return {'status': 'passed', 'cap': 6}
        previous = module.runtime_parity
        with patch.object(module, 'audit', side_effect=audit):
            report = self.recovery.audit(self.recovery.run, 1000)
        self.assertIsNot(seen[0], previous)
        self.assertIs(module.runtime_parity, previous)
        self.assertFalse(report['frozen_experiment_files_modified'])


class OverlayBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name).resolve()
        recovery = Recovery.__new__(Recovery)
        recovery.root = root
        recovery.experiment = root / 'experiment'
        recovery.original_path = recovery.experiment / 'scripts/staleness_guard.py'
        recovery.overlay = root / 'overlay.toml'
        for relative in ['scripts/staleness_guard.py', 'source-manifest.json', 'data/manifest.json']:
            path = recovery.experiment / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{}')
        recovery.item = {
            'guard_sha256': digest(recovery.original_path),
            'source_sha256': digest(recovery.experiment / 'source-manifest.json'),
            'data_sha256': digest(recovery.experiment / 'data/manifest.json'),
        }
        recovery.spec = {'caps': {}}
        recovery.original = SimpleNamespace(static=lambda _: (_ for _ in ()).throw(RuntimeError('original static reached')))
        self.recovery = recovery

    def verify(self, source):
        self.recovery.overlay.write_text(source)
        self.recovery.spec['overlay_sha256'] = digest(self.recovery.overlay)
        names = ['guard.py', 'health_probe.py', 'overlay.toml', 'spec.json', 'train-stale6.sh', 'validate_plan.py']
        for name in names:
            path = self.recovery.root / name
            if not path.exists():
                path.write_text('{}')
        manifest = {name: digest(self.recovery.root / name) for name in names}
        (self.recovery.root / 'package-manifest.json').write_text(json.dumps(manifest))
        self.recovery.static()

    def overlay(self):
        return (Path(__file__).parent / 'overlay.toml').read_text()

    def test_supported_startup_timeout_reaches_original_guard(self):
        with self.assertRaisesRegex(RuntimeError, 'original static reached'):
            self.verify(self.overlay())

    def test_different_timeout_is_rejected_even_with_updated_hashes(self):
        with self.assertRaisesRegex(ValueError, 'topology and startup timeout'):
            self.verify(self.overlay().replace('7200', '1200'))

    def test_missing_timeout_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'topology and startup timeout'):
            self.verify(self.overlay().split('[orchestrator.ckpt]')[0])

    def test_optimizer_change_is_rejected_even_with_updated_hashes(self):
        with self.assertRaisesRegex(ValueError, 'topology and startup timeout'):
            self.verify(self.overlay() + '\n[trainer.optim]\nlr = 0.01\n')

    def test_trainer_count_change_is_rejected_even_with_updated_hashes(self):
        with self.assertRaisesRegex(ValueError, 'topology and startup timeout'):
            self.verify(self.overlay().replace('num_train_gpus = 2', 'num_train_gpus = 3'))


if __name__ == '__main__':
    unittest.main()
