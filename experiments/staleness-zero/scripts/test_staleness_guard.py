import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch


script = Path(__file__).with_name('staleness_guard.py')
spec = importlib.util.spec_from_file_location('guard', script)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
experiments = script.parents[2]


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / 'dapo-qwen25-math15b-grpo-stale0'
        shutil.copytree(experiments / self.root.name, self.root)
        self.run = Path(self.temp.name) / 'run'
        (self.run / 'configs/resolved').mkdir(parents=True)
        (self.run / 'configs/resolved/orchestrator.json').write_text('{"max_off_policy_steps": 0}')
        self.metrics = []
        for step in range(1, 6):
            row = {key: 0.0 for key in ['off_policy/mean', 'off_policy/max', 'off_policy/in_flight/mean', 'off_policy/in_flight/max', 'off_policy/in_queue/mean', 'off_policy/in_queue/max']}
            row.update(step=step, **{'optim/grad_norm': 0.3, 'loss/mean': -0.01})
            self.metrics.append(row)
            self.write_trace(step, {'start': step - 1, 'end': step - 1})
        self.write_metrics()
        for suffix in ['trainer/.metadata', 'orchestrator/progress.pt']:
            p = self.run / 'checkpoints/step_5' / suffix
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('fixture')

    def tearDown(self):
        self.temp.cleanup()

    def write_metrics(self):
        (self.run / 'metrics.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in self.metrics))

    def write_trace(self, step, policy):
        p = self.run / f'rollouts/step_{step}/train/effective/traces.jsonl'
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({'run': {'work': {'policy': policy}}, 'traces': [{'id': f'trace-{step}'}]}) + '\n')

    def test_five_fresh_updates_pass(self):
        self.assertEqual(guard.audit(self.root, self.run, 5)['status'], 'passed')

    def test_nonzero_metric_rejected(self):
        self.metrics[3]['off_policy/max'] = 1
        self.write_metrics()
        with self.assertRaisesRegex(ValueError, 'off_policy/max'):
            guard.audit(self.root, self.run, 5)

    def test_missing_update_rejected(self):
        self.metrics.pop(2)
        self.write_metrics()
        with self.assertRaisesRegex(ValueError, 'Step 3'):
            guard.audit(self.root, self.run, 5)

    def test_nonfinite_gradient_rejected(self):
        self.metrics[2]['optim/grad_norm'] = float('nan')
        self.write_metrics()
        with self.assertRaisesRegex(ValueError, 'gradient'):
            guard.audit(self.root, self.run, 5)

    def test_stale_trace_rejected_even_if_metrics_say_zero(self):
        self.write_trace(5, {'start': 3, 'end': 4})
        with self.assertRaisesRegex(ValueError, 'wrong policy'):
            guard.audit(self.root, self.run, 5)

    def test_future_trace_rejected_even_if_clamped_metric_says_zero(self):
        self.write_trace(5, {'start': 5, 'end': 5})
        with self.assertRaisesRegex(ValueError, 'wrong policy'):
            guard.audit(self.root, self.run, 5)

    def test_missing_provenance_rejected(self):
        self.write_trace(5, None)
        with self.assertRaisesRegex(ValueError, 'wrong policy'):
            guard.audit(self.root, self.run, 5)

    def test_missing_checkpoint_half_rejected(self):
        (self.run / 'checkpoints/step_5/orchestrator/progress.pt').unlink()
        with self.assertRaisesRegex(ValueError, 'paired checkpoint'):
            guard.audit(self.root, self.run, 5)

    def test_runtime_cap_two_rejected(self):
        (self.run / 'configs/resolved/orchestrator.json').write_text('{"max_off_policy_steps": 2}')
        with self.assertRaisesRegex(ValueError, 'cap'):
            guard.audit(self.root, self.run, 5)

    def test_changed_token_limit_rejected(self):
        p = self.root / 'config/main.toml'
        p.write_text(p.read_text().replace('3072', '4096'))
        with self.assertRaisesRegex(ValueError, 'source changed'):
            guard.static(self.root)

    def test_noncontiguous_gpu_allocation_preserved(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,2,4,6,7'}):
            self.assertEqual(guard.devices(self.root), '6,7,0,2,4')

    def test_duplicate_gpu_rejected(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,2,4,6,6'}):
            with self.assertRaisesRegex(ValueError, 'GPU allocation'):
                guard.devices(self.root)

    def test_wrong_gpu_count_rejected(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5'}):
            with self.assertRaisesRegex(ValueError, 'GPU allocation'):
                guard.devices(self.root)

    def test_all_launchers_use_isolated_output_paths(self):
        for root in experiments.glob('*-stale0'):
            guard.static(root)
            train = (root / 'scripts/train_job.sh').read_text()
            smoke = (root / 'scripts/smoke_job.sh').read_text()
            self.assertIn(f'RUN_DIR="$RL_INFRA/outputs/{root.name}/$RUN_NAME"', train)
            self.assertIn(f'SMOKE_DIR="$RL_INFRA/outputs/{root.name}/$SMOKE_NAME"', smoke)
            self.assertIn('staleness_guard.py" gate', train)
            self.assertIn('staleness_guard.py" smoke', smoke)
            self.assertNotIn('--resume', (root / 'scripts/smoke_job.sh').read_text())
            for sh in (root / 'scripts').glob('*.sh'):
                subprocess.run(['bash', '-n', str(sh)], check=True)


if __name__ == '__main__':
    unittest.main()
