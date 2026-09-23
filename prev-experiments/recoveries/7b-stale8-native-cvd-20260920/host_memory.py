import datetime
import json
import os
from pathlib import Path

memory = {key: int(value.split()[0]) * 1024 for key, value in (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines()) if key in ['MemTotal', 'MemFree', 'MemAvailable', 'Cached', 'AnonPages']}
required = 384 * 1024**3
result = {'at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'job_id': os.environ['SLURM_JOB_ID'], 'restart_count': os.environ.get('SLURM_RESTART_COUNT', '0'), 'memory_bytes': memory, 'required_available_bytes': required, 'passed': memory['MemAvailable'] >= required}
root = Path(__file__).resolve().parent / 'validation'
root.mkdir(exist_ok=True)
(root / f"host-memory-job-{result['job_id']}-restart-{result['restart_count']}.json").write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
if not result['passed']:
    raise RuntimeError(f"Only {memory['MemAvailable'] / 1024**3:.1f} GiB host RAM available; the checkpoint continuation requests 384 GiB")
