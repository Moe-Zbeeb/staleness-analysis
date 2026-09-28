import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

root = Path('/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study')
control = root / 'launches/deepseek7b-dapo17k-b32-hp-20260929-v1'
release = control / 'release'
model_id = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
revision = '916b56a44061fd5cd7d6a8fb632557ed4f724f60'
from huggingface_hub import snapshot_download
model = control / 'assets/model'
snapshot_download(model_id, revision=revision, local_dir=model, allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', 'LICENSE', 'README.md'], max_workers=4)
def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()
manifest = {'repo_id': model_id, 'revision': revision, 'path': str(model), 'files': [{'name': p.name, 'size': p.stat().st_size, 'sha256': digest(p)} for p in sorted(model.iterdir()) if p.is_file() and not p.name.startswith('.')]}
(release / 'manifests/model.json').write_text(json.dumps(manifest, indent=2) + '\n')
package = {str(p.relative_to(release)): digest(p) for p in release.rglob('*') if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts and 'vendor' not in p.relative_to(release).parts and p.name != 'PACKAGE_SHA256.json'}
(release / 'PACKAGE_SHA256.json').write_text(json.dumps(package, indent=2) + '\n')
base = json.loads((control / 'preparation-study.json').read_text())
subprocess.run([sys.executable, '-m', 'deepseek_study.cli', 'prepare', '--model', str(model), '--dataset', base['dataset_path'], '--destination', base['prepared_model_path'], '--manifest', str(release / 'manifests/model.json')], check=True)
subprocess.run([sys.executable, '-m', 'deepseek_study.cli', 'prepare-data', str(control / 'preparation-study.json')], check=True)
prepared = json.loads(Path(base['data_manifest']).read_text())
original = json.loads((root / 'launches/deepseek15b-dapo17k-async-b32-hp-20260928-v1/assets/train-manifest-deepseek15b.json').read_text())
assert prepared['contract']['source_sha256'] == original['contract']['source_sha256']
assert [(r['id'], r['included']) for r in prepared['records']] == [(r['id'], r['included']) for r in original['records']]
with (control / 'preflight.json').open('w') as stream:
    subprocess.run([sys.executable, '-m', 'deepseek_study.cli', 'check', str(control / 'preparation-study.json')], stdout=stream, check=True)
receipt = {'model_id': model_id, 'revision': revision, 'model_bytes': sum(r['size'] for r in manifest['files']), 'included_rows': prepared['included_rows'], 'excluded_rows': prepared['excluded_rows'], 'question_order_matches_15b': True, 'gpu_execution_verified': False}
(control / 'preparation-complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt), flush=True)
