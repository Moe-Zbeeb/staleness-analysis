import json
from pathlib import Path

from common import ROOT, digest, write_json


def main():
    if (ROOT / 'manifest.json').exists() or (ROOT / 'outputs').exists():
        raise FileExistsError('Verify model inputs before freezing a new run')
    spec = json.loads((ROOT / 'spec.json').read_text())
    checked = {}
    for model in spec['models']:
        weights = [f for f in model['files'] if f['name'].endswith('.safetensors')]
        assert weights and all(f['sha256'] for f in weights)
        for item in weights:
            path = Path(model['target']) / item['name']
            assert path.stat().st_size == item['size'], str(path)
            assert digest(path) == item['sha256'], str(path)
        checked[model['tag']] = {'revision': model['revision'], 'weight_shards': len(weights), 'status': 'passed'}
        print(model['tag'] + ': pinned weight checksums verified', flush=True)
    write_json(ROOT / 'model-validation.json', {'status': 'passed', 'models': checked})


if __name__ == '__main__':
    main()
