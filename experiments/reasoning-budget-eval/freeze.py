import json
from pathlib import Path

from common import ROOT, digest, write_json


def freeze(root):
    root = Path(root)
    if (root / 'manifest.json').exists() or (root / 'outputs').exists():
        raise FileExistsError('Refusing to replace a manifest or change a run with outputs')
    spec = json.loads((root / 'spec.json').read_text())
    check = json.loads((root / 'tokenizer-validation.json').read_text())
    if check['status'] != 'passed' or set(check['models']) != {m['tag'] for m in spec['models']}:
        raise ValueError('All model tokenizer checks must pass before freezing')
    weights = json.loads((root / 'model-validation.json').read_text())
    if weights['status'] != 'passed' or set(weights['models']) != {m['tag'] for m in spec['models']}:
        raise ValueError('All pinned model weight checks must pass before freezing')
    files = set(root.glob('*.py')) | set(root.glob('*.sh')) | set(root.glob('requirements-*.txt'))
    files |= {root / n for n in ['spec.json', 'experiment.json', 'datasets.lock.json', 'preparation.json', 'tokenizer-validation.json', 'runtime-versions.json', 'model-validation.json']}
    files |= {p for p in (root / 'inputs').rglob('*') if p.is_file() and not any(part.startswith('.') for part in p.relative_to(root).parts)}
    for name in ['samples'] + [m['tag'] for m in spec['models']]:
        if not (root / 'inputs' / (name + '.jsonl')).is_file():
            raise FileNotFoundError(name)
    manifest = {'version': 1, 'files': {str(p.relative_to(root)): digest(p) for p in sorted(files)}}
    write_json(root / 'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    freeze(ROOT)
    print(ROOT / 'manifest.json')
