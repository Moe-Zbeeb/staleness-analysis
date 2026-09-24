import argparse
import json
from pathlib import Path

from common import ROOT, write_json


def configure(root, datasets, models, verification_python):
    root = Path(root).resolve()
    if (root / 'spec.json').exists() or (root / 'manifest.json').exists() or (root / 'outputs').exists():
        raise FileExistsError('Configure a fresh experiment directory; existing run files must be preserved')
    spec = json.loads((root / 'experiment.json').read_text())
    datasets = Path(datasets).expanduser().resolve()
    models = Path(models).expanduser().resolve()
    verification_python = Path(verification_python).expanduser().resolve()
    if not verification_python.is_file():
        raise FileNotFoundError(verification_python)
    for dataset in spec['datasets']:
        source = datasets / 'data/processed' / dataset / 'train.parquet'
        if not source.is_file():
            raise FileNotFoundError(source)
    for model in spec['models']:
        target = models / model['tag']
        for name in ['config.json', 'tokenizer.json', 'tokenizer_config.json']:
            if not (target / name).is_file():
                raise FileNotFoundError(target / name)
        model['target'] = str(target)
        model['tokenizer_path'] = str(root / 'inputs/tokenizers' / model['tag']) if model['tag'] in {'deepseek', 'acereason', 'phi4'} else str(target)
    spec.update(root=str(root), dataset_root=str(datasets), verification_python=str(verification_python))
    write_json(root / 'spec.json', spec)
    return spec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--models-root', type=Path, required=True)
    parser.add_argument('--verification-python', type=Path, required=True)
    args = parser.parse_args()
    configure(ROOT, args.dataset_root, args.models_root, args.verification_python)
    print(ROOT / 'spec.json')


if __name__ == '__main__':
    main()
