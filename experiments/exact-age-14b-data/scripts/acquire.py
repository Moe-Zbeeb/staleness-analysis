import argparse
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--kind', choices=['model', 'dataset'], required=True)
    args = parser.parse_args()
    root = args.root
    sources = json.loads((root / 'configs/sources.lock.json').read_text())
    api = HfApi()
    receipts = []
    (root / 'manifests').mkdir(parents=True, exist_ok=True)
    for source in sources:
        if source['repo_type'] != args.kind:
            continue
        target = root / ('models' if args.kind == 'model' else 'data/raw') / source['key']
        target.mkdir(parents=True, exist_ok=True)
        info = api.repo_info(source['repo_id'], repo_type=args.kind, revision=source['revision'], files_metadata=True)
        existing = None
        files = []
        for item in info.siblings:
            name = item.rfilename
            if name == '.gitattributes' or (args.kind == 'model' and name.startswith('figures/')):
                continue
            destination = target / name
            expected = item.lfs.sha256 if item.lfs else None
            reused = False
            if existing and name.endswith('.safetensors') and not destination.exists():
                candidate = existing / name
                if candidate.is_file() and candidate.stat().st_size == item.size and digest(candidate) == expected:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    os.link(candidate, destination)
                    reused = True
            if not reused:
                hf_hub_download(source['repo_id'], name, repo_type=args.kind, revision=source['revision'], local_dir=target)
            actual = digest(destination)
            if expected and actual != expected:
                raise RuntimeError(f'Checksum mismatch: {source["key"]}/{name}')
            if item.size is not None and destination.stat().st_size != item.size:
                raise RuntimeError(f'Size mismatch: {source["key"]}/{name}')
            files.append({'path': str(destination.relative_to(root)), 'bytes': destination.stat().st_size, 'sha256': actual, 'upstream_sha256': expected, 'reused_verified_base_weights': reused})
            print(json.dumps({'source': source['key'], 'file': name, 'verified': True, 'reused': reused}), flush=True)
        if args.kind == 'model':
            index = json.loads((target / 'model.safetensors.index.json').read_text())
            for shard in set(index['weight_map'].values()):
                if not (target / shard).is_file():
                    raise RuntimeError(f'Missing model shard {shard}')
        receipt = {**source, 'local_path': str(target), 'files': files}
        receipts.append(receipt)
        (root / 'manifests' / f'{source["key"]}.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (root / 'manifests' / f'{args.kind}-acquisition.json').write_text(json.dumps(receipts, indent=2) + '\n')


if __name__ == '__main__':
    main()
