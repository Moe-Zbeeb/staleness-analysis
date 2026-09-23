import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    publication = json.loads((root / 'release/publication.json').read_text())
    assert publication['status'] == 'published_and_verified'
    names = ['skywork', 'deepscaler', 'merged']
    for name in names:
        entry = publication['datasets'][name]
        assert entry['status'] == 'published_and_verified'
        assert digest(root / 'release' / name / 'data/train.parquet') == entry['parquet_sha256']
    model_indexes = [root / 'models' / name / 'model.safetensors.index.json' for name in ['qwen3-14b', 'deepseek-r1-distill-qwen-14b']]
    model_hashes = {str(p.relative_to(root)): digest(p) for p in model_indexes}
    targets = [root / 'data' / name for name in ['raw', 'benchmarks', 'quarantine']]
    targets += [root / 'data/processed' / name / filename for name in names for filename in ['clean.jsonl', 'clean.parquet', 'reference_parseable.jsonl']]
    targets += list((root / 'reports').rglob('*.jsonl'))
    targets += [p for p in (root / 'cache').rglob('datasets--*') if p.is_dir()]
    targets = sorted(set(p for p in targets if p.exists() or p.is_symlink()))
    entries = []
    for path in targets:
        assert path.is_relative_to(root) and path != root
        if not path.is_symlink():
            assert path.resolve().is_relative_to(root)
        files = [p for p in path.rglob('*') if p.is_file() and not p.is_symlink()] if path.is_dir() and not path.is_symlink() else [path]
        entries.append({'path': str(path.relative_to(root)), 'file_count': len(files), 'bytes': sum(p.lstat().st_size for p in files)})
    receipt = {'status': 'planned', 'created_at': datetime.now(timezone.utc).isoformat(), 'publication_collection': publication['collection_url'], 'targets': entries, 'bytes_removed': sum(x['bytes'] for x in entries), 'preserved': ['release', 'models', 'source code', 'aggregate reports and manifests']}
    receipt_path = root / 'reports/dataset-cleanup.json'
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    if not args.execute:
        print(json.dumps(receipt), flush=True)
        return
    for path in targets:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    for name in names:
        folder = root / 'data/processed' / name
        folder.mkdir(parents=True, exist_ok=True)
        link = folder / 'train.parquet'
        destination = root / 'release' / name / 'data/train.parquet'
        if link.is_symlink():
            assert link.resolve() == destination.resolve()
        else:
            assert not link.exists()
            link.symlink_to(destination)
    assert all(not p.exists() and not p.is_symlink() for p in targets)
    assert {str(p.relative_to(root)): digest(p) for p in model_indexes} == model_hashes
    for name in names:
        assert digest(root / 'data/processed' / name / 'train.parquet') == publication['datasets'][name]['parquet_sha256']
    receipt.update(status='completed', deleted_targets=len(targets), retained_rows={name: publication['datasets'][name]['rows'] for name in names}, model_indexes_unchanged=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
