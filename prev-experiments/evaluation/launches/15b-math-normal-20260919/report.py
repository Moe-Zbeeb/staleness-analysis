import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import newdispatcher


def main():
    project = Path('/mnt/nfs/home/mohamadzbib/projects/rl-infra')
    source_root = project / 'evaluation-runs/15b-math-20260919/source'
    original_root = project / 'outputs/math-sweep-15b-20260919'
    output_root = project / 'outputs/math-sweep-15b-normal-20260919'
    launcher_sha = newdispatcher.verify_package()
    source, source_sha = newdispatcher.load_source(source_root)
    args = SimpleNamespace(plan=original_root / 'plan.json', data=original_root / 'data.json', prepared_root=original_root / 'prepared', tokenizer_root=project)
    entries, identity = source.prepared_entries(args, source_sha)
    partition, decisions = newdispatcher.partition_entries(entries, 2)
    decisions.update(source_identity=identity, launcher_sha256=launcher_sha)
    partition_path = output_root / 'results/.partial-dispatch' / identity['plan_sha256'][:24] / 'partition.json'
    if json.loads(partition_path.read_text()) != decisions:
        raise ValueError('Recorded partition differs from the frozen complete source plan')
    originals = {entry['source_cell_id']: entry for entry in entries}
    seen = set()
    derived = set()
    for shard in range(2):
        index = json.loads((output_root / f'prepared/preparation-index-shard-{shard}.json').read_text())
        if index['status'] != 'complete' or index['shard'] != shard or index['launcher_sha256'] != launcher_sha:
            raise ValueError('Derived shard index has incompatible provenance')
        if index['source_identity'] != identity or index['partition_sha256'] != newdispatcher.file_hash(partition_path):
            raise ValueError('Derived shard source identity changed')
        if set(index['cells']) != {entry['source_cell_id'] for entry in partition[shard]}:
            raise ValueError('Derived shard membership differs from the frozen partition')
        for source_cell_id, item in index['cells'].items():
            if source_cell_id in seen or source_cell_id not in originals or item['cell_id'] in derived:
                raise ValueError('Duplicate or unexpected evaluation cell')
            seen.add(source_cell_id)
            derived.add(item['cell_id'])
            if item['original_cell_id'] != originals[source_cell_id]['cell']['cell_id']:
                raise ValueError('Derived source-to-original cell mapping changed')
            directory = output_root / 'prepared' / item['cell_id']
            if newdispatcher.file_hash(directory / 'preparation.json') != item['preparation_sha256']:
                raise ValueError('Prepared evaluation cell changed')
            entry = {'source_cell_id': source_cell_id, 'prepared_dir': str(directory), 'cell': json.loads((directory / 'cell.json').read_text())}
            if source.verify_complete(entry, output_root / 'results') is None:
                raise ValueError('Evaluation cell is incomplete')
    if len(seen) != 140 or seen != set(originals):
        raise ValueError('The combined report requires all 140 prepared evaluation cells')
    subprocess.run([sys.executable, str(source_root / 'evaluation/sweep.py'), 'report', '--prepared-root', str(output_root / 'prepared'), '--output-root', str(output_root / 'results'), '--report', str(output_root / 'report.json')], check=True)
    print(json.dumps({'status': 'complete', 'cells': len(seen), 'source_sha256': source_sha, 'report': str(output_root / 'report.json')}), flush=True)


if __name__ == '__main__':
    main()
