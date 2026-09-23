import argparse
from pathlib import Path

import dispatch
from sweep import report_results


def main():
    parser = argparse.ArgumentParser()
    for name in ('plan', 'data', 'prepared-root', 'results-root', 'tokenizer-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    sha = dispatch.verify_package()
    entries, identity = dispatch.prepared_entries(args, sha)
    if len(entries) != 140:
        raise ValueError('Expected all 140 frozen production cells')
    for entry in entries:
        dispatch.verify_complete(entry, args.results_root)
    args.output_root = args.results_root
    args.report = args.results_root.parent / 'report.json'
    args.allow_partial = False
    args.bootstrap_samples = 2000
    report_results(args)
    dispatch.write_json(args.results_root.parent / 'report-provenance.json', {'status': 'complete', 'cells': len(entries), **identity, 'report_sha256': dispatch.file_hash(args.report)})


if __name__ == '__main__':
    main()
