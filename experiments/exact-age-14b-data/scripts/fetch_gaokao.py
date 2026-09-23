import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root
    spec = json.loads((root / 'configs/gaokao-source.json').read_text())
    repository = spec['repository'].removeprefix('https://github.com/')
    url = f"https://raw.githubusercontent.com/{repository}/{spec['revision']}/{spec['path']}"
    with urlopen(url, timeout=60) as response:
        content = response.read()
    assert hashlib.sha256(content).hexdigest() == spec['sha256']
    destination = root / 'data/benchmarks/gaokao-2023en-test.jsonl'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)


if __name__ == '__main__':
    main()
