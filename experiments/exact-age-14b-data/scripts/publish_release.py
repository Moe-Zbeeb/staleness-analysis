import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=Path, required=True)
    args = parser.parse_args()
    code_root = Path(__file__).resolve().parents[1]
    spec = json.loads((code_root / 'configs/release.json').read_text())
    api = HfApi()
    assert api.whoami()['name'] == spec['namespace']
    public = HfApi(token=False)
    receipt_path = args.folder / 'publication.json'
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {'datasets': {}}
    if 'collection_slug' not in receipt:
        collection = api.create_collection(title=spec['collection_title'], namespace=spec['namespace'], description=spec['collection_description'], private=spec['private'], exists_ok=True)
        receipt.update(collection_slug=collection.slug, collection_url=collection.url)
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    allowed = {'README.md', 'PREPARATION.md', 'manifest.json', 'source-revisions.json', 'data/train.parquet'}
    for name, entry in spec['datasets'].items():
        folder = args.folder / name
        assert {str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file()} == allowed
        manifest = json.loads((folder / 'manifest.json').read_text())
        assert manifest['rows'] == entry['expected_rows']
        assert digest(folder / 'data/train.parquet') == manifest['train_parquet_sha256']
        repo = entry['repo_id']
        if name not in receipt['datasets']:
            api.create_repo(repo, repo_type='dataset', private=spec['private'], exist_ok=True)
            assert api.dataset_info(repo).private is spec['private']
            existing = set(api.list_repo_files(repo, repo_type='dataset'))
            assert existing <= allowed | {'.gitattributes'}, existing
            commit = api.upload_folder(repo_id=repo, repo_type='dataset', folder_path=folder, allow_patterns=sorted(allowed), commit_message=f"Publish {manifest['rows']:,} deduplicated parser-compatible math questions").oid
            receipt['datasets'][name] = {'repo_id': repo, 'commit': commit, 'rows': manifest['rows'], 'url': 'https://huggingface.co/datasets/' + repo}
            receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
        commit = receipt['datasets'][name]['commit']
        info = public.dataset_info(repo, revision=commit, files_metadata=True)
        remote = {s.rfilename: s for s in info.siblings}
        assert set(remote) == allowed | {'.gitattributes'}
        for filename in sorted(allowed):
            local = folder / filename
            item = remote[filename]
            assert item.size == local.stat().st_size
            if item.lfs:
                assert item.lfs.sha256 == digest(local)
            else:
                downloaded = Path(hf_hub_download(repo, filename, repo_type='dataset', revision=commit, token=False))
                assert digest(downloaded) == digest(local)
        assert info.private is False
        api.add_collection_item(receipt['collection_slug'], item_id=repo, item_type='dataset', note=f"{manifest['rows']:,} questions; independently deduplicated" if name != 'merged' else f"{manifest['rows']:,} questions; cross-source deduplicated merge", exists_ok=True)
        receipt['datasets'][name].update(status='published_and_verified', parquet_sha256=manifest['train_parquet_sha256'])
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps(receipt['datasets'][name]), flush=True)
    collection = public.get_collection(receipt['collection_slug'])
    assert not collection.private
    assert {(x.item_id, x.item_type) for x in collection.items} == {(x['repo_id'], 'dataset') for x in spec['datasets'].values()}
    receipt['status'] = 'published_and_verified'
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
