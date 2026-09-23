import hashlib,json,os
from pathlib import Path

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify():
    root=Path(__file__).resolve().parent
    provenance=json.loads((root/'provenance.json').read_text())
    manifest=json.loads((root/'package-manifest.json').read_text())
    assert {'gpu_devices.py','train-stale6.sh','verify_package.py','provenance.json','test_gpu_devices.py'}<=set(manifest)
    for name,wanted in manifest.items():
        path=(root/name).resolve()
        assert path.is_relative_to(root) and sha(path)==wanted,name
    base=Path(provenance['base_recovery_root'])
    assert sha(base/'package-manifest.json')==provenance['base_package_manifest_sha256']
    for name,wanted in json.loads((base/'package-manifest.json').read_text()).items():
        assert sha(base/name)==wanted,name
    original=base/'train-stale6.sh'
    assert sha(original)==provenance['base_launcher_sha256']
    expected=original.read_text()
    for change in provenance['changes']:
        assert expected.count(change['before'])==1
        expected=expected.replace(change['before'],change['after'])
    assert (root/'train-stale6.sh').read_text()==expected
    assert sha(base/'validation.json')==provenance['base_validation_sha256']
    validation=json.loads((base/'validation.json').read_text())
    assert validation['status']=='passed' and validation['package_manifest_sha256']==provenance['base_package_manifest_sha256']
    assert validation['caps']['6']['trainer_saved_step']==125 and validation['caps']['6']['orchestrator_next_step']==126
    print(json.dumps({'status':'passed','job_id':os.environ.get('SLURM_JOB_ID'),'restart_count':os.environ.get('SLURM_RESTART_COUNT','0'),'mapping_package_manifest_sha256':sha(root/'package-manifest.json'),'base_package_manifest_sha256':provenance['base_package_manifest_sha256'],'changes':'GPU mapping and allocated-device occupancy checks only','gpu_layout':provenance['gpu_layout']}),flush=True)

if __name__=='__main__':
    verify()
