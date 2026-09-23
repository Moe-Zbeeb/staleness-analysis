import datetime,hashlib,importlib.util,json,os,subprocess,sys
from dataclasses import asdict
from pathlib import Path
import torch
import torch.distributed.checkpoint as dcp
from prime_rl.configs.rl import RLConfig
from prime_rl.utils.config import cli,dump_resolved_config

root=Path(__file__).resolve().parent
spec=json.loads((root/'spec.json').read_text())
result={'status':'passed','at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':os.environ.get('SLURM_JOB_ID'),'caps':{}}
allowed={'.deployment.gpus_per_node','.deployment.num_infer_gpus','.inference.vllm.data_parallel_size','.inference.vllm.api_server_count','.trainer.weight_broadcast.inference_world_size','.orchestrator.weight_broadcast.inference_world_size'}
for cap,item in spec['caps'].items():
 experiment=Path(item['experiment_root'])
 subprocess.run([sys.executable,str(root/'guard.py'),'gate','--experiment-root',str(experiment)],check=True)
 module_spec=importlib.util.spec_from_file_location('original_guard_'+cap,experiment/'scripts/staleness_guard.py')
 original=importlib.util.module_from_spec(module_spec)
 module_spec.loader.exec_module(original)
 sys.path.insert(0,str(experiment/'python'))
 old=sys.argv
 try:
  sys.argv=['rl','@',str(experiment/'config/main.toml')]
  frozen=cli(RLConfig)
  sys.argv=['rl','@',str(experiment/'config/main.toml'),'@',str(root/'overlay.toml')]
  current=cli(RLConfig)
 finally:
  sys.argv=old
 differences=original.differences(dump_resolved_config(frozen),dump_resolved_config(current))
 assert differences and {d['path'] for d in differences}<=allowed,differences
 assert current.deployment.gpus_per_node==6 and current.deployment.num_train_gpus==4 and current.deployment.num_infer_gpus==2
 assert current.inference.vllm.tensor_parallel_size==1 and current.inference.vllm.data_parallel_size==2
 assert current.orchestrator.max_off_policy_steps==int(cap)
 assert current.trainer.ckpt.skip_optimizer is False and current.trainer.ckpt.skip_progress is False
 checkpoint=Path(item['run_dir'])/f"checkpoints/step_{item['resume_step']}"
 payload={'app':{'progress':{'step':0}}}
 reader=dcp.FileSystemReader(checkpoint/'trainer')
 metadata=reader.read_metadata()
 dcp.load(payload,storage_reader=reader,no_dist=True)
 assert payload['app']['progress']['step']==item['resume_step'],payload
 keys=list(metadata.state_dict_metadata)
 assert any(k.startswith('app.model.') for k in keys)
 assert any(k.startswith('app.optimizers.') for k in keys),keys[:10]
 state=torch.load(checkpoint/'orchestrator/progress.pt',map_location='cpu',weights_only=False)
 progress=asdict(state['progress'])
 assert progress['step']==item['resume_step']+1,progress
 assert 'dapo-math' in state['train_source']['envs']
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,2,3,4,6,7')
 mapped=subprocess.check_output([sys.executable,str(root/'guard.py'),'devices','--experiment-root',str(experiment)],env=env,text=True).strip()
 assert mapped=='6,7,0,2,3,4',mapped
 result['caps'][cap]={'resume_step':item['resume_step'],'trainer_saved_step':payload['app']['progress']['step'],'orchestrator_next_step':progress['step'],'changes':differences,'noncontiguous_device_mapping':mapped,'source_sha256':item['source_sha256'],'data_sha256':item['data_sha256'],'tensor_metadata_entries':len(keys)}
manifest=json.loads((root/'package-manifest.json').read_text())
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in manifest.items())
result['package_manifest_sha256']=hashlib.sha256((root/'package-manifest.json').read_bytes()).hexdigest()
(root/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
