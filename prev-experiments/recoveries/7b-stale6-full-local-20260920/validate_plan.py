import datetime,hashlib,importlib.util,json,os,subprocess,sys
from dataclasses import asdict
from pathlib import Path
import torch
import torch.distributed.checkpoint as dcp
from prime_rl.configs.rl import RLConfig
from prime_rl.utils.config import cli,dump_resolved_config

root=Path(__file__).resolve().parent
spec=json.loads((root/'spec.json').read_text())
result={'status':'passed','at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':os.environ.get('SLURM_JOB_ID'),'restart_count':os.environ.get('SLURM_RESTART_COUNT','0'),'caps':{}}
allowed={'.orchestrator.ckpt.wait_for_weights_timeout'}
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
 assert {d['path']:(d['expected'],d['actual']) for d in differences}=={'.orchestrator.ckpt.wait_for_weights_timeout':(None,7200)},differences
 assert current.deployment.gpus_per_node==8 and current.deployment.num_train_gpus==4 and current.deployment.num_infer_gpus==4
 assert current.inference.vllm.tensor_parallel_size==1 and current.inference.vllm.data_parallel_size==4
 assert current.orchestrator.max_off_policy_steps==int(cap)
 assert current.trainer.ckpt.skip_optimizer is False and current.trainer.ckpt.skip_progress is False and current.trainer.ckpt.skip_scheduler is False
 assert current.inference.vllm.api_server_count==4 and current.trainer.weight_broadcast.inference_world_size==4 and current.orchestrator.weight_broadcast.inference_world_size==4
 assert current.monitors.file.path.name=='metrics.jsonl'
 checkpoint_root=Path(item['run_dir'])/'checkpoints'
 paired=[int(p.name[5:]) for p in checkpoint_root.glob('step_*') if p.name[5:].isdigit() and all((p/name).is_file() and (p/name).stat().st_size>0 for name in ['trainer/.metadata','orchestrator/progress.pt',*[f'trainer/__{rank}_0.distcp' for rank in range(4)]])]
 selected_step=max(paired)
 assert item['resume_step']<=selected_step<=1000
 checkpoint=checkpoint_root/f'step_{selected_step}'
 payload={'app':{'progress':{'step':0}}}
 reader=dcp.FileSystemReader(checkpoint/'trainer')
 metadata=reader.read_metadata()
 dcp.load(payload,storage_reader=reader,no_dist=True)
 assert payload['app']['progress']['step']==selected_step,payload
 keys=list(metadata.state_dict_metadata)
 assert any(k.startswith('app.model.') for k in keys)
 assert any(k.startswith('app.optimizers.') for k in keys),keys[:10]
 assert any('scheduler' in k for k in keys),keys[:10]
 state=torch.load(checkpoint/'orchestrator/progress.pt',map_location='cpu',weights_only=False)
 progress=asdict(state['progress'])
 assert progress['step']==selected_step+1 if selected_step<1000 else progress['step'] in [1000,1001],progress
 assert 'dapo-math' in state['train_source']['envs']
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7')
 mapped=subprocess.check_output([sys.executable,str(root/'guard.py'),'devices','--experiment-root',str(experiment)],env=env,text=True).strip()
 assert mapped=='4,5,6,7,0,1,2,3',mapped
 result['caps'][cap]={'resume_step':selected_step,'trainer_saved_step':payload['app']['progress']['step'],'orchestrator_next_step':progress['step'],'changes':differences,'noncontiguous_device_mapping':mapped,'source_sha256':item['source_sha256'],'data_sha256':item['data_sha256'],'tensor_metadata_entries':len(keys)}
manifest=json.loads((root/'package-manifest.json').read_text())
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in manifest.items())
result['package_manifest_sha256']=hashlib.sha256((root/'package-manifest.json').read_bytes()).hexdigest()
(root/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
