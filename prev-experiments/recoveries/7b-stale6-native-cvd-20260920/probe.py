import ctypes,csv,datetime,hashlib,io,json,os,re,socket,subprocess,uuid
from pathlib import Path
r={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'node':socket.gethostname(),'environment':{k:os.environ.get(k) for k in ['SLURM_JOB_ID','SLURM_JOB_GPUS','SLURM_STEP_GPUS','CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER']}}
assert r['node'].split('.')[0]=='deep-chungus-9'
cvd=os.environ['CUDA_VISIBLE_DEVICES']
assert re.fullmatch(r'\d+(,\d+){4}',cvd) and len(set(cvd.split(',')))==5
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
driver=ctypes.CDLL('libcuda.so.1')
def call(name,types,*args):
 fn=getattr(driver,name);fn.argtypes=types;fn.restype=ctypes.c_int
 status=fn(*args)
 assert status==0,(name,status)
call('cuInit',[ctypes.c_uint],0)
count=ctypes.c_int()
call('cuDeviceGetCount',[ctypes.POINTER(ctypes.c_int)],ctypes.byref(count))
assert count.value==5,count.value
r['cuda_visible_devices_unchanged']=os.environ['CUDA_VISIBLE_DEVICES']==cvd
r['visible_devices']=[]
for ordinal in range(count.value):
 d=ctypes.c_int();call('cuDeviceGet',[ctypes.POINTER(ctypes.c_int),ctypes.c_int],ctypes.byref(d),ordinal)
 uid=(ctypes.c_ubyte*16)();call('cuDeviceGetUuid_v2',[ctypes.c_void_p,ctypes.c_int],ctypes.byref(uid),d.value)
 bus=ctypes.create_string_buffer(64);call('cuDeviceGetPCIBusId',[ctypes.c_void_p,ctypes.c_int,ctypes.c_int],bus,64,d.value)
 r['visible_devices'].append({'local_ordinal':ordinal,'slurm_visible_selector':cvd.split(',')[ordinal],'uuid':'GPU-'+str(uuid.UUID(bytes=bytes(uid))),'pci_bus_id':bus.value.decode()})
selector='--id='+','.join(d['uuid'] for d in r['visible_devices'])
for key,query in [('memory','--query-gpu=uuid,name,memory.total,memory.free'),('processes','--query-compute-apps=gpu_uuid,pid,used_gpu_memory')]:
 p=subprocess.run(['nvidia-smi',selector,query,'--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=30,check=True)
 r[key]=[[v.strip() for v in row] for row in csv.reader(io.StringIO(p.stdout))]
r['proc_gpu_inventory']=[{'path':str(p),'contents':p.read_text()} for p in Path('/proc/driver/nvidia/gpus').glob('*/information')]
p=Path('/usr/local/etc/gres.conf');r['gres_config_sha256']=hashlib.sha256(p.read_bytes()).hexdigest();r['gres_lines']=[line for line in p.read_text().splitlines() if 'chungus' in line]
r['selected_all_idle']=not r['processes'] and all(int(v[3])>=0.85*int(v[2]) for v in r['memory'])
root=Path('/mnt/nfs/home/mohamadzbib/projects/rl-infra/recoveries/7b-stale6-native-cvd-20260920')
(root/f'probe-{os.environ["SLURM_JOB_ID"]}.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2),flush=True)
assert r['selected_all_idle'],r['processes']
