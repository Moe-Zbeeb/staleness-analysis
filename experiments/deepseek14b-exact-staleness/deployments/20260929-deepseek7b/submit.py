import json
import subprocess
from pathlib import Path

root = Path('/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study')
control = root / 'launches/deepseek7b-dapo17k-b32-hp-20260929-v1'
assert (control / 'preparation-complete.json').exists()
assert not (control / 'submission.json').exists()
record = {'jobs': {}, 'commands': {}, 'fields': {}, 'allowed_gpu_nodes': ['deep-chungus-9', 'deep-chungus-10', 'deep-chungus-11'], 'requested_peak_gpus': 18, 'gpu_execution_verified': False}
record['observed_qos_limits'] = subprocess.check_output(['sacctmgr', 'show', 'qos', 'high-priority', 'format=Name,MaxTRESPU,MaxJobsPU,MaxSubmitPU', '-P'], text=True)
common = ['sbatch', '--parsable', '--account=grad-students', '--partition=high-priority', '--qos=high-priority', '--nodes=1', '--ntasks=1', '--no-requeue', f'--chdir={control}/release']
def submit(role, command):
    job = subprocess.check_output(command, text=True).strip().split(';')[0]
    assert job.isdigit(), job
    record['jobs'][role] = job
    record['commands'][role] = command
    (control / 'submission.json').write_text(json.dumps(record, indent=2) + '\n')
    fields = subprocess.check_output(['scontrol', 'show', 'job', '-o', job], text=True).strip()
    record['fields'][role] = fields
    (control / 'submission.json').write_text(json.dumps(record, indent=2) + '\n')
    for token in ['Account=grad-students', 'Partition=high-priority', 'QOS=high-priority', 'Requeue=0', 'NumNodes=1']:
        assert token in fields, fields
    return job
validation = submit('validation', common + ['--job-name=ds7b-config-check', '--cpus-per-task=4', '--mem=24G', '--time=00:20:00', f'--output={control}/validation-%j.out', f'--error={control}/validation-%j.err', str(control / 'validate.sh')])
gpu = common + ['--exclude=deep-chungus-[1-8],deep-gpu-[10-11]', '--time=45-00:00:00', '--kill-on-invalid-dep=yes']
main = submit('learner', gpu + ['--job-name=ds7b-dapo-b32-k256', '--exclusive', '--gres=gpu:a100:8', '--cpus-per-task=128', '--mem=0', f'--dependency=afterok:{validation}', f'--output={control}/learner/slurm-%j.out', f'--error={control}/learner/slurm-%j.err', str(control / 'learner/job.sh')])
submit('worker', gpu + ['--job-name=ds7b-dapo-b32-history', '--gres=gpu:a100:4', '--cpus-per-task=64', '--mem=256G', f'--dependency=afterok:{validation},after:{main}', f'--output={control}/worker/slurm-%j.out', f'--error={control}/worker/slurm-%j.err', str(control / 'worker/job.sh'), main])
submit('onpolicy', gpu + ['--job-name=ds7b-dapo-b32-k0', '--gres=gpu:a100:6', '--cpus-per-task=96', '--mem=384G', f'--dependency=afterok:{validation}', f'--output={control}/onpolicy/slurm-%j.out', f'--error={control}/onpolicy/slurm-%j.err', str(control / 'onpolicy/job.sh')])
for role, count in [('learner', 8), ('worker', 4), ('onpolicy', 6)]:
    assert f'gres/gpu:a100={count}' in record['fields'][role], record['fields'][role]
print(json.dumps(record, indent=2))
