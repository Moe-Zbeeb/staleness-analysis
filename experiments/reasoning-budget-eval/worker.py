import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import socket
import time

from common import ROOT, digest, read_rows, validate_manifest, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-index', type=int, required=True)
    parser.add_argument('--slot', type=int, required=True)
    args = parser.parse_args()
    validate_manifest()
    spec = json.loads((ROOT / 'spec.json').read_text())
    model = spec['models'][args.model_index]
    prep = json.loads((ROOT / 'preparation.json').read_text())['models'][model['tag']]
    output = ROOT / 'outputs' / model['tag']
    output.mkdir(parents=True, exist_ok=True)
    import torch
    assert torch.cuda.device_count() == 1
    props = torch.cuda.get_device_properties(0)
    assert 'A100' in props.name and props.total_memory >= 79_000_000_000, str(props)
    probe = torch.ones((128, 128), device='cuda', dtype=torch.bfloat16)
    assert torch.isfinite(probe @ probe).all().item()
    torch.cuda.synchronize()
    del probe
    torch.cuda.empty_cache()
    health = {'job_id': os.environ['SLURM_JOB_ID'], 'slot': args.slot, 'uuid': os.environ['CUDA_VISIBLE_DEVICES'], 'name': props.name, 'total_memory': props.total_memory, 'host': socket.gethostname(), 'status': 'passed'}
    write_json(output / f'health-{args.slot}.json', health)
    deadline = time.monotonic() + 300
    while True:
        healthy = []
        for slot in range(spec['replicas']):
            p = output / f'health-{slot}.json'
            if p.exists():
                value = json.loads(p.read_text())
                if value['job_id'] == health['job_id']:
                    healthy.append(value)
        if len(healthy) == spec['replicas']:
            assert len({h['uuid'] for h in healthy}) == spec['replicas']
            break
        if time.monotonic() > deadline:
            raise RuntimeError('Not all eight allocated GPUs passed health checks')
        time.sleep(1)
    from vllm import LLM, SamplingParams
    from vllm.tokenizers import get_tokenizer
    tokenizer = get_tokenizer(model['tokenizer_path'], trust_remote_code=False, local_files_only=True)
    samples = read_rows(ROOT / 'inputs/samples.jsonl')
    prompts = {r['key']: r for r in read_rows(ROOT / 'inputs' / (model['tag'] + '.jsonl'))}
    selected = samples[args.slot::spec['replicas']]
    assert len(selected) == 64
    for row in selected:
        p = prompts[row['key']]
        assert tokenizer.encode(p['rendered_prompt'], add_special_tokens=False) == p['prompt_token_ids']
    engine = {'model': model['target'], 'tokenizer': model['tokenizer_path'], 'dtype': 'bfloat16', 'tensor_parallel_size': 1, 'max_model_len': prep['max_model_len'], 'gpu_memory_utilization': 0.90, 'enforce_eager': True, 'max_num_seqs': spec['batch_size'], 'max_num_batched_tokens': 8192, 'enable_chunked_prefill': True, 'enable_prefix_caching': False, 'generation_config': 'vllm', 'seed': spec['seed'], 'trust_remote_code': False}
    llm = LLM(**engine)
    tokenizer = llm.get_tokenizer()
    for row in selected:
        p = prompts[row['key']]
        assert tokenizer.encode(p['rendered_prompt'], add_special_tokens=False) == p['prompt_token_ids']
    provenance = {'model': model['repo_id'], 'revision': model['revision'], 'engine': engine, 'health': health, 'sampling': model['sampling'], 'versions': {p: importlib.metadata.version(p) for p in ['torch', 'vllm', 'transformers']}, 'manifest_sha256': digest(ROOT / 'manifest.json')}
    gate_prompt = {'prompt_token_ids': prompts[selected[0]['key']]['prompt_token_ids']}
    for eos in prep['eos_token_ids']:
        result = llm.generate([gate_prompt], SamplingParams(temperature=0, max_tokens=8, stop_token_ids=prep['eos_token_ids'], logit_bias={eos: 100}, skip_special_tokens=False), use_tqdm=False)[0].outputs[0]
        assert result.finish_reason == 'stop' and len(result.token_ids) <= 1
    gate = llm.generate([gate_prompt], SamplingParams(temperature=0, max_tokens=8, min_tokens=8, ignore_eos=True, logprobs=1), use_tqdm=False)[0].outputs[0]
    assert len(gate.token_ids) == 8 and all(math.isfinite(p[t].logprob) for t, p in zip(gate.token_ids, gate.logprobs))
    provenance['decoding_gate'] = 'passed'
    write_json(output / f'provenance-{args.slot}.json', provenance)
    print(json.dumps({'phase': 'ready', 'model': model['tag'], 'slot': args.slot}), flush=True)
    for budget in spec['budgets']:
        folder = output / str(budget) / f'worker-{args.slot}'
        folder.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(selected), spec['batch_size']):
            batch = selected[start:start + spec['batch_size']]
            path = folder / f'batch-{start:04d}.json'
            if path.exists():
                saved = json.loads(path.read_text())
                assert saved['manifest_sha256'] == provenance['manifest_sha256']
                assert [r['key'] for r in saved['records']] == [r['key'] for r in batch]
                continue
            parameters = [SamplingParams(**model['sampling'], max_tokens=budget, seed=r['seed'], stop_token_ids=prep['eos_token_ids'], ignore_eos=False, skip_special_tokens=False, repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0) for r in batch]
            began = time.monotonic()
            replies = llm.generate([{'prompt_token_ids': prompts[r['key']]['prompt_token_ids']} for r in batch], parameters, use_tqdm=False)
            elapsed = time.monotonic() - began
            assert len(replies) == len(batch)
            records = []
            for row, reply in zip(batch, replies):
                result = reply.outputs[0]
                ids = list(result.token_ids)
                assert reply.prompt_token_ids == prompts[row['key']]['prompt_token_ids']
                assert result.finish_reason in ('stop', 'length') and len(ids) <= budget
                eos_positions = [i for i, t in enumerate(ids) if t in prep['eos_token_ids']]
                assert not eos_positions or eos_positions == [len(ids) - 1]
                text = tokenizer.decode(ids[:-1] if eos_positions else ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                records.append({'key': row['key'], 'dataset': row['dataset'], 'id': row['id'], 'sample_index': row['sample_index'], 'model': model['tag'], 'budget': budget, 'seed': row['seed'], 'completion': text, 'raw_completion': result.text, 'completion_ids': ids, 'completion_tokens': len(ids), 'prompt_tokens': len(reply.prompt_token_ids), 'finish_reason': result.finish_reason, 'stop_reason': result.stop_reason, 'prefilled_think': prompts[row['key']]['prefilled_think'], 'slot': args.slot, 'job_id': health['job_id']})
            write_json(path, {'manifest_sha256': provenance['manifest_sha256'], 'batch_seconds': elapsed, 'records': records})
            print(json.dumps({'phase': 'batch_complete', 'model': model['tag'], 'budget': budget, 'slot': args.slot, 'completed': start + len(batch), 'total': len(selected), 'seconds': round(elapsed, 1)}), flush=True)
    write_json(output / f'worker-{args.slot}-complete.json', {'status': 'complete', 'responses': len(selected) * len(spec['budgets']), 'manifest_sha256': provenance['manifest_sha256']})


if __name__ == '__main__':
    main()
