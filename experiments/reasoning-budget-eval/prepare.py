from collections import Counter
import importlib.metadata
import hashlib
import json
import os
import random
from pathlib import Path
import time

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from common import ROOT, digest, write_json


def main():
    if (ROOT / 'manifest.json').exists() or (ROOT / 'outputs').exists():
        raise FileExistsError('Use a fresh directory; frozen inputs and existing outputs must not be replaced')
    spec = json.loads((ROOT / 'spec.json').read_text())
    publication = json.loads((ROOT / 'datasets.lock.json').read_text())
    data = ROOT / 'inputs'
    data.mkdir(exist_ok=True)
    rows = []
    summary = {'datasets': {}, 'models': {}, 'created_at': time.time(), 'job_id': os.environ.get('SLURM_JOB_ID')}
    for dataset in spec['datasets']:
        source = Path(spec['dataset_root']) / 'data' / 'processed' / dataset / 'train.parquet'
        expected = publication['datasets'][dataset]
        assert digest(source) == expected['parquet_sha256']
        table = pq.read_table(source, columns=['id', 'prompt', 'answers'])
        assert len(table) == expected['rows']
        indices = random.Random(spec['seed']).sample(range(len(table)), spec['sample_size'])
        selected = table.take(indices).to_pylist()
        assert len({r['id'] for r in selected}) == spec['sample_size']
        for i, (row, source_index) in enumerate(zip(selected, indices)):
            assert row['prompt'].strip() and row['answers']
            row.update(dataset=dataset, sample_index=i, source_index=source_index, key=dataset + ':' + row['id'])
            row['seed'] = spec['seed'] + len(rows)
            rows.append(row)
        summary['datasets'][dataset] = {**expected, 'path': str(source.resolve()), 'seed': spec['seed'], 'source_indices': indices, 'samples': len(selected), 'reference_components': dict(Counter(len(r['answers']) for r in selected))}
    assert len(rows) == 512
    (data / 'samples.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    summary['shared_question_ids'] = len({r['id'] for r in rows if r['dataset'] == 'skywork'} & {r['id'] for r in rows if r['dataset'] == 'deepscaler'})
    for model in spec['models']:
        tokenizer = AutoTokenizer.from_pretrained(model['target'], local_files_only=True, trust_remote_code=False)
        config = json.loads((Path(model['target']) / 'config.json').read_text())
        prompts = []
        for row in rows:
            messages = [{'role': 'user', 'content': row['prompt'] + '\n\n' + spec['instruction']}]
            kwargs = {'enable_thinking': True} if model['tag'] == 'qwen3' else {}
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
            ids = tokenizer.encode(rendered, add_special_tokens=False)
            assert tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **kwargs) == ids
            prompts.append({'key': row['key'], 'prompt_token_ids': ids, 'rendered_prompt': rendered, 'prefilled_think': rendered.rstrip().endswith('<think>')})
        longest = max(len(p['prompt_token_ids']) for p in prompts)
        context = max(16384, ((longest + max(spec['budgets']) + 1023) // 1024) * 1024)
        assert context <= config['max_position_embeddings'], (model['tag'], longest, context)
        (data / (model['tag'] + '.jsonl')).write_text(''.join(json.dumps(p, ensure_ascii=False) + '\n' for p in prompts))
        eos = config['eos_token_id']
        eos = eos if isinstance(eos, list) else [eos]
        assert tokenizer.eos_token_id in eos, (model['tag'], tokenizer.eos_token_id, eos)
        summary['models'][model['tag']] = {'max_prompt_tokens': longest, 'max_model_len': context, 'native_context': config['max_position_embeddings'], 'eos_token_ids': eos, 'sampling': model['sampling'], 'template_sha256': hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(), 'example_prompt_suffix': prompts[0]['rendered_prompt'][-100:]}
        print(json.dumps({'prepared_model': model['tag'], **summary['models'][model['tag']]}), flush=True)
    summary['versions'] = {p: importlib.metadata.version(p) for p in ['transformers', 'pyarrow', 'math-verify']}
    summary['status'] = 'passed'
    write_json(ROOT / 'preparation.json', summary)
    print(json.dumps({'prepared_datasets': summary['datasets'], 'shared_question_ids': summary['shared_question_ids']}), flush=True)


if __name__ == '__main__':
    main()
