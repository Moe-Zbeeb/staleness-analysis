import importlib.metadata
import json
from pathlib import Path

from tokenizers import Tokenizer
from vllm.tokenizers import get_tokenizer

from common import ROOT, read_rows, write_json


def main():
    if (ROOT / 'manifest.json').exists() or (ROOT / 'outputs').exists():
        raise FileExistsError('Use a fresh directory; frozen inputs and existing outputs must not be replaced')
    spec = json.loads((ROOT / 'spec.json').read_text())
    results = {}
    names = ['tokenizer.json', 'config.json', 'added_tokens.json', 'special_tokens_map.json', 'tokenizer.model', 'merges.txt', 'vocab.json', 'chat_template.jinja']
    for model in spec['models']:
        original = Path(model['target'])
        target = Path(model['tokenizer_path'])
        if target != original:
            target.mkdir(parents=True, exist_ok=True)
            for name in names:
                source = original / name
                if source.is_file() and not (target / name).exists():
                    (target / name).symlink_to(source)
            config = json.loads((original / 'tokenizer_config.json').read_text())
            original_class = config.get('tokenizer_class')
            config['tokenizer_class'] = 'PreTrainedTokenizerFast'
            write_json(target / 'tokenizer_config.json', config)
        else:
            original_class = json.loads((original / 'tokenizer_config.json').read_text()).get('tokenizer_class')
        tokenizer = get_tokenizer(str(target), trust_remote_code=False, local_files_only=True)
        raw = Tokenizer.from_file(str(original / 'tokenizer.json'))
        prompts = read_rows(ROOT / 'inputs' / (model['tag'] + '.jsonl'))
        mismatches = []
        for row in prompts:
            ids = row['prompt_token_ids']
            encoded = tokenizer.encode(row['rendered_prompt'], add_special_tokens=False)
            raw_ids = raw.encode(row['rendered_prompt'], add_special_tokens=False).ids
            decoded = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            raw_decoded = raw.decode(ids, skip_special_tokens=False)
            if list(encoded) != ids or raw_ids != ids or decoded != raw_decoded:
                mismatches.append(row['key'])
        assert not mismatches, (model['tag'], mismatches[:3], len(mismatches))
        results[model['tag']] = {'status': 'passed', 'prompts': len(prompts), 'tokenizer_path': str(target), 'runtime_class': type(tokenizer).__name__, 'original_declared_class': original_class, 'native_json_encoding_matches': True, 'native_json_decoding_matches': True, 'prepared_input_ids_unchanged': True}
        print(json.dumps({model['tag']: results[model['tag']]}), flush=True)
    write_json(ROOT / 'tokenizer-validation.json', {'status': 'passed', 'models': results, 'versions': {p: importlib.metadata.version(p) for p in ['vllm', 'transformers', 'tokenizers']}})


if __name__ == '__main__':
    main()
