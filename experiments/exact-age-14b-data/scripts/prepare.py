import argparse
import hashlib
import json
import logging
import re
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasketch import MinHash, MinHashLSH
from math_verify import LatexExtractionConfig, parse, verify
from rapidfuzz.fuzz import ratio


RULES = {'normalization': 'NFKC, layout whitespace, leading question numbering, math delimiters and sizing commands; preserve mathematical numbers and operators', 'near_duplicate': {'minhash_permutations': 128, 'minhash_seed': 42, 'candidate_jaccard': 0.8, 'confirmed_jaccard': 0.9, 'confirmed_character_similarity': 0.95, 'require_identical_numeric_operator_signature': True}, 'benchmark_screen': {'shingle_width': 5, 'jaccard': 0.65, 'character_similarity': 0.9, 'containment': 0.95, 'minimum_length_ratio_for_containment': 0.5, 'reviewed_pair_exceptions': 'configs/benchmark-reviewed-pairs.json', 'policy': 'Conservative lexical overlap exclusion with inspected distinct-parameter exceptions'}, 'answer_conflicts': 'Quarantine the entire duplicate cluster when ordered reference answers cannot be established equivalent; cross-source conflicts affect only merged output', 'verifiability': 'Reference-parser compatibility and numeric-answer coverage, not independently proven label correctness or measured model-output grading accuracy', 'split': 'Unsplit cleaned pools; benchmark sets are separate. Make group-aware train/validation splits before training.'}


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def normalize(text):
    text = unicodedata.normalize('NFKC', text).replace('\u200b', '').replace('\ufeff', '')
    text = re.sub(r'^\s*(?:Problem\s+|Question\s+)?\d{1,3}[.)]\s+', '', text, flags=re.I)
    text = text.replace('−', '-').replace('–', '-').replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
    text = re.sub(r'\\(?:left|right|displaystyle)\b', '', text)
    text = re.sub(r'\\[,;!]|\$', '', text)
    return ' '.join(re.findall(r'\\[A-Za-z]+|\w+|[^\w\s]', text)).rstrip(' .?')


def tokens(text):
    return re.findall(r'\\[A-Za-z]+|\w+|[^\w\s]', text)


def shingles(text):
    ts = tokens(text)
    return {' '.join(ts[i:i + 5]) for i in range(max(1, len(ts) - 4))} if ts else set()


def signature(text):
    return re.findall(r'\d+(?:\.\d+)?|[+\-*/=<>^]', text)


def near_duplicate(left, right, a=None, b=None):
    if left == right:
        return True
    if min(len(left), len(right)) < 80 or signature(left) != signature(right):
        return False
    a = a if a is not None else shingles(left)
    b = b if b is not None else shingles(right)
    jaccard = len(a & b) / max(1, len(a | b))
    return jaccard >= 0.9 and ratio(left, right) >= 95


def boxed(text):
    start = text.rfind('\\boxed')
    if start < 0:
        return None
    start = text.find('{', start)
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[start + 1:i].strip()
    return None


def answer_values(value, decode_json=False):
    if decode_json and isinstance(value, str):
        try:
            decoded = json.loads(value)
            if isinstance(decoded, list):
                value = decoded
        except (ValueError, TypeError):
            pass
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    return [str(x).strip() for x in values if x is not None and str(x).strip()]


@lru_cache(maxsize=100000)
def parsed_answer(answer):
    answer = answer.strip().strip('$')
    if '\\boxed' not in answer:
        answer = '\\boxed{' + answer + '}'
    try:
        result = parse(answer, extraction_config=[LatexExtractionConfig()], fallback_mode='no_fallback', parsing_timeout=2)
        return [x for x in result if not isinstance(x, str)]
    except Exception:
        return []


def equivalent_answers(left, right):
    if len(left) != len(right):
        return False
    for a, b in zip(left, right):
        if normalize(a) == normalize(b):
            continue
        pa_, pb_ = parsed_answer(a), parsed_answer(b)
        if not pa_ or not pb_:
            return False
        try:
            if not (verify(pa_, pb_, timeout_seconds=2) and verify(pb_, pa_, timeout_seconds=2)):
                return False
        except Exception:
            return False
    return True


def rows_at(folder):
    for path in sorted(folder.rglob('*')):
        if '.cache' in path.parts:
            continue
        if path.suffix == '.parquet':
            for batch in pq.ParquetFile(path).iter_batches(batch_size=2048):
                for row in batch.to_pylist():
                    yield str(path.relative_to(folder)), row
        elif path.suffix == '.jsonl':
            for line in path.read_text().splitlines():
                if line.strip():
                    yield str(path.relative_to(folder)), json.loads(line)
        elif path.suffix == '.json' and path.name == 'deepscaler.json':
            for row in json.loads(path.read_text()):
                yield str(path.relative_to(folder)), row


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def benchmarks(root, sources):
    result = []
    counts = Counter()
    for source in sources:
        key = source['key']
        if source['repo_type'] != 'dataset' or key in ['skywork', 'deepscaler']:
            continue
        for i, (file, row) in enumerate(rows_at(root / 'data/raw' / key)):
            question = row.get('problem') or row.get('question') or row.get('Question')
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f'Missing benchmark prompt: {key}/{file}/{i}')
            answers = answer_values(row.get('answer', row.get('final_answer')))
            if not answers and isinstance(row.get('solution'), str):
                answers = answer_values(boxed(row['solution']))
            result.append({'id': f'{key}:{i}', 'benchmark': key, 'prompt': question.strip(), 'answers': answers, 'source_file': file, 'source_revision': source['revision']})
            counts[key] += 1
    path = root / 'data/benchmarks/gaokao-2023en-test.jsonl'
    for i, line in enumerate(path.read_text().splitlines()):
        row = json.loads(line)
        result.append({'id': f'gaokao:{i}', 'benchmark': 'gaokao', 'prompt': row['question'].strip(), 'answers': answer_values(row['answer']), 'source_file': path.name, 'source_revision': json.loads((root / 'configs/gaokao-source.json').read_text())['revision']})
        counts['gaokao'] += 1
    if set(counts) != {'math500','aime24','aime25','aime26','amc23','amc24','minerva_math','olympiadbench','gaokao'}:
        raise ValueError(f'Incomplete benchmark coverage: {counts}')
    write_jsonl(root / 'data/benchmarks/all.jsonl', result)
    return result, dict(counts)


def prompt_leakage_marker(text):
    return re.search(r'\b(?:Answer|Answers|Solution|Solutions)\s*[:.]\s*\S|\[hide=(?:[Ss]olution|[Aa]nswer)[^\]]*\]|<summary>\s*(?:Solution|Answer)\s*</summary>', text)


class BenchmarkIndex:
    def __init__(self, rows):
        self.rows = rows
        self.normalized = [normalize(r['prompt']).lower() for r in rows]
        self.sets = [shingles(x) for x in self.normalized]
        review_path = Path(__file__).resolve().parents[1] / 'configs/benchmark-reviewed-pairs.json'
        reviews = json.loads(review_path.read_text()) if review_path.exists() else []
        self.distinct_pairs = {(r['normalized_prompt_sha256'], r['benchmark_id']) for r in reviews if r['decision'] == 'distinct_variant'}
        self.exact = defaultdict(list)
        self.index = defaultdict(list)
        for i, (text, values) in enumerate(zip(self.normalized, self.sets)):
            self.exact[text].append(i)
            for value in values:
                self.index[value].append(i)

    @lru_cache(maxsize=180000)
    def match(self, prompt):
        text = normalize(prompt).lower()
        if text in self.exact:
            return [{'benchmark_id': self.rows[i]['id'], 'method': 'normalized_exact', 'score': 1.0} for i in self.exact[text]]
        a = shingles(text)
        counts = Counter(i for value in a for i in self.index.get(value, []))
        hits = []
        for i, intersection in counts.items():
            if (sha(text), self.rows[i]['id']) in self.distinct_pairs:
                continue
            b = self.sets[i]
            if min(len(a), len(b)) < 8:
                continue
            containment = intersection / min(len(a), len(b))
            if containment < 0.4:
                continue
            jaccard = intersection / (len(a) + len(b) - intersection)
            length_ratio = min(len(text), len(self.normalized[i])) / max(len(text), len(self.normalized[i]))
            similarity = ratio(text, self.normalized[i]) / 100 if jaccard >= 0.25 else 0
            if jaccard >= 0.65 or similarity >= 0.9 or (containment >= 0.95 and length_ratio >= 0.5):
                hits.append({'benchmark_id': self.rows[i]['id'], 'method': 'near_overlap', 'jaccard': jaccard, 'character_similarity': similarity, 'containment': containment})
        return sorted(hits, key=lambda x: x['benchmark_id'])


def load_source(root, key, quarantine, filter_prompt_leakage=True):
    result = []
    count = Counter()
    profiles = Counter()
    for i, (file, row) in enumerate(rows_at(root / 'data/raw' / key)):
        count['raw_rows'] += 1
        if key == 'skywork' and row.get('ability') != 'math':
            count['non_math_rows'] += 1
            continue
        count['math_rows'] += 1
        messages = row.get('prompt') if key == 'skywork' else [{'role': 'user', 'content': row.get('problem')}]
        prompt = messages[0].get('content') if isinstance(messages, list) and len(messages) == 1 and messages[0].get('role') == 'user' else None
        raw_answer = row.get('reward_model', {}).get('ground_truth') if key == 'skywork' else row.get('answer')
        answers = answer_values(raw_answer, decode_json=key == 'skywork')
        count['rows_with_reference_answer'] += bool(answers)
        source_id = f'{key}:{file}:{i}'
        reason = None
        if not isinstance(prompt, str) or not prompt.strip():
            reason = 'missing_prompt_or_unsupported_message_structure'
        elif not answers:
            reason = 'missing_reference_answer'
        elif '\ufffd' in prompt or '\x00' in prompt:
            reason = 'invalid_text_encoding'
        elif filter_prompt_leakage and prompt_leakage_marker(prompt):
            reason = 'suspected_answer_or_solution_section_in_prompt'
        elif re.search(r'<image>|!\[[^\]]*\]\(|\\includegraphics|\[asy\]|<img\b', prompt, re.I):
            reason = 'requires_image_or_diagram_rendering'
        if reason:
            count[reason] += 1
            quarantine.append({'source_id': source_id, 'reason': reason, 'source_file': file, 'prompt': prompt, 'reference_answer': raw_answer})
            continue
        prompt = prompt.strip()
        source = row.get('data_source', 'deepscaler')
        profiles[source] += 1
        result.append({'id': sha(key + '\n' + normalize(prompt)), 'prompt': prompt, 'answers': answers, 'reference_answer_raw': raw_answer, 'datasets': [key], 'source': source, 'source_records': [{'dataset': key, 'row_id': source_id, 'file': file, 'extra_info': row.get('extra_info')}], '_members': [{'source_id': source_id, 'prompt': prompt, 'answers': answers}]})
    count['valid_before_deduplication'] = len(result)
    return result, dict(count), dict(profiles)


def deduplicate(rows, label, receipts, quarantine):
    normalized = [normalize(r['prompt']) for r in rows]
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)
    exact = {}
    index = MinHashLSH(threshold=0.8, num_perm=128)
    for i, text in enumerate(normalized):
        if text in exact:
            previous = exact[text]
            union(i, previous)
            receipts.append({'version': label, 'row_id': rows[i]['id'], 'representative_id': rows[previous]['id'], 'source_records': rows[i]['source_records'], 'method': 'normalized_exact'})
            continue
        exact[text] = i
        values = shingles(text)
        fingerprint = MinHash(num_perm=128, seed=42)
        fingerprint.update_batch([x.encode() for x in sorted(values)])
        for candidate in sorted(map(int, index.query(fingerprint))):
            if near_duplicate(text, normalized[candidate], values):
                union(i, candidate)
                receipts.append({'version': label, 'row_id': rows[i]['id'], 'representative_id': rows[candidate]['id'], 'source_records': rows[i]['source_records'], 'method': 'high_similarity_same_numeric_operator_signature'})
        index.insert(str(i), fingerprint)
        if i % 5000 == 0:
            print(json.dumps({'stage': 'deduplicate', 'version': label, 'row': i, 'total': len(rows)}), flush=True)
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[find(i)].append(i)
    result = []
    conflicts = 0
    for indices in groups.values():
        first = rows[indices[0]]
        if any(not equivalent_answers(first['answers'], rows[i]['answers']) for i in indices[1:]):
            conflicts += len(indices)
            quarantine.append({'version': label, 'reason': 'conflicting_or_unresolved_reference_answers', 'rows': [{k: v for k, v in rows[i].items() if not k.startswith('_')} for i in indices]})
            continue
        item = dict(first)
        item['datasets'] = sorted({d for i in indices for d in rows[i]['datasets']})
        item['source_records'] = [s for i in indices for s in rows[i]['source_records']]
        item['_members'] = [m for i in indices for m in rows[i]['_members']]
        item['id'] = sha(normalize(item['prompt']))
        result.append(item)
    return result, {'input_rows': len(rows), 'duplicate_clusters': sum(len(v) > 1 for v in groups.values()), 'redundant_rows': len(rows) - len(groups), 'answer_conflict_rows_quarantined': conflicts, 'unique_rows_after_conflicts': len(result)}


def verifier_profile(row):
    parsed = [bool(parsed_answer(a)) for a in row['answers']]
    numeric = all(re.fullmatch(r'[+-]?(?:\d+(?:\.\d+)?|\d+/\d+)', a.strip()) is not None for a in row['answers'])
    return {'reference_present': bool(row['answers']), 'all_reference_components_parse': all(parsed), 'any_reference_component_parses': any(parsed), 'plain_numeric_reference': numeric, 'reference_components': len(parsed), 'label_correctness_independently_verified': False}


def export(root, name, rows):
    records = []
    coverage = Counter()
    sources = Counter()
    for i, row in enumerate(rows):
        item = {k: v for k, v in row.items() if not k.startswith('_')}
        if 'verifiability' not in item:
            item['verifiability'] = verifier_profile(item)
        item['messages'] = [{'role': 'user', 'content': item['prompt']}]
        records.append(item)
        for key, value in item['verifiability'].items():
            if isinstance(value, bool):
                coverage[key] += value
        sources[item['source']] += 1
        if i % 5000 == 0:
            print(json.dumps({'stage': 'verifier_profile', 'version': name, 'row': i, 'total': len(rows)}), flush=True)
    folder = root / 'data/processed' / name
    checksum = write_jsonl(folder / 'clean.jsonl', records)
    table_rows = [{**r, 'reference_answer_raw': json.dumps(r['reference_answer_raw'], ensure_ascii=False), 'source_records': json.dumps(r['source_records'], ensure_ascii=False)} for r in records]
    pq.write_table(pa.Table.from_pylist(table_rows), folder / 'clean.parquet', compression='zstd')
    compatible = [r for r in records if r['verifiability']['all_reference_components_parse']]
    write_jsonl(folder / 'reference_parseable.jsonl', compatible)
    write_jsonl(root / 'reports' / f'{name}-unparseable-references.jsonl', [r for r in records if not r['verifiability']['all_reference_components_parse']])
    return {'clean_rows': len(rows), 'reference_parseable_rows': len(compatible), 'verifiability_counts': dict(coverage), 'verifiability_percent': {k: round(100 * v / len(rows), 4) for k, v in coverage.items()} if rows else {}, 'source_counts': dict(sources), 'sha256_clean_jsonl': checksum}


def main():
    logging.getLogger('math_verify').setLevel(logging.CRITICAL)
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    sources = json.loads((root / 'configs/sources.lock.json').read_text())
    evaluation, benchmark_counts = benchmarks(root, sources)
    index = BenchmarkIndex(evaluation)
    report = {'rules': RULES, 'benchmark_counts': benchmark_counts, 'versions': {}, 'sources_lock_sha256': hashlib.sha256((root / 'configs/sources.lock.json').read_bytes()).hexdigest()}
    cleaned = {}
    for name in ['skywork', 'deepscaler']:
        quarantine, receipts, contamination = [], [], []
        rows, raw_profile, source_profile = load_source(root, name, quarantine, filter_prompt_leakage=False)
        unique, dedup_profile = deduplicate(rows, name, receipts, quarantine)
        kept, leakage = [], []
        for i, row in enumerate(unique):
            flagged = [m['source_id'] for m in row['_members'] if prompt_leakage_marker(m['prompt'])]
            if flagged:
                leakage.append({'reason': 'suspected_answer_or_solution_section_in_prompt', 'row': {k: v for k, v in row.items() if not k.startswith('_')}, 'flagged_source_ids': flagged})
                continue
            hits = []
            for member in row['_members']:
                matched = index.match(member['prompt'])
                if matched:
                    hits.append({'source_id': member['source_id'], 'matches': matched})
            if hits:
                contamination.append({'id': row['id'], 'prompt': row['prompt'], 'source_records': row['source_records'], 'hits': hits})
            else:
                kept.append(row)
            if i % 10000 == 0:
                print(json.dumps({'stage': 'benchmark_screen', 'version': name, 'row': i, 'total': len(unique)}), flush=True)
        cleaned[name] = kept
        report['versions'][name] = {'raw_profile': raw_profile, 'raw_valid_source_counts': source_profile, 'deduplication': dedup_profile, 'benchmark_overlap_unique_rows_removed': len(contamination), 'suspected_prompt_leakage_unique_rows_removed': len(leakage), **export(root, name, kept)}
        write_jsonl(root / 'data/quarantine' / f'{name}-prompt-leakage.jsonl', leakage)
        write_jsonl(root / 'reports' / f'{name}-duplicate-receipts.jsonl', receipts)
        write_jsonl(root / 'data/quarantine' / f'{name}.jsonl', quarantine)
        write_jsonl(root / 'reports' / f'{name}-benchmark-overlap.jsonl', contamination)
        (root / 'reports/preparation-progress.json').write_text(json.dumps(report, indent=2) + '\n')
    quarantine, receipts = [], []
    merged, profile = deduplicate(cleaned['skywork'] + cleaned['deepscaler'], 'merged', receipts, quarantine)
    report['versions']['merged'] = {'deduplication': profile, 'rows_with_both_dataset_memberships': sum(len(r['datasets']) == 2 for r in merged), **export(root, 'merged', merged)}
    write_jsonl(root / 'reports/merged-duplicate-receipts.jsonl', receipts)
    write_jsonl(root / 'data/quarantine/merged.jsonl', quarantine)
    for name, rows in {**cleaned, 'merged': merged}.items():
        keys = [normalize(r['prompt']) for r in rows]
        assert len(keys) == len(set(keys)), name
        assert all(not index.match(r['prompt']) for r in rows), name
        assert all(r['answers'] for r in rows), name
    report['validation'] = {'unique_normalized_prompts': True, 'no_detected_benchmark_overlap_under_declared_rules': True, 'all_clean_rows_have_reference_answers': True, 'standalone_versions_processed_independently': True, 'semantic_paraphrase_contamination_exhaustively_excluded': False, 'training_launched': False}
    (root / 'reports/preparation-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': 'complete', 'counts': {k: v['clean_rows'] for k, v in report['versions'].items()}}), flush=True)


if __name__ == '__main__':
    main()
