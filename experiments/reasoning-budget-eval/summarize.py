import csv
import html
import json

from common import ROOT, digest, validate_manifest, write_json


def main():
    validate_manifest()
    spec = json.loads((ROOT / 'spec.json').read_text())
    rows = []
    completed = []
    for model in spec['models']:
        path = ROOT / 'outputs' / model['tag'] / 'summary.json'
        if not path.exists():
            continue
        summary = json.loads(path.read_text())
        assert summary['manifest_sha256'] == digest(ROOT / 'manifest.json')
        assert summary['scored_sha256'] == digest(path.parent / 'scored.jsonl')
        assert summary['responses'] == 1536
        rows.extend(summary['results'])
        completed.append(model['tag'])
    report = ROOT / 'reports'
    report.mkdir(exist_ok=True)
    complete = len(completed) == len(spec['models']) and not any(r['pending_review'] for r in rows)
    write_json(report / 'results.json', {'complete': complete, 'completed_models': completed, 'expected_responses': 7680, 'scored_responses': 1536 * len(completed), 'results': rows})
    fields = ['model', 'dataset', 'budget', 'n', 'correct', 'accuracy', 'mean_response_tokens', 'median_response_tokens', 'p90_response_tokens', 'budget_hit_rate', 'answer_parse_rate', 'finished_count', 'mean_finished_response_tokens']
    with (report / 'results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Reasoning model budget comparison', '', f'Status: {"complete" if complete else "partial"}; {len(completed)}/5 models scored.', '', '256 fixed random samples from each cleaned dataset, seed 42. One response per question at each output-token cap. Accuracy is final-answer equivalence against dataset references using Math-Verify 0.8.0; unfinished reasoning and unparseable answers are counted as incorrect. These are sampled training-pool results, not held-out benchmark scores.', '', '| Dataset | Model | Budget | Correct | Accuracy | 95% CI | Mean tokens | Hit limit |', '| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |']
    table = []
    for r in rows:
        low, high = r['accuracy_ci95_wilson'] or (0, 1)
        accuracy = f'{r["accuracy"]:.1%}' if r['accuracy'] is not None else f'{r["accuracy_bounds"][0]:.1%}–{r["accuracy_bounds"][1]:.1%} pending review'
        interval = f'{low:.1%}–{high:.1%}' if r['accuracy_ci95_wilson'] else 'Pending review'
        values = [r['dataset'], r['model'], str(r['budget']), f'{r["correct"]}/{r["n"]}', accuracy, interval, f'{r["mean_response_tokens"]:.0f}', f'{r["budget_hit_rate"]:.1%}']
        lines.append('| ' + ' | '.join(values) + ' |')
        table.append('<tr>' + ''.join('<td>' + html.escape(v) + '</td>' for v in values) + '</tr>')
    lines.extend(['', 'Every model uses its native template. Qwen3 thinking is enabled. Phi-4 uses its native built-in system prompt and temperature 0.8/top-k 50; Qwen3 uses temperature 0.6/top-k 20; DeepSeek and both Nemotron models use temperature 0.6 with top-k disabled. Top-p is 0.95 for all. These are model-specific decoding settings, fixed across budgets.', '', 'The 4K/8K/12K caps are 4,096/8,192/12,288 generated tokens and include reasoning and final answer. Prompts are never truncated. Independent decoding calls use the same per-question seed at each budget; identical prefixes are not guaranteed. Each model is evaluated on the same inputs.', '', 'Per-response text, token IDs, finish reason, extracted answer, correctness, seeds, model revisions, input hashes, GPU mapping, and runtime versions are preserved under outputs/. Wilson intervals describe uncertainty within each sampled cell; paired comparisons share the same questions.', ''])
    (report / 'RESULTS.md').write_text('\n'.join(lines))
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Reasoning budget comparison</title><style>body{font:16px system-ui;margin:40px;color:#172130;background:#f6f8fa}main{max-width:1250px;margin:auto}table{border-collapse:collapse;background:white;width:100%}th,td{padding:10px 14px;text-align:right;border-bottom:1px solid #dde3e8}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}th{background:#e7edf4;position:sticky;top:0}p{line-height:1.6}section{overflow:auto}small{color:#42546b}</style><main><h1>Reasoning model budget comparison</h1><p>' + html.escape(f'{len(completed)}/5 models scored · 256 questions per cleaned dataset · 3 response budgets') + '</p><section><table><thead><tr>' + ''.join('<th>' + h + '</th>' for h in ['Dataset', 'Model', 'Budget', 'Correct', 'Accuracy', '95% CI', 'Mean tokens', 'Hit limit']) + '</tr></thead><tbody>' + ''.join(table) + '</tbody></table></section><p><small>Native model templates and model-specific decoding settings. Token caps include reasoning and final answer. Final-answer equivalence uses Math-Verify; unfinished reasoning and unparseable answers count as incorrect. Results describe these sampled training pools. See RESULTS.md for methods.</small></p></main></html>'
    (report / 'dashboard.html').write_text(page)
    print(json.dumps({'complete': complete, 'completed_models': completed, 'cells': len(rows)}), flush=True)


if __name__ == '__main__':
    main()
