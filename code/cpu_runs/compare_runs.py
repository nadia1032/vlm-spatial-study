"""Combine actual result metrics after checking dataset and selected-image equality."""
import argparse
import csv
import json
from pathlib import Path
import statistics

from common import read_jsonl


def compare(analyses, out):
    out = Path(out)
    if out.exists():
        raise ValueError('Comparison file already exists. Choose a new --out.')
    rows, reference = [], None
    for directory in analyses:
        directory = Path(directory)
        run = json.loads((directory.parent/'predictions.run.json').read_text(encoding='utf-8'))
        config = run['config']
        signature = (config['dataset_sha256'], tuple(sorted(config['selected_image_ids'])))
        if reference is not None and signature != reference:
            raise ValueError('Runs use different datasets or image selections; compare matching runs only.')
        reference = signature
        coverage = json.loads((directory/'coverage.json').read_text(encoding='utf-8'))
        if coverage['missing_selected_images'] or coverage['evaluated_images'] != len(config['selected_image_ids']):
            raise ValueError('Comparison requires complete predictions for the selected images.')
        with (directory/'metrics.csv').open(newline='', encoding='utf-8') as stream:
            metrics = {r['metric']: r for r in csv.DictReader(stream)}
        predictions = read_jsonl(directory.parent/'predictions.jsonl')
        times = [float(r['seconds']) for r in predictions]
        row = dict(model=config['model'], commit=run['resolved_revision'],
                   images=coverage['evaluated_images'], scene_groups=coverage['evaluated_scene_groups'],
                   device=config['device'], dtype=config['dtype'], threads=config['threads'],
                   prompt_policy=config.get('prompt_policy', 'legacy_smol_prompt'),
                   mean_seconds_per_image=statistics.mean(times),
                   mean_seconds_excluding_first=statistics.mean(times[1:]) if len(times)>1 else '',
                   total_image_seconds=sum(times), analysis_folder=str(directory.resolve()))
        for name in ('image_accuracy', 'spatial_pair_accuracy', 'invariant_pair_accuracy',
                     'spatial_sensitivity', 'prediction_invariance', 'valid_answer_rate'):
            row[name] = metrics[name]['value']
            row[name+'_n'] = metrics[name]['n']
        rows.append(row)
    if not rows:
        raise ValueError('No analyses supplied')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Comparison saved: {out}')
    print('Check device, dtype, threads and prompt-policy columns before interpreting differences.')
    print('Latency comparisons require matching hardware. BLIP is VQA-specific; chat models use their native templates.')
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analyses', nargs='+', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    compare(args.analyses, args.out)
