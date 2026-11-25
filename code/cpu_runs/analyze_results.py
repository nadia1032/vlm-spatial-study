"""Compute correctness-aware paired metrics and scene-group bootstrap intervals."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import dataset_hash, parse_answer, read_jsonl, unique_index, validate_dataset


def paired_records(manifest, pairs, predictions):
    images = unique_index(manifest, 'image_id')
    predictions = unique_index(predictions, 'image_id')
    records = []
    for pair in pairs:
        if pair['base_id'] not in predictions or pair['modified_id'] not in predictions:
            continue
        a, b = images[pair['base_id']], images[pair['modified_id']]
        pa, pb = parse_answer(predictions[a['image_id']]['raw_answer']), parse_answer(predictions[b['image_id']]['raw_answer'])
        valid = pa is not None and pb is not None
        ac, bc = pa == a['answer'], pb == b['answer']
        changed = valid and pa != pb
        consistent = valid and pa == pb
        records.append(dict(**pair, base_gt=a['answer'], modified_gt=b['answer'],
                            base_prediction=pa, modified_prediction=pb, base_correct=ac,
                            modified_correct=bc, valid_pair=valid, both_correct=ac and bc,
                            changed=changed, consistent=consistent,
                            appropriate_response=changed if pair['expected_change'] else consistent))
    return records


def cluster_interval(records, key, eligible=None, repetitions=1000, seed=42):
    """Resample scene groups, retaining their correlated transformation pairs."""
    records = records if eligible is None else [r for r in records if eligible(r)]
    groups = defaultdict(lambda: [0, 0])
    for row in records:
        groups[row['group_id']][0] += int(row[key])
        groups[row['group_id']][1] += 1
    if not groups:
        return None, None, None, 0
    counts = np.asarray(list(groups.values()), dtype=float)
    estimate = float(counts[:, 0].sum()/counts[:, 1].sum())
    # One cluster cannot support an empirical between-scene uncertainty estimate.
    if len(counts) < 2:
        return estimate, None, None, int(counts[:, 1].sum())
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repetitions):
        sampled = counts[rng.integers(0, len(counts), len(counts))]
        samples.append(sampled[:, 0].sum()/sampled[:, 1].sum())
    low, high = np.quantile(samples, [0.025, 0.975])
    return estimate, float(low), float(high), int(counts[:, 1].sum())


def summarize(records, repetitions=1000):
    changing = [r for r in records if r['expected_change']]
    invariant = [r for r in records if not r['expected_change']]
    specs = [
        ('pair_accuracy', records, 'both_correct', None),
        ('spatial_pair_accuracy', changing, 'both_correct', None),
        ('invariant_pair_accuracy', invariant, 'both_correct', None),
        ('spatial_sensitivity', changing, 'changed', None),
        ('sensitivity_given_correct_base', changing, 'both_correct', lambda r: r['base_correct']),
        ('prediction_invariance', invariant, 'consistent', None),
        ('invariance_given_correct_base', invariant, 'both_correct', lambda r: r['base_correct']),
        ('appropriate_response_rate', records, 'appropriate_response', None),
        ('valid_pair_rate', records, 'valid_pair', None),
    ]
    result = []
    for name, subset, key, eligible in specs:
        estimate, low, high, count = cluster_interval(subset, key, eligible, repetitions)
        result.append(dict(metric=name, value=estimate, ci_low=low, ci_high=high, n=count,
                           scene_groups=len({r['group_id'] for r in subset if eligible is None or eligible(r)})))
    return result


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('data/pilot/manifest.jsonl'))
    parser.add_argument('--pairs', type=Path)
    parser.add_argument('--predictions', type=Path, default=Path('results/pilot/predictions.jsonl'))
    parser.add_argument('--out', type=Path, default=Path('results/pilot/analysis'))
    parser.add_argument('--allow-incomplete', action='store_true', help='Explicitly analyze only completed records; coverage is reported')
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error('--bootstrap must be >=100')
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit('Analysis output folder is not empty. Use a new --out.')
    pairs_path = args.pairs or args.manifest.with_name('pairs.jsonl')
    manifest, pairs = read_jsonl(args.manifest), read_jsonl(pairs_path)
    images = validate_dataset(manifest, pairs, args.manifest.parent)
    predictions = read_jsonl(args.predictions)
    predicted = unique_index(predictions, 'image_id')
    if not predicted or not set(predicted).issubset(images):
        raise SystemExit('No predictions, or unknown image IDs in predictions.')
    selected = set(images)
    run_path = args.predictions.with_suffix('.run.json')
    run = None
    if run_path.exists():
        run = json.loads(run_path.read_text(encoding='utf-8'))
        if dataset_hash(args.manifest, pairs_path, manifest) != run['config']['dataset_sha256']:
            raise SystemExit('Dataset bytes changed since inference. Analysis refused.')
        selected = set(run['config']['selected_image_ids'])
        if not selected.issubset(images) or not set(predicted).issubset(selected):
            raise SystemExit('Run selection does not match dataset/predictions.')
        if any(r.get('run_id') != run['run_id'] for r in predictions):
            raise SystemExit('Mixed or incorrect prediction run IDs.')
    missing = selected-set(predicted)
    if missing and not args.allow_incomplete:
        raise SystemExit(f'{len(missing)} selected images are missing. Resume inference or explicitly use --allow-incomplete.')
    records = paired_records(manifest, pairs, predictions)
    if not records:
        raise SystemExit('No complete pairs to analyze.')
    selected_pairs = [p for p in pairs if p['base_id'] in selected and p['modified_id'] in selected]
    args.out.mkdir(parents=True)
    image_rows = []
    for row in manifest:
        if row['image_id'] not in predicted:
            continue
        raw = predicted[row['image_id']]['raw_answer']
        answer = parse_answer(raw)
        image_rows.append(dict(image_id=row['image_id'], group_id=row['group_id'], domain=row['domain'],
                               relation=row['relation'], variant=row['variant'], ground_truth=row['answer'],
                               prediction=answer, correct=answer == row['answer'], valid=answer is not None,
                               raw_answer=raw))
    overall = summarize(records, args.bootstrap)
    # Per-image accuracy counts each evaluated image once, even when a base is reused in four pairs.
    for name, key in [('image_accuracy', 'correct'), ('valid_answer_rate', 'valid')]:
        value, low, high, count = cluster_interval(image_rows, key, repetitions=args.bootstrap)
        overall.append(dict(metric=name, value=value, ci_low=low, ci_high=high, n=count,
                            scene_groups=len({r['group_id'] for r in image_rows})))
    sliced = []
    for field in ('relation', 'transform', 'domain'):
        for value in sorted({r[field] for r in records}):
            subset = [r for r in records if r[field] == value]
            for metric in summarize(subset, args.bootstrap):
                sliced.append(dict(slice_field=field, slice_value=value, **metric))
    write_csv(args.out / 'metrics.csv', overall)
    write_csv(args.out / 'metrics_by_category.csv', sliced)
    write_csv(args.out / 'pair_details.csv', records)
    write_csv(args.out / 'image_details.csv', image_rows)
    # All raw answers retained for manual parsing audit and error categorization.
    write_csv(args.out / 'failures.csv', [r for r in records if not r['both_correct']])
    coverage = dict(dataset_images=len(images), selected_images=len(selected), evaluated_images=len(predicted),
                    missing_selected_images=len(missing), dataset_pairs=len(pairs), selected_pairs=len(selected_pairs),
                    evaluated_pairs=len(records), dataset_scene_groups=len({r['group_id'] for r in manifest}),
                    evaluated_scene_groups=len({r['group_id'] for r in image_rows}),
                    inference_run=run['run_id'] if run else None,
                    note='Model results only if predictions came from run_vlm.py. Unit-test fixtures are not model evidence.')
    (args.out / 'coverage.json').write_text(json.dumps(coverage, indent=2), encoding='utf-8')
    # Plot correctness-aware metrics, not raw invariance alone.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fields = ('image_accuracy', 'spatial_pair_accuracy', 'invariant_pair_accuracy')
    labels = ('Image accuracy', 'Spatial pair accuracy', 'Invariant pair accuracy')
    lookup = {row['metric']: row for row in overall}
    values = [lookup[field]['value'] for field in fields]
    fig, ax = plt.subplots(figsize=(8, 4.6), layout='constrained')
    ax.bar(labels, [v if v is not None else 0 for v in values], color=['#315d88', '#bd7043', '#47785f'])
    for i, field in enumerate(fields):
        row = lookup[field]
        if row['ci_low'] is not None:
            ax.vlines(i, row['ci_low'], row['ci_high'], color='black', linewidth=1.5)
            ax.hlines([row['ci_low'], row['ci_high']], i-0.06, i+0.06, color='black')
        ax.text(i, min((values[i] or 0)+0.04, 1.07), 'N/A' if values[i] is None else f'{values[i]:.1%}', ha='center')
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Proportion correct')
    interval_note = ('95% bootstrap intervals resampled by scene group'
                     if any(row['ci_low'] is not None for row in overall)
                     else 'One scene group: uncertainty intervals are unavailable')
    ax.set_title('Spatial VQA evaluation\n' + interval_note)
    ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(args.out / 'accuracy.png', dpi=160)
    plt.close(fig)
    print(json.dumps(coverage, indent=2))
    for row in overall:
        value = 'N/A' if row['value'] is None else f'{row["value"]:.3f}'
        print(f'{row["metric"]}: {value} (n={row["n"]})')
    print(f'Analysis saved in {args.out}')


if __name__ == '__main__':
    main()
