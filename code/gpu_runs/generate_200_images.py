"""Generate exactly 200 images in 40 complete, label-balanced scene groups.

Open in Spyder and press F5. No model download, GPU or training is needed.
Companion files: generate_dataset.py and common.py.
"""
import argparse
from collections import Counter, defaultdict
import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile

from common import RELATIONS, VARIANTS, dataset_hash, read_jsonl, validate_dataset, write_jsonl
from generate_dataset import contact_sheet, generate

# Settings for Spyder / VS Code. Use a new folder to preserve prior datasets.
OUTPUT_DIR = Path(__file__).resolve().parent / 'data' / 'final_200'
SEED = 2026
# 200 images / 5 variants = 40 groups. Forty cannot divide equally by six.
# Even group counts preserve equal Yes/No labels within every relation.
GROUPS_PER_RELATION = dict(left=8, right=8, above=6, below=6, inside=6, outside=6)


def generate_200(out=OUTPUT_DIR, seed=SEED):
    out = Path(out).expanduser().resolve()
    if out.exists():
        raise FileExistsError(f'Output already exists; use a new folder: {out}')
    out.parent.mkdir(parents=True, exist_ok=True)
    # Build in a temporary directory, then publish only a validated complete set.
    with tempfile.TemporaryDirectory(prefix='build_final200_', dir=out.parent) as temporary:
        temporary = Path(temporary)
        pool, built = temporary/'pool', temporary/'dataset'
        with contextlib.redirect_stdout(io.StringIO()):
            generate(pool, per_relation=8, seed=seed)
        candidates = read_jsonl(pool/'manifest.jsonl')
        by_relation = defaultdict(list)
        seen = set()
        for row in candidates:
            if row['group_id'] not in seen:
                by_relation[row['relation']].append(row['group_id'])
                seen.add(row['group_id'])
        selected = {group for relation in RELATIONS
                    for group in by_relation[relation][:GROUPS_PER_RELATION[relation]]}
        rows = [r for r in candidates if r['group_id'] in selected]
        pairs = [p for p in read_jsonl(pool/'pairs.jsonl') if p['group_id'] in selected]
        (built/'images').mkdir(parents=True)
        for row in rows:
            shutil.copyfile(pool/row['image_path'], built/row['image_path'])
        validate_dataset(rows, pairs, built)
        assert len(rows)==200 and len(pairs)==160 and len(selected)==40
        assert Counter(r['answer'] for r in rows)=={'yes':100,'no':100}
        assert Counter(r['variant'] for r in rows)=={v:40 for v in VARIANTS}
        for relation in RELATIONS:
            subset = [r for r in rows if r['relation']==relation]
            assert Counter(r['answer'] for r in subset)=={'yes':len(subset)//2,'no':len(subset)//2}
        write_jsonl(built/'manifest.jsonl', rows)
        write_jsonl(built/'pairs.jsonl', pairs)
        summary = dict(seed=seed, images=200, scene_groups=40, pairs=160,
                       spatial_pairs=40, invariant_pairs=120,
                       groups_per_relation=GROUPS_PER_RELATION,
                       images_per_relation=dict(Counter(r['relation'] for r in rows)),
                       images_per_variant=dict(Counter(r['variant'] for r in rows)),
                       labels=dict(Counter(r['answer'] for r in rows)),
                       selection='Fixed even group quotas; independent of model performance.',
                       note='Exact 200-image set. Relations have 40 or 30 images; labels are balanced within each relation. Review images before evaluation.')
        (built/'dataset_info.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
        contact_sheet(rows, built)
        fingerprint = dataset_hash(built/'manifest.jsonl', built/'pairs.jsonl', rows)
        (built/'dataset.sha256').write_text(fingerprint+'\n', encoding='utf-8')
        # Renaming a directory fails rather than replacing an existing nonempty set.
        if out.exists():
            raise FileExistsError(f'Output appeared during generation: {out}')
        built.rename(out)
    print(json.dumps(summary, indent=2))
    print(f'\nDATASET FOLDER: {out}')
    print(f'Dataset SHA256: {fingerprint}')
    print('Review contact_sheet.png and the images, then share this exact folder with your teammate.')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()
    generate_200(args.out, args.seed)


if __name__=='__main__':
    main()
