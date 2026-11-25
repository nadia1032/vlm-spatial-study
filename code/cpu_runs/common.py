"""Shared validation, answer parsing and deterministic selection. No model imports."""
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

RELATIONS = ('left', 'right', 'above', 'below', 'inside', 'outside')
VARIANTS = ('base', 'spatial', 'brightness', 'clutter', 'occlusion')


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def parse_answer(raw):
    """Accept a leading yes/no, reject conflicting yes/no labels. Never guess."""
    if not isinstance(raw, str):
        return None
    raw = raw.strip().lower()
    first = re.match(r'^(yes|no)\b', raw)
    labels = set(re.findall(r'\b(?:yes|no)\b', raw))
    return first.group(1) if first and len(labels) == 1 else None


def unique_index(rows, key):
    indexed = {}
    for row in rows:
        value = row[key]
        if value in indexed:
            raise ValueError(f'Duplicate {key}: {value}')
        indexed[value] = row
    return indexed


def validate_dataset(manifest, pairs, root, check_images=True):
    root = Path(root).resolve()
    images = unique_index(manifest, 'image_id')
    unique_index(pairs, 'pair_id')
    if not images or not pairs:
        raise ValueError('The dataset must contain images and pairs.')
    for row in manifest:
        if row['answer'] not in ('yes', 'no') or row['relation'] not in RELATIONS:
            raise ValueError(f'Invalid answer or relation: {row["image_id"]}')
        if row['variant'] not in VARIANTS or row['domain'] not in ('synthetic', 'real'):
            raise ValueError(f'Invalid variant or domain: {row["image_id"]}')
        if not row['question'].strip():
            raise ValueError('Empty question')
        path = (root / row['image_path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Image paths must stay inside the dataset folder.')
        if check_images and not path.is_file():
            raise FileNotFoundError(path)
    paired_ids = set()
    for pair in pairs:
        a, b = images[pair['base_id']], images[pair['modified_id']]
        for field in ('question', 'group_id', 'relation', 'domain'):
            if a[field] != b[field] or pair[field] != a[field]:
                raise ValueError(f'Pair {pair["pair_id"]}: inconsistent {field}')
        if a['variant'] != 'base' or b['variant'] != pair['transform']:
            raise ValueError('Incorrect pair variants')
        expected = pair['transform'] == 'spatial'
        if pair['expected_change'] is not expected or (a['answer'] != b['answer']) != expected:
            raise ValueError('Pair labels do not match the transformation')
        paired_ids.update((a['image_id'], b['image_id']))
    if paired_ids != set(images):
        raise ValueError('Every image must belong to at least one pair.')
    return images


def select_groups(manifest, limit=None):
    """Round-robin over relation categories, then retain all variants of a group."""
    by_relation = defaultdict(list)
    seen = set()
    for row in manifest:
        group = row['group_id']
        if group not in seen:
            by_relation[row['relation']].append(group)
            seen.add(group)
    ordered = []
    for i in range(max(map(len, by_relation.values()), default=0)):
        for relation in RELATIONS:
            if i < len(by_relation[relation]):
                ordered.append(by_relation[relation][i])
    if limit is not None and limit < 1:
        raise ValueError('--limit-groups must be positive')
    selected = set(ordered if limit is None else ordered[:limit])
    return [row for row in manifest if row['group_id'] in selected]


def dataset_hash(manifest_path, pairs_path, rows):
    """Hash metadata AND image bytes so stale predictions cannot be resumed."""
    digest = hashlib.sha256()
    for path in (manifest_path, pairs_path):
        digest.update(Path(path).read_bytes())
    for row in rows:
        digest.update(row['image_id'].encode())
        digest.update((Path(manifest_path).parent / row['image_path']).read_bytes())
    return digest.hexdigest()
