"""Generate balanced, geometry-checked scene groups without a VLM or internet."""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFont

from common import RELATIONS, VARIANTS, validate_dataset, write_jsonl

SIZE = 320
COLORS = {'red': '#d53832', 'blue': '#225eb0', 'green': '#278045',
          'purple': '#814ba3', 'orange': '#e08820'}
SHAPES = ('circle', 'square', 'triangle')
PHRASES = {'left': 'to the left of', 'right': 'to the right of',
           'above': 'above', 'below': 'below', 'inside': 'inside', 'outside': 'outside'}


def bounds(center, radius):
    x, y = center
    return [x-radius, y-radius, x+radius, y+radius]


def geometric_answer(relation, subject, reference):
    """Use full bounding boxes, with a positive gap; reject boundary ambiguity."""
    sx0, sy0, sx1, sy1 = subject
    rx0, ry0, rx1, ry1 = reference
    if relation == 'left':
        return sx1 < rx0
    if relation == 'right':
        return sx0 > rx1
    if relation == 'above':
        return sy1 < ry0
    if relation == 'below':
        return sy0 > ry1
    inside = sx0 > rx0 and sy0 > ry0 and sx1 < rx1 and sy1 < ry1
    outside = sx1 < rx0 or sx0 > rx1 or sy1 < ry0 or sy0 > ry1
    if not inside and not outside:
        raise ValueError('Containment boundary is ambiguous')
    return inside if relation == 'inside' else outside


def make_geometry(relation, answer, rng, radius):
    jitter = lambda: rng.randint(-9, 9)
    if relation in ('left', 'right'):
        low, high = (78+jitter(), 160+jitter()), (242+jitter(), 160+jitter())
        subject, reference = (low, high) if ((relation == 'left') == answer) else (high, low)
        moved_subject, moved_reference = reference, subject
        return bounds(subject, radius), bounds(reference, radius+3), bounds(moved_subject, radius), bounds(moved_reference, radius+3)
    if relation in ('above', 'below'):
        low, high = (160+jitter(), 78+jitter()), (160+jitter(), 242+jitter())
        subject, reference = (low, high) if ((relation == 'above') == answer) else (high, low)
        return bounds(subject, radius), bounds(reference, radius+3), bounds(reference, radius), bounds(subject, radius+3)
    # A single large outlined square serves as the reference container.
    container_left = rng.choice([True, False])
    reference = bounds(((111 if container_left else 209)+jitter(), 160+jitter()), 69)
    inside_center = ((reference[0]+reference[2])//2+jitter(), (reference[1]+reference[3])//2+jitter())
    outside_center = ((263 if container_left else 57)+jitter(), 160+jitter())
    inside_first = ((relation == 'inside') == answer)
    base, moved = (inside_center, outside_center) if inside_first else (outside_center, inside_center)
    return bounds(base, radius), reference, bounds(moved, radius), reference


def intersects(a, b, margin=0):
    return not (a[2]+margin < b[0] or b[2]+margin < a[0] or a[3]+margin < b[1] or b[3]+margin < a[1])


def draw_shape(draw, box, shape, color, outline=False):
    if outline:
        draw.rectangle(box, outline=color, width=15)
    elif shape == 'circle':
        draw.ellipse(box, fill=color)
    elif shape == 'square':
        draw.rectangle(box, fill=color)
    else:
        x0, y0, x1, y1 = box
        draw.polygon([((x0+x1)/2, y0), (x1, y1), (x0, y1)], fill=color)


def render(subject, reference, subject_shape, reference_shape, color_a, color_b,
           containment, rng, variant):
    # Supersampling keeps shape edges clean. No captions or answer text are rendered.
    scale = 3
    image = Image.new('RGB', (SIZE*scale, SIZE*scale), '#f2f2ee')
    draw = ImageDraw.Draw(image)
    scaled = lambda box: [int(v*scale) for v in box]
    details = {}
    if variant == 'clutter':
        clutter = []
        for _ in range(500):
            x, y = rng.randint(12, SIZE-25), rng.randint(12, SIZE-25)
            box = [x, y, x+rng.randint(7, 17), y+rng.randint(7, 17)]
            if not intersects(box, subject, 13) and not intersects(box, reference, 13):
                draw.ellipse(scaled(box), fill=rng.choice(['#c7c7c1', '#b7b7b2', '#d0d0c7']))
                clutter.append(box)
                if len(clutter) == 18:
                    break
        details['clutter_bboxes'] = clutter
    # Draw reference first; the subject stays visible inside a container.
    draw_shape(draw, scaled(reference), reference_shape, COLORS[color_b], containment)
    draw_shape(draw, scaled(subject), subject_shape, COLORS[color_a])
    if variant == 'occlusion':
        # Cover a narrow side strip, preserving the center and relation evidence.
        x0, y0, x1, y1 = subject
        width = max(3, round((x1-x0)*0.15))
        left_side = rng.choice([True, False])
        box = [x0-2, y0-2, x0+width, y1+2] if left_side else [x1-width, y0-2, x1+2, y1+2]
        draw.rectangle(scaled(box), fill='#858580')
        details['occluder_bbox'] = box
        details['occluder_bbox_width_fraction'] = 0.15
        details['manual_visibility_review_required'] = True
    image = image.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    if variant == 'brightness':
        factor = rng.choice([0.75, 1.20])
        image = ImageEnhance.Brightness(image).enhance(factor)
        details['brightness_factor'] = factor
    return image, details


def contact_sheet(rows, out):
    selected = []
    for relation in RELATIONS:
        group = next(row['group_id'] for row in rows if row['relation'] == relation)
        selected.extend(row for row in rows if row['group_id'] == group)
    width, cell_h, thumb = 260, 325, 246
    canvas = Image.new('RGB', (width*5, cell_h*6+48), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=15)
    draw.text((12, 10), 'Generated examples and ground truth: these are not model predictions', font=font, fill='black')
    for i, row in enumerate(selected):
        x, y = (i % 5)*width, (i//5)*cell_h+48
        with Image.open(out / row['image_path']) as source:
            canvas.paste(source.resize((thumb, thumb)), (x+7, y+30))
        draw.text((x+8, y+6), f'{row["relation"]} | {row["variant"]} | {row["answer"]}', font=font, fill='black')
        # Wrap using actual text bounds, not a fixed character estimate.
        lines, line = [], ''
        for word in row['question'].split():
            candidate = f'{line} {word}'.strip()
            if draw.textlength(candidate, font=font) > width-16:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
        for j, line in enumerate(lines):
            draw.text((x+8, y+279+j*16), line, font=font, fill='black')
    canvas.save(out / 'contact_sheet.png')


def generate(out, per_relation=4, seed=42):
    out = Path(out)
    if per_relation < 2 or per_relation % 2:
        raise ValueError('--per-relation must be an even integer >=2 for balanced labels')
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'{out} is not empty. Use a new output folder to preserve existing data.')
    (out / 'images').mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rows, pairs = [], []
    for relation in RELATIONS:
        for index in range(per_relation):
            group = f'syn_s{seed}_{relation}_{index:04d}'
            base_yes = index % 2 == 0
            color_a, color_b = rng.sample(list(COLORS), 2)
            shape_a, shape_b = rng.sample(SHAPES, 2)
            containment = relation in ('inside', 'outside')
            if containment:
                shape_a, shape_b = rng.choice(('circle', 'triangle')), 'square'
            radius = rng.randint(22, 29)
            subject, reference, moved_subject, moved_reference = make_geometry(relation, base_yes, rng, radius)
            ref_description = f'{color_b} outlined square' if containment else f'{color_b} {shape_b}'
            question = f'Is the {color_a} {shape_a} {PHRASES[relation]} the {ref_description}?'
            for variant in VARIANTS:
                a, b = (moved_subject, moved_reference) if variant == 'spatial' else (subject, reference)
                answer = 'yes' if geometric_answer(relation, a, b) else 'no'
                expected = not base_yes if variant == 'spatial' else base_yes
                if (answer == 'yes') != expected:
                    raise AssertionError('Geometry does not match expected label')
                image, transform = render(a, b, shape_a, shape_b, color_a, color_b, containment, rng, variant)
                image_id = f'{group}_{variant}'
                image_path = f'images/{image_id}.png'
                image.save(out / image_path)
                row = dict(image_id=image_id, group_id=group, domain='synthetic', relation=relation,
                           variant=variant, image_path=image_path, question=question, answer=answer,
                           subject_bbox=a, reference_bbox=b, subject_shape=shape_a,
                           reference_shape=shape_b, subject_color=color_a, reference_color=color_b,
                           transform_details=transform)
                rows.append(row)
                if variant != 'base':
                    pairs.append(dict(pair_id=f'{group}_{variant}', group_id=group,
                                      base_id=f'{group}_base', modified_id=image_id, question=question,
                                      relation=relation, domain='synthetic', transform=variant,
                                      expected_change=variant == 'spatial'))
    validate_dataset(rows, pairs, out)
    write_jsonl(out / 'manifest.jsonl', rows)
    write_jsonl(out / 'pairs.jsonl', pairs)
    summary = dict(seed=seed, per_relation=per_relation, groups=6*per_relation, images=len(rows),
                   pairs=len(pairs), labels=dict(Counter(r['answer'] for r in rows)),
                   notes='All labels computed from full object bounding boxes. Review images before evaluation.')
    (out / 'dataset_info.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    contact_sheet(rows, out)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('data/pilot'))
    parser.add_argument('--per-relation', type=int, default=4, help='Even number of independent scene groups per relation')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    generate(args.out, args.per_relation, args.seed)


if __name__ == '__main__':
    main()
