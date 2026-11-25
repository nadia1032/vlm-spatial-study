"""Run one registered model with checked resume and analysis-compatible outputs."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

from common import dataset_hash, parse_answer, read_jsonl, select_groups, unique_index, validate_dataset
from model_adapters import ADAPTER_VERSION, MODELS, ModelAdapter, choose_device_dtype


def code_hash():
    digest = hashlib.sha256()
    for name in ('model_adapters.py', 'run_model.py', 'common.py'):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=list(MODELS), required=True)
    parser.add_argument('--manifest', type=Path, default=Path('data/pilot/manifest.jsonl'))
    parser.add_argument('--pairs', type=Path)
    parser.add_argument('--out', type=Path, required=True, help='Predictions JSONL; choose a separate folder per model/protocol')
    parser.add_argument('--groups', type=int, default=1, help='1=5-image check; 12=balanced 60; omit via --groups 0 for all')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--dtype', choices=('auto', 'float32', 'float16', 'bfloat16'), default='auto')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--max-new-tokens', type=int, default=8)
    parser.add_argument('--revision', default='main')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.groups < 0 or args.threads < 1 or args.max_new_tokens < 1:
        parser.error('Groups must be >=0; threads and max-new-tokens must be positive.')
    import torch
    device, dtype_name, dtype = choose_device_dtype(args.device, args.dtype, torch)
    if args.model == 'gemma' and dtype_name == 'float16':
        raise SystemExit('Gemma 3 can be numerically unstable in FP16. Use a BF16-capable GPU '
                         '(for example RTX 4090, L4 or A100) with --dtype bfloat16, '
                         'or CPU float32 with sufficient RAM. A T4 is suitable for the other four models, '
                         'but this reference runner does not claim a validated Gemma FP16/quantized path.')
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    if device == 'cuda':
        torch.cuda.manual_seed_all(0)
    manifest, pairs_path = read_jsonl(args.manifest), args.pairs or args.manifest.with_name('pairs.jsonl')
    pairs = read_jsonl(pairs_path)
    validate_dataset(manifest, pairs, args.manifest.parent)
    selected = select_groups(manifest, args.groups or None)
    spec = MODELS[args.model]
    versions = {name: importlib.metadata.version(name) for name in (
        'torch', 'torchvision', 'transformers', 'Pillow', 'numpy', 'huggingface-hub',
        'tokenizers', 'safetensors', 'num2words', 'accelerate', 'sentencepiece', 'protobuf')}
    config = dict(schema_version=2, adapter_version=ADAPTER_VERSION, code_sha256=code_hash(),
                  model=spec.repo_id, model_key=args.model, family=spec.family,
                  requested_revision=args.revision, device=device, dtype=dtype_name,
                  threads=args.threads, max_new_tokens=args.max_new_tokens, do_sample=False,
                  seed=0, image_preprocessing='checkpoint_defaults',
                  prompt_policy='spatial_question_only' if spec.family == 'blip' else 'spatial_question_plus_yes_no_instruction',
                  dataset_sha256=dataset_hash(args.manifest, pairs_path, manifest),
                  selected_image_ids=[r['image_id'] for r in selected],
                  python=platform.python_version(), versions=versions)
    run_path = args.out.with_suffix('.run.json')
    prior, predictions = None, {}
    if args.out.exists() or run_path.exists():
        if not (args.resume and args.out.exists() and run_path.exists()):
            raise SystemExit('Output exists or is incomplete. Use --resume with both files, or a new output folder.')
        prior = json.loads(run_path.read_text(encoding='utf-8'))
        if prior['config'] != config:
            raise SystemExit('Resume refused: protocol, code, data, selection, hardware mode or environment changed.')
        try:
            predictions = unique_index(read_jsonl(args.out), 'image_id')
        except (ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(f'Inspect prediction JSONL before resume: {exc}')
        if not set(predictions).issubset(config['selected_image_ids']):
            raise SystemExit('Unknown prediction IDs.')
        if any(r.get('run_id') != prior['run_id'] for r in predictions.values()):
            raise SystemExit('Mixed prediction run IDs.')
        if len(predictions) == len(selected):
            print(f'Already complete: {len(predictions)} images. Model inference skipped.', flush=True)
            return
    print(f'Loading {spec.repo_id} | {device} | {dtype_name} | {len(selected)-len(predictions)} images remain', flush=True)
    if args.model == 'gemma':
        print('Gemma requires accepted model terms and Hugging Face authentication. GPU is recommended.', flush=True)
    adapter = ModelAdapter(spec, args.revision, device, dtype)
    if prior and prior.get('resolved_revision') != adapter.resolved_revision:
        raise SystemExit('Resume refused: model commit changed. Use the recorded commit or start a new run.')
    run_id = prior['run_id'] if prior else datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not prior:
        meta = dict(run_id=run_id, config=config, resolved_revision=adapter.resolved_revision,
                    hardware=dict(platform=platform.platform(), cpu=platform.processor(),
                                  gpu=torch.cuda.get_device_name() if device == 'cuda' else None,
                                  gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory if device == 'cuda' else None,
                                  cuda=torch.version.cuda),
                    processor=adapter.processor_metadata())
        run_path.write_text(json.dumps(meta, indent=2, default=str), encoding='utf-8')
        packages = sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions() if d.metadata['Name'])
        args.out.with_name('environment.txt').write_text('\n'.join(packages)+'\n', encoding='utf-8')
    with args.out.open('a', encoding='utf-8', newline='\n') as stream:
        for row in selected:
            if row['image_id'] in predictions:
                continue
            if device == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            inputs = adapter.prepare(args.manifest.parent/row['image_path'], row['question'])
            with torch.inference_mode():
                generated = adapter.model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens)
            raw = adapter.decode(generated, inputs)
            if device == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
            output = dict(image_id=row['image_id'], raw_answer=raw, parsed_answer=parse_answer(raw),
                          seconds=elapsed, run_id=run_id,
                          input_tokens=int(inputs['input_ids'].shape[1]),
                          tensor_shapes={k:list(v.shape) for k,v in inputs.items() if hasattr(v, 'shape')},
                          generated_tokens=int(generated.shape[1]) if spec.family == 'blip' else int(generated.shape[1]-inputs['input_ids'].shape[1]),
                          prompt_text=adapter.prompt_text(row['question']),
                          timestamp=datetime.now(timezone.utc).isoformat())
            stream.write(json.dumps(output, ensure_ascii=False)+'\n')
            stream.flush()
            predictions[row['image_id']] = output
            print(f'[{len(predictions)}/{len(selected)}] {row["image_id"]}: {raw!r} ({elapsed:.2f}s)', flush=True)
    del adapter
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    print(f'Finished: {args.out}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted. Keep completed predictions and resume with unchanged settings.')
        raise SystemExit(130)
