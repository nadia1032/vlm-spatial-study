"""Sequential model runs in separate processes, with analysis and comparison."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys

from model_adapters import MODELS

ROOT = Path(__file__).resolve().parent


def execute(command):
    print('\n' + subprocess.list2cmdline(list(map(str, command))), flush=True)
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    env['TOKENIZERS_PARALLELISM'] = 'false'
    env['MPLBACKEND'] = 'Agg'
    process = subprocess.Popen(list(map(str, command)), cwd=ROOT, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding='utf-8', errors='replace')
    try:
        for line in process.stdout:
            print(line, end='', flush=True)
        return process.wait()
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    finally:
        process.stdout.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=list(MODELS), default=['smol500', 'qwen2vl', 'gitvqa'])
    parser.add_argument('--groups', type=int, default=1)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--dtype', choices=('auto', 'float32', 'float16', 'bfloat16'), default='auto')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--max-new-tokens', type=int, default=8)
    parser.add_argument('--manifest', type=Path, default=ROOT/'data/pilot/manifest.jsonl')
    parser.add_argument('--output', type=Path, default=ROOT/'results')
    parser.add_argument('--tag', default='v2_pilot', help='Use a new tag when intentionally changing settings')
    args = parser.parse_args(argv)
    if not args.tag or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.tag):
        parser.error('--tag must contain only letters, numbers, underscores and hyphens')
    if len(args.models) != len(set(args.models)):
        parser.error('List each model once')
    failed, analyses = [], []
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    for key in args.models:
        folder = args.output/f'{key}_{args.tag}_{args.groups}groups_{args.device}_{args.dtype}'
        predictions = folder/'predictions.jsonl'
        command = [sys.executable, '-u', ROOT/'run_model.py', '--model', key,
                   '--manifest', args.manifest, '--out', predictions,
                   '--groups', args.groups, '--device', args.device, '--dtype', args.dtype,
                   '--threads', args.threads, '--max-new-tokens', args.max_new_tokens]
        if predictions.exists() and predictions.with_suffix('.run.json').exists():
            command.append('--resume')
        code = execute(command)
        if code:
            failed.append((key, code))
            print(f'{key} failed with exit code {code}; retaining completed outputs and continuing.', flush=True)
            continue
        analysis = folder/f'analysis_{stamp}'
        code = execute([sys.executable, '-u', ROOT/'analyze_results.py', '--manifest', args.manifest,
                        '--predictions', predictions, '--out', analysis])
        if code:
            failed.append((key+'_analysis', code))
            continue
        analyses.append(analysis)
    if analyses:
        code = execute([sys.executable, ROOT/'report_runs.py', '--analyses', *analyses,
                        '--out', args.output/f'report_{args.tag}_{stamp}'])
        if code:
            failed.append(('comparison', code))
    print('\nCompleted analyses:', *map(str, analyses), sep='\n', flush=True)
    if failed:
        print('Unfinished stages:', failed, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nStopped. Completed image outputs were retained; resume with the same settings.')
        raise SystemExit(130)
