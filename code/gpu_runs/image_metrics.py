"""Class metrics: invalid answers stay in the denominator as errors."""
from collections import defaultdict
import numpy as np


def class_values(rows):
    recalls = {}
    for label in ('yes', 'no'):
        subset = [r for r in rows if r['ground_truth'] == label]
        recalls[label+'_recall'] = (sum(r['prediction'] == label for r in subset)/len(subset)
                                   if subset else None)
    recalls['balanced_accuracy'] = ((recalls['yes_recall']+recalls['no_recall'])/2
                                    if all(v is not None for v in recalls.values()) else None)
    return recalls


def class_metrics(rows, repetitions=1000, seed=42):
    groups = defaultdict(list)
    for r in rows:
        groups[r['group_id']].append(r)
    names = list(groups)
    values = class_values(rows)
    samples = {k:[] for k in values}
    if len(names)>1:
        rng = np.random.default_rng(seed)
        for _ in range(repetitions):
            sampled = [r for i in rng.integers(0,len(names),len(names)) for r in groups[names[i]]]
            for key,v in class_values(sampled).items():
                if v is not None:samples[key].append(v)
    output=[]
    for key,value in values.items():
        # An interval is unavailable if too many resamples lose a true class.
        valid = value is not None and len(samples[key])>=0.95*repetitions
        low,high = np.quantile(samples[key],[0.025,0.975]) if valid else (None,None)
        eligible = rows if key=='balanced_accuracy' else [r for r in rows if r['ground_truth']==key.split('_')[0]]
        output.append(dict(metric=key,value=value,ci_low=None if low is None else float(low),
                           ci_high=None if high is None else float(high),n=len(eligible),
                           scene_groups=len({r['group_id'] for r in eligible})))
    return output
