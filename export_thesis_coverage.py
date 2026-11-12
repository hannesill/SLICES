"""Export aggregate evidence for the review additions. No patient records.

Usage: python export_evidence.py FROZEN_EXPORT OUTPUT_JSON
Requires pandas and a parquet engine. The frozen export is read-only.
"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

source, output = map(Path, sys.argv[1:])
per_seed = pd.read_parquet(source / 'per_seed_results.parquet')
aggregate = pd.read_parquet(source / 'aggregated_results.parquet')
assert len(per_seed) == 2310
core = aggregate[
    (aggregate.experiment_class == 'core_ssl_benchmark')
    & (aggregate.model_size == 'default')
    & (aggregate.label_fraction == 1)
]
assert len(core) == 84
classification = core[core.task != 'los_remaining']
assert len(classification) == 63
keys = ['dataset', 'task', 'protocol', 'paradigm', 'model_size', 'label_fraction']
subgroups = {}
for attribute, expected in [('gender', 63), ('age_group', 63), ('race', 42)]:
    metric = f'fairness/{attribute}/auprc_gap/mean'
    valid = classification[classification[metric].notna()]
    assert len(valid) == expected
    subgroups[attribute] = {
        'metric': metric, 'rows': len(valid),
        'min': float(valid[metric].min()), 'max': float(valid[metric].max()),
        'cells': valid[keys + [metric]].to_dict('records'),
    }
capacity = per_seed[
    (per_seed.experiment_class == 'capacity_study')
    & (per_seed.dataset == 'miiv')
    & (per_seed.task == 'mortality_24h')
    & (per_seed.protocol == 'full_finetune')
]
capacity_pairs = []
for (size, fraction), group in capacity.groupby(['model_size', 'label_fraction']):
    paired = group.pivot(index='seed', columns='paradigm', values='test/auprc')
    assert paired.shape == (5, 2) and paired.notna().all().all()
    difference = paired.mae - paired.supervised
    capacity_pairs.append({
        'model_size': size, 'label_fraction': fraction, 'n_pairs': len(paired),
        'mae_mean': float(paired.mae.mean()),
        'supervised_mean': float(paired.supervised.mean()),
        'paired_delta_mean': float(difference.mean()),
        'paired_delta_sample_sd': float(difference.std(ddof=1)),
    })
assert len(capacity_pairs) == 6
los = core[(core.task == 'los_remaining') & (core.protocol == 'linear_probe')
           & (core.paradigm == 'contrastive')]
r2 = los[keys + ['test/r2/mean']].to_dict('records')
assert len(r2) == 3
result = {
    'source_files': {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                     for name in ['aggregated_results.parquet', 'per_seed_results.parquet']},
    'evidence_policy': 'Descriptive cell means and paired seed differences. No confirmatory inference.',
    'evaluation_rows': len(per_seed),
    'evaluation_rows_by_class': per_seed.experiment_class.value_counts().sort_index().to_dict(),
    'launch_matrix': {'total': 2455, 'pretraining_budget': 145,
                      'source': 'experiment_protocol.md, final launch matrix; not a completion audit of all pretraining runs'},
    'core_filter': {'experiment_class': 'core_ssl_benchmark', 'model_size': 'default', 'label_fraction': 1},
    'subgroup_classification': subgroups,
    'capacity': {'dataset': 'miiv', 'task': 'mortality_24h', 'protocol': 'full_finetune',
                 'metric': 'test/auprc', 'pairs': capacity_pairs},
    'contrastive_probe_los_r2': r2,
}
output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
print(json.dumps({'evaluations': len(per_seed), 'core_cells': len(core),
                  'subgroups': {k: (v['rows'], v['min'], v['max']) for k,v in subgroups.items()},
                  'capacity_mean_delta_range': [min(x['paired_delta_mean'] for x in capacity_pairs),
                                                max(x['paired_delta_mean'] for x in capacity_pairs)],
                  'los_r2': r2}))
