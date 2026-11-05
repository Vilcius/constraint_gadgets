"""Read only true target-depth results for each QAOA+ comparison budget."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .budgets import HERE, LABELS
from .studies import metrics
from tuning.storage import digest
from final_experiment.random_cops.uncompiled_analysis import prefix_timing


def load_comparisons(output, budgets, references):
    output = Path(output)
    rows, completion = [], []
    original_cache = {}
    manifest_hash = digest(list(references.values()))

    def original_depths(key, iid):
        cache_key = (key, iid)
        if cache_key not in original_cache:
            source = HERE/'results'/key/iid/'depths.json'
            original_cache[cache_key] = json.loads(source.read_text()) if source.exists() else []
        return original_cache[cache_key]

    def uncompiled_depths(key, iid):
        source = HERE/'results/uncompiled'/key/iid/'depths.json'
        if not source.exists():
            return []
        rows = json.loads(source.read_text())
        if rows:
            prefix_timing(HERE/'results/uncompiled', key, iid, rows[-1]['depth'], manifest_hash)
        return rows

    def recorded_timing(depths):
        # Missing measurements remain missing rather than appearing as zero time.
        result = {}
        for field, label in (('optimize_seconds','optimization_seconds'),
                             ('compile_seconds','compilation_seconds')):
            result[label] = (sum(row[field] for row in depths)
                             if depths and all(field in row for row in depths) else np.nan)
        return result

    for key, label in LABELS.items():
        directories = [output/key, *sorted(output.glob(f'{key}_qaoa_p*')),
                       *sorted(output.glob(f'{key}_shard_*'))]
        directories = list(dict.fromkeys(path for path in directories if path.is_dir()))
        for directory in directories:
            metadata = directory/'run.json'
            if metadata.exists():
                run = json.loads(metadata.read_text())
                if run['manifest_hash'] != manifest_hash:
                    raise ValueError(f'Comparison dataset mismatch: {directory}')
                if run['config'].get('profile') != 'full' or run['config'].get('depth_cap') is not None:
                    raise ValueError(f'Expected uncapped full results: {directory}')
        for reference_depth in sorted(budgets.reference_depth.unique()):
            selected = budgets[(budgets.method == key) & (budgets.reference_depth == reference_depth)]
            available = imported = running = failed = 0
            for budget in selected.to_dict('records'):
                iid = budget['instance_id']
                path = next((directory/iid/f'comparison_qaoa_p{reference_depth}.json'
                             for directory in directories
                             if (directory/iid/f'comparison_qaoa_p{reference_depth}.json').exists()), None)
                if path is not None and budget['target_depth'] > 5:
                    data = json.loads(path.read_text())
                    for name in ('target_depth','reference_depth','gate_budget','actual_gates'):
                        if data[name] != budget[name]:
                            raise ValueError(f'Incompatible comparison file: {path}, {name}')
                    if data['instance_id'] != iid or data['method'] != key:
                        raise ValueError(f'Comparison identity mismatch: {path}')
                    row = {**budget, **data, 'method': label, 'source': str(path)}
                    imported += (budget['target_depth'] <= 5 and
                                 data.get('new_optimization_seconds', 0.) == 0.)
                elif budget['target_depth'] <= 5:
                    data = next((r for r in uncompiled_depths(key,iid)
                                 if r['depth'] == budget['target_depth']), None)
                    if data is None:
                        continue
                    probabilities = np.asarray(data['counts'],dtype=float)/sum(data['counts'])
                    row = {**budget, **metrics(probabilities,references[iid]),
                           'method': label, 'depth': budget['target_depth'],
                           'source': 'uncompiled_saved_depth', 'new_optimization_seconds': 0.,
                           'n_total': data['n_total'], 'n_slack': data['n_slack']}
                    imported += 1
                else:
                    status_path = next((directory/iid/'status.json' for directory in directories
                                        if (directory/iid/'status.json').exists()), None)
                    if status_path is not None:
                        status = json.loads(status_path.read_text()).get('status')
                        running += status == 'running'
                        failed += status == 'failed'
                    continue
                saved = original_depths(key,iid)
                prefix = [item for item in saved if item['depth'] <= min(budget['target_depth'],5)]
                timing = recorded_timing(prefix)
                if path is not None and budget['target_depth'] > 5:
                    timing['optimization_seconds'] = (
                        data['imported_optimization_seconds'] + data['new_optimization_seconds']
                        if all(k in data for k in ('imported_optimization_seconds','new_optimization_seconds'))
                        else np.nan)
                row.update(timing)
                row['recorded_training_seconds'] = sum(timing.values())
                prefix=recorded_timing(prefix)['optimization_seconds']
                remeasured=prefix_timing(HERE/'results/uncompiled',key,iid,
                                        min(budget['target_depth'],5),manifest_hash)
                row['uncompiled_total_optimization_seconds'] = (
                    timing['optimization_seconds']-prefix+remeasured)
                if budget['target_depth'] <= 5:
                    row.update(optimization_seconds=remeasured, compilation_seconds=0.,
                               recorded_training_seconds=remeasured,
                               uncompiled_total_optimization_seconds=remeasured)
                baseline_rows = uncompiled_depths(key, iid)
                baseline = next((item for item in baseline_rows if item['depth']==5),None)
                row['original_depth5_p_feasible'] = (
                    metrics(np.asarray(baseline['counts'],dtype=float)/sum(baseline['counts']),
                            references[iid])['p_feasible'] if baseline is not None else np.nan)
                row['feasible_fraction'] = references[iid]['feasible_fraction']
                row['overlap_level'] = references[iid]['o']
                rows.append(row)
                available += 1
            completion.append({'Method': label,'QAOA+ Depth':reference_depth,
                               'Expected':len(selected),'Available':available,
                               'Read From Uncompiled':imported,'Running':running,'Failed':failed,
                               'Remaining':len(selected)-available})
    for iid, reference in references.items():
        path = HERE/'results/uncompiled/qaoa_plus'/iid/'depths.json'
        if not path.exists():
            continue
        data = uncompiled_depths('qaoa_plus', iid)
        for depth in sorted(budgets.reference_depth.unique()):
            item = next((r for r in data if r['depth']==depth),None)
            if item is None:
                continue
            probabilities = np.asarray(item['counts'],dtype=float)/sum(item['counts'])
            timing = recorded_timing([row for row in data if row['depth']<=depth])
            rows.append(dict(instance_id=iid,method='QAOA+',reference_depth=depth,
                             depth=depth,n_x=reference['n_x'],regime=reference['regime'],
                             source=str(path),
                             n_total=item['n_total'],n_slack=item['n_slack'],
                             **timing,recorded_training_seconds=sum(timing.values()),
                             uncompiled_total_optimization_seconds=prefix_timing(
                                 HERE/'results/uncompiled','qaoa_plus',iid,depth,
                                 manifest_hash),
                             overlap_level=reference['o'],feasible_fraction=reference['feasible_fraction'],
                             **metrics(probabilities,reference)))
    frame = pd.DataFrame(rows)
    for column in ('instance_id','method','reference_depth','minimum_depth_overrun'):
        if column not in frame:
            frame[column] = pd.Series(dtype=object)
    return frame,pd.DataFrame(completion)


def paired_results(results, method, reference_depth=1, include_overruns=True):
    left = results[(results.method==method)&(results.reference_depth==reference_depth)].copy()
    if left.empty:
        return pd.DataFrame()
    if not include_overruns:
        left = left[~left.minimum_depth_overrun.fillna(False).astype(bool)]
    right = results[(results.method=='QAOA+')&(results.reference_depth==reference_depth)]
    measures=['p_optimal','p_feasible','ar_feas']
    pair = left.merge(right[['instance_id',*measures]],on='instance_id',
                      suffixes=('_pc','_qaoa'),validate='one_to_one')
    for measure in measures:
        pair[f'difference_{measure}']=pair[f'{measure}_pc']-pair[f'{measure}_qaoa']
    return pair
