"""One uncapped trajectory per COP, with independently saved comparison points."""
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np

from .optimization import optimize_resumable, TrainingInterrupted
from .task_storage import TaskStore
from tuning.storage import digest, write_json
from .simulator import Simulator


def seed_for(seed, *parts):
    return int(digest([seed, *parts])[:8], 16)


def metrics(probabilities, instance):
    feasible = np.asarray(instance['feasible_mask'], dtype=bool)
    values = np.asarray(instance['objective_values'])
    pf = float(probabilities[feasible].sum())
    po = float(probabilities[np.asarray(instance['optimal_indices'], dtype=int)].sum())
    average = float(probabilities[feasible] @ values[feasible] / pf) if pf else None
    spread = instance['f_max_feasible'] - instance['f_star']
    ar = (None if abs(spread) < 1e-12 else
          ((instance['f_max_feasible'] - average) / spread if pf else 0.))
    return {'p_optimal': po, 'p_feasible': pf, 'ar_feas': ar,
            'conditional_objective': average}


def import_depths(directory, instance, config, simulator, limit):
    directory = Path(directory)
    run = json.loads((directory / 'run.json').read_text())
    if run['manifest_hash'] != config['manifest_hash']:
        raise ValueError('Import dataset mismatch')
    for name in ('steps', 'restarts', 'learning_rate', 'shots', 'training_seed', 'sampling_seed'):
        if run['config'][name] != config.get('import_settings', config)[name]:
            raise ValueError(f'Import optimization setting mismatch: {name}')
    if run.get('method', run['config'].get('method')) != config['method']:
        raise ValueError('Import method mismatch')
    task = directory / instance['id']
    legacy = task / 'depths.json'
    if legacy.exists():
        paths = [legacy]
        rows = json.loads(legacy.read_text())
    else:
        paths = sorted((task / 'depths').glob('depth_*.json'))
        rows = [json.loads(path.read_text()) for path in paths]
    selected = [row for row in rows if row['depth'] <= limit]
    if (not selected or [r['depth'] for r in selected] != list(range(1, len(selected) + 1))
            or len(selected) < min(limit, 5)):
        raise ValueError(f'Missing saved prefix for {instance["id"]}')
    final = selected[-1]
    probabilities = simulator.probabilities(final['angles'])
    np.testing.assert_allclose(probabilities, final['probabilities'], atol=2e-8, rtol=1e-6)
    loss = float((abs(simulator.state(final['angles']))**2) @ simulator.energy)
    if not np.isclose(loss, final['loss'], rtol=1e-7, atol=1e-7):
        raise ValueError(f'Imported loss does not match circuit: {instance["id"]}')
    for row in selected:
        row['origin'] = 'imported'
        row['imported_compile_seconds'] = row.get('compile_seconds', 0.)
        row.update(metrics(np.asarray(row['counts']) / sum(row['counts']), instance))
    provenance = {'run_identifier': run['identifier'], 'directory': str(task),
                  'files': {str(p.relative_to(task)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in paths}}
    return selected, provenance


def depth_path(directory, depth):
    return Path(directory) / 'depths' / f'depth_{depth:05d}.json'


def choose_import_source(sources, instance, limit):
    """Prefer the longest committed prefix, retaining original depths as fallback."""
    if isinstance(sources, (str, Path)):
        return Path(sources)
    candidates = []
    for source in sources:
        directory = Path(source)
        task = directory / instance['id']
        legacy = task / 'depths.json'
        if legacy.exists():
            depths = {row['depth'] for row in json.loads(legacy.read_text())}
        else:
            depths = {int(p.stem.removeprefix('depth_'))
                      for p in (task / 'depths').glob('depth_*.json')}
        prefix = 0
        while prefix < limit and prefix + 1 in depths:
            prefix += 1
        if prefix >= min(limit, 5):
            candidates.append((prefix, directory))
    if not candidates:
        raise ValueError(f'Missing saved prefix for {instance["id"]}')
    return max(candidates, key=lambda item: item[0])[1]


def run_instance(method, instance, budgets, config, output, run_id, source,
                 should_stop=lambda: False):
    identity = digest([run_id, instance['id'], budgets])
    store = TaskStore(Path(output) / instance['id'], identity)
    result_path = store.directory / 'result.json'
    if result_path.exists():
        if json.loads(result_path.read_text())['identity'] != identity:
            raise ValueError('Completed task identity mismatch')
        return
    maximum = max(row['target_depth'] for row in budgets)
    task = store.load()
    store.status('running', stage='preparing_and_validating_import', maximum_depth=maximum)
    started = time.perf_counter()
    simulator = Simulator(instance, method, config['database'])
    preparation_seconds = time.perf_counter() - started

    def publish(row):
        for budget in budgets:
            if budget['target_depth'] != row['depth']:
                continue
            comparison = dict(
                identity=identity, instance_id=instance['id'], method=method,
                **{k: v for k, v in budget.items() if k not in ('instance_id', 'method')},
                **metrics(np.asarray(row['counts']) / sum(row['counts']), instance),
                **{f'exact_{k}': v for k, v in metrics(np.asarray(row['probabilities']), instance).items()},
                depth=row['depth'], n_total=simulator.n, n_slack=simulator.model.n_slack,
                new_optimization_seconds=task['new_optimization_seconds'],
                imported_optimization_seconds=task['imported_optimization_seconds'],
                imported_depths=task['imported_depths'], compile_seconds=0.,
                prepare_seconds=task['prepare_seconds'],
                peak_memory_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                import_provenance=task['import_provenance'])
            write_json(store.directory / f'comparison_qaoa_p{budget["reference_depth"]}.json', comparison)

    def finish_depth(row):
        if row['identity'] != identity or row['depth'] != task['next_depth']:
            raise ValueError('Depth-record identity or order mismatch')
        if row['origin'] == 'imported':
            task['imported_depths'] += 1
            task['imported_optimization_seconds'] += row.get('optimize_seconds', 0.)
        else:
            task['new_optimization_seconds'] += row['optimize_seconds']
        publish(row)
        task['next_depth'] += 1
        task['optimizer'] = None
        store.save(task)

    try:
        if task is None:
            selected_source = choose_import_source(source, instance, maximum)
            rows, provenance = import_depths(selected_source, instance, config, simulator, maximum)
            task = dict(next_depth=1, optimizer=None, import_provenance=provenance,
                        prepare_seconds=preparation_seconds, imported_depths=0,
                        imported_optimization_seconds=0., new_optimization_seconds=0.)
            for row in rows:
                row['identity'] = identity
                write_json(depth_path(store.directory, row['depth']), row)
            for row in rows:
                finish_depth(row)
        else:
            task['prepare_seconds'] += preparation_seconds
        while task['next_depth'] <= maximum:
            depth = task['next_depth']
            path = depth_path(store.directory, depth)
            # Recover a depth committed just before an interrupted checkpoint.
            if path.exists():
                finish_depth(json.loads(path.read_text()))
                continue
            if should_stop():
                store.save(task)
                raise TrainingInterrupted('Stopped before next depth')
            previous = json.loads(depth_path(store.directory, depth - 1).read_text())['angles']

            def checkpoint(optimizer):
                task['optimizer'] = optimizer
                # Only active optimizer state and scalar totals are checkpointed.
                # Completed depth records are written once to separate files.
                store.save(task)
                store.status('running', depth=depth, maximum_depth=maximum,
                             restart=optimizer['restart'], step=optimizer['step'])

            optimized = optimize_resumable(
                simulator.loss_and_gradient, shape=(depth, 2), steps=config['steps'],
                restarts=config['restarts'], learning_rate=config['learning_rate'],
                seed=seed_for(config['training_seed'], instance['id'], depth),
                previous=previous, state=task['optimizer'], checkpoint=checkpoint,
                checkpoint_steps=config['checkpoint_steps'], should_stop=should_stop)
            probabilities = simulator.probabilities(optimized['best_angles'])
            sampling_seed = seed_for(config['sampling_seed'], instance['id'], depth)
            counts = np.random.default_rng(sampling_seed).multinomial(config['shots'], probabilities)
            row = dict(identity=identity, depth=depth, angles=optimized['best_angles'],
                       loss=optimized['best_loss'], probabilities=probabilities, counts=counts,
                       shots=config['shots'], sampling_seed=sampling_seed,
                       history=optimized['history'], restarts=optimized['restart_results'],
                       n_total=simulator.n, n_slack=simulator.model.n_slack,
                       compile_seconds=0., origin='numpy_adjoint',
                       optimize_seconds=optimized['optimize_seconds'],
                       **metrics(counts / config['shots'], instance),
                       **{f'exact_{k}': v for k, v in metrics(probabilities, instance).items()})
            write_json(path, row)
            finish_depth(row)
        comparisons = []
        for budget in budgets:
            path = store.directory / f'comparison_qaoa_p{budget["reference_depth"]}.json'
            if not path.exists():
                publish(json.loads(depth_path(store.directory, budget['target_depth']).read_text()))
            comparisons.append(json.loads(path.read_text()))
        write_json(result_path, dict(identity=identity, instance_id=instance['id'], method=method,
                   maximum_depth=maximum, completed_comparisons=[r['reference_depth'] for r in comparisons],
                   new_optimization_seconds=task['new_optimization_seconds'],
                   imported_optimization_seconds=task['imported_optimization_seconds'],
                   compile_seconds=0., prepare_seconds=task['prepare_seconds']))
        store.status('complete', depth=maximum)
        print(f'{method} {instance["id"]}: ' + ', '.join(
            f'QAOA+ p={r["reference_depth"]}: PC p={r["depth"]}, P(opt)={r["p_optimal"]:.4f}'
            for r in comparisons), flush=True)
    except Exception as error:
        import traceback
        store.status('interrupted' if isinstance(error, TrainingInterrupted) else 'failed',
                     error=str(error), traceback=traceback.format_exc())
        raise
