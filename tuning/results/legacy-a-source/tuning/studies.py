"""Study orchestration using the existing circuit classes and shared optimizer."""
import copy
import json
from pathlib import Path
import resource
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np

from core import qaoa_base as base, constraint_handler as ch
from core.vcg import VCG, state_quality
from core.penalty_qaoa import PenaltyQAOA
from core.penalty_heuristics import coefficient_penalties
from core.optimization import make_adam_functions, optimize_resumable, TrainingInterrupted
from storage import TaskStore, digest, write_json, write_pickle


def seed_for(seed, *parts):
    # Deliberately omit the competing loss/policy from initialization seeds.
    return int(digest([seed, *parts])[:8], 16)


def loss_args(name):
    if name == 'feasibility':
        return 0, 1.0
    if name == 'fidelity':
        return 1, 1.0
    if name.startswith('lambda_'):
        return 2, float(name.split('_')[1])
    raise ValueError(f'Unknown loss {name}')


def quality_metrics(state, mask):
    p_f, fidelity, conditional = map(float, state_quality(state, mask))
    conditional_probs = np.abs(np.asarray(state)[mask])**2 / max(p_f, 1e-12)
    positive = conditional_probs[conditional_probs > 0]
    count = int(np.sum(mask))
    entropy = float(-np.sum(positive*np.log(positive))/np.log(count)) if count > 1 else 1.0
    return dict(p_feasible=p_f, fidelity=fidelity, conditional_fidelity=conditional,
                entropy=entropy, min_conditional_probability=float(conditional_probs.min()),
                coverage=float(np.mean(conditional_probs >= 1e-3/count)))


def probability_metrics(probabilities, reference):
    probabilities = np.asarray(probabilities, dtype=float)
    mask = np.asarray(reference['feasible_mask'], dtype=bool)
    optimal = np.asarray(reference['optimal_indices'], dtype=int)
    p_f = float(probabilities[mask].sum())
    p_opt = float(probabilities[optimal].sum())
    values = np.asarray(reference['objective_values'])
    conditional_mean = float(probabilities[mask] @ values[mask] / p_f) if p_f else None
    spread = reference['f_max_feasible']-reference['f_star']
    quality = ((reference['f_max_feasible']-conditional_mean)/spread
               if spread and p_f else (1.0 if p_f else None))
    return dict(p_feasible=p_f, p_optimal=p_opt, conditional_objective=conditional_mean,
                conditional_quality=quality, weighted_quality=p_f*quality if p_f else 0.0)


def settings_b(instance, config):
    anchors = coefficient_penalties(np.asarray(instance['Q']),
                                   ch.parse_constraints(instance['constraints']))
    result = [dict(id=f'{name}_{multiplier:g}', rule=name, multiplier=multiplier,
                   delta=anchor*multiplier, selectable=True)
              for name, anchor in anchors.items()
              for multiplier in config['study_b']['multipliers']]
    result.append(dict(id='paper', rule='paper', multiplier=1.0, selectable=False,
                       delta=5+2*abs(instance['f_min_unconstrained'])))
    return result


def optimizer_phase(task, store, phase, cost, step, *, config, shape, steps,
                    restarts, lr, seed, args, previous=None, starting=None,
                    should_stop=lambda: False, zero_pad_first=True):
    active = task.get('optimizer')
    if active is not None and task.get('phase') != phase:
        raise ValueError('Checkpoint phase mismatch')
    task['phase'] = phase

    def checkpoint(state):
        task['optimizer'] = state
        store.save(task)
        store.status('running', phase=phase, restart=state['restart'], step=state['step'])

    result = optimize_resumable(cost, step, shape=shape, steps=steps,
        restarts=restarts, learning_rate=lr, seed=seed, args=args,
        previous=previous, starting=starting, state=active, zero_pad_first=zero_pad_first,
        checkpoint=checkpoint, checkpoint_steps=config['checkpoint_steps'],
        should_stop=should_stop)
    return result


def timing(result):
    return dict(compile_seconds=float(result['compile_seconds']),
                optimize_seconds=float(result['optimize_seconds']))


def run_a_instance(instance, config, directory, run_id, manifest_hash, should_stop):
    cfg = config['study_a']
    gadget = VCG([instance['constraint']])
    num_gamma = len(gadget.constraint_Ham.ops)
    mask = np.asarray(instance['feasible_mask'], dtype=bool)
    cache = {}

    def functions(strategy, depth):
        key = strategy, depth
        if key not in cache:
            objective, state_fn = gadget.tuning_functions(strategy, depth)
            cost, step = make_adam_functions(objective, cfg['learning_rate'])
            cache[key] = cost, step, state_fn
        return cache[key]

    for loss in cfg['losses']:
        identity = digest([run_id, manifest_hash, instance['id'], loss])
        store = TaskStore(Path(directory)/'a'/instance['id']/loss, identity)
        if (store.directory/'result.json').exists():
            existing = json.loads((store.directory/'result.json').read_text())
            if existing['identity'] != identity:
                raise ValueError('Completed task identity mismatch')
            continue
        task = store.load() or dict(warm=None, depths=[], optimizer=None, best=None)
        try:
            mode, weight = loss_args(loss)
            args = (jnp.asarray(mode), jnp.asarray(weight))
            if task['warm'] is None:
                cost, step, _ = functions('QAOA', 1)
                result = optimizer_phase(task, store, 'warmup', cost, step,
                    config=config, shape=(1, 2), steps=cfg['qaoa_steps'],
                    restarts=cfg['qaoa_restarts'], lr=cfg['learning_rate'],
                    seed=seed_for(config['training_seed'], instance['id'], 'warmup'),
                    args=args, should_stop=should_stop)
                task['warm'] = dict(angles=result['best_angles'], loss=result['best_loss'],
                                    history=result['history'], **timing(result))
                task['optimizer'] = None
                store.save(task)
            for depth in range(len(task['depths'])+1, cfg['max_depth']+1):
                if task['depths'] and task['depths'][-1]['fidelity'] >= cfg['fidelity_threshold']:
                    break
                cost, step, state_fn = functions('ma-QAOA', depth)
                previous = task['depths'][-1]['angles'] if task['depths'] else None
                starting = (base.convert_qaoa_to_ma_angles(
                    task['warm']['angles'], num_gamma, gadget.n_x, 1) if depth == 1 else None)
                result = optimizer_phase(task, store, f'depth_{depth}', cost, step,
                    config=config, shape=(depth, num_gamma+gadget.n_x),
                    steps=cfg['ma_steps'], restarts=cfg['ma_restarts'], lr=cfg['learning_rate'],
                    seed=seed_for(config['training_seed'], instance['id'], depth), args=args,
                    previous=previous, starting=starting, should_stop=should_stop,
                    zero_pad_first=False)
                state = np.asarray(state_fn(result['best_angles']))
                row = dict(depth=depth, angles=result['best_angles'], loss=result['best_loss'],
                           history=result['history'], restarts=result['restart_results'],
                           **quality_metrics(state, mask), **timing(result))
                task['depths'].append(row)
                if task['best'] is None or row['fidelity'] > task['best']['fidelity']:
                    task['best'] = dict(row, state=state)
                task['optimizer'] = None
                store.save(task)
                write_json(store.directory/'depths.json', task['depths'])
            best = task['best']
            entry = dict(opt_angles=np.asarray(best['angles']), n_layers=best['depth'],
                         ar=best['p_feasible'], entropy=best['entropy'],
                         single_feasible_bitstring=None, dicke_superposition_weights=None,
                         num_gamma=num_gamma, num_beta=gadget.n_x,
                         parameter_layout='legacy_pauli_slots_then_rx',
                         constraints=[instance['constraint']], loss_mode=loss,
                         state=np.asarray(best['state']), fidelity=best['fidelity'],
                         support=instance['support'], schema_version=1)
            write_pickle(store.directory/'gadget.pkl', entry)
            summary = {key: best[key] for key in quality_metrics(best['state'], mask)}
            all_timing = [task['warm']] + task['depths']
            summary.update(identity=identity, instance_id=instance['id'], condition=loss,
                           family=instance['family'], support=instance['support'],
                           selected_depth=best['depth'], final_loss=best['loss'],
                           evaluated_depths=len(task['depths']),
                           converged=best['fidelity'] >= cfg['fidelity_threshold'],
                           compile_seconds=sum(r['compile_seconds'] for r in all_timing),
                           optimize_seconds=sum(r['optimize_seconds'] for r in all_timing),
                           process_peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
            write_json(store.directory/'result.json', summary)
            store.status('complete')
            print(f"A {instance['id']} {loss}: F={best['fidelity']:.6f}, depth={best['depth']}", flush=True)
        except TrainingInterrupted:
            store.status('interrupted', phase=task.get('phase'))
            raise
        except Exception:
            store.status('failed', traceback=traceback.format_exc(), phase=task.get('phase'))
            raise


def run_b_instance(instance, config, directory, run_id, manifest_hash, should_stop):
    cfg = config['study_b']
    template = PenaltyQAOA(np.asarray(instance['Q']), instance['constraints'],
                          penalty=1.0, angle_strategy='QAOA', compile_on_init=False)
    cache = {}

    def functions(depth):
        if depth not in cache:
            solver = copy.copy(template)
            solver.n_layers = depth
            objective, probs_fn = solver.tuning_functions()
            cost, step = make_adam_functions(objective, cfg['learning_rate'])
            cache[depth] = cost, step, probs_fn
        return cache[depth]

    for setting in settings_b(instance, config):
        identity = digest([run_id, manifest_hash, instance['id'], setting])
        store = TaskStore(Path(directory)/'b'/instance['id']/setting['id'], identity)
        if (store.directory/'result.json').exists():
            existing = json.loads((store.directory/'result.json').read_text())
            if existing['identity'] != identity:
                raise ValueError('Completed task identity mismatch')
            continue
        task = store.load() or dict(depths=[], optimizer=None)
        try:
            for depth in range(len(task['depths'])+1, cfg['max_depth']+1):
                cost, step, probs_fn = functions(depth)
                previous = task['depths'][-1]['angles'] if task['depths'] else None
                args = (jnp.asarray(setting['delta']),)
                result = optimizer_phase(task, store, f'depth_{depth}', cost, step,
                    config=config, shape=(depth, 2), steps=cfg['steps'],
                    restarts=cfg['restarts'], lr=cfg['learning_rate'],
                    seed=seed_for(config['training_seed'], instance['id'], depth),
                    args=args, previous=previous, should_stop=should_stop)
                probabilities = np.asarray(probs_fn(result['best_angles'], *args))
                probabilities = np.clip(probabilities, 0, None)
                probabilities /= probabilities.sum()
                sample_seed = seed_for(config['sampling_seed'], instance['id'], setting['id'], depth)
                counts = np.random.default_rng(sample_seed).multinomial(cfg['shots'], probabilities)
                exact = probability_metrics(probabilities, instance)
                sampled = probability_metrics(counts/cfg['shots'], instance)
                row = dict(depth=depth, angles=result['best_angles'], loss=result['best_loss'],
                           history=result['history'], restarts=result['restart_results'],
                           counts=counts.tolist(), probabilities=probabilities.tolist(),
                           sampling_seed=sample_seed, shots=cfg['shots'],
                           **sampled, **{f'exact_{k}': v for k, v in exact.items()}, **timing(result))
                task['depths'].append(row)
                task['optimizer'] = None
                store.save(task)
                write_json(store.directory/'depths.json', task['depths'])
            final = task['depths'][-1]
            summary = dict(identity=identity, instance_id=instance['id'], condition=setting['id'],
                           **{k: v for k, v in setting.items() if k != 'id'},
                           n_x=instance['n_x'], regime=instance['regime'],
                           n_slack=template.n_slack, n_total=template.n_total,
                           depth=final['depth'], p_optimal=final['p_optimal'],
                           p_feasible=final['p_feasible'], exact_p_optimal=final['exact_p_optimal'],
                           exact_p_feasible=final['exact_p_feasible'],
                           compile_seconds=sum(r['compile_seconds'] for r in task['depths']),
                           optimize_seconds=sum(r['optimize_seconds'] for r in task['depths']),
                           process_peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
            write_json(store.directory/'result.json', summary)
            store.status('complete')
            print(f"B {instance['id']} {setting['id']}: P(opt)={final['p_optimal']:.4f} at p={final['depth']}", flush=True)
        except TrainingInterrupted:
            store.status('interrupted', phase=task.get('phase'))
            raise
        except Exception:
            store.status('failed', traceback=traceback.format_exc(), phase=task.get('phase'))
            raise
