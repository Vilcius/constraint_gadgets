"""Resource estimates from saved tuning data. Never trains or simulates states."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'tuning/.cache/matplotlib'))

import numpy as np
import pandas as pd
import pennylane as qml
import pennylane.estimator as qre

from core import constraint_handler as ch, qaoa_base as base
from core import resource_estimation as re
from core.vcg import VCG
from core.penalty_qaoa import PenaltyQAOA
from analyze import mean_intervals, summary_plot
from storage import atomic_bytes, digest, write_json


def coalesced_tuning_hamiltonian(model, delta):
    """Match PenaltyQAOA.tuning_functions' fixed union of Pauli words.

    Keep terms whose weight cancels at a particular delta: the trained QNode
    still emitted those zero-angle rotations. Do not simplify a second time.
    """
    unit = base.build_penalty_hamiltonian(model.parsed_constraints,
        model.slack_info, 1.0, fallback_wire=model.x_wires[0])
    q_rep, p_rep = model.qubo_Ham.pauli_rep, unit.pauli_rep
    words = list(dict.fromkeys([*q_rep, *p_rep]))
    words = [w for w in words if q_rep.get(w, 0) != 0 or p_rep.get(w, 0) != 0]
    coeffs = [float(np.real(q_rep.get(w, 0) + delta * p_rep.get(w, 0))) for w in words]
    return qml.Hamiltonian(coeffs,
        [w.operation(wire_order=model.all_wires) for w in words])


def restore_gadget(gadget, entry):
    """Restore the saved legacy layout, including its unused identity slot."""
    if entry.get('parameter_layout', 'legacy_pauli_slots_then_rx') != 'legacy_pauli_slots_then_rx':
        raise ValueError('Unsupported saved VCG parameter layout')
    gadget.opt_angles = entry['opt_angles']
    gadget.n_layers = int(entry['n_layers'])
    gadget._single_feasible_bitstring = entry.get('single_feasible_bitstring')
    gadget._dicke_superposition_weights = entry.get('dicke_superposition_weights')
    if gadget.opt_angles is not None:
        expected = (gadget.n_layers, len(gadget.constraint_Ham.ops) + gadget.n_x)
        if np.shape(gadget.opt_angles) != expected:
            raise ValueError(f'Saved angle shape {np.shape(gadget.opt_angles)} != {expected}')
    return gadget


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _row(resource, **metadata):
    counts = resource['gate_counts']
    return dict(**metadata, **resource,
        one_qubit_gates=sum(n for g, n in counts.items() if g in re._1Q_GATES_SET),
        two_qubit_gates=sum(n for g, n in counts.items() if g in re._2Q_GATES_SET),
        t_gates=counts.get('T', 0))


class EstimateCache:
    """One atomic, content-identified file per instance; safe to resume analysis."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        sources = [Path(__file__), ROOT/'core/resource_estimation.py',
            ROOT/'core/vcg.py', ROOT/'core/qaoa_base.py',
            ROOT/'core/constraint_handler.py', ROOT/'core/dicke_state_prep.py',
            ROOT/'core/penalty_qaoa.py']
        self.policy = dict(schema=1, pennylane=qml.__version__,
            gate_sets={k: sorted(v) for k, v in re.GATE_SETS.items()},
            synthesis_config=str(qre.ResourceConfig()),
            source_hashes={str(p.relative_to(ROOT)): _sha(p) for p in sources},
            gates='structural decompositions; no angle pruning or global transpilation')

    def run(self, name, inputs, calculate):
        key = digest(dict(policy=self.policy, inputs=inputs))
        path = self.directory/f'{name}.json'
        if path.exists():
            saved = json.loads(path.read_text())
            if saved['identifier'] == key:
                return saved['rows']
        started = time.perf_counter()
        rows = calculate()
        write_json(path, dict(identifier=key, policy=self.policy, inputs=inputs,
            seconds=time.perf_counter()-started, rows=rows))
        return rows


def estimate_study_a(output, cache_dir=None):
    """All completed A gadgets, plus one independent penalty component per constraint."""
    output = Path(output)
    cache = EstimateCache(cache_dir or output/'resource_estimates')
    manifest = json.loads((output/'manifest_a.json').read_text())
    rows = []
    for instance in manifest['instances']:
        paths = sorted((output/'a'/instance['id']).glob('*/gadget.pkl'))
        if not paths:
            continue
        inputs = dict(instance=instance, gadget_files={p.parent.name: _sha(p) for p in paths})

        def calculate(instance=instance, paths=paths):
            c, n = instance['constraint'], instance['support']
            metadata = dict(instance_id=instance['id'], constraint=c,
                family=instance['family'], support=n, status='complete')
            gadget = VCG([c])
            result, depth_cache = [], {}
            for path in paths:
                entry = pickle.loads(path.read_bytes())
                restore_gadget(gadget, entry)
                if gadget.n_layers not in depth_cache:
                    depth_cache[gadget.n_layers] = re.estimate_vcg_resources(gadget)
                for gs, resources in depth_cache[gadget.n_layers].items():
                    result.append(_row(resources, **metadata, gate_set=gs,
                        representation='gadget', condition=path.parent.name,
                        selected_depth=gadget.n_layers, fidelity=entry['fidelity'],
                        n_decision=n, n_slack=0, n_allocated=n))
            parsed = ch.parse_constraints([c])
            _, n_slack = ch.determine_slack_variables(parsed, n)
            for gs, resources in re.estimate_constraint_penalty(c, n, 1.0, coalesce=True).items():
                result.append(_row(resources, **metadata, gate_set=gs,
                    representation='penalty_cost_layer', condition='unit_penalty',
                    n_decision=n, n_slack=n_slack, n_allocated=n+n_slack))
            return result
        rows.extend(cache.run(instance['id'], inputs, calculate))
    return _export(rows, cache.directory/'study_a_resources.csv')


def estimate_study_b(output, vcg_db_path, cache_dir=None):
    """B constraint components and full executed PenaltyQAOA circuits at depths 1--5.

    Missing VCGs are flagged explicitly. No PC-QAOA success comparison is made.
    """
    output, db_path = Path(output), Path(vcg_db_path)
    cache = EstimateCache(cache_dir or output/'resource_estimates')
    db = pickle.loads(db_path.read_bytes()) if db_path.exists() else {}
    manifest = json.loads((output/'manifest_b.json').read_text())
    rows = []
    for instance in manifest['instances']:
        paths = sorted((output/'b'/instance['id']).glob('*/result.json'))
        if not paths:
            continue
        inputs = dict(instance=instance, results={p.parent.name: _sha(p) for p in paths},
            db_hash=_sha(db_path) if db_path.exists() else None)

        def calculate(instance=instance, paths=paths):
            n = instance['n_x']
            metadata = dict(instance_id=instance['id'], n_decision=n,
                support=n, regime=instance['regime'])
            result = []
            for i, (c, family) in enumerate(zip(instance['constraints'], instance['families'])):
                parsed = ch.parse_constraints([c])
                _, n_slack = ch.determine_slack_variables(parsed, n)
                component = re.estimate_constraint_resources(c, n, 1.0, db, coalesce_penalty=True)
                for view in ['gadget', 'penalty']:
                    resources = component[view]
                    missing = resources.get('missing', False)
                    for gs in re.GATE_SETS:
                        fields = dict(**metadata, condition='constraint_component',
                            constraint=c, constraint_index=i, family=family,
                            representation='gadget' if view=='gadget' else 'penalty_cost_layer',
                            status='missing_vcg' if missing else 'complete',
                            gadget_kind=component['gadget']['kind'],
                            n_slack=n_slack if view=='penalty' else 0,
                            n_allocated=n+n_slack if view=='penalty' else len(parsed[0].variables),
                            gate_set=gs)
                        result.append(fields if missing else _row(resources[gs], **fields))
            # Topology is shared across the five dynamic penalty settings.
            model = PenaltyQAOA(np.asarray(instance['Q']), instance['constraints'],
                penalty=1.0, angle_strategy='QAOA', n_layers=1, compile_on_init=False)
            first = json.loads(paths[0].read_text())
            ham = coalesced_tuning_hamiltonian(model, first['delta'])
            coalesced = re.estimate_penalty_resources(model, circuit_hamiltonian=ham)
            expanded = re.estimate_penalty_resources(model)
            for path in paths:
                saved = json.loads(path.read_text())
                for gs in re.GATE_SETS:
                    for depth in range(1, int(saved['depth'])+1):
                        sections = coalesced[gs]
                        full = sections['sp'].add_series(sections['layer'].multiply_series(depth))
                        result.append(_row(re.resources_to_dict(full), **metadata,
                            gate_set=gs, condition=path.parent.name, delta=saved['delta'],
                            representation='full_penalty_qaoa', depth=depth,
                            status='complete', n_slack=model.n_slack, n_allocated=model.n_total,
                            prep_gates=sum(sections['sp'].gate_counts.values()),
                            layer_gates=sum(sections['layer'].gate_counts.values()),
                            expanded_layer_gates=sum(expanded[gs]['layer'].gate_counts.values())))
            return result
        rows.extend(cache.run(instance['id'], inputs, calculate))
    return _export(rows, cache.directory/'study_b_resources.csv')


def _export(rows, path):
    frame = pd.DataFrame(rows)
    atomic_bytes(path, frame.to_csv(index=False).encode())
    return frame


def resource_figures(frame, study, writer, selected_loss='fidelity'):
    """Study-separated figures with pointwise 95% CIs across instances."""
    import matplotlib.pyplot as plt
    if frame.empty:
        return []
    frame = frame[frame.status == 'complete']
    paths = []
    if study == 'a':
        gadgets = frame[frame.representation == 'gadget']
        for gs in re.GATE_SETS:
            f = gadgets[gadgets.gate_set == gs]
            metric = 'two_qubit_gates' if gs == 'nisq' else 't_gates'
            label = 'CNOT Gates' if gs == 'nisq' else 'T Gates'
            fig = summary_plot(f, 'support', metric, 'condition',
                f'Study A: {gs.upper()} Gadget {label}', label)
            paths.append(writer.save(fig, f'a_gadget_{gs}'))
            selected = frame[(frame.gate_set == gs) &
                ((frame.condition == selected_loss) | (frame.representation == 'penalty_cost_layer'))].copy()
            selected['view'] = selected.representation.map({
                'gadget': 'Selected VCG Preparation', 'penalty_cost_layer': 'One Penalty Cost Layer'})
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
            for ax, metric, label in zip(axes, ['total_gates', 'n_allocated'], ['Total Gates', 'Allocated Qubits']):
                for idx, (view, subset) in enumerate(selected.groupby('view', sort=False)):
                    summary = mean_intervals(subset.groupby('support')[metric].agg(['mean','std','count']))
                    ax.errorbar(summary.index, summary['mean'], yerr=summary.ci95_half_width,
                        marker=['o','s'][idx], color=['#3e8fb0','#eb6f92'][idx], label=view)
                ax.set(xlabel='Constraint Support Size', ylabel=label, title=label)
                ax.set_xticks(sorted(selected.support.unique()))
            axes[0].legend(fontsize=9, frameon=False)
            fig.suptitle(f'Study A: {gs.upper()} Constraint Components')
            paths.append(writer.save(fig, f'a_components_{gs}'))
    else:
        full = frame[frame.representation == 'full_penalty_qaoa']
        # Each COP contributes once: all settings use the same traced topology.
        full = full[full.condition == 'range']
        for gs in re.GATE_SETS:
            f = full[(full.gate_set == gs) & (full.depth == full.depth.max())]
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
            for ax, metric, label in zip(axes, ['total_gates', 'two_qubit_gates' if gs=='nisq' else 't_gates'],
                ['Total Gates', 'CNOT Gates' if gs=='nisq' else 'T Gates']):
                for idx, (regime, subset) in enumerate(f.groupby('regime')):
                    ax.scatter(subset.support, subset[metric], marker=['o','s'][idx],
                        color=['#3e8fb0','#31748f'][idx], label=regime.title(), alpha=.8)
                summary = mean_intervals(f.groupby('support')[metric].agg(['mean','std','count']))
                ax.errorbar(summary.index, summary['mean'], yerr=summary.ci95_half_width,
                    color='#9063cd', marker='D', linestyle='--', label='Mean (95% CI)')
                ax.set(xlabel='Decision Qubits', ylabel=label, title=label)
                ax.set_xticks(sorted(f.support.unique()))
            axes[0].legend(fontsize=9, frameon=False)
            fig.suptitle(f'Study B: {gs.upper()} PenaltyQAOA At Depth {int(full.depth.max())}')
            paths.append(writer.save(fig, f'b_circuit_{gs}'))
        f = full[(full.gate_set=='nisq') & (full.depth==full.depth.max())]
        fig, ax = plt.subplots(figsize=(6.4, 3.7))
        ax.scatter(f.expanded_layer_gates, f.layer_gates, color='#9063cd', marker='o')
        lo, hi = 0, max(f.expanded_layer_gates.max(), f.layer_gates.max())
        ax.plot([lo,hi],[lo,hi], '--', color='#6e6a86', label='Equal Counts')
        ax.set(xlabel='Expanded Penalty Layer Gates', ylabel='Executed Merged Layer Gates',
            title='Study B: Pauli-Term Merging')
        ax.legend(frameon=False)
        paths.append(writer.save(fig, 'b_resource_merging'))
    return paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=['a','b','both'], default='both')
    parser.add_argument('--a-output', type=Path, default=ROOT/'tuning/results/full')
    parser.add_argument('--b-output', type=Path, default=ROOT/'tuning/results/full-b-five-methods')
    parser.add_argument('--vcg-db', type=Path, default=ROOT/'tuning/artifacts/selected_vcg_db.pkl')
    args = parser.parse_args()
    for study in ['a','b']:
        if args.study not in [study,'both']:
            continue
        frame = estimate_study_a(args.a_output) if study=='a' else estimate_study_b(args.b_output, args.vcg_db)
        print(f'Study {study.upper()}: {len(frame)} resource rows; statuses {frame.status.value_counts().to_dict()}', flush=True)
