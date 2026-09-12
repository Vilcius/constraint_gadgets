"""Compare resource traces with executable circuits, without training."""
from pathlib import Path
import json
import pickle
from collections import Counter

import numpy as np
import pennylane as qml
import pennylane.estimator as qre
import pytest

from core import resource_estimation as re
from core import constraint_handler as ch, qaoa_base as base
from core.vcg import VCG
from core.penalty_qaoa import PenaltyQAOA
from resource_analysis import (restore_gadget, coalesced_tuning_hamiltonian,
    estimate_study_a, estimate_study_b, EstimateCache)


def _actual_cost_counts(operations):
    """Independent decomposition of the emitted MultiRZ and single-qubit gates."""
    counts = Counter()
    for op in operations:
        if op.name == 'MultiRZ':
            counts['RZ'] += 1
            counts['CNOT'] += 2 * (len(op.wires)-1)
        else:
            counts[op.name] += 1
    return {g: n for g, n in counts.items() if n}


def test_resources_dict_works_without_convenience_properties():
    class Resource:
        gate_counts = {'CNOT': 2, 'RZ': 1}
        algo_wires = 3
        zeroed_wires = 1
        any_state_wires = 2
    saved = re.resources_to_dict(Resource())
    assert saved['total_gates'] == 3
    assert saved['total_wires'] == 6


def test_saved_vcg_matches_actual_circuit_and_refuses_bad_layout(monkeypatch):
    monkeypatch.setattr(VCG, 'train', lambda *a, **k: pytest.fail('Retraining'))
    gadget = VCG(['2*x_0 + 3*x_1 + x_2 <= 3'])
    entry = dict(opt_angles=np.full((2, len(gadget.constraint_Ham.ops)+3), .31),
        n_layers=2, parameter_layout='legacy_pauli_slots_then_rx')
    restore_gadget(gadget, entry)
    with qml.queuing.AnnotatedQueue() as queue:
        gadget.opt_circuit()
    counts = _actual_cost_counts(qml.tape.QuantumScript.from_queue(queue).operations)
    result = re.estimate_vcg_resources(gadget)
    assert result['nisq']['gate_counts'] == counts
    assert set(result['ftqc']['gate_counts']) <= re.FTQC_GATE_SET
    with pytest.raises(ValueError, match='layout'):
        restore_gadget(gadget, dict(entry, parameter_layout='wrong'))
    with pytest.raises(ValueError, match='shape'):
        restore_gadget(gadget, dict(entry, opt_angles=np.zeros((2,2))))


def test_penalty_resources_include_decision_hadamards_and_match_executed_layer():
    model = PenaltyQAOA(np.array([[2,1],[0,-1]]), ['x_0 + 2*x_1 <= 1'],
        penalty=3, angle_strategy='QAOA', n_layers=3, compile_on_init=False)
    ham = coalesced_tuning_hamiltonian(model, 3)
    with qml.queuing.AnnotatedQueue() as queue:
        model.qaoa_circuit(np.full((3,2), .23), hamiltonian=ham)
    actual = _actual_cost_counts(qml.tape.QuantumScript.from_queue(queue).operations)
    estimate = re.estimate_penalty_resources(model, circuit_hamiltonian=ham)
    assert estimate['nisq']['sp'].gate_counts == {'Hadamard': model.n_total}
    assert re.resources_to_dict(estimate['nisq']['full'])['gate_counts'] == actual
    # The reconstructed Hamiltonian agrees with classical squared residuals.
    raw = qml.matrix(model.full_Ham, wire_order=model.all_wires)
    merged = qml.matrix(ham, wire_order=model.all_wires)
    np.testing.assert_allclose(raw, merged)


def test_task_and_instance_penalty_resources_agree():
    Q = np.array([[2,1],[0,-1]])
    c = ['x_0 + 2*x_1 <= 1']
    model = PenaltyQAOA(Q, c, penalty=3, n_layers=2, compile_on_init=False)
    task = re.estimate_from_task(dict(n_x=2, qubo_idx=0, constraints=c),
        {2:[{'Q':Q}]}, n_layers=2, penalty_weight=3)
    direct = re.estimate_penalty_resources(model)
    for gs in re.GATE_SETS:
        assert re.resources_to_dict(task[gs]['penalty']) == re.resources_to_dict(direct[gs]['full'])
    assert task['vcg_missing_pc']


def test_xy_mirror_matches_actual_mixer():
    actual = qre.estimate(lambda: base.apply_xy_mixer(.31,[0,1,2]),
        gate_set=re.NISQ_GATE_SET)()
    mirror = qre.estimate(lambda: re._qre_xy_mixer([0,1,2]),
        gate_set=re.NISQ_GATE_SET)()
    assert mirror.gate_counts == actual.gate_counts


def test_missing_gadget_is_explicit_and_does_not_train(monkeypatch):
    monkeypatch.setattr(VCG,'train',lambda *a,**k: pytest.fail('Retraining'))
    out = re.estimate_constraint_resources('2*x_0 + 3*x_1 + x_2 <= 3',3,1,{})
    assert out['gadget'] == {'kind':'vcg','missing':True}
    assert all(out['penalty'][gs]['total_gates'] > 0 for gs in re.GATE_SETS)


def test_cache_reuses_and_invalidates(tmp_path):
    cache = EstimateCache(tmp_path)
    seen = []
    def compute():
        seen.append(1)
        return [{'total_gates':len(seen)}]
    assert cache.run('instance', {'angles':'a'}, compute)[0]['total_gates'] == 1
    assert cache.run('instance', {'angles':'a'}, compute)[0]['total_gates'] == 1
    assert cache.run('instance', {'angles':'b'}, compute)[0]['total_gates'] == 2
    cache.policy['schema'] += 1
    assert cache.run('instance', {'angles':'b'}, compute)[0]['total_gates'] == 3


def test_small_saved_analysis_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(VCG,'train',lambda *a,**k: pytest.fail('Retraining'))
    c = '2*x_0 + 3*x_1 + x_2 <= 3'
    g = VCG([c])
    entry = dict(opt_angles=np.full((1,len(g.constraint_Ham.ops)+3),.31),
        n_layers=1, fidelity=.9, parameter_layout='legacy_pauli_slots_then_rx')
    (tmp_path/'manifest_a.json').write_text(json.dumps({'instances':[
        dict(id='a',constraint=c,support=3,family='positive')]}))
    p = tmp_path/'a/a/fidelity'; p.mkdir(parents=True)
    (p/'gadget.pkl').write_bytes(pickle.dumps(entry))
    a = estimate_study_a(tmp_path)
    assert len(a)==4
    cached = estimate_study_a(tmp_path)
    assert a.to_json()==cached.to_json()
    b = dict(id='b',n_x=3,Q=np.eye(3,dtype=int).tolist(),
        constraints=[c], families=['knapsack'], regime='disjoint')
    (tmp_path/'manifest_b.json').write_text(json.dumps({'instances':[b]}))
    p = tmp_path/'b/b/range'; p.mkdir(parents=True)
    (p/'result.json').write_text(json.dumps(dict(delta=3,depth=2)))
    out = estimate_study_b(tmp_path,tmp_path/'missing.pkl')
    assert len(out[out.status=='missing_vcg'])==2
    assert len(out[out.representation=='full_penalty_qaoa'])==4
    for gs in re.GATE_SETS:
        f=out[(out.gate_set==gs)&(out.representation=='full_penalty_qaoa')]
        np.testing.assert_array_equal(f.total_gates, f.prep_gates+f.depth*f.layer_gates)


def test_isolated_penalty_coalescing_preserves_energy_and_reduces_counts():
    c = 'x_0*x_1 + 2*x_1*x_2 + x_2 <= 2'
    parsed = ch.parse_constraints([c])
    slack, _ = ch.determine_slack_variables(parsed,3)
    raw = base.build_penalty_hamiltonian(parsed,slack,1)
    simple = raw.simplify()
    words = list(raw.wires)
    np.testing.assert_allclose(qml.matrix(raw,wire_order=words),qml.matrix(simple,wire_order=words))
    expanded = re.estimate_constraint_penalty(c,3,1)
    merged = re.estimate_constraint_penalty(c,3,1,coalesce=True)
    assert merged['nisq']['total_gates'] < expanded['nisq']['total_gates']


def test_legacy_vcg_resource_script_accepts_tuning_database(tmp_path):
    from analyze_results.compute_vcg_resources import build_vcg_resource_table
    c='2*x_0 + 3*x_1 + x_2 <= 3'
    g=VCG([c])
    entry=dict(constraints=[c],support=3,n_layers=2,
        opt_angles=np.full((2,len(g.constraint_Ham.ops)+3),.31),
        single_feasible_bitstring=None,dicke_superposition_weights=None)
    p=tmp_path/'db.pkl';p.write_bytes(pickle.dumps({c:entry}))
    table=build_vcg_resource_table(str(p),output_prefix=None)
    assert len(table)==1
    restore_gadget(g,entry)
    assert table.iloc[0].sp_gates==re.estimate_vcg_resources(g)['nisq']['gate_counts']
