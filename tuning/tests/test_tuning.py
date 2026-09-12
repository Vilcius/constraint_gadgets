import copy
import json
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pennylane as qml
import pytest

from core.vcg import VCG, state_quality, loss_from_state
from core.penalty_qaoa import PenaltyQAOA
from core import qaoa_base as base, constraint_handler as ch
from core.optimization import make_adam_functions, optimize_resumable, TrainingInterrupted
from core.penalty_heuristics import coefficient_penalties
from datasets import generate_a, generate_b, objective_reference, exact_case, feasible_mask
from storage import TaskStore, ROOT, ensure_run
from studies import probability_metrics, settings_b
from analyze import select_loss, select_penalty, FigureWriter


@pytest.fixture
def config():
    return json.loads((ROOT/'tuning/config.json').read_text())


def test_fidelity_distinguishes_phase_bias_and_infeasibility():
    mask = np.array([True, True, False, False])
    uniform = np.array([1, 1, 0, 0])/np.sqrt(2)
    flipped = np.array([1, -1, 0, 0])/np.sqrt(2)
    np.testing.assert_allclose(state_quality(uniform, mask), [1, 1, 1], atol=1e-12)
    np.testing.assert_allclose(state_quality(flipped, mask), [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(state_quality([1, 0, 0, 0], mask), [1, .5, .5])
    np.testing.assert_allclose(state_quality([0, 0, 1, 0], mask), [0, 0, 0])
    assert np.isfinite(float(loss_from_state([1e-9, 0, np.sqrt(1-1e-18), 0], mask, 2, 2.)))


def test_vcg_circuit_and_gradients_match_existing_conventions():
    gadget = VCG(['2*x_0 + 3*x_1 + x_2 <= 3'])
    angles = jnp.asarray([[0.23, -0.6]])
    objective, state = gadget.tuning_functions('QAOA', 1)
    @qml.qnode(qml.device('default.qubit', wires=3), interface='jax')
    def reference(params):
        gadget._circuit(params, 'QAOA', 1)
        return qml.expval(gadget.constraint_Ham)
    np.testing.assert_allclose(objective(angles, 0, 1.), reference(angles), atol=1e-12)
    np.testing.assert_allclose(jax.grad(objective)(angles, 0, 1.), jax.grad(reference)(angles), atol=1e-10)
    gradient = np.asarray(jax.grad(objective)(angles, 1, 1.))
    for i in range(2):
        shift = np.zeros((1, 2)); shift[0, i] = 1e-6
        numerical = (objective(angles+shift, 1, 1.)-objective(angles-shift, 1, 1.))/2e-6
        np.testing.assert_allclose(gradient[0, i], numerical, atol=1e-7)
    count = len(gadget.constraint_Ham.ops)
    converted = base.convert_qaoa_to_ma_angles(angles, count, 3, 1)
    _, ma_state = gadget.tuning_functions('ma-QAOA', 1)
    np.testing.assert_allclose(state(angles), ma_state(converted), atol=1e-12)
    _, deeper = gadget.tuning_functions('ma-QAOA', 2)
    np.testing.assert_allclose(ma_state(converted), deeper(np.vstack([converted, np.zeros_like(converted)])), atol=1e-12)


def test_dynamic_penalty_circuit_and_gradient():
    Q = np.array([[-1, 2, 0], [0, -2, 1], [0, 0, 1]])
    constraints = ['x_0*x_1 + x_2 <= 1', 'x_0 + x_1 >= 1']
    angles = jnp.asarray([[.2, -.4], [.1, .3]])
    dynamic = PenaltyQAOA(Q, constraints, n_layers=2, angle_strategy='QAOA', compile_on_init=False)
    expectation, probs = dynamic.tuning_functions()
    for delta in [.5, 1., 2.]:
        fixed = PenaltyQAOA(Q, constraints, penalty=delta, n_layers=2,
                            angle_strategy='QAOA', compile_on_init=False)
        @qml.qnode(qml.device('default.qubit', wires=fixed.all_wires), interface='jax')
        def reference(params):
            fixed.qaoa_circuit(params)
            return qml.expval(fixed.full_Ham)
        @qml.qnode(qml.device('default.qubit', wires=fixed.all_wires), interface='jax')
        def reference_probs(params):
            fixed.qaoa_circuit(params)
            return qml.probs(wires=fixed.x_wires)
        np.testing.assert_allclose(expectation(angles, delta), reference(angles), atol=1e-10)
        np.testing.assert_allclose(probs(angles, delta), reference_probs(angles), atol=1e-10)
        np.testing.assert_allclose(jax.grad(expectation)(angles, delta), jax.grad(reference)(angles), atol=1e-9)


def test_penalty_diagonal_matches_classical_residuals():
    Q = np.array([[-2, 1, 2], [0, 1, -1], [0, 0, -1]])
    constraints = ['2*x_0*x_0 + x_0*x_1 + x_2 <= 2', 'x_1 + x_2 >= 1']
    solver = PenaltyQAOA(Q, constraints, penalty=2., compile_on_init=False)
    diagonal = np.real(np.diag(qml.matrix(solver.full_Ham, wire_order=solver.all_wires)))
    for index, actual in enumerate(diagonal):
        bits = format(index, f'0{solver.n_total}b')
        x = np.array([int(b) for b in bits[:3]])
        penalty = 0.
        for constraint, slack in zip(solver.parsed_constraints, solver.slack_info):
            value = ch.evaluate_lhs(constraint, bits[:3])
            slack_value = sum(int(bits[slack.slack_start_wire+j]) * 2**j for j in range(slack.n_slack))
            residual = value - slack.effective_rhs + (slack_value if slack.operator == 'leq' else -slack_value)
            penalty += residual**2
        np.testing.assert_allclose(actual, x@Q@x+2*penalty, atol=1e-10)


def test_optimizer_resume_final_loss_and_rng():
    objective = lambda x, target: jnp.sum((x-target)**2)
    cost, step = make_adam_functions(objective, .05)
    kwargs = dict(shape=(1, 2), steps=7, restarts=2, learning_rate=.05,
                  seed=9, args=(jnp.asarray(.3),), checkpoint_steps=1)
    complete = optimize_resumable(cost, step, **kwargs)
    saved = []
    def interrupt(state):
        if state['restart'] == 0 and state['step'] == 3:
            saved.append(jax.tree.map(lambda x: np.asarray(x) if isinstance(x, jax.Array) else x, state))
            raise TrainingInterrupted()
    with pytest.raises(TrainingInterrupted):
        optimize_resumable(cost, step, checkpoint=interrupt, **kwargs)
    resumed = optimize_resumable(cost, step, state=saved[0], **kwargs)
    np.testing.assert_array_equal(complete['best_angles'], resumed['best_angles'])
    np.testing.assert_array_equal(complete['key'], resumed['key'])
    assert complete['best_loss'] == resumed['best_loss']
    assert complete['history'] == resumed['history']
    assert complete['best_loss'] == float(cost(complete['best_angles'], *kwargs['args']))


def test_checkpoint_recovery_and_identity(tmp_path):
    store = TaskStore(tmp_path/'one', 'a')
    store.save({'value': 1})
    store.save({'value': 2})
    store.path.write_bytes(b'corrupt')
    with pytest.warns(UserWarning):
        assert store.load() == {'value': 1}
    store.save({'value': 3})
    assert store.load() == {'value': 3}
    with pytest.raises(ValueError, match='does not match'):
        TaskStore(tmp_path/'one', 'b').load()
    other = TaskStore(tmp_path/'two', 'a')
    other.save({'value': 4})
    assert store.load()['value'] == 3 and other.load()['value'] == 4


def test_run_rejects_changed_configuration(tmp_path, config):
    ensure_run(tmp_path, config)
    config['training_seed'] += 1
    with pytest.raises(ValueError, match='changed'):
        ensure_run(tmp_path, config)


def test_generation_quotas_and_existing_coefficient_conventions(config):
    rows = generate_a(config)
    assert len(rows) == 50
    for n in range(4, 9):
        subset = [r for r in rows if r['support'] == n]
        assert len(subset) == 10
        assert sorted(Counter(r['family'] for r in subset).values()) == [2, 2, 3, 3]
    for row in rows:
        pc = ch.parse_constraint(row['constraint'])
        assert not exact_case(row['constraint'], np.array(row['feasible_mask']), row['support'])
        coefficients = list(pc.quadratic.values()) if row['family'] == 'quadratic' else list(pc.linear.values())
        assert all(1 <= abs(c) <= (3 if row['family'] == 'quadratic' else 5) for c in coefficients)
        if row['family'] == 'quadratic':
            assert len(pc.quadratic) == row['support']*(row['support']+1)//2
    cops = generate_b(config)
    assert len(cops) == 20
    for n in range(4, 9):
        assert Counter(r['regime'] for r in cops if r['n_x'] == n) == {'disjoint': 2, 'overlapping': 2}
    for row in cops:
        Q = np.array(row['Q'])
        assert np.all(Q == np.triu(Q)) and Q.min() >= -5 and Q.max() <= 4
        assert row['optimal_indices'] and 2 <= len(row['constraints']) <= 3
        assert row['n_structural'] + row['n_penalty'] == len(row['constraints'])


from collections import Counter


def test_all_optima_and_penalty_scales():
    reference = objective_reference(np.zeros((2, 2)), ['x_0 + x_1 == 1'])
    result = probability_metrics([.1, .2, .3, .4], reference)
    assert result['p_optimal'] == .5 and result['conditional_quality'] == 1
    Q = np.array([[-2., 3.], [0, 4.]])
    rules = coefficient_penalties(Q, ch.parse_constraints(['x_0 + x_1 <= 1']))
    assert rules == {'range': 10., 'maximum': 5., 'local': 8.}


def test_five_penalties_require_no_optimum(config):
    from core.penalty_heuristics import practical_penalties
    from datasets import find_feasible_point
    constraints = ['x_0 + x_1 == 1']
    Q = np.array([[-2., -3.], [0., 4.]])
    parsed = ch.parse_constraints(constraints)
    rules = practical_penalties(Q, parsed, '01')
    assert rules == {'range': 10., 'feasible': 10., 'verma_lewis': 6.,
                     'local': 8., 'maximum': 5.}
    # Symmetric and upper-triangular matrices describe the same polynomial.
    np.testing.assert_equal(rules, practical_penalties((Q+Q.T)/2, parsed, '01'))
    point = find_feasible_point(constraints, 2, 101)
    repeat = find_feasible_point(constraints, 2, 101)
    assert point['feasible_point'] == repeat['feasible_point']
    assert point['feasibility_search_attempts'] == repeat['feasibility_search_attempts']
    assert point['feasibility_search_seconds'] >= 0
    instance = dict(Q=Q, constraints=constraints, **point)
    settings = settings_b(instance, config)  # No optimum/reference fields supplied.
    assert [s['id'] for s in settings] == config['study_b']['methods']
    assert len(settings) == 5 and all(s['selectable'] for s in settings)
    assert all('multiplier' not in s for s in settings)
    with pytest.raises(ValueError, match='violates'):
        practical_penalties(Q, parsed, '00')
    changed = copy.deepcopy(config)
    changed['study_b']['multipliers'] = [1.]
    with pytest.raises(ValueError, match='without multipliers'):
        settings_b(instance, changed)


@pytest.mark.parametrize('winner', ['range', 'feasible', 'verma_lewis', 'local', 'maximum'])
def test_all_five_penalty_methods_selectable(config, winner):
    names = config['study_b']['methods']
    rows = pd.DataFrame([dict(instance_id=i, condition=c, depth=5, selectable=True,
                             p_optimal=.9 if c == winner else .2,
                             exact_p_optimal=.1 if c == winner else .99)
                         for i in ['a', 'b'] for c in names])
    assert select_penalty(rows, ['a', 'b'], names, 5) == winner
    assert select_penalty(rows.iloc[:-1], ['a', 'b'], names, 5) is None


@pytest.mark.parametrize('winner', ['feasibility', 'fidelity', 'lambda_0.5', 'lambda_1', 'lambda_2'])
def test_all_losses_eligible(config, winner):
    conditions = config['study_a']['losses']
    rows = pd.DataFrame([dict(instance_id=i, condition=c, fidelity=.9 if c == winner else .8)
                         for i in ['x', 'y'] for c in conditions])
    assert select_loss(rows, ['x', 'y'], conditions) == winner
    assert select_loss(rows.iloc[:-1], ['x', 'y'], conditions) is None


def test_penalty_selection_uses_sampled_final_depth():
    frame = pd.DataFrame([dict(instance_id=i, condition=c, depth=5,
                              selectable=c != 'paper', p_optimal=p, exact_p_optimal=e)
                          for i in ['x', 'y'] for c, p, e in
                          [('range_1', .8, .4), ('local_1', .7, .9), ('paper', 1., 1.)]])
    assert select_penalty(frame, ['x', 'y'], ['range_1', 'local_1', 'paper'], 5) == 'range_1'
    assert select_penalty(frame, ['x', 'y'], ['range_1', 'local_1', 'paper'], 4) is None


def test_exported_layout_loads_without_training(tmp_path, monkeypatch):
    from core.pc_qaoa import _load_vcg_gadget
    constraint = '2*x_0 + 3*x_1 + x_2 <= 3'
    gadget = VCG([constraint])
    angles = np.zeros((1, len(gadget.constraint_Ham.ops)+3))
    entry = dict(opt_angles=angles, n_layers=1, ar=.5, entropy=.8,
                 single_feasible_bitstring=None, dicke_superposition_weights=None)
    path = tmp_path/'db.pkl'
    path.write_bytes(pickle.dumps({ch.normalize_constraint(constraint): entry}))
    monkeypatch.setattr(VCG, 'train', lambda *a, **k: pytest.fail('Unexpected retraining'))
    loaded = _load_vcg_gadget([constraint], db_path=str(path))
    np.testing.assert_array_equal(loaded.opt_angles, angles)


def test_builtin_font_fallback(tmp_path):
    import matplotlib.pyplot as plt
    writer = FigureWriter(tmp_path, latex=False)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1]); ax.set_ylabel(r'$P(\mathrm{opt})$')
    png = writer.save(fig, 'fallback')
    assert png.exists() and (tmp_path/'fallback.pdf').exists()
    assert 'Matplotlib' in json.loads((tmp_path/'font_status.json').read_text())['font']


def test_latex_failure_uses_builtin_fonts(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    writer = FigureWriter(tmp_path)
    writer.latex = True
    fig, ax = plt.subplots()
    ax.set_ylabel(r'$F_{\mathrm{cond}}$')
    save = fig.savefig
    def fail_pgf(*args, **kwargs):
        if kwargs.get('backend') == 'pgf':
            raise RuntimeError('Simulated missing LaTeX package')
        return save(*args, **kwargs)
    monkeypatch.setattr(fig, 'savefig', fail_pgf)
    with pytest.warns(UserWarning, match='using built-in'):
        writer.save(fig, 'latex_failure')
    assert not writer.latex and (tmp_path/'latex_failure.png').exists()


def test_study_resume_preserves_angles_and_sampled_counts(tmp_path, config):
    from studies import run_b_instance
    config['checkpoint_steps'] = 1
    config['study_b'].update(max_depth=2, restarts=2, steps=3)
    Q, constraints = np.array([[-1, 2], [0, -2]]), ['x_0 + x_1 <= 1']
    instance = dict(id='test', n_x=2, regime='overlapping', Q=Q.tolist(), constraints=constraints, feasible_point='00',
                    **objective_reference(Q, constraints))
    run_b_instance(instance, config, tmp_path/'normal', 'r', 'm', lambda: False)
    counter = [0]
    def stop():
        counter[0] += 1
        return counter[0] >= 5
    with pytest.raises(TrainingInterrupted):
        run_b_instance(instance, config, tmp_path/'resumed', 'r', 'm', stop)
    run_b_instance(instance, config, tmp_path/'resumed', 'r', 'm', lambda: False)
    for setting in settings_b(instance, config):
        def records(root):
            return json.loads((tmp_path/root/'b/test'/setting['id']/'depths.json').read_text())
        for uninterrupted, resumed in zip(records('normal'), records('resumed')):
            assert uninterrupted['counts'] == resumed['counts']
            assert uninterrupted['angles'] == resumed['angles']
            assert uninterrupted['loss'] == resumed['loss']


def test_remapped_quadratic_penalty_has_real_gradient():
    constraint = 'x_0*x_0 + 2*x_0*x_2 + 3*x_2*x_1 + 2*x_1*x_1 <= 3'
    solver = PenaltyQAOA(np.diag([-1, -2, -3]), [constraint],
                         angle_strategy='QAOA', compile_on_init=False)
    objective, _ = solver.tuning_functions()
    angles = jnp.asarray([[.1, .2]])
    assert jnp.isrealobj(objective(angles, 2.))
    assert np.isfinite(jax.grad(objective)(angles, 2.)).all()


def test_inherited_layer_initialization_matches_each_original_trainer():
    cost, step = make_adam_functions(lambda x: jnp.sum(x*x), .05)
    previous = jnp.asarray([[.2, -.3]])
    for zero_pad in [True, False]:
        initial = []
        def capture(state):
            if state['angles'] is not None and state['step'] == 0:
                initial.append(np.asarray(state['angles']).copy())
        optimize_resumable(cost, step, shape=(2, 2), steps=0, restarts=2,
                           learning_rate=.05, seed=17, previous=previous,
                           zero_pad_first=zero_pad, checkpoint=capture)
        key = jax.random.PRNGKey(17)
        for restart, angles in enumerate(initial):
            key, subkey = jax.random.split(key)
            expected = (jnp.zeros(2) if zero_pad and restart == 0 else
                        jax.random.uniform(subkey, (2,), minval=-2*jnp.pi, maxval=2*jnp.pi))
            np.testing.assert_array_equal(angles[0], previous[0])
            np.testing.assert_array_equal(angles[1], expected)


def test_coalesced_remapped_quadratic_matches_original_circuit():
    constraints = ['x_0*x_0 + 2*x_0*x_2 + 3*x_2*x_1 + 2*x_1*x_1 <= 3']
    solver = PenaltyQAOA(np.diag([-1, -2, -3]), constraints, penalty=2.,
                         n_layers=2, angle_strategy='QAOA', compile_on_init=False)
    objective, probabilities = solver.tuning_functions()
    angles = jnp.asarray([[.1, .2], [.3, -.1]])
    @qml.qnode(qml.device('default.qubit', wires=solver.all_wires), interface='jax')
    def reference(params):
        solver.qaoa_circuit(params)
        return qml.expval(solver.full_Ham), qml.probs(wires=solver.x_wires)
    value, probs = reference(angles)
    np.testing.assert_allclose(objective(angles, 2.), value, atol=1e-10)
    np.testing.assert_allclose(probabilities(angles, 2.), probs, atol=1e-10)
    np.testing.assert_allclose(jax.grad(objective)(angles, 2.),
                              jax.grad(lambda p: jnp.real(reference(p)[0]))(angles), atol=1e-9)


def test_run_lock_and_concurrent_task_files(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import importlib.util
    spec = importlib.util.spec_from_file_location('tuning_cli', ROOT/'tuning/run.py')
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    output_lock = cli.output_lock
    with output_lock(tmp_path):
        with pytest.raises(RuntimeError, match='Another process'):
            with output_lock(tmp_path):
                pytest.fail('Concurrent run acquired the same output directory')
    def write_task(index):
        store = TaskStore(tmp_path/str(index), str(index))
        for step in range(4):
            store.save({'owner': index, 'step': step})
        return store.load()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(write_task, range(4)))
    assert results == [{'owner': index, 'step': 3} for index in range(4)]
