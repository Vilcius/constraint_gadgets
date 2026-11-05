"""Uncompiled versions of the same four circuits and seeded Adam procedure."""
import json
from pathlib import Path
import resource
import time
import traceback
import jax
import numpy as np
import pennylane as qml
from core import qaoa_base as base
from core.penalty_qaoa import PenaltyQAOA
from core.qaoa_plus import QAOAPlus
from core.optimization import optimize_resumable, TrainingInterrupted
from final_experiment.random_cops.gate_budget.simulator import Simulator as PCSimulator
from final_experiment.random_cops import studies as original
from final_experiment.random_cops.storage import TaskStore, digest, write_json


class Simulator(PCSimulator):
    def __init__(self, instance, method, database):
        if method in ('pcqaoa','pcqaoa_stateprep'):
            super().__init__(instance,method,database)
            return
        if method=='penaltyqaoa':
            self.model=PenaltyQAOA(np.asarray(instance['Q']),instance['constraints'],
                penalty=1.0,angle_strategy='QAOA',compile_on_init=False)
            # Match the existing runner's dynamic-penalty tuning_functions,
            # including its coalesced coefficient construction and wire order.
            unit=base.build_penalty_hamiltonian(self.model.parsed_constraints,
                self.model.slack_info,1.0,fallback_wire=self.model.x_wires[0])
            qr,pr=self.model.qubo_Ham.pauli_rep,unit.pauli_rep
            words=list(dict.fromkeys([*qr,*pr]))
            words=[w for w in words if qr.get(w,0)!=0 or pr.get(w,0)!=0]
            ham=qml.Hamiltonian([float(np.real(qr.get(w,0)))+float(instance['delta'])*float(np.real(pr.get(w,0)))
                                 for w in words],
                                [w.operation(wire_order=self.model.all_wires) for w in words])
            self.model.full_Ham=ham
        elif method=='qaoa_plus':
            self.model=QAOAPlus(np.asarray(instance['Q']),instance['constraints'],
                angle_strategy='QAOA',feasible_states=instance['feasible_bitstrings'],compile_on_init=False)
            self.model.n_slack=0
            ham=self.model.problem_ham
        else:raise ValueError(method)
        self.n=self.model.n_total
        indices=np.arange(1<<self.n)
        self.energy=np.zeros(len(indices));self.phase_energy=np.zeros(len(indices))
        for word,coefficient in ham.pauli_rep.items():
            diagonal=np.ones(len(indices))
            for wire,pauli in word.items():
                if pauli!='Z':raise ValueError('Non-diagonal Hamiltonian')
                diagonal*=1-2*((indices>>(self.n-1-wire))&1)
            contribution=float(np.real(coefficient))*diagonal
            self.energy+=contribution
            if word:self.phase_energy+=contribution
        self.operations=[]
        if method=='penaltyqaoa':
            self.initial=np.ones(len(indices),complex)/np.sqrt(len(indices))
            for wire in self.model.all_wires:
                mask=1<<(self.n-1-wire);low=indices[(indices&mask)==0]
                self.operations.append(('rx',low,low^mask))
        else:
            self.initial=np.zeros(len(indices),complex)
            self.initial[int(self.model.initial_state,2)]=1
            for move in self.model.mixer_moves:
                if move['kind']=='bridge':
                    low=np.array([int(move['states'][0],2)]);high=np.array([int(move['states'][1],2)])
                    self.operations.append(('rx',low,high));continue
                wires=move['wires'];a=1<<(self.n-1-wires[0])
                for controls,values in move['clauses']:
                    selected=(indices&a)==0
                    for wire,value in zip(controls,values):
                        selected&=((indices>>(self.n-1-wire))&1)==value
                    mask=a
                    if move['kind']=='swap':
                        b=1<<(self.n-1-wires[1]);selected&=(indices&b)!=0;mask|=b
                    low=indices[selected];self.operations.append(('rx',low,low^mask))


def numpy_step(simulator,learning_rate):
    """Optax Adam's update with NumPy arithmetic, preserving its state layout."""
    def step(angles,state,*args):
        angles=np.asarray(angles,dtype=float)
        loss,gradient=simulator.loss_and_gradient(angles)
        adam,empty=state;count=int(adam.count)+1
        mu=.9*np.asarray(adam.mu)+.1*gradient
        nu=.999*np.asarray(adam.nu)+.001*gradient**2
        update=(mu/(1-.9**count))/(np.sqrt(nu/(1-.999**count))+1e-8)
        return angles-learning_rate*update,(type(adam)(np.int32(count),mu,nu),empty),loss
    return step


def run_instance(method,instance,config,directory,run_id,manifest_hash,should_stop=lambda:False):
    identity=digest([run_id,manifest_hash,method,instance['id']])
    store=TaskStore(Path(directory)/instance['id'],identity)
    path=store.directory/'result.json'
    if path.exists():
        if json.loads(path.read_text())['identity']!=identity:raise ValueError('Result identity mismatch')
        return
    task=store.load() or {'depths':[],'optimizer':None}
    start=time.perf_counter()
    database=Path(config['vcg_database'])
    if not database.is_absolute():database=original.ROOT/database
    sim=Simulator(instance,method,database)
    task['prepare_seconds']=task.get('prepare_seconds',0.)+time.perf_counter()-start
    step=numpy_step(sim,config['learning_rate'])
    def cost(angles,*args):
        state=sim.state(angles)
        return float(np.vdot(state,sim.energy*state).real)
    try:
        # This context also prevents PRNG/Optax helper JITs; no QNode is differentiated.
        with jax.disable_jit():
            for depth in range(len(task['depths'])+1,config['max_depth']+1):
                task['phase']=f'depth_{depth}'
                def checkpoint(state):
                    task['optimizer']=state;store.save(task)
                    store.status('running',depth=depth,restart=state['restart'],step=state['step'])
                optimized=optimize_resumable(cost,step,shape=(depth,2),steps=config['steps'],
                    restarts=config['restarts'],learning_rate=config['learning_rate'],
                    seed=original.seed_for(config['training_seed'],instance['id'],depth),
                    previous=task['depths'][-1]['angles'] if task['depths'] else None,
                    state=task['optimizer'],checkpoint=checkpoint,
                    checkpoint_steps=config['checkpoint_steps'],compile_warmup=False,should_stop=should_stop)
                begin=time.perf_counter();probabilities=sim.probabilities(optimized['best_angles'])
                probability_seconds=time.perf_counter()-begin
                row=dict(depth=depth,angles=optimized['best_angles'],loss=optimized['best_loss'],
                    history=optimized['history'],restarts=optimized['restart_results'],
                    n_slack=sim.model.n_slack,n_total=sim.n,probability_seconds=probability_seconds,
                    compile_seconds=0.,optimize_seconds=optimized['optimize_seconds'],
                    **original._sample(probabilities,instance,config,depth))
                if method=='qaoa_plus':
                    row['mixer_diagnostics']={**sim.model.mixer_diagnostics,
                        'mixer_gate_count':sim.model.mixer_gate_count(),'connected':True,
                        'initial_state':sim.model.initial_state}
                task['depths'].append(row);task['optimizer']=None;store.save(task)
                write_json(store.directory/'depths.json',task['depths'])
        final=task['depths'][-1]
        summary=dict(identity=identity,instance_id=instance['id'],method=method,
            n_x=instance['n_x'],depth=final['depth'],n_total=sim.n,n_slack=sim.model.n_slack,
            compiled=False,execution_backend=config['execution_backend'],compile_seconds=0.,
            optimize_seconds=sum(r['optimize_seconds'] for r in task['depths']),
            prepare_seconds=task['prepare_seconds'],
            probability_seconds=sum(r['probability_seconds'] for r in task['depths']),
            process_peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            **{k:final[k] for k in ('p_optimal','p_feasible','exact_p_optimal','exact_p_feasible')})
        write_json(path,summary);store.status('complete')
        print(f'{method} {instance["id"]}: uncompiled P(opt)={final["p_optimal"]:.4f}',flush=True)
    except Exception as error:
        store.status('interrupted' if isinstance(error,TrainingInterrupted) else 'failed',traceback=traceback.format_exc())
        raise
