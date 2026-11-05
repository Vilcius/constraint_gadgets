"""Fixed, outcome-independent layer budgets from the saved gate tables."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / 'final_experiment/random_cops'
LABELS = {'pcqaoa':'PC-QAOA', 'pcqaoa_stateprep':'PC-QAOA (StatePrep)'}


def budget_table(resource_dir=None, stateprep_resources=None, reference_depths=(1,)):
    base = Path(resource_dir or HERE/'results/resource_estimates')
    original = pd.read_csv(base/'final_circuit_resources.csv')
    exact = pd.read_csv(stateprep_resources or
                        HERE/'results/stateprep_resource_estimates/stateprep_circuit_resources.csv')
    references = {depth: original[(original.method=='QAOA+') &
                   (original.gate_set=='nisq') & (original.depth==depth)]
                   .set_index('instance_id').total_gates for depth in reference_depths}
    if not references or not set(references).issubset({1,5}):
        raise ValueError('Reference depths must be 1 and/or 5')
    rows = []
    for key,label in LABELS.items():
        frame = original if key=='pcqaoa' else exact
        frame = frame[(frame.method==label)&(frame.gate_set=='nisq')]
        first = frame[frame.depth==1].set_index('instance_id')
        fifth = frame[frame.depth==5].set_index('instance_id')
        if not first.index.is_unique or not fifth.index.is_unique:
            raise ValueError('Duplicate resource rows')
        if any(not ref.index.is_unique or set(first.index)!=set(ref.index)
               or set(fifth.index)!=set(ref.index) for ref in references.values()):
            raise ValueError('Resource tables do not describe the same COPs')
        for reference_depth,reference in references.items():
            for instance_id,row in first.iterrows():
                layer = (fifth.loc[instance_id,'total_gates']-row.total_gates)/4
                preparation = row.total_gates-layer
                if layer<=0 or preparation<0 or layer!=int(layer):
                    raise ValueError('Invalid additive circuit-resource counts')
                budget = int(reference.loc[instance_id])
                unconstrained = int(np.floor((budget-preparation)/layer+1e-10))
                requested = max(1,unconstrained)
                target = requested
                used = int(preparation+target*layer)
                rows.append(dict(instance_id=instance_id,method=key,method_label=label,
                                 n_x=int(row.n_x),regime=row.regime,reference_depth=int(reference_depth),
                                 gate_budget=budget,preparation_gates=int(preparation),
                                 layer_gates=int(layer),uncapped_depth=requested,
                                 target_depth=target,additional_layers=max(0,target-5),
                                 uncapped_additional_layers=max(0,requested-5),
                                 minimum_depth_overrun=unconstrained<1,
                                 actual_gates=used,budget_fraction=used/budget,
                                 unused_gates=max(0,budget-used),overrun_gates=max(0,used-budget)))
    return pd.DataFrame(rows).sort_values(['method','instance_id','reference_depth']).reset_index(drop=True)
