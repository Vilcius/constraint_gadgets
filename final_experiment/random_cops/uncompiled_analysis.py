"""Read isolated uncompiled reruns without changing original performance data."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

LABELS={'pcqaoa':'PC-QAOA','pcqaoa_stateprep':'PC-QAOA (StatePrep)',
        'penaltyqaoa':'PenaltyQAOA','qaoa_plus':'QAOA+'}

def prefix_timing(root,key,iid,depth,manifest_hash=None):
    directory=Path(root)/key
    metadata=directory/'run.json';path=directory/iid/'depths.json'
    if not metadata.exists() or not path.exists():return np.nan
    run=json.loads(metadata.read_text())
    if not run['config'].get('uncompiled') or run['config'].get('profile')!='full':
        raise ValueError(f'Not an uncompiled full run: {directory}')
    if manifest_hash and run['manifest_hash']!=manifest_hash:raise ValueError('Timing dataset mismatch')
    rows=[r for r in json.loads(path.read_text()) if r['depth']<=depth]
    if [r['depth'] for r in rows]!=list(range(1,depth+1)):return np.nan
    if any(r['compile_seconds']!=0 for r in rows):raise ValueError('Unexpected compilation time')
    return sum(r['optimize_seconds'] for r in rows)

def load_timing_pairs(here):
    here=Path(here);manifest=json.loads((here/'artifacts/manifest.json').read_text())
    rows=[]
    for key,label in LABELS.items():
        directory=here/'results/uncompiled'/key
        for path in sorted(directory.glob('final_*/result.json')):
            new=json.loads(path.read_text());iid=new['instance_id']
            oldpath=here/'results'/key/iid/'result.json'
            if not oldpath.exists():continue
            old=json.loads(oldpath.read_text())
            if new['depth']!=5 or old['depth']!=5:continue
            duration=prefix_timing(here/'results/uncompiled',key,iid,5,manifest['hash'])
            rows.append(dict(instance_id=iid,method=label,n_x=new['n_x'],n_total=new['n_total'],
                uncompiled_seconds=duration,compiled_seconds=old['optimize_seconds'],
                p_optimal_difference=new['p_optimal']-old['p_optimal'],
                exact_p_optimal_difference=new['exact_p_optimal']-old['exact_p_optimal']))
    return pd.DataFrame(rows)
