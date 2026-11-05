"""Two-worker, resumable gate-budget comparison; no JAX circuit compilation."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import traceback
import warnings

from . import runtime as cpu
from final_experiment.random_cops.analyze import status_summary
from tuning.storage import digest,environment,write_json
from .budgets import ROOT,HERE,budget_table

warnings.filterwarnings('ignore',message=r'Contains tensors of types .*',category=UserWarning)


def worker(job):
    from .studies import run_instance
    from .optimization import TrainingInterrupted
    if cpu.STOP.is_set():return {'status':'interrupted'}
    try:
        run_instance(*job,should_stop=cpu.STOP.is_set)
        return {'status':'complete'}
    except TrainingInterrupted:
        return {'status':'interrupted'}
    except Exception:
        from .task_storage import TaskStore
        method,instance,budget,config,output,run_id,_ = job
        TaskStore(Path(output)/instance['id'],digest([run_id,instance['id'],budget])).status(
            'failed',traceback=traceback.format_exc())
        return {'status':'failed','traceback':traceback.format_exc()}


def export_results(output):
    import pandas as pd
    output=Path(output)
    rows=[json.loads(p.read_text()) for p in sorted(output.glob('final_*/comparison_qaoa_p*.json'))]
    temporary=output/'comparisons.csv.tmp'
    pd.DataFrame(rows).to_csv(temporary,index=False)
    temporary.replace(output/'comparisons.csv')


def main(method):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run','resume','smoke','status','export'])
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--threads',type=int,default=6)
    parser.add_argument('--reference-depth',choices=['1'],default='1',
                        help='The gate budget is QAOA+ at one layer')
    parser.add_argument('--reuse-run',type=Path,action='append',default=[],
                        help='Read completed depths from an earlier gate-budget run')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--manifest',type=Path,default=HERE/'artifacts/manifest.json')
    parser.add_argument('--source',type=Path,default=HERE/'results'/method)
    parser.add_argument('--database',type=Path,default=ROOT/'tuning/results/full/selected_vcg_db.pkl')
    parser.add_argument('--resource-dir',type=Path,default=HERE/'results/resource_estimates')
    parser.add_argument('--stateprep-resources',type=Path,default=HERE/'results/stateprep_resource_estimates/stateprep_circuit_resources.csv')
    parser.add_argument('--shard-count',type=int,default=1)
    parser.add_argument('--shard-index',type=int,default=0)
    args=parser.parse_args()
    if min(args.workers,args.threads,args.shard_count)<1 or not 0<=args.shard_index<args.shard_count:
        parser.error('Positive workers, threads and a valid shard index are required')
    smoke=args.command=='smoke'
    references=[1]
    suffix='_qaoa_p1'
    output=(args.output or HERE/'results/gate_budget'/
            (f'smoke_{method}{suffix}' if smoke else f'{method}{suffix}')).resolve()
    if args.command=='status':
        report=status_summary(output)
        report['comparison_results']={str(p):len(list(output.glob(f'final_*/comparison_qaoa_p{p}.json'))) for p in references}
        print(json.dumps(report,indent=2));return 0
    if args.command=='export':export_results(output);return 0
    cpu.runtime_environment(args.threads)
    os.environ['JAX_ENABLE_X64']='true'
    manifest=json.loads(args.manifest.read_text())
    if manifest['hash']!=digest(manifest['instances']):raise ValueError('Manifest hash mismatch')
    table=budget_table(args.resource_dir,args.stateprep_resources,reference_depths=references)
    instances={r['id']:r for r in manifest['instances']}
    if set(table.instance_id)!=set(instances):raise ValueError('Resource/dataset ID mismatch')
    planning=HERE/'artifacts/gate_budget'
    planning.mkdir(parents=True,exist_ok=True)
    # prepare is deterministic planning only; execution writes its own snapshot.
    if args.command=='prepare':
        path=planning/f'budgets_uncapped_qaoa_{args.reference_depth}.csv'
        temporary=path.with_suffix('.csv.tmp');table.to_csv(temporary,index=False);temporary.replace(path)
        for name,group in table.groupby(['method','reference_depth']):
            print(name,{'COPs':len(group),'existing_target_depth':int((group.target_depth<=5).sum()),
                        'extensions':int((group.target_depth>5).sum()),'maximum_depth':int(group.target_depth.max()),
                        'minimum_depth_overruns':int(group.minimum_depth_overrun.sum()),
                        'additional_depth_optimizations':int(group.additional_layers.sum())})
        print(path);return 0
    config=json.loads((HERE/'config.json').read_text())
    config.update(method=method,manifest_hash=manifest['hash'],reference_depths=references,
                  depth_cap=None,
                  backend='NumPy manual adjoint; no JAX compilation',
                  database=str(args.database.resolve()),
                  random_generator='NumPy PCG64; same integer seed derivation as original runner',
                  profile='smoke' if smoke else 'full')
    selected=[group.to_dict('records') for _,group in
              table[table.method==method].groupby('instance_id',sort=True)]
    if smoke:
        config['import_settings']={name:config[name] for name in (
            'steps','restarts','learning_rate','shots','training_seed','sampling_seed')}
        short=[rows for rows in selected if max(r['target_depth'] for r in rows)<=5]
        first=short[0] if short else min(selected,key=lambda rows:max(r['target_depth'] for r in rows))
        second=next(rows for rows in selected if max(r['target_depth'] for r in rows)>5 and rows!=first)
        selected=[first,second]
        for rows in selected:
            for row in rows:
                row['smoke_original_target']=row['target_depth']
                row['target_depth']=min(6,row['target_depth'])
                row['actual_gates']=row['preparation_gates']+row['target_depth']*row['layer_gates']
                row['budget_fraction']=row['actual_gates']/row['gate_budget']
                row['unused_gates']=max(0,row['gate_budget']-row['actual_gates'])
                row['overrun_gates']=max(0,row['actual_gates']-row['gate_budget'])
                row['additional_layers']=max(0,row['target_depth']-5)
        config.update(steps=2,restarts=2,shots=1000,checkpoint_steps=1)
    selected=[row for i,row in enumerate(selected) if i%args.shard_count==args.shard_index]
    config['source_run_identifier']=json.loads((args.source/'run.json').read_text())['identifier']
    # Keep the earlier two-reference output intact and import its finished depths.
    # New single-reference results use a different directory and provenance.
    reuse = [HERE/'results/gate_budget'/method, *args.reuse_run]
    sources = [args.source.resolve()]
    reused_runs = {}
    for directory in reuse:
        directory = directory.resolve()
        metadata = directory/'run.json'
        if directory == output or directory in sources or not metadata.exists():
            continue
        run = json.loads(metadata.read_text())
        if (run['manifest_hash'] != manifest['hash'] or
                run['config'].get('method') != method or
                run['config'].get('profile') != 'full' or
                run['config'].get('depth_cap') is not None):
            raise ValueError(f'Incompatible reuse source: {directory}')
        for name in ('steps','restarts','learning_rate','shots','training_seed','sampling_seed'):
            if run['config'][name] != config.get('import_settings',config)[name]:
                raise ValueError(f'Incompatible reuse optimization setting: {name}')
        sources.append(directory)
        reused_runs[str(directory)] = run['identifier']
    config['reused_run_identifiers'] = reused_runs
    files=[Path(__file__).parent/name for name in
           ('runner.py','studies.py','simulator.py','budgets.py')]+[
        ROOT/'core/pc_qaoa.py',ROOT/'core/qaoa_base.py',ROOT/'core/vcg.py',
        ROOT/'core/constraint_handler.py',ROOT/'core/constraint_state_prep.py',
        ROOT/'core/dicke_state_prep.py',Path(__file__).with_name('optimization.py'),
        Path(__file__).with_name('task_storage.py'),Path(__file__).with_name('runtime.py'),args.database,
        args.resource_dir/'final_circuit_resources.csv',args.stateprep_resources]
    hashes={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):
            hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    provenance=dict(config=config,manifest_hash=manifest['hash'],source=hashes,
                    environment=environment(),budgets=selected)
    identifier=digest(provenance)
    with cpu.output_lock(output):
        path=output/'run.json'
        if path.exists() and json.loads(path.read_text())['identifier']!=identifier:
            raise ValueError('Configuration, source, dataset, budget or environment changed; use a new output directory')
        write_json(path,dict(identifier=identifier,**provenance))
        write_json(output/'selection.json',{'instance_ids':[rows[0]['instance_id'] for rows in selected]})
        jobs=[]
        for rows in selected:
            instance_id=rows[0]['instance_id']
            result=output/instance_id/'result.json'
            if result.exists():
                if json.loads(result.read_text())['identity']!=digest([identifier,instance_id,rows]):
                    raise ValueError('Completed result identity mismatch')
                continue
            jobs.append((method,instances[instance_id],rows,config,str(output),identifier,
                         [str(source) for source in sources]))
        cpus=cpu.physical_cpus();workers=min(args.workers,len(cpus),len(jobs))
        if not jobs:print('All selected tasks are complete');return 0
        threads=min(args.threads,max(1,len(cpus)//workers))
        print(f'{len(jobs)} COPs; {workers} workers x {threads} physical cores; uncapped QAOA+ budgets {references}; no compilation',flush=True)
        context=mp.get_context('spawn');stop=context.Event();slots=context.Queue()
        for i in range(workers):slots.put(cpus[i*threads:(i+1)*threads])
        old={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
        for s in old:signal.signal(s,lambda *_:stop.set())
        failures=0
        try:
            with ProcessPoolExecutor(max_workers=workers,mp_context=context,
                    initializer=cpu.initialize_worker,initargs=(slots,stop,threads)) as pool:
                for future in as_completed([pool.submit(worker,job) for job in jobs]):
                    result=future.result()
                    if result['status']=='failed':
                        failures+=1;print(result['traceback'],file=sys.stderr,flush=True)
        finally:
            for s,handler in old.items():signal.signal(s,handler)
            slots.close()
        export_results(output)
        return 130 if stop.is_set() else (1 if failures else 0)
