"""Rebuild report tables/figures from completed saved results; never trains."""
from pathlib import Path
import os, sys, json, shutil, hashlib, subprocess
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tuning/.cache/matplotlib'))
sys.path.insert(0,str(ROOT/'tuning'))
import numpy as np
import pandas as pd
from analyze import read_results, ranking, mean_intervals, FigureWriter, figures, LABELS
PENALTY_LABELS = {name: rf'$\delta_{{\mathrm{{{subscript}}}}}$' for name, subscript in [('range', 'range'), ('feasible', 'feasible'), ('verma_lewis', 'VL'), ('local', 'local'), ('maximum', 'max')]}
LABELS.update(PENALTY_LABELS)
A=ROOT/'tuning/results/full'; B=ROOT/'tuning/results/full-b-five-methods'
d=read_results(A,B); a,b=d['a'],d['b']
assert len(a)==250 and len(b)==100
assert a.groupby('condition').instance_id.nunique().eq(50).all()
assert b.groupby('condition').instance_id.nunique().eq(20).all()
assert b.depth.eq(5).all()
assert set(b.condition)=={'range','feasible','verma_lewis','local','maximum'}
for st,out in [('a',A),('b',B)]:
 for p in (out/st).glob('*/*/status.json'): assert json.loads(p.read_text())['status']=='complete'
(HERE/'figures').mkdir(exist_ok=True);(HERE/'tables').mkdir(exist_ok=True);(HERE/'data').mkdir(exist_ok=True)
writer=FigureWriter(HERE/'figures')
for st in ['a','b']:figures(d,'datasets',writer,study=st)
for section in ['loss','examples','penalty','resources']:figures(d,section,writer)
labels={'feasibility':r'$L_{\mathrm{feas}}$','fidelity':r'$L_{\mathrm{fid}}$','lambda_0.5':r'$L_{0.5}$','lambda_1':r'$L_1$','lambda_2':r'$L_2$', 'range':'Coefficient range','maximum':'Maximum coefficient','local':'Absolute local change','verma_lewis':'Verma--Lewis','feasible':'Feasible-solution bound'}
labels.update(PENALTY_LABELS)
def table(name,columns,rows,align):
 text='\\begin{tabular}{'+align+'}\n\\toprule\n'+' & '.join(columns)+r' \\'+'\n\\midrule\n'
 text+='\n'.join(' & '.join(row)+r' \\' for row in rows)+'\n\\bottomrule\n\\end{tabular}\n'
 (HERE/'tables'/name).write_text(text)
ra=ranking(a,'fidelity'); rb=ranking(b,'p_optimal')
assert ra.index[0]==json.loads((A/'selected_loss.json').read_text())['condition']
assert rb.index[0]==json.loads((B/'selected_penalty.json').read_text())['condition']
rows=[]
for name,r in ra.iterrows():
 f=a[a.condition==name]
 rows.append([labels[name],f'{r["mean"]:.6f}',f'[{r.ci95_lower:.6f}, {r.ci95_upper:.6f}]',f'{f.p_feasible.mean():.6f}',f'{int(f.converged.sum())}/50',f'{f.selected_depth.mean():.2f}',f'{f.evaluated_depths.mean():.2f}'])
table('study_a.tex',['Loss','Mean $F$','95\\% CI','Mean $P_{\\mathcal F}$','$F\\geq0.999$','Selected $p$','Evaluated $p$'],rows,'lrrrrrr')
rows=[]
for name,r in rb.iterrows():
 f=b[b.condition==name]
 rows.append([labels[name],f'{100*r["mean"]:.3f}',f'[{100*r.ci95_lower:.3f}, {100*r.ci95_upper:.3f}]',f'{100*f.exact_p_optimal.mean():.3f}',f'{100*f.p_feasible.mean():.3f}'])
table('study_b.tex',['Method','Sampled $P(\\mathrm{opt})$','95\\% CI','Exact $P(\\mathrm{opt})$','Sampled $P(\\mathcal F)$'],rows,'lrrrr')
p=b.pivot(index='instance_id',columns='condition',values='p_optimal'); pairs=[]
for name in rb.index[1:]:
 diff=p['range']-p[name]
 stat=mean_intervals(pd.DataFrame({'mean':[diff.mean()],'std':[diff.std()],'count':[len(diff)]})).iloc[0]
 pairs.append(dict(method=name,mean_difference=stat['mean'],ci95_lower=stat.ci95_lower,ci95_upper=stat.ci95_upper,wins=int((diff>0).sum()),ties=int((diff==0).sum()),losses=int((diff<0).sum())))
table('paired_b.tex',['Comparator','Mean Difference (pp)','95\\% CI (pp)','Wins / Ties / Losses'],[[labels[r['method']],f'{100*r["mean_difference"]:.3f}',f'[{100*r["ci95_lower"]:.3f}, {100*r["ci95_upper"]:.3f}]',f'{r["wins"]} / {r["ties"]} / {r["losses"]}'] for r in pairs],'lrrr')
rows=[]
for n,f in b.groupby('n_x'):
 means=f.groupby('condition').p_optimal.mean()
 rows.append([str(n)]+[f'{100*means[k]:.3f}' for k in ['range','feasible','verma_lewis','local','maximum']])
table('size_b.tex',['$n$'] + [labels[name] for name in ['range','feasible','verma_lewis','local','maximum']],rows,'rrrrrr')
for name,frame in [('a_results',a),('b_results',b),('a_ranking',ra),('b_ranking',rb),('paired_b',pd.DataFrame(pairs)),('a_instances',d['a_instances']),('b_instances',d['b_instances'])]: frame.to_csv(HERE/'data'/f'{name}.csv',index=name.endswith('ranking'))
for st,out in [('a',A),('b',B)]:
 for filename in ['run.json',f'manifest_{st}.json','selected_loss.json' if st=='a' else 'selected_penalty.json']:
  shutil.copy2(out/filename,HERE/'data'/f'{st}_{filename}')
prov={'study_a_run_id':d['run']['identifier'],'study_b_run_id':d['b_run']['identifier'], 'analysis_sha256':hashlib.sha256((ROOT/'tuning/analyze.py').read_bytes()).hexdigest(),'counts':{'a':len(a),'b':len(b)},'interval':'pointwise Student t, 95%, across instances','paired_comparisons':pairs}
(HERE/'data/report_provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
# A matched example shows the state under two losses for the same constraint.
import matplotlib.pyplot as plt
from analyze import constraint_title
import pickle
example='a_4_mixed_01'; instance=d['a_instances'].set_index('id').loc[example]
mask=np.asarray(instance.feasible_mask)
fig,axes=plt.subplots(1,2,figsize=(8,3.4),sharey=True)
for ax,loss in zip(axes,['feasibility','fidelity']):
 entry=pickle.loads((A/'a'/example/loss/'gadget.pkl').read_bytes())
 ax.bar(np.arange(len(mask)),np.abs(entry['state'])**2,color=np.where(mask,'#3e8fb0','#eb6f92'))
 ax.axhline(1/mask.sum(),color='#6e6a86',linestyle='--',linewidth=1,label='Uniform Feasible-State Probability')
 ax.set(xlabel='Basis-State Index',title=labels[loss]+f', $F={entry["fidelity"]:.4f}$')
axes[0].set_ylabel('Probability');fig.suptitle(constraint_title(instance.constraint));writer.save(fig,'a_matched_distribution')
print('Generated tables and figures from 250 A tasks and 100 B tasks.',flush=True)
print('Uniform decision baseline:',np.mean([len(r.optimal_indices)/2**r.n_x for r in d['b_instances'].itertuples()]),flush=True)
if '--no-compile' not in sys.argv:
 for _ in range(2):
  result=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','summary.tex'],cwd=HERE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
  if result.returncode: print(result.stdout[-5000:]);raise SystemExit(result.returncode)
 print('Built',HERE/'summary.pdf')
