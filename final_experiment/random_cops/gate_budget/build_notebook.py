"""Build the uncapped PC-QAOA comparison against one QAOA+ layer."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[3]
nb=nbf.v4.new_notebook();cells=[]
def md(s):cells.append(nbf.v4.new_markdown_cell(s))
def code(s):cells.append(nbf.v4.new_code_cell(s))
md(r'''# Paired PC-QAOA and QAOA+ Comparisons at Matched Gate Budgets

We compare QAOA+ at one layer against deep PC-QAOA circuits selected using the same total NISQ gate budget. There is no depth cap:

$$p_i=\max\left(1,\left\lfloor\frac{G_{\mathrm{QAOA+},i}(1)-G_{\mathrm{prep},i}}{G_{\mathrm{layer},i}}\right\rfloor\right).$$

StatePrep PC-QAOA is the primary comparison. Regular VCG PC-QAOA uses its own resource-derived depths and is shown wherever its target results exist. No fixed-depth PC-QAOA reference is included. The original comparison notebook already reports fixed-depth results. Here, each PC-QAOA result is taken at its own gate-budget target depth.

Whole layers usually prevent exactly equal counts. PC-QAOA uses the largest depth that stays within the budget. The minimum-one-layer rule is retained, and all 500 COPs remain eligible for the plots. Cases where a single PC-QAOA layer exceeds the QAOA+ budget are included and counted explicitly. They are minimum-depth exceptions, not equal-gate comparisons. Gate-use figures show the remaining mismatch explicitly.

All performance figures compare the same COPs within each method pair. Each saved result must be at the predetermined target depth; unfinished deeper targets are not replaced by shallower results. We retain all completed observations, including perfect solutions. Optimization still uses standard QAOA angles, Adam with learning rate 0.01, ten restarts, fifty steps at every depth and 10,000 final shots. NumPy adjoint gradients run without circuit compilation. Matching gates per circuit does not equalize classical training effort.

QAOA+ performance and timing references come only from the new uncompiled runs. PC-QAOA targets at depths one through five also use those reruns, as do the depth-five diagnostic baselines. Missing uncompiled references remain missing; compiled results are not substituted. Deeper PC-QAOA results retain their existing optimized trajectories, which were warm-started from the original runs. Their timing totals replace the original prefix measurements with the new uncompiled measurements.

This notebook only reads saved results and gate-count tables. It does not start optimization or resource estimation. Mean intervals are pointwise 95% Student-$t$ intervals across COPs. Paired intervals use per-COP differences.''')
code(r'''from pathlib import Path
import os,sys,json
candidates=[Path.cwd(),*Path.cwd().parents,Path.cwd()/"code-tuning"]
REPO=next(p for p in candidates if (p/"final_experiment/random_cops/config.json").exists())
sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/"tuning"))
os.environ.setdefault("MPLCONFIGDIR",str(REPO/"final_experiment/random_cops/.cache/matplotlib"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import t
from IPython.display import display
from figure_style import FigureWriter,PALETTE,METHOD_COLORS,apply_publication_style
from final_experiment.random_cops.gate_budget.budgets import budget_table,LABELS
from final_experiment.random_cops.gate_budget.analysis import load_comparisons,paired_results
HERE=REPO/"final_experiment/random_cops"
OUTPUT=Path(os.environ.get("PCQAOA_GATE_BUDGET_RESULTS",HERE/"results/gate_budget"))
FIGURE_DIR=OUTPUT/"figures"
INCLUDE_BUDGET_OVERRUNS=True
REFERENCE_DEPTH=1
manifest=json.loads((HERE/"artifacts/manifest.json").read_text())
references={r["id"]:r for r in manifest["instances"]}
instances=pd.DataFrame(manifest["instances"])
budgets=budget_table(reference_depths=(REFERENCE_DEPTH,))
METHODS=["PC-QAOA (StatePrep)","PC-QAOA"]
COLORS=METHOD_COLORS
GRAY=PALETTE["neutral_dark"]
MARKERS={"PC-QAOA (StatePrep)":"P","PC-QAOA":"o","QAOA+":"D"}
OVERLAP_STYLES={0:"-",1:"--",2:":"}
OVERLAP_MARKERS={0:"o",1:"s",2:"D"}
OVERLAP_LABELS={0:r"Disjoint ($o=0$)",1:r"One Overlap ($o=1$)",2:r"Two Overlaps ($o=2$)"}
SIZE_CMAP=LinearSegmentedColormap.from_list("paper_size",[PALETTE["green_1"],PALETTE["teal"],PALETTE["blue_main"]])
METRICS=[("p_optimal",r"$P(\mathrm{opt})$"),("p_feasible",r"$P(\mathrm{feas})$"),("ar_feas",r"$\mathrm{AR}_{\mathrm{feas}}$")]
SLUGS={"PC-QAOA (StatePrep)":"stateprep","PC-QAOA":"vcg"}
apply_publication_style()
writer=FigureWriter(FIGURE_DIR,latex=True,display_inline=True)

def mean_ci(frame,groups,value):
    stats=frame.groupby(groups)[value].agg(["mean","std","count"]).reset_index()
    stats["ci95"]=t.ppf(.975,stats["count"]-1)*stats["std"]/np.sqrt(stats["count"])
    stats.loc[stats["count"]<2,"ci95"]=np.nan
    return stats

def pair(method,r):
    return paired_results(results,method,r,include_overruns=INCLUDE_BUDGET_OVERRUNS)

def common_pairs(r):
    frames={method:pair(method,r) for method in METHODS}
    if any(frame.empty for frame in frames.values()):return {}
    ids=set.intersection(*(set(frame.instance_id) for frame in frames.values()))
    return {method:frame[frame.instance_id.isin(ids)].copy()
            for method,frame in frames.items()} if ids else {}

def pair_lines(ax,frame,metric,method):
    for column,label,color,marker in [(f"{metric}_pc",method,COLORS[method],MARKERS[method]),(f"{metric}_qaoa","QAOA+",COLORS["QAOA+"],MARKERS["QAOA+"])]:
        stats=mean_ci(frame,"n_x",column)
        ax.errorbar(stats.n_x,stats["mean"],yerr=stats.ci95.fillna(0),color=color,marker=marker,linewidth=2,capsize=3,label=label)
    ax.legend(frameon=False,fontsize=9)

def method_overlap_legend(ax,methods,overlaps):
    handles=[Line2D([],[],color=COLORS[method],linewidth=2.4,label=method)
             for method in methods]
    handles += [Line2D([],[],color=GRAY,linestyle=OVERLAP_STYLES[int(overlap)],
                      marker=OVERLAP_MARKERS[int(overlap)],linewidth=1.8,
                      label=OVERLAP_LABELS[int(overlap)])
                for overlap in sorted(set(overlaps))]
    ax.legend(handles=handles,frameon=False,fontsize=9,ncol=2)

def retire_figure_exports(names):
    # Called only after the consolidated replacement has been exported.
    for name in names:
        for extension in ("pdf","png"):
            (writer.directory/f"{name}.{extension}").unlink(missing_ok=True)

print("Results:",OUTPUT,"\nFigures:",FIGURE_DIR,"\nFonts:",writer.font_status)
print("Minimum-Depth Budget Overruns Included:",INCLUDE_BUDGET_OVERRUNS)
''')
md('''## Completion and Pair Coverage

Comparison files are saved as soon as their target depth finishes. Existing depths one through five are read directly when they already contain the target result. The regular VCG comparison may have fewer pairs until its optional run completes. Every plot and paired summary uses the available intersection for its specific method pair. Reusable targets appear immediately, before any new gate-budget optimization is run; a previously saved depth-five result is used only if five is that COP's actual gate-budget target.''')
code(r'''results,completion=load_comparisons(OUTPUT,budgets,references)
for row in completion.to_dict("records"):
    print(f"{row['Method']}: {row['Expected']} expected; {row['Available']} completed/available "
          f"({row['Read From Uncompiled']} reused, {row['Available']-row['Read From Uncompiled']} newly optimized); "
          f"{row['Running']} running; {row['Failed']} failed; {row['Remaining']} remaining")
qaoa_count=results.loc[results.method=="QAOA+","instance_id"].nunique()
print(f"QAOA+ at p=1: {qaoa_count}/{len(references)} saved reference results")
display(completion)
coverage=[]
for r in [REFERENCE_DEPTH]:
    for method in METHODS:
        planned=budgets[(budgets.method_label==method)&(budgets.reference_depth==r)]
        coverage.append({"Method":method,"QAOA+ Depth":r,"Planned COPs":len(planned),"Minimum-Depth Overruns":int(planned.minimum_depth_overrun.sum()),"Available Paired COPs":len(pair(method,r))})
display(pd.DataFrame(coverage))
''')
md('''## Required PC-QAOA Layers

The distributions show all planned COPs, including minimum-depth exceptions. Depths are determined only from gate counts. There is no cap. Each method uses its own preparation and per-layer gate counts.''')
code(r'''edges=[0,5,10,25,50,100,250,500]
labels=["1-5","6-10","11-25","26-50","51-100","101-250","251-500"]
for r in [REFERENCE_DEPTH]:
    fig,ax=plt.subplots(figsize=(8.0,4.3));x=np.arange(len(labels))
    for offset,method in zip([-.19,.19],METHODS):
        subset=budgets[(budgets.method_label==method)&(budgets.reference_depth==r)]
        counts=pd.cut(subset.target_depth,bins=edges,labels=labels).value_counts(sort=False)
        bars=ax.bar(x+offset,counts.values,width=.36,color=COLORS[method],label=method)
        for bar,count in zip(bars,counts):
            if count:ax.text(bar.get_x()+bar.get_width()/2,count+3,str(count),ha="center",fontsize=9)
    ax.set_xticks(x,labels,rotation=25)
    ax.set(xlabel=r"Required PC-QAOA Depth ($p$)",ylabel="COP Count",title=fr"COP Count vs. Required Depth: QAOA+ $p={r}$ Budget")
    ax.set_ylim(0,370);ax.legend(frameon=False,fontsize=9);writer.save(fig,f"required_depth_distribution_qaoa_p{r}")
    fig,ax=plt.subplots(figsize=(8.0,4.3))
    additional_labels=["0","1-5","6-20","21-45","46-95","96-245","246-495"]
    for offset,method in zip([-.19,.19],METHODS):
        subset=budgets[(budgets.method_label==method)&(budgets.reference_depth==r)]
        counts=pd.cut(subset.additional_layers,bins=[-1,0,5,20,45,95,245,495],labels=additional_labels).value_counts(sort=False)
        bars=ax.bar(x+offset,counts.values,width=.36,color=COLORS[method],label=method)
        for bar,count in zip(bars,counts):
            if count:ax.text(bar.get_x()+bar.get_width()/2,count+3,str(count),ha="center",fontsize=9)
    ax.set_xticks(x,additional_labels,rotation=25)
    ax.set(xlabel="Additional Training Layers",ylabel="COP Count",title=fr"COP Count vs. Additional Layers: QAOA+ $p={r}$ Budget")
    ax.set_ylim(0,370);ax.legend(frameon=False,fontsize=9);writer.save(fig,f"additional_layers_qaoa_p{r}")
    fig,ax=plt.subplots(figsize=(6.8,4))
    for method in METHODS:
        subset=budgets[(budgets.method_label==method)&(budgets.reference_depth==r)]
        stats=mean_ci(subset,"n_x","target_depth")
        ax.errorbar(stats.n_x,stats["mean"],yerr=stats.ci95.fillna(0),color=COLORS[method],marker=MARKERS[method],capsize=3,label=method)
    ax.set(xlabel=r"Problem Size ($n$)",ylabel=r"PC-QAOA Depth ($p$)",title=fr"Required Depth vs. Problem Size: QAOA+ $p={r}$ Budget")
    ax.legend(frameon=False,fontsize=9);writer.save(fig,f"target_depth_by_size_qaoa_p{r}")
display(budgets.groupby(["method_label","reference_depth"]).agg(COPs=("instance_id","size"),Maximum_Depth=("target_depth","max"),Median_Depth=("target_depth","median"),Existing_Targets=("target_depth",lambda s:(s<=5).sum()),Overruns=("minimum_depth_overrun","sum"),Additional_Depths=("additional_layers","sum")))
''')
md(r'''## Paired Metric Scatter Plots

Each point is the same COP: QAOA+ performance is on the horizontal axis and PC-QAOA at the corresponding gate-budget depth is on the vertical axis. Points above the diagonal favor PC-QAOA. Color indicates decision size. Separate figures show $P(\mathrm{opt})$, $P(\mathrm{feas})$ and $\mathrm{AR}_{\mathrm{feas}}$ for QAOA+ at one layer. A zoom appears when optimal-solution probabilities are concentrated near zero. No observations are removed based on their performance.''')
code(r'''paired_summary=[]
for r in [REFERENCE_DEPTH]:
    for method in METHODS:
        frame=pair(method,r)
        if frame.empty:continue
        for metric,ylabel in METRICS:
            values=frame[[f"{metric}_pc",f"{metric}_qaoa"]].dropna()
            if values.empty:continue
            fig,ax=plt.subplots(figsize=(5.9,4.7))
            valid=frame.loc[values.index]
            points=ax.scatter(values[f"{metric}_qaoa"],values[f"{metric}_pc"],c=valid.n_x,cmap=SIZE_CMAP,s=30,alpha=.7)
            fig.colorbar(points,ax=ax,label=r"Problem Size ($n$)")
            bound=min(1.02,max(.05,float(values.to_numpy().max())*1.04))
            ax.plot([0,bound],[0,bound],"--",color=GRAY,linewidth=1.3)
            ax.set(xlabel=fr"QAOA+ $p={r}$ {ylabel}",ylabel=fr"{method} {ylabel}",title=fr"{method} vs. QAOA+ $p={r}$: {ylabel}",xlim=(-.01*bound,bound),ylim=(-.01*bound,bound))
            ax.set_aspect("equal",adjustable="box")
            near=(values[f"{metric}_pc"]<=.05)&(values[f"{metric}_qaoa"]<=.05)
            if metric=="p_optimal" and near.sum()>=10 and near.mean()>.3 and bound>.15:
                inset=ax.inset_axes([.58,.1,.36,.36])
                inset.scatter(values[f"{metric}_qaoa"],values[f"{metric}_pc"],c=valid.n_x,cmap=SIZE_CMAP,s=10,alpha=.65)
                inset.plot([0,.05],[0,.05],"--",color=GRAY,linewidth=1)
                inset.set(xlim=(-.001,.05),ylim=(-.001,.05));inset.tick_params(labelsize=8)
                ax.indicate_inset_zoom(inset,edgecolor=GRAY,alpha=.5)
            writer.save(fig,f"paired_{SLUGS[method]}_{metric}_qaoa_p{r}")
            difference=values[f"{metric}_pc"]-values[f"{metric}_qaoa"]
            ci=t.ppf(.975,len(difference)-1)*difference.std(ddof=1)/np.sqrt(len(difference)) if len(difference)>1 else np.nan
            paired_summary.append({"Method":method,"QAOA+ Depth":r,"Metric":metric,"COPs":len(difference),"Mean PC Minus QAOA+":difference.mean(),"95% CI Half-Width":ci,"PC Wins":int((difference>1e-12).sum()),"Ties":int((abs(difference)<=1e-12).sum()),"QAOA+ Wins":int((difference< -1e-12).sum())})
paired_summary=pd.DataFrame(paired_summary)
display(paired_summary)
''')
md(r'''## Paired Differences and Performance by Size

Each performance figure has three curves: QAOA+ at one layer, regular VCG PC-QAOA at its gate-budget depth, and StatePrep PC-QAOA at its gate-budget depth. These figures use only COPs with completed target results for all three methods. The table reports how many common COPs are available at each problem size. The scatter plots above retain each variant's full available pair set.

Difference plots use those same common COPs and subtract QAOA+ from each PC-QAOA variant before averaging. Positive differences favor PC-QAOA. The QAOA+ reference is zero. Probability differences are expressed in percentage points. Means have pointwise 95% Student-$t$ intervals across COPs.''')
code(r'''for r in [REFERENCE_DEPTH]:
    frames={method:pair(method,r) for method in METHODS}
    if any(frame.empty for frame in frames.values()):
        print("Three-method size plots require target results from both PC-QAOA variants.")
        continue
    common_ids=set.intersection(*(set(frame.instance_id) for frame in frames.values()))
    if not common_ids:
        print("No COPs have completed target results for all three methods yet.")
        continue
    frames={method:frame[frame.instance_id.isin(common_ids)].copy()
            for method,frame in frames.items()}
    baseline=frames[METHODS[0]]
    print(f"Three-method size comparisons: {len(common_ids)} common COPs")
    display(baseline.groupby("n_x").size().rename("Common COPs").to_frame())
    for metric,ylabel in METRICS:
        valid_ids=set.intersection(*(set(frame.loc[
            np.isfinite(frame[f"{metric}_pc"]) & np.isfinite(frame[f"{metric}_qaoa"]),
            "instance_id"]) for frame in frames.values()))
        if not valid_ids:continue
        selected={method:frame[frame.instance_id.isin(valid_ids)]
                  for method,frame in frames.items()}
        reference=baseline[baseline.instance_id.isin(valid_ids)]
        fig,ax=plt.subplots(figsize=(7.0,4.2))
        for method,frame in selected.items():
            scale=100 if metric in ("p_optimal","p_feasible") else 1
            frame=frame.assign(paired_difference=scale*frame[f"difference_{metric}"])
            stats=mean_ci(frame,"n_x","paired_difference")
            ax.errorbar(stats.n_x,stats["mean"],yerr=stats.ci95.fillna(0),color=COLORS[method],marker=MARKERS[method],capsize=3,linewidth=2,label=method)
        ax.axhline(0,color=COLORS["QAOA+"],linestyle="--",linewidth=1.5,label="QAOA+ Reference")
        unit=" (Percentage Points)" if metric in ("p_optimal","p_feasible") else ""
        ax.set(xlabel=r"Problem Size ($n$)",ylabel=f"PC Minus QAOA+ {ylabel}{unit}",title=fr"Paired {ylabel} Difference vs. Problem Size: QAOA+ $p={r}$")
        ax.set_xticks(sorted(instances.n_x.unique()))
        ax.legend(frameon=False,fontsize=9);writer.save(fig,f"paired_difference_{metric}_qaoa_p{r}")
        fig,ax=plt.subplots(figsize=(7.0,4.2))
        for method in ["QAOA+",*METHODS]:
            frame=reference if method=="QAOA+" else selected[method]
            column=f"{metric}_qaoa" if method=="QAOA+" else f"{metric}_pc"
            stats=mean_ci(frame,"n_x",column)
            ax.errorbar(stats.n_x,stats["mean"],yerr=stats.ci95.fillna(0),
                        color=COLORS[method],marker=MARKERS[method],linewidth=2,
                        capsize=3,label=method)
        ax.set(xlabel=r"Problem Size ($n$)",ylabel=ylabel,
               title=fr"{ylabel} vs. Problem Size: QAOA+ $p={r}$ Budget")
        ax.set_xticks(sorted(instances.n_x.unique()));ax.legend(frameon=False,fontsize=9)
        writer.save(fig,f"paired_size_all_methods_{metric}_qaoa_p{r}")
        retire_figure_exports([f"paired_size_{SLUGS[method]}_{metric}_qaoa_p{r}"
                               for method in METHODS])
''')
md(r'''## Performance by Constraint Overlap

Disjoint supports and overlap levels one and two share one figure for each PC-QAOA variant. Method is shown by color. Solid circles indicate disjoint supports; dashed squares indicate one overlap; dotted diamonds indicate two overlaps. The legend lists method colors and overlap styles separately.

Both methods use the same COPs within each size and overlap group. Every available pair is retained. The table reports group counts; a single observation has no estimated confidence interval. The overlap level counts variable occurrences across supports beyond the number of distinct variables in those supports. It does not count how many constraints PC-QAOA penalizes.''')
code(r'''overlap_counts=[]
for method in METHODS:
    frame=pair(method,REFERENCE_DEPTH)
    if frame.empty:continue
    counts=frame.groupby(["n_x","overlap_level"]).size().rename("Paired COPs").reset_index()
    counts.insert(0,"Method",method);overlap_counts.append(counts)
    fig,ax=plt.subplots(figsize=(7.2,4.5))
    for overlap,subset in frame.groupby("overlap_level",sort=True):
        overlap=int(overlap)
        for column,label in [("p_optimal_pc",method),("p_optimal_qaoa","QAOA+")]:
            stats=mean_ci(subset,"n_x",column)
            ax.errorbar(stats.n_x,stats["mean"],yerr=stats.ci95.fillna(0),
                        color=COLORS[label],linestyle=OVERLAP_STYLES[overlap],
                        marker=OVERLAP_MARKERS[overlap],markersize=5,linewidth=1.8,
                        capsize=2.5,alpha=.8)
    ax.set(xlabel=r"Problem Size ($n$)",ylabel=r"$P(\mathrm{opt})$",
           title=fr"$P(\mathrm{{opt}})$ vs. Problem Size by Overlap: QAOA+ $p={REFERENCE_DEPTH}$ Budget")
    ax.set_xticks(sorted(instances.n_x.unique()))
    method_overlap_legend(ax,[method,"QAOA+"],frame.overlap_level.unique())
    writer.save(fig,f"paired_overlap_{SLUGS[method]}_qaoa_p{REFERENCE_DEPTH}")
    retire_figure_exports([f"paired_{regime}_{SLUGS[method]}_qaoa_p{REFERENCE_DEPTH}"
                           for regime in ("disjoint","overlapping")])
if overlap_counts:display(pd.concat(overlap_counts,ignore_index=True))
''')
md(r'''## Distribution of Paired Improvements

This cumulative plot includes every available COP pair. The horizontal axis is the PC-QAOA minus QAOA+ difference in $P(\mathrm{opt})$, in percentage points. The height at any value is the fraction of COPs whose difference is at or below that value. Negative differences favor QAOA+; positive differences favor PC-QAOA. A curve farther to the right indicates larger paired improvements.

This view shows the distribution that a mean and confidence interval summarize. It does not add a new comparison or remove unusual observations. Each variant uses its own available COP pairs, so the sample sizes can differ. The printed win, tie and loss counts distinguish exact ties from losses at zero.''')
code(r'''fig,ax=plt.subplots(figsize=(7.0,4.3));plotted=False
for method in METHODS:
    frame=pair(method,REFERENCE_DEPTH)
    if frame.empty:continue
    differences=100*frame["difference_p_optimal"].dropna().to_numpy()
    if not len(differences):continue
    values,counts=np.unique(differences,return_counts=True)
    cumulative=np.cumsum(counts)/len(differences)
    margin=max(.25,.03*float(np.ptp(values)))
    x=np.r_[values[0]-margin,values,values[-1]+margin]
    y=np.r_[0.,cumulative,1.]
    ax.step(x,y,where="post",color=COLORS[method],linewidth=2,
            label=f"{method} ({len(differences)} COPs)")
    print(f"{method}: {np.count_nonzero(differences>1e-10)} wins, "
          f"{np.count_nonzero(np.abs(differences)<=1e-10)} ties, "
          f"{np.count_nonzero(differences< -1e-10)} losses")
    plotted=True
if plotted:
    ax.axvline(0,color=GRAY,linestyle="--",linewidth=1.2)
    ax.set(xlabel=r"Paired $P(\mathrm{opt})$ Difference (Percentage Points)",
           ylabel="Cumulative Fraction of Paired COPs",
           title=r"Cumulative COP Fraction vs. Paired $P(\mathrm{opt})$ Difference",
           ylim=(0,1.02))
    ax.legend(frameon=False,fontsize=9)
    writer.save(fig,f"paired_improvement_ecdf_qaoa_p{REFERENCE_DEPTH}")
else:plt.close(fig)
''')
md('''## Win, Tie and Loss Fractions

One figure shows all three metrics. Each row counts per-COP comparisons rather than weighting by the size of an improvement. All available pairs are retained, including minimum-depth exceptions. A small numerical tolerance identifies ties. The paired-difference table reports the magnitudes as well.''')
code(r'''if not paired_summary.empty:
    order={metric:i for i,(metric,_) in enumerate(METRICS)}
    subset=paired_summary[paired_summary["QAOA+ Depth"]==REFERENCE_DEPTH].copy()
    subset["metric_order"]=subset.Metric.map(order)
    subset=subset.sort_values(["metric_order","Method"])
    fig,ax=plt.subplots(figsize=(8.2,max(3.6,.55*len(subset)+1.2)))
    left=np.zeros(len(subset))
    for column,label,color in [("PC Wins","PC-QAOA Wins",PALETTE["violet"]),("Ties","Ties",PALETTE["neutral"]),("QAOA+ Wins","QAOA+ Wins",COLORS["QAOA+"])]:
        fractions=subset[column].to_numpy()/subset.COPs.to_numpy()
        ax.barh(np.arange(len(subset)),fractions,left=left,color=color,label=label,height=.62)
        left+=fractions
    metric_labels=dict(METRICS)
    labels=[f"{row.Method}: {metric_labels[row.Metric]}" for row in subset.itertuples()]
    ax.set_yticks(np.arange(len(subset)),labels);ax.invert_yaxis()
    ax.set(xlabel="Fraction of Paired COPs",title=r"Win, Tie and Loss Fractions by Metric: QAOA+ $p=1$",xlim=(0,1))
    ax.legend(frameon=False,fontsize=9,ncol=3,loc="upper center",bbox_to_anchor=(.5,-.18))
    writer.save(fig,f"wins_all_metrics_qaoa_p{REFERENCE_DEPTH}")
    retire_figure_exports([f"wins_{metric}_qaoa_p{REFERENCE_DEPTH}" for metric,_ in METRICS])
''')
md('''## Gate Matching and Circuit Resources

The resource figures combine QAOA+ at one layer and both PC-QAOA variants at their own gate-budget depths. All curves use the same COPs with completed target results for all three methods. The table reports counts by size.

The gate-matching scatter combines both PC-QAOA variants. The diagonal is exact equality; points below it are within budget, and points above it are minimum-one-layer exceptions. These exceptions remain included under the notebook's selected setting.

NISQ total gates determine target depths. Counts in the other gate set describe the same circuits, without a second budget selection. Estimates include preparation and the repeated preparation/inverse operations in gadget mixers. They count the decomposed circuits rather than the numerical projector shortcut used during simulation.''')
code(r'''original_resources=pd.read_csv(HERE/"results/resource_estimates/final_circuit_resources.csv")
exact_resources=pd.read_csv(HERE/"results/stateprep_resource_estimates/stateprep_circuit_resources.csv")
resource_frames=common_pairs(REFERENCE_DEPTH)
if not resource_frames:
    print("Resource comparisons require completed target results from all three methods.")
else:
    first_frame=resource_frames[METHODS[0]]
    display(first_frame.groupby("n_x").size().rename("Common COPs").to_frame())
    fig,ax=plt.subplots(figsize=(6.6,4.8))
    low=np.inf;high=0.
    for method,frame in resource_frames.items():
        ax.scatter(frame.gate_budget,frame.actual_gates,color=COLORS[method],
                   marker=MARKERS[method],s=22,alpha=.6,label=method)
        low=min(low,frame.gate_budget.min(),frame.actual_gates.min())
        high=max(high,frame.gate_budget.max(),frame.actual_gates.max())
    low=max(1,low*.8);high*=1.2
    ax.plot([low,high],[low,high],"--",color=COLORS["QAOA+"],linewidth=1.5,
            label="QAOA+ Gate Budget")
    ax.set_xscale("log");ax.set_yscale("log")
    ax.set(xlabel=r"QAOA+ $p=1$ Gates",ylabel="PC-QAOA Gates",
           title=r"PC-QAOA Gates vs. QAOA+ $p=1$ Gates",xlim=(low,high),ylim=(low,high))
    ax.legend(frameon=False,fontsize=9)
    writer.save(fig,"gate_matching_all_methods_qaoa_p1")
    retire_figure_exports([f"gate_matching_{SLUGS[method]}_qaoa_p1" for method in METHODS])
    for gate_set,metric,ylabel,name in [("nisq","total_gates","Total Gates","total_gates_nisq"),("nisq","two_qubit_gates","CNOT Gates","cnot_gates"),("ftqc","total_gates","Total Gates","total_gates_ftqc"),("ftqc","t_gates","T Gates","t_gates"),("nisq","logical_qubits","Logical Register Qubits","logical_qubits"),("nisq","total_wires","Total Decomposition Wires","total_wires")]:
        baseline=original_resources[(original_resources.method=="QAOA+")&
                    (original_resources.gate_set==gate_set)&(original_resources.depth==1)].set_index("instance_id")
        series={"QAOA+":first_frame[["instance_id","n_x"]].copy()}
        series["QAOA+"]["value"]=series["QAOA+"].instance_id.map(baseline[metric])
        for method,frame in resource_frames.items():
            data=exact_resources if method=="PC-QAOA (StatePrep)" else original_resources
            first=data[(data.method==method)&(data.gate_set==gate_set)&(data.depth==1)].set_index("instance_id")
            fifth=data[(data.method==method)&(data.gate_set==gate_set)&(data.depth==5)].set_index("instance_id")
            resources=frame[["instance_id","n_x","target_depth"]].copy()
            if metric in ("logical_qubits","total_wires"):
                resources["value"]=resources.instance_id.map(first[metric])
            else:
                slopes=(fifth[metric]-first[metric])/4
                resources["value"]=resources.instance_id.map(first[metric])+(resources.target_depth-1)*resources.instance_id.map(slopes)
            series[method]=resources
        valid_ids=set.intersection(*(set(frame.loc[np.isfinite(frame.value),"instance_id"])
                                     for frame in series.values()))
        if not valid_ids:continue
        fig,ax=plt.subplots(figsize=(7.0,4.2))
        for method,frame in series.items():
            frame=frame[frame.instance_id.isin(valid_ids)]
            stats=mean_ci(frame,"n_x","value")
            ax.errorbar(stats.n_x,stats["mean"],yerr=stats.ci95.fillna(0),
                        color=COLORS[method],marker=MARKERS[method],capsize=3,linewidth=2,label=method)
        ax.set(xlabel=r"Problem Size ($n$)",ylabel=ylabel,
               title=fr"{ylabel} vs. Problem Size: QAOA+ $p=1$ Budget")
        ax.set_xticks(sorted(instances.n_x.unique()))
        if metric not in ("logical_qubits","total_wires"):ax.set_yscale("log")
        ax.legend(frameon=False,fontsize=9)
        writer.save(fig,f"paired_{name}_all_methods_qaoa_p1")
        retire_figure_exports([f"paired_{name}_{SLUGS[method]}_qaoa_p1" for method in METHODS])
''')
md(r'''## Feasibility Compared With the Original Depth-Five Runs

A gate-budget target can be less than five as well as greater than five. Therefore these figures do not always compare deeper circuits with the original depth-five experiment. The tables use the same common COPs as the three-method size figures, comparing each PC-QAOA target with its own saved depth-five result.

Feasibility is not the optimized quantity: training minimizes the penalized expected cost. Final restart selection does not guarantee monotonic improvement across depths. Disjoint instances enforce every constraint structurally, while overlapping instances retain penalties. Mixing these groups, target depths and problem sizes can produce substantial variation across COPs. The reported confidence intervals estimate uncertainty in the mean, not the spread of individual COP performance.''')
code(r'''diagnostic_frames=common_pairs(REFERENCE_DEPTH)
for method,frame in diagnostic_frames.items():
    frame=frame.assign(depth_group=np.select(
        [frame.target_depth<5,frame.target_depth==5],["Below Five","Five"],default="Above Five"),
        feasibility_change=frame.p_feasible_pc-frame.original_depth5_p_feasible)
    for group_name in ["depth_group","overlap_level"]:
        table=frame.groupby(group_name).agg(
            COPs=("instance_id","size"),Mean_Target_Depth=("target_depth","mean"),
            Original_Depth_Five=("original_depth5_p_feasible","mean"),
            Gate_Budget_Target=("p_feasible_pc","mean"),
            Mean_Change=("feasibility_change","mean"),
            Across_COP_SD=("p_feasible_pc","std"))
        print(method,group_name);display(table)
''')
md('''## Uncompiled Total Optimization Time

The scatter uses the main notebook's simulated-qubit columns, colors, markers, point size and opacity. QAOA+ uses newly measured uncompiled depth-one optimization time. Each PC-QAOA total replaces the old compiled depth-one-through-five timings with the new uncompiled prefix measurements, then adds its already-recorded uncompiled extension time. Targets below five use only the required prefix. No compilation time is included, and saved reuse is not treated as zero-cost training.

These totals reconstruct the optimization workload from separate measurements. They are not elapsed times of a newly executed deep trajectory. All three curves require completed target results and the needed uncompiled prefix measurements for the same COPs. Different machines, CPU loads and optimized states can affect timings. A controlled single-machine timing benchmark requires measuring the complete trajectories there. These scatter points do not have mean confidence intervals. Problem-size scatter plots show the same observations on linear and logarithmic time axes. Horizontal offsets and small deterministic jitter separate methods and COPs without changing their problem sizes. The tables report times and paired PC-QAOA/QAOA+ time ratios by size. Total time includes optimizing every depth up to the target, not only the final circuit.''')
code(r'''timed=results[np.isfinite(results.uncompiled_total_optimization_seconds)&
              (results.uncompiled_total_optimization_seconds>0)].copy()
methods=["QAOA+",*METHODS]
parts={method:timed[timed.method==method] for method in methods}
if any(frame.empty for frame in parts.values()):
    print("Waiting for uncompiled prefix timings for QAOA+ and both PC-QAOA variants.")
else:
    ids=set.intersection(*(set(frame.instance_id) for frame in parts.values()))
    if ids:
        parts={method:frame[frame.instance_id.isin(ids)] for method,frame in parts.items()}
        combined=pd.concat(parts.values(),ignore_index=True)
        offsets=dict(zip(methods,np.linspace(-.30,.30,len(methods))))
        ticks=np.arange(int(combined.n_total.min()),int(combined.n_total.max())+1)
        fig,ax=plt.subplots(figsize=(6.8,4.0))
        for method,frame in parts.items():
            ax.scatter(frame.n_total+offsets[method],frame.uncompiled_total_optimization_seconds,
                       color=COLORS[method],marker=MARKERS[method],s=22,alpha=.62,label=method)
        ax.set_xticks(ticks);ax.set_xlim(ticks[0]-.6,ticks[-1]+.6)
        ax.set(xlabel="Simulated Qubits",ylabel="Total Optimization Time (s)",
               title="Optimization Time vs. Simulated Qubits")
        ax.legend(frameon=False);writer.save(fig,"uncompiled_total_optimization_runtime")
        # Method offsets and deterministic jitter separate observations at each size.
        for logarithmic in [False, True]:
            fig,ax=plt.subplots(figsize=(6.8,4.0))
            for method,frame in parts.items():
                frame=frame.sort_values("instance_id")
                jitter=np.random.default_rng(20261029).uniform(-.065,.065,len(frame))
                ax.scatter(frame.n_x+offsets[method]+jitter,
                           frame.uncompiled_total_optimization_seconds,
                           color=COLORS[method],marker=MARKERS[method],s=18,alpha=.62,label=method)
            ticks=sorted(combined.n_x.unique())
            ax.set_xticks(ticks);ax.set_xlim(min(ticks)-.6,max(ticks)+.6)
            if logarithmic:ax.set_yscale("log")
            ax.set(xlabel=r"Problem Size ($n$)",ylabel="Total Optimization Time (s)",
                   title=r"Optimization Time vs. Problem Size ($n$)")
            ax.legend(frameon=False)
            writer.save(fig,"uncompiled_optimization_time_by_size"+("_log" if logarithmic else ""))
        timing_by_size=combined.groupby(["n_x","method"]).uncompiled_total_optimization_seconds.agg(["mean","median","max","count"])
        print("Reconstructed Optimization Time by Problem Size (Seconds):")
        display(timing_by_size)
        timing_by_size.to_csv(OUTPUT/"optimization_time_by_size.csv")
        paired_times=combined.pivot(index=["instance_id","n_x"],columns="method",values="uncompiled_total_optimization_seconds")
        ratios=[]
        for method in METHODS:
            ratio=paired_times[method]/paired_times["QAOA+"]
            ratio=ratio.rename("time_ratio").reset_index()
            ratio["method"]=method
            ratios.append(ratio)
        timing_ratios=pd.concat(ratios,ignore_index=True)
        print("Paired PC-QAOA / QAOA+ Time Ratios by Problem Size:")
        display(timing_ratios.groupby(["n_x","method"]).time_ratio.agg(["median","mean","count"]))
        print("Common COPs with Uncompiled Timing:",len(ids))
        display(combined.groupby("method").uncompiled_total_optimization_seconds.agg(["mean","median","max","count"]))
    else:print("No common COPs have all required uncompiled timing measurements yet.")
retire_figure_exports(["uncompiled_additional_optimization_runtime",
                       "paired_optimization_time_all_methods_qaoa_p1",
                       "paired_compilation_time_all_methods_qaoa_p1",
                       "paired_recorded_training_time_all_methods_qaoa_p1"])
''')
md('''## Reproducibility

Completed depth records are saved once; frequent checkpoints contain the active optimizer state and scalar progress. NumPy uses the existing integer seed derivation with a different random sequence from JAX. Original saved depths are reused without retraining.''')
code(r'''for key,label in LABELS.items():
    for directory in [OUTPUT/key,*sorted(OUTPUT.glob(f"{key}_qaoa_p*")),*sorted(OUTPUT.glob(f"{key}_shard_*"))]:
        path=directory/"run.json"
        if path.exists():
            data=json.loads(path.read_text());print(label,directory,"Run Identifier:",data["identifier"])
print("Figure Fonts:",writer.font_status)
print("Figure Directory:",FIGURE_DIR)
''')
nb.cells=cells
nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3'}}
path=ROOT/'final_experiment/random_cops/notebooks/gate_budget_results.ipynb'
nbf.write(nb,path);print(path)
