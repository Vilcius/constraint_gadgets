"""Read-only analysis of task results, plus explicit table/figure exports."""
from collections import Counter
import json
from pathlib import Path
import pickle
import re
import shutil
import subprocess
import warnings

import numpy as np
import pandas as pd

from storage import write_json, write_pickle, atomic_bytes

COLORS = {'feasibility': '#6e6a86', 'fidelity': '#3e8fb0',
          'lambda_0.5': '#9063cd', 'lambda_1': '#31748f', 'lambda_2': '#eb6f92',
          'range': '#3e8fb0', 'maximum': '#9063cd', 'local': '#31748f', 'paper': '#6e6a86'}
LABELS = {'feasibility': r'$L_{\mathrm{feas}}$', 'fidelity': r'$L_{\mathrm{fid}}$',
          'lambda_0.5': r'$L_{0.5}$', 'lambda_1': r'$L_1$', 'lambda_2': r'$L_2$'}


def read_results(output):
    output = Path(output)
    if not (output/'run.json').exists():
        raise FileNotFoundError(f'No run.json in {output}; generate or run a study first')
    run = json.loads((output/'run.json').read_text())
    result = dict(run=run, output=output)
    for study in ['a', 'b']:
        manifest_path = output/f'manifest_{study}.json'
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'instances': []}
        instances = {r['id']: r for r in manifest['instances']}
        summaries, depths = [], []
        for path in sorted((output/study).glob('*/*/result.json')):
            row = json.loads(path.read_text())
            if row['instance_id'] not in instances:
                continue
            summaries.append(row)
        # Include completed depths of interrupted tasks, not only completed tasks.
        for path in sorted((output/study).glob('*/*/depths.json')):
            instance_id, condition = path.parent.parent.name, path.parent.name
            if instance_id not in instances:
                continue
            metadata = instances[instance_id]
            for row in json.loads(path.read_text()):
                depths.append(dict(instance_id=instance_id, condition=condition,
                    size=metadata.get('support', metadata.get('n_x')),
                    **{k: v for k, v in row.items()
                       if k not in ['angles', 'history', 'restarts', 'counts', 'probabilities']}))
        result[study], result[f'{study}_depths'] = pd.DataFrame(summaries), pd.DataFrame(depths)
        result[f'{study}_instances'] = pd.DataFrame(manifest['instances'])
    return result


def status_summary(output):
    data = read_results(output)
    result = dict(profile=data['run']['config']['profile'])
    for study in ['a', 'b']:
        count = len(data[f'{study}_instances'])
        settings = data['run']['config'][f'study_{study}']
        conditions = len(settings['losses']) if study == 'a' else 3*len(settings['multipliers'])+1
        counts = Counter()
        for path in (Path(output)/study).glob('*/*/status.json'):
            counts[json.loads(path.read_text())['status']] += 1
        expected = count*conditions
        counts['pending'] = max(0, expected-sum(counts.values()))
        frame = data[study]
        result[study] = dict(expected=expected, completed_results=len(frame), statuses=dict(counts))
        if len(frame):
            result[study].update(compile_seconds=float(frame.compile_seconds.sum()),
                optimize_seconds=float(frame.optimize_seconds.sum()),
                largest_worker_peak_rss_mb=float(frame.process_peak_rss_mb.max()))
    return result


def select_loss(frame, instance_ids, conditions):
    expected = {(i, c) for i in instance_ids for c in conditions}
    actual = set(zip(frame.instance_id, frame.condition)) if len(frame) else set()
    if not expected or actual != expected or len(frame) != len(expected):
        return None
    means = frame.groupby('condition').fidelity.mean()
    return max(conditions, key=lambda name: means[name])


def select_penalty(frame, instance_ids, condition_names, depth):
    expected = {(i, c) for i in instance_ids for c in condition_names}
    actual = set(zip(frame.instance_id, frame.condition)) if len(frame) else set()
    if not expected or actual != expected or len(frame) != len(expected):
        return None
    if not (frame.depth == depth).all():
        return None
    eligible = frame[frame.selectable]
    # This intentionally uses sampled p_optimal, never exact_p_optimal.
    return eligible.groupby('condition').p_optimal.mean().idxmax()


def ranking(frame, metric):
    if frame.empty:
        return pd.DataFrame()
    return frame.groupby('condition')[metric].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)


def export_results(output):
    output = Path(output)
    data = read_results(output)
    for study, metric in [('a', 'fidelity'), ('b', 'p_optimal')]:
        frame = data[study]
        if frame.empty:
            continue
        atomic_bytes(output/f'{study}_results.csv', frame.to_csv(index=False).encode())
        atomic_bytes(output/f'{study}_depths.csv', data[f'{study}_depths'].to_csv(index=False).encode())
        table = ranking(frame, metric)
        atomic_bytes(output/f'{study}_ranking.csv', table.to_csv().encode())
    config = data['run']['config']
    a_instances = data['a_instances']
    winner_a = select_loss(data['a'], a_instances.id if len(a_instances) else [], config['study_a']['losses'])
    if winner_a:
        database = {}
        for row in a_instances.to_dict('records'):
            path = output/'a'/row['id']/winner_a/'gadget.pkl'
            database[row['constraint']] = pickle.loads(path.read_bytes())
        write_pickle(output/'selected_vcg_db.pkl', database)
        write_json(output/'selected_loss.json', dict(condition=winner_a,
            mean_fidelity=float(data['a'][data['a'].condition == winner_a].fidelity.mean()),
            criterion='mean total fidelity across constraints', run_id=data['run']['identifier'],
            database='selected_vcg_db.pkl', n_constraints=len(database)))
        print(f'Selected VCG loss: {winner_a}')
    b_names = [f'{r}_{m:g}' for r in ['range', 'maximum', 'local']
               for m in config['study_b']['multipliers']] + ['paper']
    b_instances = data['b_instances']
    winner_b = select_penalty(data['b'], b_instances.id if len(b_instances) else [],
                              b_names, config['study_b']['max_depth'])
    if winner_b:
        selected = data['b'][data['b'].condition == winner_b]
        write_json(output/'selected_penalty.json', dict(condition=winner_b,
            rule=selected.iloc[0]['rule'], multiplier=float(selected.iloc[0].multiplier),
            mean_sampled_p_optimal=float(selected.p_optimal.mean()),
            criterion='mean sampled P(opt) at final depth', depth=config['study_b']['max_depth'],
            shots=config['study_b']['shots'], n_cops=len(selected), run_id=data['run']['identifier']))
        print(f'Selected penalty: {winner_b}')
    return winner_a, winner_b


class FigureWriter:
    """PGF/LaTeX PDF + matching PNG previews; explicit Computer Modern fallback."""
    def __init__(self, directory, latex=True):
        import matplotlib as mpl
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.latex = latex and bool(shutil.which('pdflatex'))
        self.font_status = 'LaTeX Computer Modern (PGF)' if self.latex else 'Matplotlib Computer Modern math / DejaVu Serif'
        mpl.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
            'mathtext.fontset': 'cm', 'font.size': 11, 'axes.titlesize': 12,
            'axes.spines.top': False, 'axes.spines.right': False,
            'axes.grid': True, 'grid.alpha': 0.22, 'grid.linewidth': 0.5,
            'figure.facecolor': 'white', 'axes.facecolor': 'white',
            'savefig.facecolor': 'white', 'savefig.dpi': 300, 'pdf.fonttype': 42,
            'pgf.texsystem': 'pdflatex', 'pgf.rcfonts': False, 'text.usetex': False})

    def save(self, figure, name):
        import matplotlib.pyplot as plt
        pdf, png = self.directory/f'{name}.pdf', self.directory/f'{name}.png'
        figure.tight_layout()
        if self.latex:
            try:
                figure.savefig(pdf, backend='pgf', bbox_inches='tight')
                if shutil.which('pdftocairo'):
                    subprocess.run(['pdftocairo', '-png', '-singlefile', '-r', '300',
                                    str(pdf), str(png.with_suffix(''))], check=True, capture_output=True)
                else:
                    figure.savefig(png, backend='pgf', bbox_inches='tight')
            except Exception as error:
                self.latex = False
                self.font_status = 'Matplotlib Computer Modern math / DejaVu Serif (LaTeX failed)'
                warnings.warn(f'LaTeX figure rendering failed; using built-in fonts: {error}')
        if not self.latex:
            figure.savefig(pdf, bbox_inches='tight')
            figure.savefig(png, bbox_inches='tight')
        write_json(self.directory/'font_status.json', dict(font=self.font_status))
        plt.close(figure)
        return png


def summary_plot(frame, x, y, group, title, ylabel):
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(6.4, 3.7))
    markers = ['o', 's', '^', 'D', 'v']
    for index, (label, subset) in enumerate(frame.groupby(group, sort=False)):
        stats = subset.groupby(x)[y].agg(['mean', 'std'])
        axis.errorbar(stats.index, stats['mean'], yerr=stats['std'].fillna(0),
            marker=markers[index % len(markers)], capsize=3, linewidth=1.5,
            color=COLORS.get(label, COLORS.get(str(label).split('_')[0])),
            label=LABELS.get(label, str(label)))
    axis.set(xlabel={'support': 'Constraint Support Size', 'depth': 'QAOA Depth',
                     'n_x': 'Decision Variables', 'multiplier': 'Penalty Multiplier'}.get(x, x),
             ylabel=ylabel, title=title)
    axis.legend(fontsize=9, frameon=False)
    if x in ['depth', 'support', 'n_x']:
        axis.set_xticks(sorted(frame[x].unique()))
    return figure



def constraint_title(constraint, terms_per_line=4):
    """Render saved polynomial constraints as math, wrapping long expressions."""
    lhs, relation, rhs = re.split(r'\s*(<=|>=|==|<|>)\s*', constraint)
    terms = re.findall(r'[+-]?\s*[^+-]+', lhs)
    rendered = []
    for term in terms:
        term = re.sub(r'x_(\d+)\*x_\1\b', r'x_\1^{2}', term.strip())
        term = re.sub(r'(^|[+-]\s*)1\*', r'\1', term)
        term = re.sub(r'x_(\d+)', r'x_{\1}', term)
        rendered.append(term.replace('*', r'\, '))
    lines = [' '.join(rendered[i:i+terms_per_line])
             for i in range(0, len(rendered), terms_per_line)]
    symbol = {'<=': r'\leq', '>=': r'\geq', '==': '=', '<': '<', '>': '>'}[relation]
    lines[-1] += f' {symbol} {rhs}'
    return '\n'.join(f'${line}$' for line in lines)


def figures(data, section, writer):
    """Return saved PNG paths for one notebook section; no training or selection."""
    import matplotlib.pyplot as plt
    saved = []
    a, b = data['a'], data['b']
    def save(fig, name):
        saved.append(writer.save(fig, name))
    if section == 'datasets':
        for study in ['a', 'b']:
            frame = data[f'{study}_instances']
            if frame.empty:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(8, 3.3))
            if study == 'a':
                frame.family.value_counts().plot.bar(ax=axes[0], color='#9063cd')
                axes[0].set_ylabel('Constraints')
            else:
                counts = Counter(f for families in frame.families for f in families)
                pd.Series(counts).plot.bar(ax=axes[0], color='#31748f')
                axes[0].set_ylabel('Constraint Occurrences')
            axes[0].set_xlabel('Constraint Family')
            axes[0].tick_params(axis='x', labelrotation=45, labelsize=8)
            axes[1].hist(frame.feasible_fraction, bins=10, color='#3e8fb0', edgecolor='white')
            axes[1].set(xlabel='Feasible Fraction', ylabel='Instances')
            save(fig, f'{study}_dataset')
    elif section == 'loss' and not a.empty:
        save(summary_plot(a, 'support', 'fidelity', 'condition', 'VCG Fidelity', 'Total Fidelity'), 'a_fidelity')
        for family, frame in a.groupby('family'):
            save(summary_plot(frame, 'support', 'fidelity', 'condition', family.title(), 'Total Fidelity'), f'a_fidelity_{family}')
        fig, ax = plt.subplots(figsize=(5.8, 3.8))
        for name, frame in a.groupby('condition'):
            ax.scatter(frame.p_feasible, frame.fidelity, s=25,
                       color=COLORS[name], label=LABELS[name], alpha=0.8)
        ax.set(xlabel='Feasible Probability', ylabel='Total Fidelity', title='Fidelity and Feasibility')
        ax.legend(frameon=False, fontsize=9)
        save(fig, 'a_fidelity_feasibility')
        for column, label in [('selected_depth', 'Selected Depth'), ('optimize_seconds', 'Optimization Time (s)')]:
            save(summary_plot(a, 'support', column, 'condition', label, label), f'a_{column}')
        depths = data['a_depths']
        if len(depths):
            save(summary_plot(depths, 'depth', 'fidelity', 'condition', 'Fidelity by Depth', 'Total Fidelity'), 'a_depth_fidelity')
    elif section == 'examples' and not a.empty:
        for _, selected in a.sort_values('fidelity').iloc[[0, len(a)//2, len(a)-1]].drop_duplicates().iterrows():
            directory = data['output']/'a'/selected.instance_id/selected.condition
            entry = pickle.loads((directory/'gadget.pkl').read_bytes())
            instance = data['a_instances'].set_index('id').loc[selected.instance_id]
            mask = np.asarray(instance.feasible_mask)
            probabilities = np.abs(entry['state'])**2
            formula = constraint_title(instance.constraint)
            fig, ax = plt.subplots(figsize=(7, 3 + 0.3*len(formula.splitlines())))
            ax.bar(np.arange(len(probabilities)), probabilities,
                   color=np.where(mask, '#3e8fb0', '#eb6f92'), width=1)
            ax.set(xlabel='Basis-State Index', ylabel='Probability',
                   title='Probability Distribution: ' + LABELS[selected.condition] + '\n' + formula)
            save(fig, f'a_distribution_{selected.instance_id}_{selected.condition}')
        selected = a.iloc[0]
        directory = data['output']/'a'/selected.instance_id/selected.condition
        depth = json.loads((directory/'depths.json').read_text())[0]
        history = pd.DataFrame(depth['history'])
        fig, ax = plt.subplots(figsize=(6, 3))
        for restart, frame in history.groupby('restart'):
            palette = ['#9063cd', '#3e8fb0', '#31748f', '#eb6f92', '#f6c177']
            ax.plot(frame.step, frame.loss, color=palette[int(restart) % len(palette)],
                    label=f'Restart {restart+1}')
        ax.set(xlabel='Optimizer Step (Before Update)', ylabel='Training Loss', title='Example VCG Training History')
        if history.restart.nunique() <= 5:
            ax.legend(frameon=False, fontsize=8)
        save(fig, 'a_training_history')
    elif section == 'penalty' and not b.empty:
        eligible = b[b.selectable]
        save(summary_plot(eligible, 'multiplier', 'p_optimal', 'rule', 'Penalty Tuning at Final Depth', r'Sampled $P(\mathrm{opt})$'), 'b_multipliers')
        depths = data['b_depths']
        for rule in ['range', 'maximum', 'local']:
            subset = depths[depths.condition.str.startswith(rule+'_') | (depths.condition == 'paper')]
            save(summary_plot(subset, 'depth', 'p_optimal', 'condition', rule.title(), r'Sampled $P(\mathrm{opt})$'), f'b_depth_{rule}')
        best = eligible.groupby('condition').p_optimal.mean().idxmax()
        subset = b[b.condition.isin([best, 'paper'])]
        save(summary_plot(subset, 'n_x', 'p_optimal', 'condition', 'Best Observed Setting and Manuscript Reference', r'Sampled $P(\mathrm{opt})$'), 'b_sizes')
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        ax.scatter(b.exact_p_optimal, b.p_optimal, color='#9063cd', s=18, alpha=0.7)
        ax.plot([0, 1], [0, 1], '--', color='#6e6a86', linewidth=1)
        ax.set(xlabel=r'Exact $P(\mathrm{opt})$', ylabel=r'Sampled $P(\mathrm{opt})$')
        save(fig, 'b_sampled_exact')
    elif section == 'resources' and not b.empty:
        instances = data['b_instances']
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
        axes[0].scatter(instances.n_x, instances.n_slack, color='#3e8fb0', label='PenaltyQAOA Slack')
        axes[0].scatter(instances.n_x, instances.pc_n_slack, color='#9063cd', marker='x', label='PC-QAOA Partition Slack')
        axes[0].set(xlabel='Decision Variables', ylabel='Slack Qubits')
        axes[0].legend(frameon=False, fontsize=8)
        axes[1].scatter(b.n_total, b.optimize_seconds, color='#31748f', s=15)
        axes[1].set(xlabel='Total Simulated Qubits', ylabel='Optimization Time (s)')
        save(fig, 'b_qubits_runtime')
    return saved
