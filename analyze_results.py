import re
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

#|%%--%%| <JT9SEL7RBJ|oRp6qeenrv>
r"""°°°
Single Constraint Data
°°°"""
#|%%--%%| <oRp6qeenrv|fQnYovfAXO>


ops = {'equals': '==', 'geq': '>=', 'leq': '<='}


def calculate_probability_optimal(counts: dict, outcomes: list):
    """
    Calculate the probability of getting the optimal solution from the counts.
    Args:
        counts (dict): The counts of the qubits.
        outcomes (list): The outcomes of the qubits.
    Returns:
        float: The probability of getting the optimal solution.
    """
    total_counts = sum(counts.values())
    tot_prob = 0
    for i, outcome in enumerate(outcomes):
        if outcome == -1:
            tot_prob += counts[list(counts.keys())[i]]
    return tot_prob / total_counts


def calculate_probability_optimal_qubo(counts: dict, optimal_x: list | None, n_c: int = 1, single_flag: bool = True):
    """
    Calculate the probability of getting the optimal solution from the counts.
    Args:
        counts (dict): The counts of the qubits.
        optimal_x (list): The optimal solution.
    Returns:
        float: The probability of getting the optimal solution.
    """
    if optimal_x is None:
        return -1
    if single_flag:
        n_c = 1
    total_counts = sum(counts.values())
    tot_prob = 0
    for i, outcome in enumerate(counts):
        if outcome[:-n_c] in optimal_x and outcome[-n_c:] == '0' * n_c:
            tot_prob += counts[outcome]
    return tot_prob / total_counts


def read_single_constraint_data(corp: str = '', results_dir: str = './results/') -> pd.DataFrame:
    """
    Read the single constraint data from the results directory.
    Args:
        corp (str): The prefix of the file name.
        results_dir (str): The directory where the results are stored.
    Returns:
        pd.DataFrame: The single constraint data.
    """
    df = pd.DataFrame()
    for op in ops:
        if corp == '':
            temp_df = pd.read_pickle(f'{results_dir}{corp}single_constraint_data_{op}.pkl')
            temp_df['problem'] = op
            temp_df['prob_opt'] = temp_df.apply(lambda x: calculate_probability_optimal(x['counts'], x['outcomes']), axis=1)
            temp_df['depth'] = temp_df['resources'].apply(lambda x: x.depth)
            temp_df['num_gates'] = temp_df['resources'].apply(lambda x: x.num_gates)
            temp_df['num_gates_2'] = temp_df['resources'].apply(lambda x: x.gate_sizes[2])
        elif corp == 'qubo_':
            temp_df = pd.read_pickle(f'{results_dir}{corp}single_constraint_data_{op}_3.pkl')
            temp_df['problem'] = op
            temp_df['constraints'] = temp_df['constraints'].apply(lambda x: [x])
            temp_df['prob_opt'] = temp_df.apply(lambda x: calculate_probability_optimal_qubo(x['counts'], x['optimal_x']), axis=1)
        # change 'AR' to float
        temp_df['AR'] = temp_df['AR'].apply(lambda x: np.float64(x))
        df = pd.concat([df, temp_df], ignore_index=True)
    return df


state_df = read_single_constraint_data()
qubo_df = read_single_constraint_data('qubo_')
#set the outcomes in qubo_df to the outcomes in state_df for the corresponding 'constraints'
qubo_df['outcomes'] = qubo_df['constraints'].apply(lambda x: state_df[state_df['constraints'].map(tuple) == tuple(x)]['outcomes'].values[0] if len(state_df[state_df['constraints'].map(tuple) == tuple(x)]) > 0 else None)
state_df = state_df[state_df['angle_strategy'] == 'ma-QAOA']
qubo_df = qubo_df[qubo_df['angle_strategy'] == 'ma-QAOA']


#|%%--%%| <fQnYovfAXO|DQ7E91yFUl>
r"""°°°
Plotting Functions for Single Constraint Data
°°°"""
#|%%--%%| <DQ7E91yFUl|4ADDj0liE4>


def plot_single_constraint_ar(df: pd.DataFrame, problem: str, hue='b', qubo: bool = False, moon: bool = True, confident: bool = False, save: bool = False, file: str | None = None) -> None:
    """
    Plot the approximation ratio vs number of variables for a given problem.
    Args:
        df (pd.DataFrame): The dataframe containing the data.
        problem (str): The problem to plot.
        hue (str): The column to use for the hue.
        qubo (bool): Whether to plot the qubo data or the state data.
        moon (bool): Whether to use the moon style or the dawn style.
        confident (bool): Whether to use the confident style or not.
        save (bool): Whether to save the plot or not.
        file (str): The name of the file to save the plot as.
    Returns:
        None
    """
    ops = {'equals': '=', 'geq': r'$\geq$', 'leq': r'$\leq$', 'less': '<', 'greater': '>'}
    plots_dir = './plots/'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    if moon:
        plt.style.use('rose-pine-moon')
        colors = ['C0', 'C4', 'C2', 'C1', 'C3', 'C5']
    else:
        plt.style.use('rose-pine-dawn')
        colors = ['C0', 'C5', 'C2', 'C1', 'C3', 'C4']
    dff = df.copy()
    dff = dff[dff['problem'] == problem]
    dff['b'] = dff['constraints'].apply(lambda x: r'$b=$'+f'{x[0][-1]}')
    dff['constraints'] = dff['constraints'].map(tuple)
    fig = plt.figure(figsize=(15, 8))
    if qubo:
        ax = sns.boxplot(data=dff, x='n_x', y='AR', hue=hue, palette=colors)
    else:
        ax = sns.boxplot(data=dff, x='n_x', y='AR', hue=hue, palette=colors, fill=False, linewidth=5)

    if moon:
        plt.background = 'black'
        plt.xlabel('Number of variables: '+r'$n_x$', size=18)
        plt.ylabel('AR', size=18)
        plt.tick_params(axis='both', labelsize=18)
        plt.xticks([1, 2, 3, 4, 5])
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1), frameon=True, prop={'size': 12}, title=r'$\sum_{i=0}^{n_x - 1} x_i$' + f'{ops[problem]}' + r'$b$')  # , title='Number of Constraints/Layers')
        plt.title(f'{problem.capitalize()} Constraint Gadget vs AR', size=20)
        plt.tight_layout()
        if save:
            plt.savefig(f'{plots_dir}{problem}_AR_vs_n_x_moon.png')
        plt.show()
        plt.close()
    else:
        plt.background = 'white'
        plt.xlabel('Number of variables: '+r'$n_x$', size=20)
        plt.ylabel('AR', size=20)
        plt.xticks([0, 1, 2, 3, 4, 5])
        plt.tick_params(axis='both', labelsize=18)
        if qubo:
            # Build custom legend for hatch + color
            # legend_handles = []
            # seen = set()
            # for g in group_labels:
            #     hue = g.split('_')[0]
            #     layer = g.split('L')[1]
            #     label = r"$b=$" + f"{hue}"+r", $p=$" + f"{layer}"
            #     if label not in seen:
            #         legend_handles.append(Patch(facecolor=group_colors[g],
            #                                     hatch=group_to_hatch[g],
            #                                     edgecolor='black',
            #                                     label=label))
            #         seen.add(label)
            plt.title(f'QCBOs with {problem.capitalize()} Constraints vs Approximation Ratio', size=20)
            # plt.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.15, 1), frameon=True, prop={'size': 16}, title=r'$\sum_{i=0}^{n_x - 1} x_i$' + f'{ops[problem]}' + r'$b$', title_fontsize=18)  # , title='Number of Constraints/Layers')
            plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1), frameon=True, prop={'size': 16}, title=r'$\sum_{i=0}^{n_x - 1} x_i$' + f'{ops[problem]}' + r'$b$', title_fontsize=18)  # , title='Number of Constraints/Layers')
        else:
            plt.title(f'{problem.capitalize()} Constraints vs Approximation Ratio', size=20)
            plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1), frameon=True, prop={'size': 16}, title=r'$\sum_{i=0}^{n_x - 1} x_i$' + f'{ops[problem]}' + r'$b$', title_fontsize=18)  # , title='Number of Constraints/Layers')
        plt.tight_layout()
        ax.patch.set_facecolor('white')
        plt.rcParams['figure.facecolor'] = 'white'
        fig.patch.set_facecolor('white')
        if save:
            if file is not None:
                plt.savefig(f'{plots_dir}{file}_AR_dawn.png', facecolor='white')
            elif qubo:
                plt.savefig(f'{plots_dir}{problem}_qubo_AR_dawn.png', facecolor='white')
            else:
                plt.savefig(f'{plots_dir}{problem}_AR_dawn.png', facecolor='white')
        plt.show()
        plt.close()


def plot_single_constraint_prob_opt(df: pd.DataFrame, problem: str, hue='b', qubo: bool = False, moon: bool = True, confident: bool = False, save: bool = False, file: str | None = None) -> None:
    """
    Plot the probability of optimal vs number of variables for a given problem.
    Args:
        df (pd.DataFrame): The dataframe containing the data.
        problem (str): The problem to plot.
        hue (str): The column to use for the hue.
        qubo (bool): Whether to plot the qubo data or the state data.
        moon (bool): Whether to use the moon style or the dawn style.
        confident (bool): Whether to use the confident style or not.
        save (bool): Whether to save the plot or not.
        file (str): The name of the file to save the plot as.
    Returns:
        None
    """
    ops = {'equals': '=', 'geq': r'$\geq$', 'leq': r'$\leq$', 'less': '<', 'greater': '>'}
    plots_dir = './plots/'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    if moon:
        plt.style.use('rose-pine-moon')
        colors = ['C0', 'C4', 'C2', 'C1', 'C3', 'C5']
    else:
        plt.style.use('rose-pine-dawn')
        colors = ['C0', 'C5', 'C2', 'C1', 'C3', 'C4']
    dff = df.copy()
    dff = dff[dff['problem'] == problem]
    dff['b'] = dff['constraints'].apply(lambda x: r'$b=$'+f'{x[0][-1]}')
    dff['constraints'] = dff['constraints'].map(tuple)
    fig = plt.figure(figsize=(15, 8))
    if qubo:
        ax = sns.boxplot(data=dff, x='n_x', y='prob_opt', hue=hue, palette=colors)
    else:
        ax = sns.boxplot(data=dff, x='n_x', y='prob_opt', hue=hue, palette=colors, fill=False, linewidth=5)

    if moon:
        plt.background = 'black'
        plt.xlabel('Number of variables: '+r'$n_x$', size=18)
        plt.ylabel('AR', size=18)
        plt.xticks([1, 2, 3, 4, 5])
        plt.tick_params(axis='both', labelsize=18)
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1), frameon=True, prop={'size': 12}, title=r'$\sum_{i=0}^{n_x - 1} x_i$' + f'{ops[problem]}' + r'$b$')
        if qubo:
            plt.title(f'QCBOs with {problem.capitalize()} Constraints vs Probability of Optimal Solution', size=20)
        else:
            plt.title(f'{problem.capitalize()} Constraints vs Probability of Optimal Solution', size=20)
        plt.tight_layout()
        if save:
            if qubo:
                plt.savefig(f'{plots_dir}{problem}_qubo_prob_opt_vs_n_x_moon.png')
            else:
                plt.savefig(f'{plots_dir}{problem}_prob_opt_vs_n_x_moon.png')
        plt.show()
        plt.close()
    else:
        plt.background = 'white'
        plt.xlabel('Number of variables: '+r'$n_x$', size=20)
        plt.ylabel('Probability of Optimal Solution', size=20)
        plt.xticks([0, 1, 2, 3, 4, 5])
        plt.tick_params(axis='both', labelsize=18)
        if qubo:
            # Build custom legend for hatch + color
            # legend_handles = []
            # seen = set()
            # for g in group_labels:
            #     hue = g.split('_')[0]
            #     layer = g.split('L')[1]
            #     label = r"$b=$" + f"{hue}"+r", $p=$" + f"{layer}"
            #     if label not in seen:
            #         legend_handles.append(Patch(facecolor=group_colors[g],
            #                                     hatch=group_to_hatch[g],
            #                                     edgecolor='black',
            #                                     label=label))
            #         seen.add(label)
            plt.title(f'QCBOs with {problem.capitalize()} Constraints vs Probability of Optimal Solution', size=20)
            # plt.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.15, 1), frameon=True, prop={'size': 16}, title=r'$\sum_{i=0}^{n_x - 1} x_i$' + f'{ops[problem]}' + r'$b$', title_fontsize=18)  # , title='Number of Constraints/Layers')
            plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1), frameon=True, prop={'size': 16}, title=r'$\sum_{i=0}^{n_x - 1} x_i$' + f'{ops[problem]}' + r'$b$', title_fontsize=18)  # , title='Number of Constraints/Layers')
        else:
            plt.title(f'{problem.capitalize()} Constraints vs Probability of Optimal Solution', size=20)
            plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1), frameon=True, prop={'size': 16}, title=r'$\sum_{i=0}^{n_x - 1} x_i$' + f'{ops[problem]}' + r'$b$', title_fontsize=18)  # , title='Number of Constraints/Layers')
        plt.tight_layout()
        ax.patch.set_facecolor('white')
        plt.rcParams['figure.facecolor'] = 'white'
        fig.patch.set_facecolor('white')
        if save:
            if file is not None:
                plt.savefig(f'{plots_dir}{file}_prob_opt_dawn.png', facecolor='white')
            elif qubo:
                plt.savefig(f'{plots_dir}{problem}_qubo_prob_opt_vs_n_x_dawn.png', facecolor='white')
            else:
                plt.savefig(f'{plots_dir}{problem}_prob_opt_vs_n_x_dawn.png', facecolor='white')

        plt.show()
        plt.close()


def plot_single_constraint_counts(df: pd.DataFrame, constraint: list[str], problem: str, qubo: str | None = None, moon: bool = True, save: bool = False, file: str | None = None) -> None:
    """
    Plot the counts for a given constraint.
    Args:
        df (pd.DataFrame): The dataframe containing the data.
        constraint (list[str]): The constraint to plot.
        problem (str): The problem to plot.
        qubo (str | None): The qubo string to plot. If None, plot all qubo strings.
        moon (bool): Whether to use the moon style or the dawn style.
        save (bool): Whether to save the plot or not.
        file (str | None): The name of the file to save the plot as.
    Returns:
        None
    """
    plots_dir = './plots/'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    if moon:
        plt.style.use('rose-pine-moon')
        colors = ['C0', 'C5', 'C2', 'C1', 'C3', 'C4']
        outcome_color = 'C1'
    else:
        plt.style.use('rose-pine-dawn')
        colors = ['C0', 'C5', 'C2', 'C1', 'C3', 'C4']
        #
    dff = df.copy()
    dff = dff[dff['constraints'].map(tuple) == tuple(constraint)]
    outcomes = dff['outcomes'].iloc[0]
    outcomes = [f'{i:0{dff["n_x"].iloc[0]+dff['n_c'].iloc[0]}b}' for i, o in enumerate(outcomes) if o == -1]
    if qubo is not None:
        dff = dff[dff['qubo_string'] == qubo]
        optimal = dff['optimal_x'].iloc[0]
        if optimal is not None:
            optimal = [f'{o}{'0'*dff["n_c"].iloc[0]}' for o in optimal]
    else:
        optimal = None
    dff['keys'] = dff['counts'].apply(lambda x: list(x.keys()))
    dff = dff.explode('keys')
    dff['values'] = dff.apply(lambda x: x['counts'][x['keys']], axis=1)

    fig = plt.figure(figsize=(15, 8), facecolor='white')
    # ax = sns.barplot(data=dff, x='keys', y='values', hue='type', palette=colors)
    ax = sns.barplot(data=dff, x='keys', y='values', hue='angle_strategy', palette=colors)
    labels = []
    colors = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        if optimal is not None and text in optimal:
            labels.append(r'$\mathbf{|'+text+r'\rangle}$')
            colors.append('blue')
        elif text in outcomes:
            labels.append(r'$|'+text+r'\rangle$')
            colors.append('blue')
        else:
            labels.append(r'$|'+text+r'\rangle$')
            colors.append('red')
    ax.set_xticklabels(labels, fontsize=18)
    for tick_label, color in zip(ax.get_xticklabels(), colors):
        tick_label.set_color(color)
    if moon:
        # plot vertical lines of indices of outcomes (in binary strings) that are 0
        plt.background = 'black'
        plt.xlabel('States', size=18)
        plt.ylabel('Counts', size=18)
        plt.tick_params(axis='both', labelsize=18)
        plt.tight_layout()
        if save:
            plt.savefig(f'{plots_dir}{problem}_prob_opt_vs_n_x_moon.png')
        plt.show()
        plt.close()
    else:
        plt.xlabel('States: ' + r'$|'+' '.join([f'x_{i}' for i in range(dff["n_x"].iloc[0])]) + ' '.join([f'v_{i}' for i in range(dff['n_c'].iloc[0])]) + r' \rangle$', size=18)
        plt.ylabel('Counts', size=18)
        plt.tick_params(axis='both', labelsize=18)
        plt.xticks(rotation=90)
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1), frameon=True, prop={'size': 18}, title='Constraint\nGadget', title_fontsize=18)

        def _latex_constraint(constraint):
            constraint = constraint.replace('<=', '\leq')
            constraint = constraint.replace('>=', '\geq')
            constraint = constraint.replace('==', '=')
            constraint = constraint.replace('*', '')
            return constraint

        qubo_title = ''
        if qubo is not None:
            qubo_title = r'QUBO: $' + _latex_constraint(qubo) + r'$' + '\n'
        plt.title(f'{qubo_title}' + r'Constraints $'+f'{', '.join([_latex_constraint(c) for c in constraint])}'+r'$ vs counts', size=20)
        ax.patch.set_facecolor('white')
        plt.rcParams['figure.facecolor'] = 'white'
        fig.patch.set_facecolor('white')
        plt.tight_layout()
        if save:
            if file is not None:
                plt.savefig(f'{plots_dir}{file}_counts_dawn.png', facecolor='white')
            elif qubo is None:
                plt.savefig(f'{plots_dir}alex_constr_vs_{df[df["constraints"].map(tuple) == tuple(constraint)].index[0]}_counts_dawn.png', facecolor='white')
            else:
                plt.savefig(f'{plots_dir}alex_qubo_vs_{df[df["constraints"].map(tuple) == tuple(constraint)].index[0]}_counts_dawn.png', facecolor='white')
        plt.show()
        plt.close()


qubo_df = qubo_df[qubo_df['n_layers'] == 1]
for problem in ['equals', 'leq', 'geq']:
    plot_single_constraint_ar(state_df, problem, hue='b', qubo=False, moon=False, confident=False, save=True)
    plot_single_constraint_ar(qubo_df, problem, hue='b', qubo=True, moon=False, confident=False, save=True)
    plot_single_constraint_prob_opt(state_df, problem, hue='b', qubo=False, moon=False, confident=False, save=True)
    plot_single_constraint_prob_opt(qubo_df, problem, hue='b', qubo=True, moon=False, confident=False, save=True)

#|%%--%%| <4ADDj0liE4|O4Et9M9NAy>
r"""°°°
Two Constraint Data
°°°"""
#|%%--%%| <O4Et9M9NAy|qOL3rDxKxP>


def read_two_constraint_data(corp: str = '', results_dir: str = './results/') -> pd.DataFrame:
    """
    Read the two constraint data from the results directory.
    This data is for n_layers = 1 of the GM-QAOA circuit.
    Args:
        corp (str): The prefix of the file name.
        results_dir (str): The directory where the results are stored.
    Returns:
        pd.DataFrame: The two constraint data.
    """
    df = pd.DataFrame()
    for support in [1, 2, 3]:
        for single in [True, False]:
            temp_df = pd.read_pickle(f'{results_dir}{corp}two_constraint_support_{support}_single_{single}.pkl')
            if corp == '':
                temp_df['prob_opt'] = temp_df.apply(lambda x: calculate_probability_optimal(x['counts'], x['outcomes']), axis=1)
                temp_df['depth'] = temp_df['resources'].apply(lambda x: x.depth)
                temp_df['num_gates'] = temp_df['resources'].apply(lambda x: x.num_gates)
                temp_df['num_gates_2'] = temp_df['resources'].apply(lambda x: x.gate_sizes[2])
            elif corp == 'qubo_':
                temp_df['constraints'] = temp_df['constraints'].apply(lambda x: [x])
                temp_df['prob_opt'] = temp_df.apply(lambda x: calculate_probability_optimal_qubo(x['counts'], x['optimal_x'], x['n_c'], x['single_flag']), axis=1)
            temp_df['support'] = support
            temp_df['AR'] = temp_df['AR'].astype(float)
            df = pd.concat([df, temp_df], ignore_index=True)
    return df


def read_two_constraint_data_2(corp='qubo_', results_dir: str = './results/') -> pd.DataFrame:
    """
    Read the two constraint data from the results directory.
    This data is for n_layers = 2 of the GM-QAOA circuit.
    Args:
        corp (str): The prefix of the file name.
        results_dir (str): The directory where the results are stored.
    Returns:
        pd.DataFrame: The two constraint data.
    """
    df = pd.DataFrame()
    for support in [1, 2, 3]:
        for single in [True, False]:
            temp_df = pd.read_pickle(f'{results_dir}{corp}two_constraint_support_{support}_single_{single}_n_layers_2.pkl')
            if corp == '':
                temp_df['prob_opt'] = temp_df.apply(lambda x: calculate_probability_optimal(x['counts'], x['outcomes']), axis=1)
                temp_df['depth'] = temp_df['resources'].apply(lambda x: x.depth)
                temp_df['num_gates'] = temp_df['resources'].apply(lambda x: x.num_gates)
                temp_df['num_gates_2'] = temp_df['resources'].apply(lambda x: x.gate_sizes[2])
            elif corp == 'qubo_':
                temp_df['constraints'] = temp_df['constraints'].apply(lambda x: [x])
                temp_df['prob_opt'] = temp_df.apply(lambda x: calculate_probability_optimal_qubo(x['counts'], x['optimal_x'], x['n_c'], x['single_flag']), axis=1)
            temp_df['support'] = support
            temp_df['AR'] = temp_df['AR'].astype(float)
            df = pd.concat([df, temp_df], ignore_index=True)
    return df


two_state_df = read_two_constraint_data()
two_qubo_df = read_two_constraint_data('qubo_')
two_state_df = two_state_df[two_state_df['angle_strategy'] == 'ma-QAOA']
two_qubo_df = two_qubo_df[two_qubo_df['angle_strategy'] == 'ma-QAOA']
two_qubo_df_2 = read_two_constraint_data_2('qubo_')
two_qubo_df = pd.concat([two_qubo_df, two_qubo_df_2], ignore_index=True)


#|%%--%%| <qOL3rDxKxP|Dx2d0inCcb>
r"""°°°
Plotting Functions for Two Constraint Data
°°°"""
#|%%--%%| <Dx2d0inCcb|sFN4tWGzAb>


def plot_two_constraint_prob_opt(df: pd.DataFrame, hue='single_flag', qubo: bool = False, moon: bool = True, confident: bool = False, save: bool = False, file: str | None = None) -> None:
    """
    Plot the probability of optimal vs support of the constraints for two overlapping constraints.
    Args:
        df (pd.DataFrame): The dataframe containing the data.
        hue (str): The column to use for the hue.
        qubo (bool): Whether to plot the qubo data or the state data.
        moon (bool): Whether to use the moon style or the dawn style.
        confident (bool): Whether to use the confident style or not.
        save (bool): Whether to save the plot or not.
        file (str | None): The name of the file to save the plot as.
    Returns:
        None
    """
    plots_dir = './plots/'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    if moon:
        plt.style.use('rose-pine-moon')
        colors = ['C0', 'C5', 'C2', 'C1', 'C3', 'C4']
    else:
        plt.style.use('rose-pine-dawn')
        colors = ['C0', 'C5', 'C2', 'C1', 'C3', 'C4']
    dff = df.copy()
    dff[hue] = dff[hue].astype(str)
    dff['constraints'] = dff['constraints'].map(tuple)
    dff['group'] = dff[hue].astype(str) + '_L' + dff['n_layers'].astype(str)
    group_labels = sorted(dff['group'].unique())
    hatch_map = {'1': '', '2': '//'}
    group_to_hatch = {g: hatch_map[g.split('L')[1]] for g in group_labels}
    base_colors = {'True': 'C0', "False": 'C5'}
    group_colors = {g: base_colors[g.split('_')[0]] for g in group_labels}
    hatches = np.repeat(['', '', '//', '//'], 3)

    fig = plt.figure(figsize=(15, 8))
    if qubo:
        ax = sns.boxplot(data=dff, x='support', y='prob_opt', hue='group', palette=group_colors)
        for bar, hatch in zip(ax.patches, hatches):
            bar.set_hatch(hatch)
            plt.draw()
    else:
        ax = sns.boxplot(data=dff, x='support', y='prob_opt', hue=hue, palette=colors)
    if moon:
        plt.background = 'black'
        plt.xlabel('Support of the Constraints', size=18)
        plt.ylabel('Probability of Optimal', size=18)
        plt.xticks([1, 2, 3])
        plt.tick_params(axis='both', labelsize=18)
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1), frameon=True, prop={'size': 12})  # , title=)  # , title='Number of Constraints/Layers')
        if qubo:
            plt.title('QCBOs with Two Overlapping Constraints vs Probability of Optimal Solution', size=20)
        else:
            plt.title('Two Overlapping Constraints vs Probability of Optimal Solution', size=20)
        plt.tight_layout()
        if save:
            if qubo:
                plt.savefig(f'{plots_dir}{hue}_qubo_prob_opt_vs_n_x_moon.png')
            else:
                plt.savefig(f'{plots_dir}{hue}_prob_opt_vs_n_x_moon.png')
        plt.show()
        plt.close()
    else:
        plt.background = 'white'
        plt.xlabel('Support of the Constraints', size=18)
        plt.ylabel('Probability of Optimal', size=18)
        plt.xticks(ticks=[0, 1, 2, 3])
        plt.tick_params(axis='both', labelsize=14)

        if qubo:
            # Build custom legend for hatch + color
            legend_handles = []
            seen = set()
            for g in group_labels:
                hue = g.split('_')[0]
                layer = g.split('L')[1]
                label = f"{hue}"+r", $p=$" + f"{layer}"
                if label not in seen:
                    legend_handles.append(Patch(facecolor=group_colors[g],
                                                hatch=group_to_hatch[g],
                                                edgecolor='black',
                                                label=label))
                    seen.add(label)

            ax.legend(handles=legend_handles, title=r'(Single Flag, $p$ layers)', bbox_to_anchor=(.85, 1), loc='upper left', prop={'size': 14}, title_fontsize=16)
            plt.title('QCBOs with Two Overlapping Constraints vs Probability of Optimal Solution', size=20)
        else:
            plt.legend(loc='upper right', bbox_to_anchor=(.95, 1), frameon=True, prop={'size': 14}, title='Use of Single\nFlag Qubit', title_fontsize=14)
            plt.title('Two Overlapping Constraints vs Probability of Optimal Solution', size=20)
        plt.tight_layout()
        plt.rcParams['figure.facecolor'] = 'white'
        fig.patch.set_facecolor('white')
        ax.patch.set_facecolor('white')
        if save:
            if file is not None:
                plt.savefig(f'{plots_dir}{file}_prob_opt_dawn.png', facecolor='white')
            elif qubo:
                plt.savefig(f'{plots_dir}two_qubo_vs_support_prob_opt_dawn.png', facecolor='white')
            else:
                plt.savefig(f'{plots_dir}two_constr_vs_support_prob_opt_dawn.png', facecolor='white')

        plt.show()
        plt.close()


def plot_two_constraint_ar(df: pd.DataFrame, hue='single_flag', qubo: bool = False, moon: bool = True, confident: bool = False, save: bool = False, file: str | None = None) -> None:
    """
    Plot the approximation ratio vs support of the constraints for two overlapping constraints.
    Args:
        df (pd.DataFrame): The dataframe containing the data.
        hue (str): The column to use for the hue.
        qubo (bool): Whether to plot the qubo data or the state data.
        moon (bool): Whether to use the moon style or the dawn style.
        confident (bool): Whether to use the confident style or not.
        save (bool): Whether to save the plot or not.
        file (str | None): The name of the file to save the plot as.
    Returns:
        None
    """
    plots_dir = './plots/'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    if moon:
        plt.style.use('rose-pine-moon')
        colors = ['C0', 'C5', 'C2', 'C1', 'C3', 'C4']
    else:
        plt.style.use('rose-pine-dawn')
        colors = ['C0', 'C5', 'C2', 'C1', 'C3', 'C4']
    dff = df.copy()
    dff[hue] = dff[hue].astype(str)
    dff['constraints'] = dff['constraints'].map(tuple)
    dff['group'] = dff['single_flag'].astype(str) + '_L' + dff['n_layers'].astype(str)
    group_labels = sorted(dff['group'].unique())
    hatch_map = {'1': '', '2': '//'}
    group_to_hatch = {g: hatch_map[g.split('L')[1]] for g in group_labels}
    base_colors = {'True': 'C0', "False": 'C5'}
    group_colors = {g: base_colors[g.split('_')[0]] for g in group_labels}
    hatches = np.repeat(['', '', '//', '//'], 3)

    fig = plt.figure(figsize=(15, 8))
    if qubo:
        ax = sns.boxplot(data=dff, x='support', y='AR', hue='group', palette=group_colors)
        for bar, hatch in zip(ax.patches, hatches):
            bar.set_hatch(hatch)
            plt.draw()
    else:
        ax = sns.boxplot(data=dff, x='support', y='AR', hue=hue, palette=colors)
    if moon:
        plt.background = 'black'
        plt.xlabel('Support of the Constraints', size=18)
        plt.ylabel('AR', size=18)
        plt.xticks([1, 2, 3])
        plt.tick_params(axis='both', labelsize=18)
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1), frameon=True, prop={'size': 12})
        if qubo:
            plt.title('QCBOs with Two Overlapping Constraints vs AR', size=20)
        else:
            plt.title('Two Overlapping Constraints vs AR', size=20)
        plt.tight_layout()
        if save:
            if qubo:
                plt.savefig(f'{plots_dir}{hue}_qubo_AR_vs_n_x_moon.png')
            else:
                plt.savefig(f'{plots_dir}{hue}_AR_vs_n_x_moon.png')
        plt.show()
        plt.close()
    else:
        plt.background = 'white'
        plt.xlabel('Support of the Constraints', size=18)
        plt.ylabel('AR', size=18)
        plt.xticks(ticks=[0, 1, 2, 3])
        plt.tick_params(axis='both', labelsize=14)

        if qubo:
            # Build custom legend for hatch + color
            legend_handles = []
            seen = set()
            for g in group_labels:
                hue = g.split('_')[0]
                layer = g.split('L')[1]
                label = f"{hue}"+r", $p=$" + f"{layer}"
                if label not in seen:
                    legend_handles.append(Patch(facecolor=group_colors[g],
                                                hatch=group_to_hatch[g],
                                                edgecolor='black',
                                                label=label))
                    seen.add(label)

            ax.legend(handles=legend_handles, title=r'(Single Flag, $p$ layers)', bbox_to_anchor=(.85, 1), loc='upper left', prop={'size': 14}, title_fontsize=16)
            plt.title('QCBOs with Two Overlapping Constraints vs Approximation Ratio', size=20)
        else:
            plt.legend(loc='upper right', bbox_to_anchor=(.95, 1), frameon=True, prop={'size': 14}, title='Use of Single\nFlag Qubit', title_fontsize=14)
            plt.title('Two Overlapping Constraints vs Approximation Ratio', size=20)
        plt.tight_layout()
        ax.patch.set_facecolor('white')
        plt.rcParams['figure.facecolor'] = 'white'
        fig.patch.set_facecolor('white')
        if save:
            if file is not None:
                plt.savefig(f'{plots_dir}{file}_AR_dawn.png', facecolor='white')
            elif qubo:
                plt.savefig(f'{plots_dir}two_qubo_vs_support_AR_dawn.png', facecolor='white')
            else:
                plt.savefig(f'{plots_dir}two_constr_vs_support_AR_dawn.png', facecolor='white')

        plt.show()
        plt.close()


plot_two_constraint_prob_opt(two_state_df, hue='single_flag', qubo=False, moon=False, confident=False, save=True)
plot_two_constraint_ar(two_state_df, hue='single_flag', qubo=False, moon=False, confident=False, save=True)
plot_two_constraint_prob_opt(two_qubo_df, hue='single_flag', qubo=True, moon=False, confident=False, save=True)
plot_two_constraint_ar(two_qubo_df, hue='single_flag', qubo=True, moon=False, confident=True, save=True)

#|%%--%%| <sFN4tWGzAb|2p6LZG66oa>
r"""°°°
Hamiltonian LaTeX Table
°°°"""
#|%%--%%| <2p6LZG66oa|85lclkFh7q>

state_hams_eq = state_df[(state_df['problem'] == 'equals') & (state_df['angle_strategy'] == 'ma-QAOA')][['constraints', 'Hamiltonian']]


def write_hamiltonian_latex_table(df, file_name, results_dir='./results/'):
    with open(f'{results_dir}{file_name}', 'w') as f:
        f.write('\\begin{tabular}{|c|l|}\n')
        f.write('\\hline\n')
        f.write('\\textbf{Constraints} & \\textbf{Hamiltonian} \\\\\n')
        f.write('\\hline\n')
        for i, row in df.iterrows():
            constraints = row['constraints']
            hamiltonian = row['Hamiltonian']
            flag_qubit = max(hamiltonian.wires)
            constraints_str = constraints[0]
            # remove ' and 1 * from the string, change == to =
            constraints_str = constraints_str.replace('\'', '').replace('1*', '')
            constraints_str = constraints_str.replace('==', '=')
            constraints_str = constraints_str.replace('>=', r'\geq')
            constraints_str = constraints_str.replace('<=', r'\leq')
            print(constraints_str)

            # convert Hamiltonian to LaTeX format, remove *, change (\d) to _\d using regex, remove @, and convert any decimal to its corresponding fraction using (float).as_integer_ratio() and formatting that as \frac{num}{den}
            ham_str = ''
            for i, t in enumerate(zip(hamiltonian.coeffs, hamiltonian.ops)):
                c = t[0]
                o = t[1]
                c = c.numpy().as_integer_ratio()
                if c[1] == 1 and np.abs(c[0]) != 1:
                    c_str = str(np.abs(c[0]))
                elif c[1] == 1 and np.abs(c[0]) == 1:
                    c_str = ''
                else:
                    c_str = f'\\frac{{{np.abs(c[0])}}}{{{c[1]}}}'
                if c[0] < 0:
                    c_str = ' - ' + c_str
                elif c[0] > 0 and i != 0:
                    c_str = ' + ' + c_str

                o_str = str(o)
                o_str = re.sub(r'\((\d)\)', r'_\1', o_str)
                # find max _\d and replace it with _f
                o_str = re.sub(f'_{flag_qubit}', '_f', o_str)
                o_str = o_str.replace(' @', '')

                ham_str += f'{c_str} {o_str}'
                print(ham_str)

            f.write(f'${constraints_str}$ & ${ham_str}$ \\\\\n')
            f.write('\\hline\n')
        f.write('\\end{tabular}\n')


ops = {'equals': '$=$', 'leq': r'$\leq$', 'less': '$<$', 'geq': r'$\geq$', 'greater': '$>$'}
for op in ops:
    op_df = state_df[(state_df['problem'] == op) & (state_df['angle_strategy'] == 'ma-QAOA')][['constraints', 'Hamiltonian']]
    write_hamiltonian_latex_table(op_df, f'cqaoa_hamiltonian_table_{op}.tex')

