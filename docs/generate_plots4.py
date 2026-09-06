#!/usr/bin/env python3
"""
generate_plots4.py
Generates publication-quality visualizations for Report 4 based solely on data in ~/Downloads/runs/.
"""

import os
import json
import ast
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

RUNS_DIR = os.path.expanduser('~/Downloads/runs')
RUNS2_DIR = os.path.expanduser('~/Downloads/runs2')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'figures4')
os.makedirs(OUT_DIR, exist_ok=True)

# Styling configuration
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10.5,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'figure.facecolor': 'white',
    'axes.facecolor': '#FAFAFA',
    'axes.edgecolor': '#CCCCCC',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color palette matching report theme
PALETTE = {
    'Run 1: Infix CoOp + Ordinal (Vision LoRA)': '#4C72B0', # Blue
    'Run 2: Plain LoRA Baseline (Dual LoRA)': '#DD8452',    # Orange
    'Run 3: CoOp-CSC Dual LoRA': '#55A868',                 # Green
    'Run 4: Plain LoRA + ResCls (Dual LoRA)': '#C44E52',    # Red
}

SHORT_NAMES = {
    'Run 1: Infix CoOp + Ordinal (Vision LoRA)': 'Run 1 (Infix CoOp + Ord)',
    'Run 2: Plain LoRA Baseline (Dual LoRA)': 'Run 2 (Plain Dual LoRA)',
    'Run 3: CoOp-CSC Dual LoRA': 'Run 3 (CoOp-CSC Dual LoRA)',
    'Run 4: Plain LoRA + ResCls (Dual LoRA)': 'Run 4 (Plain LoRA + ResCls)',
}

MARKERS = {
    'Run 1: Infix CoOp + Ordinal (Vision LoRA)': 'o',
    'Run 2: Plain LoRA Baseline (Dual LoRA)': 's',
    'Run 3: CoOp-CSC Dual LoRA': '^',
    'Run 4: Plain LoRA + ResCls (Dual LoRA)': 'D',
}

DATASET_NAMES = {
    'walnut': 'Walnut Grading (Initial Aug, s11-13)',
    'walnut_fixed': 'Walnut Grading (Fixed Aug, s21-23)',
    'piarom_shape': 'Piarom Date (Initial Aug, s11-13)',
    'piarom_shape_fixed': 'Piarom Date (Fixed Aug, s21-23)',
    'stanford_cars': 'Stanford Cars (196 classes)',
}

ZERO_SHOTS = {
    'walnut': 22.261,
    'walnut_fixed': 22.261,
    'piarom_shape': 24.907,
    'piarom_shape_fixed': 24.907,
    'stanford_cars': 65.539,
}

def load_data():
    records = []
    def parse_obj(obj, fname, is_runs2=False):
        if not isinstance(obj, dict):
            return
        cp = obj.get('checkpoint', '')
        out = obj.get('output_dir', '')
        combined = f'{cp} {out} {fname}'
        
        p = 'Unknown'
        if 'run1_infix_coop_ordinal' in combined or ('run1' in combined and ('coop' in combined or 'infix' in combined or 'summary-cars-run1' in fname or 'summary-piarom-run1' in fname or 'summary-walnut-run1' in fname or 'summary2' in fname or 'summary6' in fname)):
            p = 'Run 1: Infix CoOp + Ordinal (Vision LoRA)'
        elif 'run3_coop_csc_dual_lora' in combined or ('run3' in fname and 'summary' in fname) or 'cars-run3' in fname or 'summary3' in fname or 'summary8' in fname:
            p = 'Run 3: CoOp-CSC Dual LoRA'
        elif 'run4_plain_lora_rescls' in combined or ('run4' in fname and 'summary' in fname) or 'cars-run4' in fname or 'summary4' in fname or 'summary7' in fname:
            p = 'Run 4: Plain LoRA + ResCls (Dual LoRA)'
        elif 'run2_plain_lora' in combined or ('run2' in fname and 'summary' in fname) or 'cars-run2' in fname or 'summary1' in fname or 'summary5' in fname:
            p = 'Run 2: Plain LoRA Baseline (Dual LoRA)'
        
        ds = obj.get('dataset', '')
        if not ds:
            if 'cars' in combined: ds = 'stanford_cars'
            elif 'piarom' in combined: ds = 'piarom_shape'
            elif 'walnut' in combined: ds = 'walnut'

        if is_runs2:
            if ds == 'piarom_shape': ds = 'piarom_shape_fixed'
            elif ds == 'walnut': ds = 'walnut_fixed'

        shots = obj.get('shots')
        seed = obj.get('seed')
        train_acc = obj.get('final_train_acc')
        test_acc = obj.get('final_test_acc')
        zero_shot = obj.get('zero_shot_test_acc')
        time_sec = obj.get('training_time_seconds')

        records.append({
            'file': fname,
            'dataset': ds,
            'paradigm': p,
            'shots': shots,
            'seed': seed,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'zero_shot': zero_shot if zero_shot is not None else ZERO_SHOTS.get(ds),
            'time_sec': time_sec
        })

    if os.path.exists(RUNS_DIR):
        for f in sorted(os.listdir(RUNS_DIR)):
            p = os.path.join(RUNS_DIR, f)
            with open(p, 'r') as fp:
                txt = fp.read()
            try:
                data = json.loads(txt)
            except Exception:
                try:
                    data = ast.literal_eval(txt)
                except Exception as e:
                    print(f'Error reading {f}: {e}')
                    continue
            if isinstance(data, list):
                for item in data:
                    parse_obj(item, f, is_runs2=False)
            elif isinstance(data, dict):
                parse_obj(data, f, is_runs2=False)

    if os.path.exists(RUNS2_DIR):
        for f in sorted(os.listdir(RUNS2_DIR)):
            p = os.path.join(RUNS2_DIR, f)
            with open(p, 'r') as fp:
                txt = fp.read()
            try:
                data = json.loads(txt)
            except Exception:
                try:
                    data = ast.literal_eval(txt)
                except Exception as e:
                    print(f'Error reading {f}: {e}')
                    continue
            if isinstance(data, list):
                for item in data:
                    parse_obj(item, f, is_runs2=True)
            elif isinstance(data, dict):
                parse_obj(data, f, is_runs2=True)

    df = pd.DataFrame(records)
    return df

def save(fig, name):
    fig.tight_layout()
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=250, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')

def plot_few_shot_curves(df):
    """Figure 1: 2x3 comprehensive multi-benchmark few-shot performance curves with error bands and delta impact."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5), sharey=False)
    shots = [1, 2, 4, 8, 16, 32]
    
    panels = [
        (0, 0, 'walnut', 'Walnut Grading (Initial Aug, s11-13)'),
        (0, 1, 'walnut_fixed', 'Walnut Grading (Fixed Aug, s21-23)'),
        (0, 2, 'stanford_cars', 'Stanford Cars (196 classes, s11-13)'),
        (1, 0, 'piarom_shape', 'Piarom Date (Initial Aug, s11-13)'),
        (1, 1, 'piarom_shape_fixed', 'Piarom Date (Fixed Aug, s21-23)'),
    ]

    for r, c, ds, title in panels:
        ax = axes[r, c]
        sub_ds = df[df['dataset'] == ds]
        zs = ZERO_SHOTS[ds]
        ax.axhline(zs, color='#555555', linestyle=':', linewidth=1.5, label=f'Zero-Shot CLIP ({zs:.1f}%)')
        
        for p in sorted(sub_ds['paradigm'].unique()):
            sub_p = sub_ds[sub_ds['paradigm'] == p]
            grouped = sub_p.groupby('shots')['test_acc'].agg(['mean', 'std']).reindex(shots)
            means = grouped['mean']
            stds = grouped['std'].fillna(0.0)
            color = PALETTE.get(p, '#333333')
            marker = MARKERS.get(p, 'o')
            label = SHORT_NAMES.get(p, p)
            
            ax.plot(shots, means, marker=marker, markersize=6.5, linewidth=2.0, color=color, label=label)
            ax.fill_between(shots, means - stds, means + stds, color=color, alpha=0.15)
        
        ax.set_xscale('log', base=2)
        ax.set_xticks(shots)
        ax.set_xticklabels([str(s) for s in shots])
        ax.set_xlabel('Shots per Class (k)', fontweight='bold')
        ax.set_ylabel('Test Accuracy (%)', fontweight='bold')
        ax.set_title(title, fontweight='bold', fontsize=11.5)
        if r == 0 and c == 0:
            ax.legend(loc='lower right', frameon=True, framealpha=0.92, fontsize=8.5)

    # Panel (1, 2): Net Gain from Fixed Augmentation (Walnut solid, Piarom dashed)
    ax_delta = axes[1, 2]
    ax_delta.axhline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.7)
    for p in sorted(df['paradigm'].unique()):
        color = PALETTE.get(p)
        w_orig = df[(df['dataset'] == 'walnut') & (df['paradigm'] == p)].groupby('shots')['test_acc'].mean().reindex(shots)
        w_fix = df[(df['dataset'] == 'walnut_fixed') & (df['paradigm'] == p)].groupby('shots')['test_acc'].mean().reindex(shots)
        w_diff = w_fix - w_orig
        
        p_orig = df[(df['dataset'] == 'piarom_shape') & (df['paradigm'] == p)].groupby('shots')['test_acc'].mean().reindex(shots)
        p_fix = df[(df['dataset'] == 'piarom_shape_fixed') & (df['paradigm'] == p)].groupby('shots')['test_acc'].mean().reindex(shots)
        p_diff = p_fix - p_orig
        
        p_label = SHORT_NAMES.get(p).split('(')[1].replace(')', '')
        ax_delta.plot(shots, w_diff, marker='o', markersize=6, linewidth=1.8, color=color, label=f'{p_label} (Walnut)')
        ax_delta.plot(shots, p_diff, marker='s', markersize=6, linewidth=1.8, linestyle='--', color=color, alpha=0.85, label=f'{p_label} (Piarom)')

    ax_delta.set_xscale('log', base=2)
    ax_delta.set_xticks(shots)
    ax_delta.set_xticklabels([str(s) for s in shots])
    ax_delta.set_xlabel('Shots per Class (k)', fontweight='bold')
    ax_delta.set_ylabel(r'$\Delta$ Accuracy Gain (Fixed - Initial pp)', fontweight='bold')
    ax_delta.set_title(r'Fixed Augmentation Impact ($\Delta$ pp)', fontweight='bold', fontsize=11.5)
    ax_delta.legend(loc='upper right', frameon=True, framealpha=0.92, fontsize=7.5, ncol=2)

    save(fig, 'few_shot_curves_all_datasets.png')

def plot_train_accuracy_curves(df):
    """Figure 3: 2x3 training accuracy curves across shots with error bands."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5), sharey=False)
    shots = [1, 2, 4, 8, 16, 32]
    
    panels = [
        (0, 0, 'walnut', 'Walnut (Initial Aug, s11-13)', 80, 102),
        (0, 1, 'walnut_fixed', 'Walnut (Fixed Aug, s21-23)', 80, 102),
        (0, 2, 'stanford_cars', 'Stanford Cars (196 classes, s11-13)', 55, 100),
        (1, 0, 'piarom_shape', 'Piarom Date (Initial Aug, s11-13)', 80, 102),
        (1, 1, 'piarom_shape_fixed', 'Piarom Date (Fixed Aug, s21-23)', 80, 102),
    ]

    for r, c, ds, title, ymin, ymax in panels:
        ax = axes[r, c]
        sub_ds = df[df['dataset'] == ds]
        
        for p in sorted(sub_ds['paradigm'].unique()):
            sub_p = sub_ds[sub_ds['paradigm'] == p]
            grouped = sub_p.groupby('shots')['train_acc'].agg(['mean', 'std']).reindex(shots)
            means = grouped['mean']
            stds = grouped['std'].fillna(0.0)
            color = PALETTE.get(p, '#333333')
            marker = MARKERS.get(p, 'o')
            label = SHORT_NAMES.get(p, p)
            
            ax.plot(shots, means, marker=marker, markersize=6.5, linewidth=2.0, color=color, label=label)
            ax.fill_between(shots, means - stds, means + stds, color=color, alpha=0.15)
        
        ax.set_xscale('log', base=2)
        ax.set_xticks(shots)
        ax.set_xticklabels([str(s) for s in shots])
        ax.set_xlabel('Shots per Class (k)', fontweight='bold')
        ax.set_ylabel('Final Training Accuracy (%)', fontweight='bold')
        ax.set_title(title, fontweight='bold', fontsize=11.5)
        ax.set_ylim(ymin, ymax)
        if r == 0 and c == 0:
            ax.legend(loc='lower left', frameon=True, framealpha=0.92, fontsize=8.5)

    # Panel (1, 2): Empty panel
    axes[1, 2].axis('off')

    save(fig, 'train_curves_all_datasets.png')

def plot_piarom_augmentation_comparison(df):
    """Side-by-side comparison of Initial vs Fixed Augmentation on Piarom Date Shape."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.2))
    shots = [1, 2, 4, 8, 16, 32]
    
    sub_orig = df[df['dataset'] == 'piarom_shape']
    sub_fix = df[df['dataset'] == 'piarom_shape_fixed']
    
    # Left: Curves comparison (Dashed = Initial, Solid = Fixed)
    for p in sorted(sub_orig['paradigm'].unique()):
        gr_orig = sub_orig[sub_orig['paradigm'] == p].groupby('shots')['test_acc'].mean().reindex(shots)
        gr_fix = sub_fix[sub_fix['paradigm'] == p].groupby('shots')['test_acc'].mean().reindex(shots)
        color = PALETTE.get(p, '#333333')
        label_base = SHORT_NAMES.get(p, p).split('(')[1].replace(')', '')
        
        ax1.plot(shots, gr_orig, linestyle='--', marker='x', markersize=6, color=color, alpha=0.6, label=f'{label_base} (Initial)')
        ax1.plot(shots, gr_fix, linestyle='-', marker=MARKERS.get(p, 'o'), markersize=7, linewidth=2.2, color=color, label=f'{label_base} (Fixed Aug)')
        
    ax1.axhline(ZERO_SHOTS['piarom_shape'], color='gray', linestyle=':', linewidth=1.5, label='Zero-Shot CLIP (24.9%)')
    ax1.set_xscale('log', base=2)
    ax1.set_xticks(shots)
    ax1.set_xticklabels([str(s) for s in shots])
    ax1.set_xlabel('Shots per Class (k)', fontweight='bold')
    ax1.set_ylabel('Test Accuracy (%)', fontweight='bold')
    ax1.set_title('Piarom: Trajectory Comparison (Initial vs. Fixed Aug)', fontweight='bold')
    ax1.legend(loc='lower right', fontsize=8.5, frameon=True, framealpha=0.92)
    
    # Right: Net Gain (\Delta) from fixing augmentation
    for p in sorted(sub_orig['paradigm'].unique()):
        gr_orig = sub_orig[sub_orig['paradigm'] == p].groupby('shots')['test_acc'].mean().reindex(shots)
        gr_fix = sub_fix[sub_fix['paradigm'] == p].groupby('shots')['test_acc'].mean().reindex(shots)
        diff = gr_fix - gr_orig
        color = PALETTE.get(p, '#333333')
        label = SHORT_NAMES.get(p, p)
        ax2.plot(shots, diff, marker=MARKERS.get(p, 'o'), markersize=7, linewidth=2.2, color=color, label=label)
        
    ax2.axhline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.7)
    ax2.set_xscale('log', base=2)
    ax2.set_xticks(shots)
    ax2.set_xticklabels([str(s) for s in shots])
    ax2.set_xlabel('Shots per Class (k)', fontweight='bold')
    ax2.set_ylabel(r'$\Delta$ Accuracy Gain from Fixed Augmentation (pp)', fontweight='bold')
    ax2.set_title('Net Accuracy Improvement from Fixed Augmentations', fontweight='bold')
    ax2.legend(loc='upper right', frameon=True)
    
    save(fig, 'piarom_augmentation_comparison.png')

def plot_delta_over_zero_shot(df):
    """Figure 2: Net Accuracy Delta over Zero-Shot Baseline (Gain / Degradation)."""
    fig, axes = plt.subplots(1, 4, figsize=(23, 5.0), sharey=False)
    datasets = ['walnut', 'piarom_shape', 'piarom_shape_fixed', 'stanford_cars']
    shots = [1, 2, 4, 8, 16, 32]
    
    for i, ds in enumerate(datasets):
        ax = axes[i]
        sub_ds = df[df['dataset'] == ds]
        zs = ZERO_SHOTS[ds]
        
        ax.axhline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.7, label='Zero-Shot Baseline')
        
        paradigms = sorted(sub_ds['paradigm'].unique())
        for p in paradigms:
            sub_p = sub_ds[sub_ds['paradigm'] == p]
            grouped = sub_p.groupby('shots')['test_acc'].agg(['mean']).reindex(shots)
            delta = grouped['mean'] - zs
            
            color = PALETTE.get(p, '#333333')
            marker = MARKERS.get(p, 'o')
            label = SHORT_NAMES.get(p, p)
            
            ax.plot(shots, delta, marker=marker, markersize=6.5, linewidth=2.0, color=color, label=label)
        
        ax.set_xscale('log', base=2)
        ax.set_xticks(shots)
        ax.set_xticklabels([str(s) for s in shots])
        ax.set_xlabel('Shots per Class (k)', fontweight='bold')
        ax.set_ylabel(r'$\Delta$ Accuracy vs. Zero-Shot (pp)', fontweight='bold')
        ax.set_title(f'{DATASET_NAMES[ds]}: Gain vs Zero-Shot', fontweight='bold', fontsize=11.5)
        if i == 0:
            ax.legend(loc='lower right', frameon=True, framealpha=0.92, facecolor='white', edgecolor='#DDDDDD')
            
    save(fig, 'delta_over_zero_shot.png')

def plot_cardinality_collapse_cars(df):
    """Figure 3: Focused Analysis on Stanford Cars: The 1-Shot Collapse Mechanism."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    
    sub = df[df['dataset'] == 'stanford_cars']
    zs = ZERO_SHOTS['stanford_cars']
    
    shots = [1, 2, 4, 8, 16, 32]
    for p in sorted(sub['paradigm'].unique()):
        sub_p = sub[sub['paradigm'] == p]
        gr = sub_p.groupby('shots')['test_acc'].agg(['mean', 'std']).reindex(shots)
        ax1.plot(shots, gr['mean'], marker=MARKERS.get(p, 'o'), markersize=7, linewidth=2.2, 
                 color=PALETTE.get(p), label=SHORT_NAMES.get(p))
        ax1.fill_between(shots, gr['mean'] - gr['std'].fillna(0), gr['mean'] + gr['std'].fillna(0),
                         color=PALETTE.get(p), alpha=0.15)
        
    ax1.axhline(zs, color='gray', linestyle=':', linewidth=1.8, label=f'Zero-Shot CLIP ({zs:.1f}%)')
    ax1.set_xscale('log', base=2)
    ax1.set_xticks(shots)
    ax1.set_xticklabels([str(s) for s in shots])
    ax1.set_xlabel('Shots per Class (k)', fontweight='bold')
    ax1.set_ylabel('Test Accuracy (%)', fontweight='bold')
    ax1.set_title('Stanford Cars: Accuracy Trajectory Across Shots', fontweight='bold')
    ax1.legend(loc='lower right', frameon=True, framealpha=0.92)
    
    # Bar chart of 1-shot performance and delta
    one_shot = sub[sub['shots'] == 1].groupby('paradigm')['test_acc'].mean()
    p_keys = sorted(one_shot.index)
    labels = [SHORT_NAMES.get(k).replace('Run ', 'R') for k in p_keys]
    vals = [one_shot[k] for k in p_keys]
    colors = [PALETTE.get(k) for k in p_keys]
    
    bars = ax2.bar(labels, vals, color=colors, width=0.55, edgecolor='#333333', linewidth=1)
    ax2.axhline(zs, color='black', linestyle='--', linewidth=1.5, label=f'Zero-Shot ({zs:.1f}%)')
    ax2.set_ylabel('1-Shot Test Accuracy (%)', fontweight='bold')
    ax2.set_title('Extreme 1-Shot Collapse in 196-Class Domain', fontweight='bold')
    ax2.set_ylim(0, 80)
    
    for bar in bars:
        h = bar.get_height()
        diff = h - zs
        sign = '+' if diff >= 0 else ''
        ax2.annotate(f'{h:.1f}%\n({sign}{diff:.1f}pp)',
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords='offset points',
                     ha='center', va='bottom', fontsize=9.5, fontweight='bold')
    ax2.legend(loc='upper right', frameon=True)
    
    save(fig, 'cardinality_collapse_cars.png')

def plot_pareto_frontier(df):
    """Figure 4: Computational Cost (Training Time) vs 32-Shot Test Accuracy."""
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    
    datasets = ['walnut', 'piarom_shape', 'piarom_shape_fixed', 'stanford_cars']
    ds_markers = {'walnut': 'o', 'piarom_shape': 's', 'piarom_shape_fixed': 'D', 'stanford_cars': '^'}
    
    for ds in datasets:
        sub_ds = df[(df['dataset'] == ds) & (df['shots'] == 32)]
        for p in sorted(sub_ds['paradigm'].unique()):
            sub = sub_ds[sub_ds['paradigm'] == p]
            m_time = sub['time_sec'].mean()
            m_acc = sub['test_acc'].mean()
            
            ax.scatter(m_time, m_acc, color=PALETTE.get(p), marker=ds_markers[ds], s=120,
                       edgecolor='black', linewidth=0.8, alpha=0.9, zorder=4)
            # Label
            short_p = SHORT_NAMES.get(p).split('(')[1].replace(')', '')
            ax.annotate(f'{ds[:4]}-{short_p}', (m_time, m_acc), xytext=(5, 2),
                        textcoords='offset points', fontsize=8.5, alpha=0.85)
            
    # Legends
    p_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=PALETTE[p], markersize=9, label=SHORT_NAMES[p]) for p in sorted(PALETTE.keys())]
    ds_handles = [plt.Line2D([0], [0], marker=ds_markers[d], color='w', markerfacecolor='gray', markersize=9, label=DATASET_NAMES[d]) for d in datasets]
    
    leg1 = ax.legend(handles=p_handles, loc='lower right', title='Paradigm', frameon=True)
    ax.add_artist(leg1)
    ax.legend(handles=ds_handles, loc='center left', title='Dataset', frameon=True)
    
    ax.set_xlabel('32-Shot Training Time (Seconds, Log Scale)', fontweight='bold')
    ax.set_ylabel('32-Shot Test Accuracy (%)', fontweight='bold')
    ax.set_xscale('log')
    ax.set_title('Pareto Frontier: Accuracy vs. Compute Consumption (32-Shot)', fontweight='bold', fontsize=12)
    
    save(fig, 'training_time_vs_accuracy.png')

def plot_generalization_gap(df):
    """Figure 5: Generalization Gap (Train Acc - Test Acc) across shots."""
    fig, axes = plt.subplots(1, 4, figsize=(23, 5.0), sharey=True)
    datasets = ['walnut', 'piarom_shape', 'piarom_shape_fixed', 'stanford_cars']
    shots = [1, 2, 4, 8, 16, 32]
    
    for i, ds in enumerate(datasets):
        ax = axes[i]
        sub_ds = df[df['dataset'] == ds]
        
        for p in sorted(sub_ds['paradigm'].unique()):
            sub_p = sub_ds[sub_ds['paradigm'] == p]
            # Gap = Train Acc - Test Acc
            sub_p = sub_p.copy()
            sub_p['gap'] = sub_p['train_acc'] - sub_p['test_acc']
            gr = sub_p.groupby('shots')['gap'].agg(['mean', 'std']).reindex(shots)
            
            ax.plot(shots, gr['mean'], marker=MARKERS.get(p, 'o'), markersize=6.5, linewidth=2.0,
                    color=PALETTE.get(p), label=SHORT_NAMES.get(p))
            ax.fill_between(shots, gr['mean'] - gr['std'].fillna(0), gr['mean'] + gr['std'].fillna(0),
                            color=PALETTE.get(p), alpha=0.12)
            
        ax.set_xscale('log', base=2)
        ax.set_xticks(shots)
        ax.set_xticklabels([str(s) for s in shots])
        ax.set_xlabel('Shots per Class (k)', fontweight='bold')
        if i == 0:
            ax.set_ylabel('Generalization Gap (% Train - % Test)', fontweight='bold')
            ax.legend(loc='upper right', frameon=True, framealpha=0.92)
        ax.set_title(DATASET_NAMES[ds], fontweight='bold', fontsize=11.5)
        
    save(fig, 'generalization_gap_curves.png')

def plot_seed_stability(df):
    """Figure 6: Seed Variance Comparison (Std Dev of Test Accuracy)."""
    fig, axes = plt.subplots(1, 4, figsize=(23, 4.8), sharey=True)
    datasets = ['walnut', 'piarom_shape', 'piarom_shape_fixed', 'stanford_cars']
    shots = [1, 2, 4, 8, 16, 32]
    
    for i, ds in enumerate(datasets):
        ax = axes[i]
        sub_ds = df[df['dataset'] == ds]
        
        for p in sorted(sub_ds['paradigm'].unique()):
            sub_p = sub_ds[sub_ds['paradigm'] == p]
            stds = sub_p.groupby('shots')['test_acc'].std().reindex(shots).fillna(0.0)
            ax.plot(shots, stds, marker=MARKERS.get(p, 'o'), markersize=6.5, linewidth=2.0,
                    color=PALETTE.get(p), label=SHORT_NAMES.get(p))
            
        ax.set_xscale('log', base=2)
        ax.set_xticks(shots)
        ax.set_xticklabels([str(s) for s in shots])
        ax.set_xlabel('Shots per Class (k)', fontweight='bold')
        if i == 0:
            ax.set_ylabel('Test Accuracy Standard Deviation (%)', fontweight='bold')
            ax.legend(loc='upper right', frameon=True)
        ax.set_title(f'{DATASET_NAMES[ds]}: Seed Sensitivity', fontweight='bold', fontsize=11.5)
        
    save(fig, 'seed_variance_analysis.png')

def plot_heatmap(df):
    """Figure 7: Summary Heatmap across Datasets, Shots, and Paradigms."""
    fig, axes = plt.subplots(1, 4, figsize=(25, 5.5))
    datasets = ['walnut', 'piarom_shape', 'piarom_shape_fixed', 'stanford_cars']
    
    for i, ds in enumerate(datasets):
        ax = axes[i]
        sub = df[df['dataset'] == ds]
        piv = sub.pivot_table(index='shots', columns='paradigm', values='test_acc', aggfunc='mean')
        # Rename columns to short names
        piv.columns = [SHORT_NAMES.get(c, c).replace(' (Dual LoRA)', '').replace(' (Vision LoRA)', '') for c in piv.columns]
        
        sns.heatmap(piv, annot=True, fmt='.1f', cmap='YlGnBu', cbar=(i == 3), ax=ax,
                    linewidths=0.5, linecolor='white')
        ax.set_title(DATASET_NAMES[ds], fontweight='bold', fontsize=11.5)
        ax.set_xlabel('')
        ax.set_ylabel('Shots per Class (k)' if i == 0 else '')
        
    save(fig, 'performance_summary_heatmap.png')

def main():
    print(f'Loading data from {RUNS_DIR} and {RUNS2_DIR}...')
    df = load_data()
    print(f'Successfully loaded {len(df)} total records.')
    
    print('Generating analytical figures in docs/figures4/...')
    plot_few_shot_curves(df)
    plot_train_accuracy_curves(df)
    plot_piarom_augmentation_comparison(df)
    plot_delta_over_zero_shot(df)
    plot_cardinality_collapse_cars(df)
    plot_pareto_frontier(df)
    plot_generalization_gap(df)
    plot_seed_stability(df)
    plot_heatmap(df)
    print('All figures successfully generated!')

if __name__ == '__main__':
    main()
