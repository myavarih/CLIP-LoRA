#!/usr/bin/env python3
"""
Generate all comparison plots for the CLIP-LoRA experiment report.
Produces: accuracy vs shots (all configs), cross-seed variance, training time,
alpha sweep, overfitting analysis, and combined training curves.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================================
# Color Palette & Style
# ============================================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.facecolor': 'white',
    'axes.facecolor': '#FAFAFA',
    'axes.edgecolor': '#CCCCCC',
})

COLORS = {
    'baseline': '#2196F3',        # Blue
    'p_qv': '#FF9800',            # Orange
    'p_v': '#4CAF50',             # Green
    'r_1': '#F44336',             # Red
    'r_3': '#9C27B0',             # Purple
    'a_0.75': '#795548',          # Brown
    'a_1.5': '#E91E63',           # Pink
    'a_2.0': '#FF5722',           # Deep Orange
    'a1.5_d0.4': '#00BCD4',       # Cyan
    'drop_0.4': '#607D8B',        # Blue Grey
    'e_text': '#CDDC39',          # Lime
    'e_vision': '#FF6F00',        # Amber Dark
    'zs': '#9E9E9E',             # Grey
    'baseline_a1.5': '#3F51B5',   # Indigo
    'p_qv_a1.5': '#FF7043',       # Deep Orange Light
    'p_v_a1.5': '#66BB6A',        # Green Light
    'r_1_a1.5': '#EF5350',        # Red Light
}

# ============================================================================
# ALL EXPERIMENT DATA (extracted from run_summary.json files)
# ============================================================================

# --- Seed 1 baseline (experiments_round_one) ---
baseline_s1 = {1: 39.74, 2: 43.82, 4: 46.39, 8: 49.65, 16: 56.18, 32: 62.82}

# --- Seed 2 baseline (baseliens_results) ---
baseline_s2 = {1: 38.93, 2: 43.82, 4: 47.67, 8: 50.70, 16: 59.32}

# --- Seed 2 ablation: baseline (results(1)) ---
baseline_abl_s2 = {1: 38.46, 2: 44.06, 4: 48.02, 8: 50.35, 16: 59.56}

# --- Seed 2 ablations (results(1)) ---
p_qv_s2 = {1: 36.60, 2: 41.26, 4: 44.87, 8: 52.80, 16: 58.51}
p_v_s2 = {1: 40.68, 2: 40.21, 4: 46.97, 8: 50.00, 16: 58.16}
r_1_s2 = {1: 38.23, 2: 43.12, 4: 47.55, 8: 49.42}

# --- Seed 2 4-shot-only ablations (results(2)) ---
a_075_s2 = {4: 47.32}
a_15_s2 = {4: 50.00}
drop_04_s2 = {4: 48.37}
e_text_s2 = {4: 40.56}
e_vision_s2 = {4: 45.34}
r_3_s2 = {4: 48.60}

# --- Seed 3 alpha 1.5 ablation sweep (results (3) & (4)) ---
baseline_a15_s3 = {1: 39.39, 2: 42.31, 4: 49.65, 8: 51.63, 16: 58.28}
p_qv_a15_s3 = {1: 39.63, 2: 42.66, 4: 48.14, 8: 50.70, 16: 58.62}
p_v_a15_s3 = {1: 38.69, 2: 41.49, 4: 47.32, 8: 50.93, 16: 58.51}
r_1_a15_s3 = {1: 38.11, 2: 42.77, 4: 48.02, 8: 50.47}

# --- Alpha 2.0 experiments (results (1) with spaces) ---
a_20_s1 = {1: 39.98, 2: 41.61, 4: 44.29, 8: 53.61}

# --- Alpha 1.5 + Dropout 0.4 (results (2) with spaces) ---
a15_d04_s1 = {1: 40.56, 2: 43.01, 4: 44.17, 8: 52.68}

# Training times (seconds)
train_times = {
    'baseline_s1': {1: 280, 2: 569, 4: 1828, 8: 2541, 16: 5126, 32: 7980},
    'baseline_s2': {1: 358, 2: 637, 4: 1768, 8: 2574, 16: 4946},
    'a_20_s1': {1: 136, 2: 354, 4: 902, 8: 1286},
    'a15_d04_s1': {1: 143, 2: 380, 4: 919, 8: 1323},
    'baseline_a15_s3': {1: 429, 2: 946, 4: 1627, 8: 2463, 16: 4550},
}

# Train accuracies (for overfitting analysis)
train_acc = {
    'baseline_s1': {1: 100, 2: 100, 4: 100, 8: 100, 16: 100, 32: 100},
    'baseline_s2': {1: 100, 2: 100, 4: 100, 8: 100, 16: 100},
    'a_20_s1': {1: 100, 2: 100, 4: 100, 8: 95.83},
    'a15_d04_s1': {1: 100, 2: 100, 4: 95.83, 8: 89.58},
    'p_qv_s2': {1: 100, 2: 100, 4: 100, 8: 100, 16: 100},
    'p_v_s2': {1: 100, 2: 100, 4: 100, 8: 100, 16: 95.83},
    'r_1_s2': {1: 100, 2: 100, 4: 100, 8: 100},
}

test_acc_for_overfit = {
    'baseline_s2': baseline_s2,
    'p_qv_s2': p_qv_s2,
    'p_v_s2': p_v_s2,
    'r_1_s2': r_1_s2,
}

ZS_ACC = 22.26

# ============================================================================
# PLOT 1: Comprehensive Accuracy vs Shots (ALL configs)
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 8))

# Main ablations (seed 2) — solid lines, thicker
configs_main = [
    ('Baseline ($q,k,v$, $r$=2, $\\alpha$=1.0, seed 2)', baseline_s2, COLORS['baseline'], '-', 'o', 2.5),
    ('$q,v$ only', p_qv_s2, COLORS['p_qv'], '-', 's', 2.0),
    ('$v$ only', p_v_s2, COLORS['p_v'], '-', '^', 2.0),
    ('Rank 1 ($r$=1)', r_1_s2, COLORS['r_1'], '-', 'D', 2.0),
]

# Alpha / advanced ablations — dashed lines
configs_adv = [
    ('Baseline 32-shot (seed 1)', baseline_s1, '#0D47A1', '-', 'h', 2.5),
    ('$\\alpha$=1.5 baseline (seed 3)', baseline_a15_s3, COLORS['baseline_a1.5'], '--', 'o', 1.8),
    ('$\\alpha$=1.5, $q,v$ (seed 3)', p_qv_a15_s3, COLORS['p_qv_a1.5'], '--', 's', 1.8),
    ('$\\alpha$=1.5, $v$ only (seed 3)', p_v_a15_s3, COLORS['p_v_a1.5'], '--', '^', 1.8),
    ('$\\alpha$=1.5, $r$=1 (seed 3)', r_1_a15_s3, COLORS['r_1_a1.5'], '--', 'D', 1.8),
    ('$\\alpha$=2.0, $d$=0.4', a_20_s1, COLORS['a_2.0'], '--', 'v', 1.8),
    ('$\\alpha$=1.5, $d$=0.4', a15_d04_s1, COLORS['a1.5_d0.4'], '--', 'P', 1.8),
]

for label, data, color, ls, marker, lw in configs_main + configs_adv:
    shots = sorted(data.keys())
    accs = [data[s] for s in shots]
    ax.plot(shots, accs, marker=marker, linestyle=ls, linewidth=lw, color=color, 
            label=label, markersize=7, markeredgecolor='white', markeredgewidth=0.5)

# 4-shot only points
ax.scatter([4], [a_075_s2[4]], marker='*', s=150, c=COLORS['a_0.75'], zorder=5,
           edgecolors='white', linewidth=0.5, label=f"$\\alpha$=0.75 ({a_075_s2[4]:.1f}%)")
ax.scatter([4], [r_3_s2[4]], marker='*', s=150, c=COLORS['r_3'], zorder=5,
           edgecolors='white', linewidth=0.5, label=f"Rank 3 ({r_3_s2[4]:.1f}%)")
ax.scatter([4], [drop_04_s2[4]], marker='*', s=150, c=COLORS['drop_0.4'], zorder=5,
           edgecolors='white', linewidth=0.5, label=f"Dropout 0.4 ({drop_04_s2[4]:.1f}%)")
ax.scatter([4], [e_text_s2[4]], marker='x', s=100, c=COLORS['e_text'], zorder=5,
           linewidth=2, label=f"Text only ({e_text_s2[4]:.1f}%)")
ax.scatter([4], [e_vision_s2[4]], marker='x', s=100, c=COLORS['e_vision'], zorder=5,
           linewidth=2, label=f"Vision only ({e_vision_s2[4]:.1f}%)")

ax.set_xlabel('Number of Shots (images per class)', fontsize=13)
ax.set_ylabel('Test Accuracy (%)', fontsize=13)
ax.set_title('Test Accuracy vs. Training Shots — All Configurations', fontsize=15, fontweight='bold')
ax.set_xticks([1, 2, 4, 8, 16, 32])
ax.set_xscale('log', base=2)
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.legend(loc='upper left', fontsize=8, ncol=2, framealpha=0.95)
ax.set_ylim(35, 65)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'acc_vs_shots_all.png'), dpi=200)
plt.close()
print("✅ acc_vs_shots_all.png")

# ============================================================================
# PLOT 2: Cross-Seed Variance (Baseline only)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

seeds_data = {
    1: baseline_s1,
    2: baseline_s2,
}
common_shots = sorted(set(baseline_s1.keys()) & set(baseline_s2.keys()))
means = []
stds = []
for s in common_shots:
    vals = [baseline_s1[s], baseline_s2[s]]
    if s in baseline_abl_s2:
        vals.append(baseline_abl_s2[s])
    means.append(np.mean(vals))
    stds.append(np.std(vals))

x = np.arange(len(common_shots))
bars = ax.bar(x, means, width=0.5, color=COLORS['baseline'], alpha=0.8, 
              edgecolor='white', linewidth=1.5)
ax.errorbar(x, means, yerr=stds, fmt='none', ecolor='#333333', capsize=5, capthick=1.5)

for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(i, m + s + 0.8, f'{m:.1f}±{s:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels([f'{s}-shot' for s in common_shots])
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title('Baseline Accuracy Across Seeds (Mean ± Std)', fontsize=14, fontweight='bold')
ax.set_ylim(35, 65)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'cross_seed_variance.png'), dpi=200)
plt.close()
print("✅ cross_seed_variance.png")

# ============================================================================
# PLOT 3: Training Time vs Shots
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

for name, data, color, marker, ls in [
    ('Baseline', train_times['baseline_s1'], COLORS['baseline'], 'o', '-'),
]:
    shots = sorted(data.keys())
    times = [data[s] / 60 for s in shots]
    ax.plot(shots, times, marker=marker, linestyle=ls, linewidth=2.5, color=color,
            label=name, markersize=8, markeredgecolor='white', markeredgewidth=0.5)

ax.set_xlabel('Number of Shots', fontsize=12)
ax.set_ylabel('Training Time (minutes)', fontsize=12)
ax.set_title('Training Time Scaling with Number of Shots', fontsize=14, fontweight='bold')
ax.set_xticks([1, 2, 4, 8, 16, 32])
ax.set_xscale('log', base=2)
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'training_time_vs_shots.png'), dpi=200)
plt.close()
print("✅ training_time_vs_shots.png")

# ============================================================================
# PLOT 4: Alpha Sweep Comparison (at 4-shot)
# ============================================================================
fig, ax = plt.subplots(figsize=(9, 6))

alphas = [0.75, 1.0, 1.5, 2.0]
accs_4shot = [a_075_s2[4], baseline_s2[4], a_15_s2[4], a_20_s1[4]]
colors_alpha = [COLORS['a_0.75'], COLORS['baseline'], COLORS['a_1.5'], COLORS['a_2.0']]

bars = ax.bar(range(len(alphas)), accs_4shot, width=0.5, color=colors_alpha, alpha=0.85,
              edgecolor='white', linewidth=2)

for i, (a, acc) in enumerate(zip(alphas, accs_4shot)):
    ax.text(i, acc + 0.3, f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_xticks(range(len(alphas)))
ax.set_xticklabels([f'$\\alpha$={a}' for a in alphas])
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title('Alpha Scaling Factor Comparison (4-Shot, $q,k,v$, $r$=2)', fontsize=14, fontweight='bold')
ax.set_ylim(42, 52)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'alpha_sweep_4shot.png'), dpi=200)
plt.close()
print("✅ alpha_sweep_4shot.png")

# ============================================================================
# PLOT 5: Overfitting Analysis (Train vs Test Gap)
# ============================================================================
fig, ax = plt.subplots(figsize=(11, 6))

configs_gap = [
    ('Baseline', baseline_s2, train_acc['baseline_s2'], COLORS['baseline']),
    ('$q,v$ only', p_qv_s2, train_acc['p_qv_s2'], COLORS['p_qv']),
    ('$v$ only', p_v_s2, train_acc['p_v_s2'], COLORS['p_v']),
    ('$\\alpha$=2.0, $d$=0.4', a_20_s1, train_acc['a_20_s1'], COLORS['a_2.0']),
    ('$\\alpha$=1.5, $d$=0.4', a15_d04_s1, train_acc['a15_d04_s1'], COLORS['a1.5_d0.4']),
]

for label, test_data, train_data, color in configs_gap:
    common = sorted(set(test_data.keys()) & set(train_data.keys()))
    gaps = [train_data[s] - test_data[s] for s in common]
    ax.plot(common, gaps, marker='o', linewidth=2, color=color, label=label,
            markersize=7, markeredgecolor='white', markeredgewidth=0.5)

ax.set_xlabel('Number of Shots', fontsize=12)
ax.set_ylabel('Train-Test Accuracy Gap (%)', fontsize=12)
ax.set_title('Overfitting Analysis: Train − Test Accuracy Gap', fontsize=14, fontweight='bold')
ax.set_xticks([1, 2, 4, 8, 16])
ax.set_xscale('log', base=2)
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.legend(fontsize=9)
ax.axhline(y=0, color='#333333', linestyle='-', linewidth=0.5, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'overfitting_gap.png'), dpi=200)
plt.close()
print("✅ overfitting_gap.png")

# ============================================================================
# PLOT 6: Rank Comparison at 4-Shot
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 5.5))

ranks = [1, 2, 3]
rank_accs = [r_1_s2[4], baseline_s2[4], r_3_s2[4]]
rank_colors = [COLORS['r_1'], COLORS['baseline'], COLORS['r_3']]

bars = ax.bar(range(len(ranks)), rank_accs, width=0.4, color=rank_colors, alpha=0.85,
              edgecolor='white', linewidth=2)

for i, acc in enumerate(rank_accs):
    ax.text(i, acc + 0.15, f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_xticks(range(len(ranks)))
ax.set_xticklabels([f'Rank $r$={r}' for r in ranks])
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title('LoRA Rank Comparison (4-Shot, $q,k,v$, $\\alpha$=1.0)', fontsize=13, fontweight='bold')
ax.set_ylim(45, 50)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'rank_comparison_4shot.png'), dpi=200)
plt.close()
print("✅ rank_comparison_4shot.png")

# ============================================================================
# PLOT 7: Modality Comparison at 4-Shot
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 5.5))

modalities = ['Text Only', 'Vision Only', 'Both (Baseline)']
mod_accs = [e_text_s2[4], e_vision_s2[4], baseline_s2[4]]
mod_colors = ['#CDDC39', '#FF6F00', COLORS['baseline']]

bars = ax.bar(range(len(modalities)), mod_accs, width=0.4, color=mod_colors, alpha=0.85,
              edgecolor='white', linewidth=2)

for i, acc in enumerate(mod_accs):
    ax.text(i, acc + 0.3, f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_xticks(range(len(modalities)))
ax.set_xticklabels(modalities)
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title('Encoder Modality Targeting (4-Shot)', fontsize=13, fontweight='bold')
ax.set_ylim(38, 50)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'modality_comparison_4shot.png'), dpi=200)
plt.close()
print("✅ modality_comparison_4shot.png")

# ============================================================================
# PLOT 8: Shot Progression (Baseline Seed 1 — 1 to 32)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 5.5))

shots_prog = sorted(baseline_s1.keys())
accs_prog = [baseline_s1[s] for s in shots_prog]

# Gradient fill
for i in range(len(shots_prog) - 1):
    ax.fill_between([shots_prog[i], shots_prog[i+1]], [35, 35], 
                    [accs_prog[i], accs_prog[i+1]], alpha=0.15, color=COLORS['baseline'])

ax.plot(shots_prog, accs_prog, marker='o', linewidth=3, color=COLORS['baseline'],
        markersize=10, markeredgecolor='white', markeredgewidth=1.5, zorder=5)

for s, a in zip(shots_prog, accs_prog):
    ax.annotate(f'{a:.1f}%', (s, a), textcoords="offset points", xytext=(0, 12),
                ha='center', fontsize=10, fontweight='bold', color=COLORS['baseline'])

total_gain = accs_prog[-1] - ZS_ACC
ax.annotate(f'+{total_gain:.1f}% gain over\nzero-shot ({ZS_ACC}%)', 
            xy=(32, accs_prog[-1]), xytext=(16, 42),
            fontsize=10, ha='center',
            arrowprops=dict(arrowstyle='->', color='#666666'),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', edgecolor=COLORS['baseline']))

ax.set_xlabel('Number of Shots', fontsize=12)
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title('Baseline Accuracy Progression: 1-Shot → 32-Shot', fontsize=14, fontweight='bold')
ax.set_xticks(shots_prog)
ax.set_xticklabels([str(s) for s in shots_prog])
ax.set_ylim(35, 66)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'shot_progression.png'), dpi=200)
plt.close()
print("✅ shot_progression.png")

# ============================================================================
# PLOT 9: Alpha 1.5 Ablation Grid (Seed 3) — All target layers
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

configs_a15 = [
    ('Baseline ($q,k,v$) $\\alpha$=1.5', baseline_a15_s3, COLORS['baseline_a1.5'], 'o', '-'),
    ('$q,v$ $\\alpha$=1.5', p_qv_a15_s3, COLORS['p_qv_a1.5'], 's', '-'),
    ('$v$ only $\\alpha$=1.5', p_v_a15_s3, COLORS['p_v_a1.5'], '^', '-'),
    ('Rank 1 $\\alpha$=1.5', r_1_a15_s3, COLORS['r_1_a1.5'], 'D', '-'),
]

for label, data, color, marker, ls in configs_a15:
    shots = sorted(data.keys())
    accs = [data[s] for s in shots]
    ax.plot(shots, accs, marker=marker, linestyle=ls, linewidth=2.2, color=color,
            label=label, markersize=8, markeredgecolor='white', markeredgewidth=0.8)

ax.set_xlabel('Number of Shots', fontsize=12)
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title('Alpha 1.5 Ablation: Target Layer Comparison (Seed 3)', fontsize=14, fontweight='bold')
ax.set_xticks([1, 2, 4, 8, 16])
ax.set_xscale('log', base=2)
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.legend(fontsize=10)
ax.set_ylim(36, 62)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'alpha15_ablation_grid.png'), dpi=200)
plt.close()
print("✅ alpha15_ablation_grid.png")

# ============================================================================
# PLOT 10: Dropout + Alpha Interaction (α=1.5 d=0.4 vs α=2.0 d=0.4 vs baseline)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

configs_dropout = [
    ('Baseline ($\\alpha$=1.0, $d$=0.25)', baseline_s2, COLORS['baseline'], 'o', '-', 2.5),
    ('$\\alpha$=1.5, $d$=0.4', a15_d04_s1, COLORS['a1.5_d0.4'], 'P', '--', 2.0),
    ('$\\alpha$=2.0, $d$=0.4', a_20_s1, COLORS['a_2.0'], 'v', '--', 2.0),
    ('$\\alpha$=1.5, $d$=0.25 (seed 3)', baseline_a15_s3, COLORS['baseline_a1.5'], 'D', '-.', 1.8),
]

for label, data, color, marker, ls, lw in configs_dropout:
    shots = sorted(data.keys())
    accs = [data[s] for s in shots]
    ax.plot(shots, accs, marker=marker, linestyle=ls, linewidth=lw, color=color,
            label=label, markersize=7, markeredgecolor='white', markeredgewidth=0.5)

ax.set_xlabel('Number of Shots', fontsize=12)
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title('Dropout × Alpha Interaction Analysis', fontsize=14, fontweight='bold')
ax.set_xticks([1, 2, 4, 8, 16])
ax.set_xscale('log', base=2)
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.legend(fontsize=9)
ax.set_ylim(36, 62)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'dropout_alpha_interaction.png'), dpi=200)
plt.close()
print("✅ dropout_alpha_interaction.png")

# ============================================================================
# PLOT 11: Complete 4-Shot Training Progress (ALL 12 Configurations)
# ============================================================================
import json

ckpt_paths_4shot = [
    ('Baseline ($q,k,v$, $r$=2, $\\alpha$=1.0)', '/home/emmwhy/Downloads/results(1)/experiments_output/ablations/walnut_4shots_seed2_baseline/visualizations/checkpoint_accuracies.json', COLORS['baseline'], '-', 'o', 2.5),
    ('$q,v$ only', '/home/emmwhy/Downloads/results(1)/experiments_output/ablations/walnut_4shots_seed2_p_qv/visualizations/checkpoint_accuracies.json', COLORS['p_qv'], '-', 's', 2.0),
    ('$v$ only', '/home/emmwhy/Downloads/results(1)/experiments_output/ablations/walnut_4shots_seed2_p_v/visualizations/checkpoint_accuracies.json', COLORS['p_v'], '-', '^', 2.0),
    ('Rank 1 ($r$=1)', '/home/emmwhy/Downloads/results(1)/experiments_output/ablations/walnut_4shots_seed2_r_1/visualizations/checkpoint_accuracies.json', COLORS['r_1'], '-', 'D', 2.0),
    ('Rank 3 ($r$=3)', '/home/emmwhy/Downloads/results(2)/experiments_output/ablations/walnut_4shots_seed2_r_3/visualizations/checkpoint_accuracies.json', COLORS['r_3'], '-', 'P', 1.8),
    ('$\\alpha$=0.75', '/home/emmwhy/Downloads/results(2)/experiments_output/ablations/walnut_4shots_seed2_a_0.75/visualizations/checkpoint_accuracies.json', COLORS['a_0.75'], '--', '*', 1.6),
    ('$\\alpha$=1.5', '/home/emmwhy/Downloads/results(2)/experiments_output/ablations/walnut_4shots_seed2_a_1.5/visualizations/checkpoint_accuracies.json', COLORS['a_1.5'], '--', 'h', 1.8),
    ('Dropout 0.4', '/home/emmwhy/Downloads/results(2)/experiments_output/ablations/walnut_4shots_seed2_drop_0.4/visualizations/checkpoint_accuracies.json', COLORS['drop_0.4'], '-.', 'x', 1.6),
    ('Text Only', '/home/emmwhy/Downloads/results(2)/experiments_output/ablations/walnut_4shots_seed2_e_text/visualizations/checkpoint_accuracies.json', COLORS['e_text'], ':', 'v', 1.6),
    ('Vision Only', '/home/emmwhy/Downloads/results(2)/experiments_output/ablations/walnut_4shots_seed2_e_vision/visualizations/checkpoint_accuracies.json', COLORS['e_vision'], ':', '<', 1.6),
    ('$\\alpha$=1.5 (seed 3)', '/home/emmwhy/Downloads/results (3)/my_impl/my_impl/ablation_experiments_walnut/walnut_4shots_seed3_baseline_alpha_1.5/visualizations/checkpoint_accuracies.json', COLORS['baseline_a1.5'], '--', 'd', 1.8),
    ('$\\alpha$=2.0, $d$=0.4', '/home/emmwhy/Downloads/results (2)/experiments_output/alpha_2.0/walnut_4shots_seed1/visualizations/checkpoint_accuracies.json', COLORS['a_2.0'], '-.', '>', 1.8),
]

fig, ax = plt.subplots(figsize=(14, 8))

for label, path, color, ls, marker, lw in ckpt_paths_4shot:
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
        iters = sorted([int(k) for k in data.keys()])
        accs = [data[str(it)] for it in iters]
        ax.plot(iters, accs, marker=marker, linestyle=ls, linewidth=lw, color=color,
                label=label, markersize=6, markeredgecolor='white', markeredgewidth=0.5)

ax.set_xlabel('Iteration Step', fontsize=13)
ax.set_ylabel('Test Accuracy (%)', fontsize=13)
ax.set_title('Complete 4-Shot Training Progress Across All Configurations', fontsize=15, fontweight='bold')
ax.legend(loc='lower right', fontsize=8.5, ncol=2, framealpha=0.95)
ax.set_ylim(38, 52)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'training_curves_4shots.png'), dpi=200)
plt.close()
print("✅ training_curves_4shots.png (All 12 configurations included)")

print("\n🎉 All plots generated successfully!")

