"""
generate_plots2.py — generates all comparison figures for the second lab report
(Runs 1-10: CoOp-LoRA, RT-LoRA, and CSC ablation studies on Walnut dataset)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUT = os.path.join(os.path.dirname(__file__), 'figures2')
os.makedirs(OUT, exist_ok=True)

# ─── Data ──────────────────────────────────────────────────────────────────────
SHOTS = [1, 2, 4, 8, 16, 32]
ZERO_SHOT_ACC = 22.26
ZERO_SHOT_COST = 2.80
ZERO_SHOT_1OFF = 22.73
ZERO_SHOT_MAE  = 2.18

RUNS = {
    'Run 1\n(CoOp-LoRA Baseline)': {
        'color': '#4C72B0', 'ls': '-',
        'test_acc':  [29.84, 35.08, 36.83, 39.63, 42.66, 43.01],
        'mean_cost': [1.7296, 1.6387, 1.546, 1.4318, 1.4062, 1.3969],
        'acc_1off':  [44.76, 46.97, 49.42, 52.91, 53.15, 55.94],
        'mae':       [1.2226, 1.1422, 1.0909, 1.0233, 0.993, 1.0233],
        'train_time':[211.2, 520.9, 1416.0, 2068.4, 4205.0, 7373.5],
    },
    'Run 2\n(+ Residual Class Tokens)': {
        'color': '#DD8452', 'ls': '-',
        'test_acc':  [31.59, 39.86, 41.72, 46.15, 51.86, 55.01],
        'mean_cost': [1.8013, 1.5478, 1.4883, 1.4149, 1.2972, 1.1707],
        'acc_1off':  [44.41, 49.18, 49.53, 55.13, 57.58, 62.00],
        'mae':       [1.3357, 1.0793, 1.042, 1.0396, 0.9336, 0.8392],
        'train_time':[226.2, 557.2, 1479.3, 2156.4, 4343.8, 7160.9],
    },
    'Run 3\n(+ Ordinal Cost Loss)': {
        'color': '#55A868', 'ls': '-',
        'test_acc':  [29.72, 34.27, 37.53, 37.30, 43.94, 44.99],
        'mean_cost': [1.7302, 1.5315, 1.5583, 1.4749, 1.2855, 1.3339],
        'acc_1off':  [44.64, 50.00, 48.95, 51.52, 57.81, 56.29],
        'mae':       [1.1911, 1.0501, 1.1002, 1.049, 0.8986, 0.9569],
        'train_time':[214.2, 540.9, 1462.1, 2114.4, 4294.4, 7479.6],
    },
    'Run 4\n(+ PromptSRC Reg.)': {
        'color': '#C44E52', 'ls': '-',
        'test_acc':  [31.93, 35.90, 36.36, 37.88, 31.93, 37.76],
        'mean_cost': [2.2279, 1.8304, 1.715, 1.6579, 2.2558, 1.9324],
        'acc_1off':  [34.73, 42.66, 43.82, 47.09, 33.57, 40.44],
        'mae':       [1.7576, 1.3636, 1.2086, 1.2145, 1.7949, 1.479],
        'train_time':[227.9, 590.9, 1632.9, 2464.4, 5330.6, 9598.2],
    },
    'Run 5\n(Full Method / All)': {
        'color': '#8172B3', 'ls': '-',
        'test_acc':  [33.22, 39.04, 38.93, 38.93, 46.15, 45.10],
        'mean_cost': [2.0303, 1.6579, 1.8129, 1.5886, 1.4557, 1.4452],
        'acc_1off':  [38.58, 46.62, 42.42, 49.18, 52.56, 52.45],
        'mae':       [1.5396, 1.1807, 1.3403, 1.1608, 1.0513, 1.0117],
        'train_time':[221.7, 594.7, 1678.7, 2579.2, 5603.2, 10100.6],
    },
    'Run 6\n(RT-LoRA / Ours)': {
        'color': '#E91E63', 'ls': '-',
        'test_acc':  [32.75, 36.71, 40.56, 45.22, 51.86, 58.28],
        'mean_cost': [1.8403, 1.5816, 1.5350, 1.3928, 1.1906, 1.0367],
        'acc_1off':  [41.26, 48.48, 48.37, 54.43, 61.19, 65.73],
        'mae':       [1.3089, 1.1061, 1.0734, 0.9755, 0.8322, 0.7331],
        'train_time':[227.0, 590.7, 1573.3, 2293.2, 4682.4, 8046.0],
    },
    'Run 7\n(CSC + Vision LoRA)': {
        'color': '#00897B', 'ls': '-',
        'test_acc':  [30.89, 37.41, 41.14, 42.31, 50.00, 49.42],
        'mean_cost': [1.7273, 1.5408, 1.4959, 1.4621, 1.3059, 1.2389],
        'acc_1off':  [44.52, 48.48, 50.12, 51.63, 55.83, 59.79],
        'mae':       [1.2063, 1.1049, 1.0641, 1.0466, 0.9254, 0.8776],
        'train_time':[236.0, 607.6, 1632.5, 2251.1, 4445.8, 7574.0],
    },
    'Run 8\n(CSC + Dual LoRA)': {
        'color': '#D84315', 'ls': '-',
        'test_acc':  [34.15, 38.93, 39.63, 44.17, 47.20, 49.18],
        'mean_cost': [1.7570, 1.5163, 1.5559, 1.3974, 1.2955, 1.2395],
        'acc_1off':  [43.47, 49.42, 48.02, 53.73, 57.69, 59.44],
        'mae':       [1.2611, 1.0443, 1.1049, 0.9837, 0.9103, 0.8776],
        'train_time':[236.3, 612.6, 1614.7, 2334.0, 4709.7, 8202.9],
    },
    'Run 9\n(CSC 100it + Vision)': {
        'color': '#0288D1', 'ls': '-',
        'test_acc':  [27.04, 37.53, 41.14, 42.31, 50.12, 49.65],
        'mean_cost': [1.8153, 1.5373, 1.4930, 1.4662, 1.3042, 1.2284],
        'acc_1off':  [43.12, 48.60, 50.23, 51.52, 55.83, 60.02],
        'mae':       [1.2832, 1.1014, 1.0618, 1.0501, 0.9242, 0.8671],
        'train_time':[218.6, 553.5, 1520.1, 2133.5, 4309.5, 7533.8],
    },
    'Run 10\n(CSC 100it + Dual)': {
        'color': '#6A1B9A', 'ls': '-',
        'test_acc':  [34.62, 39.04, 39.86, 44.29, 46.85, 48.95],
        'mean_cost': [1.7407, 1.5093, 1.5600, 1.3992, 1.3059, 1.2471],
        'acc_1off':  [43.82, 49.65, 47.32, 53.61, 57.34, 59.21],
        'mae':       [1.2459, 1.0385, 1.1096, 0.9848, 0.9172, 0.8834],
        'train_time':[221.7, 555.9, 1511.6, 2205.3, 4533.7, 7837.2],
    },
}

RUN_KEYS = list(RUNS.keys())
COLORS   = [RUNS[r]['color'] for r in RUN_KEYS]
PALETTE  = dict(zip(RUN_KEYS, COLORS))

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
})

# ─── Helper ────────────────────────────────────────────────────────────────────
def legend_patches():
    return [mpatches.Patch(color=RUNS[r]['color'], label=r.replace('\n', ' ')) for r in RUN_KEYS]

def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  saved {name}')

# ══════════════════════════════════════════════════════════════════════════════
# 1. Test Accuracy vs Shots — all runs + zero-shot
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8.5, 5.2))
ax.axhline(ZERO_SHOT_ACC, color='gray', ls=':', lw=1.5, label='Zero-Shot CLIP (22.26%)')
for rname, rd in RUNS.items():
    ax.plot(SHOTS, rd['test_acc'], color=rd['color'], lw=2.2, label=rname.replace('\n', ' '))
ax.set_xlabel('Number of Training Shots per Class')
ax.set_ylabel('Top-1 Test Accuracy (%)')
ax.set_title('Test Accuracy vs. Shot Count — All Runs')
ax.set_xticks(SHOTS)
ax.legend(fontsize=7.5, loc='upper left', ncol=2)
save('acc_vs_shots_all_runs.png')

# ══════════════════════════════════════════════════════════════════════════════
# 2. Mean Cost Penalty vs Shots
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8.5, 5.2))
ax.axhline(ZERO_SHOT_COST, color='gray', ls=':', lw=1.5, label='Zero-Shot (2.80)')
for rname, rd in RUNS.items():
    ax.plot(SHOTS, rd['mean_cost'], color=rd['color'], lw=2.2, label=rname.replace('\n', ' '))
ax.set_xlabel('Number of Training Shots per Class')
ax.set_ylabel('Mean Cost Penalty (lower = better)')
ax.set_title('Mean Cost Penalty vs. Shot Count')
ax.set_xticks(SHOTS)
ax.legend(fontsize=7.5, ncol=2)
save('cost_vs_shots_all_runs.png')

# ══════════════════════════════════════════════════════════════════════════════
# 3. Acc@1-off (Ordinal Tolerance) vs Shots
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8.5, 5.2))
ax.axhline(ZERO_SHOT_1OFF, color='gray', ls=':', lw=1.5, label='Zero-Shot (22.73%)')
for rname, rd in RUNS.items():
    ax.plot(SHOTS, rd['acc_1off'], color=rd['color'], lw=2.2, label=rname.replace('\n', ' '))
ax.set_xlabel('Number of Training Shots per Class')
ax.set_ylabel('Acc@1-Off / Tolerance ≤1 (%)')
ax.set_title('Ordinal Tolerance Accuracy vs. Shot Count')
ax.set_xticks(SHOTS)
ax.legend(fontsize=7.5, ncol=2)
save('acc1off_vs_shots_all_runs.png')

# ══════════════════════════════════════════════════════════════════════════════
# 4. MAE vs Shots
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8.5, 5.2))
ax.axhline(ZERO_SHOT_MAE, color='gray', ls=':', lw=1.5, label='Zero-Shot MAE (2.18)')
for rname, rd in RUNS.items():
    ax.plot(SHOTS, rd['mae'], color=rd['color'], lw=2.2, label=rname.replace('\n', ' '))
ax.set_xlabel('Number of Training Shots per Class')
ax.set_ylabel('Mean Absolute Error (lower = better)')
ax.set_title('MAE vs. Shot Count')
ax.set_xticks(SHOTS)
ax.legend(fontsize=7.5, ncol=2)
save('mae_vs_shots_all_runs.png')

# ══════════════════════════════════════════════════════════════════════════════
# 5. Spider / Radar chart — 16-shot comparison (4 metrics)
# ══════════════════════════════════════════════════════════════════════════════
categories = ['Test Acc', 'Acc@1-Off', 'Low Cost\n(inv.)', 'Low MAE\n(inv.)']
N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# values at 16-shot; normalise to [0,1] using min/max across runs
idx16 = SHOTS.index(16)
raw = {
    'Test Acc':    [RUNS[r]['test_acc'][idx16]    for r in RUN_KEYS],
    'Acc@1-Off':   [RUNS[r]['acc_1off'][idx16]    for r in RUN_KEYS],
    'Low Cost\n(inv.)': [1/RUNS[r]['mean_cost'][idx16] for r in RUN_KEYS],
    'Low MAE\n(inv.)':  [1/RUNS[r]['mae'][idx16]       for r in RUN_KEYS],
}
norm = {}
for cat, vals in raw.items():
    lo, hi = min(vals), max(vals)
    span = hi - lo if hi != lo else 1
    norm[cat] = [(v - lo) / span for v in vals]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
for i, rname in enumerate(RUN_KEYS):
    values = [norm[cat][i] for cat in categories]
    values += values[:1]
    ax.plot(angles, values, color=COLORS[i], lw=2)
    ax.fill(angles, values, color=COLORS[i], alpha=0.06)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=9.5)
ax.set_yticklabels([])
ax.set_title('16-Shot: Normalised Metric Comparison', y=1.12, fontsize=11)
patches = legend_patches()
ax.legend(handles=patches, labels=[p.get_label() for p in patches],
          fontsize=7.0, loc='upper right', bbox_to_anchor=(1.56, 1.15))
save('radar_16shot.png')

# ══════════════════════════════════════════════════════════════════════════════
# 6. Bar chart: delta vs baseline (Run1) at 16-shot for each metric
# ══════════════════════════════════════════════════════════════════════════════
metrics = {
    'Test Acc (%)': ('test_acc', +1),
    'Acc@1-Off (%)': ('acc_1off', +1),
    'Mean Cost': ('mean_cost', -1),
    'MAE': ('mae', -1),
}
fig, axes = plt.subplots(1, 4, figsize=(18, 5.2), sharey=False)
base_run = RUN_KEYS[0]
for ax, (metric_label, (key, sign)) in zip(axes, metrics.items()):
    base_val = RUNS[base_run][key][idx16]
    deltas = []
    for rname in RUN_KEYS[1:]:
        delta = (RUNS[rname][key][idx16] - base_val) * sign
        deltas.append(delta)
    colors_bar = [COLORS[i+1] for i in range(len(RUN_KEYS)-1)]
    labels_bar = [r.replace('\n', ' ') for r in RUN_KEYS[1:]]
    bars = ax.bar(labels_bar, deltas, color=colors_bar, width=0.55)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_title(f'Δ {metric_label}\nvs. Run 1', fontsize=9.5)
    ax.set_ylabel('Delta (positive = better)')
    ax.tick_params(axis='x', labelsize=6.0, rotation=25)
    for bar, d in zip(bars, deltas):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + (0.01 if d >= 0 else -0.03),
                f'{d:+.2f}', ha='center', va='bottom' if d >= 0 else 'top', fontsize=7.0, color='black')
fig.suptitle('Gain Over Baseline (Run 1) at 16-Shot', fontsize=12, y=1.02)
save('delta_vs_baseline_16shot.png')

# ══════════════════════════════════════════════════════════════════════════════
# 7. Training Time Comparison — line plot
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8.5, 5.2))
for rname, rd in RUNS.items():
    ax.plot(SHOTS, [t/60 for t in rd['train_time']], color=rd['color'], lw=2.2,
            label=rname.replace('\n', ' '))
ax.set_xlabel('Number of Training Shots per Class')
ax.set_ylabel('Training Time (minutes)')
ax.set_title('Training Wall-Clock Time vs. Shot Count')
ax.set_xticks(SHOTS)
ax.legend(fontsize=7.0, ncol=2)
save('training_time_all_runs.png')

# ══════════════════════════════════════════════════════════════════════════════
# 8. Heatmap table: Test Accuracy — runs × shots
# ══════════════════════════════════════════════════════════════════════════════
data = np.array([[RUNS[r]['test_acc'][i] for i in range(len(SHOTS))] for r in RUN_KEYS])
fig, ax = plt.subplots(figsize=(10, 5.5))
cmap = plt.cm.RdYlGn
im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=data.min()-1, vmax=data.max()+1)
ax.set_xticks(range(len(SHOTS)))
ax.set_xticklabels([f'{s}-shot' for s in SHOTS])
ax.set_yticks(range(len(RUN_KEYS)))
ax.set_yticklabels([r.replace('\n', ' ') for r in RUN_KEYS], fontsize=8.0)
for i in range(len(RUN_KEYS)):
    for j in range(len(SHOTS)):
        ax.text(j, i, f'{data[i,j]:.1f}', ha='center', va='center', fontsize=8.5,
                color='black' if 35 < data[i,j] < 50 else 'white')
plt.colorbar(im, ax=ax, label='Test Accuracy (%)')
ax.set_title('Test Accuracy Heatmap — All Runs × All Shots')
save('heatmap_test_acc.png')

# ══════════════════════════════════════════════════════════════════════════════
# 9. Heatmap: Mean Cost Penalty
# ══════════════════════════════════════════════════════════════════════════════
data_cost = np.array([[RUNS[r]['mean_cost'][i] for i in range(len(SHOTS))] for r in RUN_KEYS])
fig, ax = plt.subplots(figsize=(10, 5.5))
cmap_r = plt.cm.RdYlGn_r
im = ax.imshow(data_cost, cmap=cmap_r, aspect='auto')
ax.set_xticks(range(len(SHOTS)))
ax.set_xticklabels([f'{s}-shot' for s in SHOTS])
ax.set_yticks(range(len(RUN_KEYS)))
ax.set_yticklabels([r.replace('\n', ' ') for r in RUN_KEYS], fontsize=8.0)
for i in range(len(RUN_KEYS)):
    for j in range(len(SHOTS)):
        ax.text(j, i, f'{data_cost[i,j]:.3f}', ha='center', va='center', fontsize=8.5, color='black')
plt.colorbar(im, ax=ax, label='Mean Cost Penalty (lower = better)')
ax.set_title('Mean Cost Penalty Heatmap — All Runs × All Shots')
save('heatmap_cost.png')

# ══════════════════════════════════════════════════════════════════════════════
# 10. Grouped bar: all metrics at 4 specific shots (1, 4, 16, 32)
# ══════════════════════════════════════════════════════════════════════════════
focal_shots = [1, 4, 16, 32]
focal_idx   = [SHOTS.index(s) for s in focal_shots]
x = np.arange(len(focal_shots))
width = 0.078

fig, ax = plt.subplots(figsize=(14, 5.5))
for i, rname in enumerate(RUN_KEYS):
    vals = [RUNS[rname]['test_acc'][j] for j in focal_idx]
    offset = (i - 4.5) * width
    bars = ax.bar(x + offset, vals, width, label=rname.replace('\n', ' '), color=COLORS[i])

ax.set_xlabel('Number of Shots')
ax.set_ylabel('Top-1 Test Accuracy (%)')
ax.set_title('Grouped Test Accuracy — Selected Shots')
ax.set_xticks(x)
ax.set_xticklabels([f'{s}-shot' for s in focal_shots])
ax.legend(fontsize=7.0, loc='upper left', ncol=2)
save('grouped_bar_test_acc.png')

# ══════════════════════════════════════════════════════════════════════════════
# 11. Run 4 anomaly chart — accuracy vs shots highlighting collapse
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8.5, 5.2))
ax.axhline(ZERO_SHOT_ACC, color='gray', ls=':', lw=1.3, label='Zero-Shot (22.26%)')
for rname, rd in RUNS.items():
    lw = 2.8 if 'PromptSRC' in rname else 1.8
    alpha = 1.0 if 'PromptSRC' in rname else 0.4
    ax.plot(SHOTS, rd['test_acc'], color=rd['color'], lw=lw, alpha=alpha,
            label=rname.replace('\n', ' '))
ax.annotate('Run 4 collapses\nat 16-shot', xy=(16, 31.93), xytext=(12, 25),
            arrowprops=dict(arrowstyle='->', color='#C44E52'),
            color='#C44E52', fontsize=9)
ax.set_xlabel('Number of Training Shots per Class')
ax.set_ylabel('Top-1 Test Accuracy (%)')
ax.set_title('Run 4 (PromptSRC): Accuracy Collapse at Higher Shots')
ax.set_xticks(SHOTS)
ax.legend(fontsize=7.0, ncol=2)
save('run4_collapse_highlight.png')

# ══════════════════════════════════════════════════════════════════════════════
# 12. Dual-axis: Accuracy + Cost for Run 2 (best overall run)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()
r2 = RUNS['Run 2\n(+ Residual Class Tokens)']
ax1.plot(SHOTS, r2['test_acc'], color='#DD8452', lw=2.5, label='Test Acc (Run 2)')
ax1.axhline(ZERO_SHOT_ACC, color='gray', ls=':', lw=1.2, label='ZS Acc baseline')
ax2.plot(SHOTS, r2['mean_cost'], color='#DD8452', lw=2.5, ls='--', label='Mean Cost (Run 2)')
ax2.axhline(ZERO_SHOT_COST, color='gray', ls='--', lw=1.2, alpha=0.5, label='ZS Cost baseline')
ax1.set_xlabel('Shots')
ax1.set_ylabel('Test Accuracy (%)', color='#DD8452')
ax2.set_ylabel('Mean Cost Penalty', color='#884422')
ax1.set_xticks(SHOTS)
ax1.set_title('Run 2: Accuracy and Cost Penalty vs. Shots')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
save('run2_dual_axis.png')

# ══════════════════════════════════════════════════════════════════════════════
# 13. Shot-scaling improvement factor (relative to zero-shot)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8.5, 5.2))
for rname, rd in RUNS.items():
    ratio = [acc / ZERO_SHOT_ACC for acc in rd['test_acc']]
    ax.plot(SHOTS, ratio, color=rd['color'], lw=2.2, label=rname.replace('\n', ' '))
ax.axhline(1.0, color='gray', ls=':', lw=1.5, label='Zero-Shot baseline (×1.0)')
ax.set_xlabel('Number of Training Shots per Class')
ax.set_ylabel('Accuracy / Zero-Shot Accuracy (ratio)')
ax.set_title('Relative Improvement Factor Over Zero-Shot')
ax.set_xticks(SHOTS)
ax.legend(fontsize=7.0, ncol=2)
save('relative_improvement_ratio.png')

# ══════════════════════════════════════════════════════════════════════════════
# 14. Plain LoRA Baseline (Report 1) vs. CoOp-LoRA Variants (Report 2) — Acc Curves
# ══════════════════════════════════════════════════════════════════════════════
PLAIN_LORA_ACC = [39.74, 43.82, 46.39, 49.65, 56.18, 62.82]

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.axhline(ZERO_SHOT_ACC, color='gray', ls=':', lw=1.5, label='Zero-Shot CLIP (22.26%)')
ax.plot(SHOTS, PLAIN_LORA_ACC, color='#1A1A1A', lw=2.8, ls='--', marker='o',
        label='Plain CLIP-LoRA (Report 1 Baseline)')

for rname, rd in RUNS.items():
    ax.plot(SHOTS, rd['test_acc'], color=rd['color'], lw=2.2,
            label=rname.replace('\n', ' '))

ax.set_xlabel('Number of Training Shots per Class', fontsize=11)
ax.set_ylabel('Top-1 Test Accuracy (%)', fontsize=11)
ax.set_title('Plain LoRA (Report 1) vs. CoOp-LoRA, RT-LoRA & CSC Suite (Report 2)', fontsize=12)
ax.set_xticks(SHOTS)
ax.legend(fontsize=7.0, loc='upper left', ncol=2)
save('plain_lora_vs_coop_lora_acc.png')

# ══════════════════════════════════════════════════════════════════════════════
# 15. Grouped Bar: Plain LoRA vs. CoOp-LoRA Runs Across All Shots
# ══════════════════════════════════════════════════════════════════════════════
x = np.arange(len(SHOTS))
width = 0.072

fig, ax = plt.subplots(figsize=(16, 5.5))
bars_plain = ax.bar(x - 5.5 * width, PLAIN_LORA_ACC, width, label='Plain LoRA (Report 1)', color='#333333')
for i, rname in enumerate(RUN_KEYS):
    vals = [RUNS[rname]['test_acc'][j] for j in range(len(SHOTS))]
    offset = (i - 4.5) * width
    ax.bar(x + offset, vals, width, label=rname.replace('\n', ' '), color=COLORS[i])

ax.set_xlabel('Number of Training Shots', fontsize=11)
ax.set_ylabel('Top-1 Test Accuracy (%)', fontsize=11)
ax.set_title('Direct Accuracy Comparison: Plain LoRA vs. Hybrid CoOp-LoRA & CSC Suite', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels([f'{s}-shot' for s in SHOTS])
ax.legend(fontsize=6.8, loc='upper left', ncol=3)
save('plain_lora_vs_coop_grouped_bar.png')

# ══════════════════════════════════════════════════════════════════════════════
# 16. Delta vs. Plain LoRA Baseline Across Shots
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.axhline(0, color='#1A1A1A', lw=1.2, ls='-')
for i, rname in enumerate(RUN_KEYS):
    deltas = [RUNS[rname]['test_acc'][j] - PLAIN_LORA_ACC[j] for j in range(len(SHOTS))]
    ax.plot(SHOTS, deltas, color=COLORS[i], lw=2.2, marker='s', label=rname.replace('\n', ' '))

ax.set_xlabel('Number of Training Shots per Class', fontsize=11)
ax.set_ylabel('Δ Test Accuracy vs. Plain LoRA (percentage points)', fontsize=11)
ax.set_title('Accuracy Delta: CoOp-LoRA & CSC Variants Relative to Plain LoRA', fontsize=12)
ax.set_xticks(SHOTS)
ax.legend(fontsize=7.0, loc='lower right', ncol=2)
save('plain_lora_vs_coop_delta.png')

# ══════════════════════════════════════════════════════════════════════════════
# 17. Paradigm Analysis: Low-Shot Cold-Start Gap vs. High-Shot Scaling
# ══════════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.2))

# Subplot 1: 1-Shot & 2-Shot (Cold-Start Regime)
shots_low = [1, 2]
idx_low = [0, 1]
x_low = np.arange(len(shots_low))
w = 0.072
ax1.bar(x_low - 5.5*w, [PLAIN_LORA_ACC[i] for i in idx_low], w, label='Plain LoRA (Frozen Text)', color='#333333')
for i, rname in enumerate(RUN_KEYS):
    ax1.bar(x_low + (i-4.5)*w, [RUNS[rname]['test_acc'][j] for j in idx_low], w, label=rname.replace('\n', ' '), color=COLORS[i])
ax1.set_xticks(x_low)
ax1.set_xticklabels(['1-Shot (6 samples total)', '2-Shot (12 samples total)'])
ax1.set_ylabel('Top-1 Test Accuracy (%)')
ax1.set_title('Low-Shot Regime: Cold-Start Dynamics', fontsize=10.5)
ax1.legend(fontsize=6.0, ncol=2)

# Subplot 2: 16-Shot & 32-Shot (High-Shot Regime)
shots_high = [16, 32]
idx_high = [4, 5]
x_high = np.arange(len(shots_high))
ax2.bar(x_high - 5.5*w, [PLAIN_LORA_ACC[i] for i in idx_high], w, label='Plain LoRA (Frozen Text)', color='#333333')
for i, rname in enumerate(RUN_KEYS):
    ax2.bar(x_high + (i-4.5)*w, [RUNS[rname]['test_acc'][j] for j in idx_high], w, label=rname.replace('\n', ' '), color=COLORS[i])
ax2.set_xticks(x_high)
ax2.set_xticklabels(['16-Shot (96 samples total)', '32-Shot (192 samples total)'])
ax2.set_ylabel('Top-1 Test Accuracy (%)')
ax2.set_title('High-Shot Regime: Closing the Gap with Domain Tokens', fontsize=10.5)
ax2.legend(fontsize=6.0, ncol=2)

fig.suptitle('Paradigm Comparison: Low-Shot vs. High-Shot Dynamics', fontsize=12)
save('plain_lora_vs_coop_regime_analysis.png')

print('\nAll plots generated successfully.')
