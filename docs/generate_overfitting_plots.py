#!/usr/bin/env python3
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import zipfile

OUT_DIR = '/home/emmwhy/Projects/CV_Lab/CLIP-LoRA/docs/figures2'
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
})

# Load exact Plain LoRA data from Downloads
plain_lora_data = {}
for shot in [1, 2, 4, 8, 16]:
    fpath = f'/home/emmwhy/Downloads/{shot}-shot-checkpoint_accuracies.json'
    if os.path.exists(fpath):
        with open(fpath) as fh:
            d = json.load(fh)
            iters = sorted([int(k) for k in d.keys()])
            accs = [d[str(it)] for it in iters]
            plain_lora_data[shot] = (iters, accs)

zips = {
    'Plain CLIP-LoRA (Baseline)': None,
    'Run 1 (CoOp Baseline)': ('/home/emmwhy/Downloads/results.zip', '#4C72B0', '-'),
    'Run 2 (+ Class Tokens)': ('/home/emmwhy/Downloads/results (1).zip', '#DD8452', '-'),
    'Run 3 (+ Ordinal Loss)': ('/home/emmwhy/Downloads/results(1).zip', '#55A868', '-'),
    'Run 4 (+ PromptSRC)': ('/home/emmwhy/Downloads/results (2).zip', '#C44E52', '-'),
    'Run 5 (Full Method)': ('/home/emmwhy/Downloads/results (3).zip', '#8172B3', '-'),
    'Run 6 (RT-LoRA)': ('/home/emmwhy/Downloads/results(2).zip', '#E91E63', '-'),
    'Run 7 (CSC Vision LoRA)': ('/home/emmwhy/Downloads/results(3).zip', '#00897B', '-'),
    'Run 8 (CSC Dual LoRA)': ('/home/emmwhy/Downloads/results(4).zip', '#D84315', '-'),
    'Run 9 (CSC 100it Vision)': ('/home/emmwhy/Downloads/results (4).zip', '#0288D1', '-'),
    'Run 10 (CSC 100it Dual)': ('/home/emmwhy/Downloads/results(5).zip', '#6A1B9A', '-'),
}

run_data = {}
for rname, val in zips.items():
    if val is None:
        continue
    zpath, color, ls = val
    run_data[rname] = {'color': color, 'ls': ls, 'shots': {}}
    with zipfile.ZipFile(zpath, 'r') as z:
        for shot in [1, 2, 4, 8, 16, 32]:
            ckpts = [n for n in z.namelist() if 'checkpoint_accuracies.json' in n and f'walnut_{shot}shots' in n]
            if ckpts:
                data = json.loads(z.read(ckpts[0]).decode('utf-8'))
                iters = sorted([int(k) for k in data.keys()])
                accs = [data[str(it)]['acc'] if isinstance(data[str(it)], dict) else data[str(it)] for it in iters]
                run_data[rname]['shots'][shot] = (iters, accs)

# Multi-panel comparison: 1-shot, 2-shot, 4-shot, 16-shot (2x2 grid)
selected_shots = [1, 2, 4, 16]
fig, axes = plt.subplots(2, 2, figsize=(15, 9.5), sharex=True)

for idx, shot in enumerate(selected_shots):
    ax = axes[idx // 2, idx % 2]
    
    # Plot Plain LoRA exact checkpoints
    if shot in plain_lora_data:
        p_iters, p_accs = plain_lora_data[shot]
        p_pairs = [(it, ac) for it, ac in zip(p_iters, p_accs) if it <= 2000]
        ax.plot([p[0] for p in p_pairs], [p[1] for p in p_pairs],
                color='#1A1A1A', lw=2.5, ls='--', marker='o', markersize=4.5,
                label='Plain CLIP-LoRA (Report 1 Baseline)')
    
    # Plot prompt tuning runs
    for rname, rinfo in run_data.items():
        if shot in rinfo['shots']:
            iters, accs = rinfo['shots'][shot]
            ax.plot(iters, accs, color=rinfo['color'], lw=1.9, ls=rinfo['ls'],
                    label=rname)
    
    ax.set_title(f'{shot}-Shot', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=10.5)
    if idx // 2 == 1:
        ax.set_xlabel('Training Iteration Step', fontsize=10.5)
    ax.set_xlim(50, 2050)
    
    if idx == 0:
        ax.legend(fontsize=7.0, loc='lower right', framealpha=0.9, ncol=2)

plt.tight_layout()
save_path = os.path.join(OUT_DIR, 'prompt_tuning_overfitting_comparison.png')
plt.savefig(save_path, dpi=200, bbox_inches='tight')
plt.close()
print('Generated:', save_path)

# Dedicated 4-shot comparison plot
fig, ax = plt.subplots(figsize=(10.5, 5.8))
if 4 in plain_lora_data:
    p_iters, p_accs = plain_lora_data[4]
    p_pairs = [(it, ac) for it, ac in zip(p_iters, p_accs) if it <= 2000]
    ax.plot([p[0] for p in p_pairs], [p[1] for p in p_pairs],
            color='#1A1A1A', lw=2.8, ls='--', marker='o', markersize=5,
            label='Plain CLIP-LoRA (Report 1 Baseline)')

for rname, rinfo in run_data.items():
    if 4 in rinfo['shots']:
        iters, accs = rinfo['shots'][4]
        ax.plot(iters, accs, color=rinfo['color'], lw=2.0, ls=rinfo['ls'], marker='s', markersize=3.5,
                label=rname)

ax.set_xlabel('Training Iteration Step', fontsize=11.5)
ax.set_ylabel('Test Accuracy (%)', fontsize=11.5)
ax.set_title('4-Shot', fontsize=12, fontweight='bold')
ax.legend(fontsize=7.5, loc='lower right', framealpha=0.95, ncol=2)
ax.set_xlim(50, 2050)
save_path_4shot = os.path.join(OUT_DIR, 'training_curves_4shots_comparison.png')
plt.tight_layout()
plt.savefig(save_path_4shot, dpi=200, bbox_inches='tight')
plt.close()
print('Generated:', save_path_4shot)
