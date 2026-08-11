import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")

ABLATIONS_DIR = "/home/emmwhy/Downloads/results(2)/experiments_output/ablations"
FIGURES_DIR = "/home/emmwhy/Projects/CV_Lab/CLIP-LoRA/docs/figures"

def parse_results():
    results = []
    checkpoint_data = []

    if not os.path.exists(ABLATIONS_DIR):
        print(f"Directory {ABLATIONS_DIR} does not exist.")
        return [], []

    for item in os.listdir(ABLATIONS_DIR):
        item_path = os.path.join(ABLATIONS_DIR, item)
        if os.path.isdir(item_path):
            # Parse directory name: walnut_{shots}shots_seed2_{ablation}
            parts = item.split('_')
            if len(parts) >= 4 and parts[0] == 'walnut':
                shots = int(parts[1].replace('shots', ''))
                # Handle cases where ablation name has underscores
                ablation = '_'.join(parts[3:])
                
                vis_dir = os.path.join(item_path, 'visualizations')
                run_summary_path = os.path.join(vis_dir, 'run_summary.json')
                ckpt_acc_path = os.path.join(vis_dir, 'checkpoint_accuracies.json')
                
                if os.path.exists(run_summary_path):
                    with open(run_summary_path, 'r') as f:
                        data = json.load(f)
                        res = data.get('results', {})
                        res['shots'] = shots
                        res['ablation'] = ablation
                        results.append(res)
                
                if os.path.exists(ckpt_acc_path):
                    with open(ckpt_acc_path, 'r') as f:
                        data = json.load(f)
                        # Data is step: accuracy
                        for step_str, acc in data.items():
                            checkpoint_data.append({
                                'shots': shots,
                                'ablation': ablation,
                                'step': int(step_str),
                                'accuracy': acc
                            })
                            
    return results, checkpoint_data

def plot_acc_vs_shots(results):
    df = pd.DataFrame(results)
    if df.empty:
        print("No summary data to plot.")
        return
        
    plt.figure(figsize=(8, 6))
    
    # Plot Zero-Shot as a baseline (dashed line)
    zs_acc = df['zero_shot_test_acc'].mean()
    plt.axhline(y=zs_acc, color='r', linestyle='--', label=f'Zero-Shot ({zs_acc:.2f}%)')

    sns.lineplot(data=df, x='shots', y='final_test_acc', hue='ablation', marker='o', linewidth=2, markersize=8)
    
    plt.title('Final Test Accuracy vs. Number of Shots', fontsize=14)
    plt.xlabel('Number of Shots', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xticks([1, 2, 4, 8, 16])
    plt.legend(title='Ablation Type')
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'acc_vs_shots_2.png'), dpi=300)
    plt.close()
    print("Generated acc_vs_shots_2.png")

def plot_training_curves(checkpoint_data, target_shots=4):
    df = pd.DataFrame(checkpoint_data)
    if df.empty:
        print("No checkpoint data to plot.")
        return
        
    # Filter for target shots
    df_filtered = df[df['shots'] == target_shots]
    
    if df_filtered.empty:
        print(f"No checkpoint data for {target_shots} shots.")
        return
        
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_filtered, x='step', y='accuracy', hue='ablation', linewidth=2)
    
    plt.title(f'Test Accuracy during Training ({target_shots} Shots)', fontsize=14)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(title='Ablation Type')
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f'training_curves_{target_shots}shots_2.png'), dpi=300)
    plt.close()
    print(f"Generated training_curves_{target_shots}shots_2.png")

if __name__ == "__main__":
    results, checkpoint_data = parse_results()
    print(f"Parsed {len(results)} summary files and {len(checkpoint_data)} checkpoint records.")
    
    plot_acc_vs_shots(results)
    plot_training_curves(checkpoint_data, target_shots=4)
