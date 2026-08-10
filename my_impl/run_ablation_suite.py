import os
import sys
import json
import shutil
import argparse
import subprocess
import pandas as pd
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Automated CLIP-LoRA Targeted Ablation Suite")
    parser.add_argument('--dataset', type=str, default='walnut', help='Dataset name (default: walnut)')
    parser.add_argument('--root_path', type=str, default='', help='Dataset root path (default: empty for auto-resolution)')
    parser.add_argument('--shots', type=int, nargs='+', default=[1, 2, 4, 8, 16], help='List of shots to run')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and evaluation')
    parser.add_argument('--seed', type=int, default=1, help='Fixed random seed')
    parser.add_argument('--backbone', type=str, default='ViT-B/16', help='CLIP backbone')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--n_iters', type=int, default=500, help='Number of iterations per shot')
    parser.add_argument('--alpha', type=int, default=1, help='LoRA alpha scaling')
    parser.add_argument('--dropout_rate', type=float, default=0.25, help='LoRA dropout rate')
    parser.add_argument('--output_dir', type=str, default='ablation_experiments_output', help='Base directory to save results and visualizations')
    parser.add_argument('--checkpoints_dir', type=str, default='ablation_checkpoints', help='Base directory to save model checkpoints')
    parser.add_argument('--force', action='store_true', help='Force re-run even if saved experiment exists')
    return parser.parse_args()


# Baseline Configuration
BASELINE = {
    'params': ['q', 'k', 'v'],
    'r': 2,
    'encoder': 'both',
    'position': 'all'
}

# Ablation Variations (Targeted one-parameter-at-a-time changes)
ABLATIONS = [
    # 0. Baseline
    BASELINE,
    # 1. Attention Matrices (params)
    {**BASELINE, 'params': ['v']},
    {**BASELINE, 'params': ['q', 'v']},
    # 2. Rank (r)
    {**BASELINE, 'r': 1},
    {**BASELINE, 'r': 3},
    # 3. Encoders
    {**BASELINE, 'encoder': 'text'},
    {**BASELINE, 'encoder': 'vision'},
    # 4. Layer Positions
    {**BASELINE, 'position': 'bottom'},
    {**BASELINE, 'position': 'mid'},
    {**BASELINE, 'position': 'up'}
]


def get_config_name(config):
    """Generates a unique string identifier for the given ablation configuration."""
    name_parts = []
    if config['params'] != BASELINE['params']:
        name_parts.append('p_' + ''.join(config['params']))
    if config['r'] != BASELINE['r']:
        name_parts.append(f"r_{config['r']}")
    if config['encoder'] != BASELINE['encoder']:
        name_parts.append(f"e_{config['encoder']}")
    if config['position'] != BASELINE['position']:
        name_parts.append(f"pos_{config['position']}")
        
    if not name_parts:
        return "baseline"
    return "_".join(name_parts)


def is_experiment_completed(checkpoints_dir, output_dir, backbone, dataset, shot, seed, config_name):
    """
    Checks if both the model checkpoint and the run summary JSON exist and are valid for this config.
    """
    backbone_clean = backbone.replace('/', '').replace('-', '').lower()
    
    # We pass checkpoints_dir as checkpoints_dir/config_name to main.py to avoid collision
    ckpt_base = os.path.join(checkpoints_dir, config_name)
    checkpoint_file = os.path.join(ckpt_base, backbone_clean, dataset, f"{shot}shots", f"seed{seed}", "lora_weights.pt")
    
    exp_dir = os.path.join(output_dir, f"{dataset}_{shot}shots_seed{seed}_{config_name}")
    summary_file = os.path.join(exp_dir, "visualizations", "run_summary.json")
    summary_file_alt = os.path.join(exp_dir, "run_summary.json")
    
    has_checkpoint = os.path.exists(checkpoint_file) and os.path.getsize(checkpoint_file) > 0
    has_summary = (os.path.exists(summary_file) and os.path.getsize(summary_file) > 0) or \
                  (os.path.exists(summary_file_alt) and os.path.getsize(summary_file_alt) > 0)
                  
    if has_checkpoint and has_summary:
        target_summary = summary_file if os.path.exists(summary_file) else summary_file_alt
        try:
            with open(target_summary, 'r') as f:
                data = json.load(f)
            return True, data, checkpoint_file, exp_dir
        except Exception:
            return False, None, checkpoint_file, exp_dir
            
    return False, None, checkpoint_file, exp_dir


def get_summary_table_string(results):
    lines = []
    lines.append("\n" + "=" * 110)
    lines.append("                                📊 ABLATION EXPERIMENTS SUMMARY 📊")
    lines.append("=" * 110)
    header = f"{'Config':<20} | {'Shot':<6} | {'Zero-Shot Acc':<15} | {'Final Train Acc':<16} | {'Final Test Acc':<15} | {'Train Time':<12}"
    lines.append(header)
    lines.append("-" * 110)
    for r in results:
        cfg = r['config']
        shot_str = f"{r['shots']}s"
        zs_acc = f"{r.get('zero_shot_test_acc', 0.0):.2f}%"
        train_acc = f"{r.get('final_train_acc', 0.0):.2f}%"
        test_acc = f"{r.get('final_test_acc', 0.0):.2f}%"
        train_time = f"{r.get('training_time_seconds', 0.0):.2f}s"
        lines.append(f"{cfg:<20} | {shot_str:<6} | {zs_acc:<15} | {train_acc:<16} | {test_acc:<15} | {train_time:<12}")
    lines.append("=" * 110 + "\n")
    return "\n".join(lines)


def print_summary_table(results):
    print(get_summary_table_string(results))


def run_experiments(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_script = os.path.join(script_dir, "main.py")
    
    print(f"🚀 Starting CLIP-LoRA Targeted Ablation Suite")
    print(f"📁 Dataset: {args.dataset}")
    print(f"🎯 Target Shots: {args.shots}")
    print(f"📦 Batch Size: {args.batch_size}")
    print(f"🎲 Seed: {args.seed}")
    print(f"💾 Checkpoints Dir: {args.checkpoints_dir}")
    print(f"🖼️ Output Dir: {args.output_dir}\n")
    print(f"Total ablation configurations: {len(ABLATIONS)}")
    
    all_results = []
    
    for config in ABLATIONS:
        config_name = get_config_name(config)
        print(f"\n{'='*80}")
        print(f"🔄 Starting Ablation Config: [{config_name}]")
        print(f"   Params: {config['params']} | Rank: {config['r']} | Encoder: {config['encoder']} | Position: {config['position']}")
        print(f"{'='*80}")
        
        for shot in args.shots:
            print(f"\n{'-'*60}")
            print(f"🔬 Processing: {args.dataset} | {shot} Shot(s) | Config: {config_name}")
            print(f"{'-'*60}")
            
            completed, prev_data, ckpt_path, exp_dir = is_experiment_completed(
                args.checkpoints_dir, args.output_dir, args.backbone, args.dataset, shot, args.seed, config_name
            )
            
            if completed and not args.force:
                print(f"✅ Experiment for {shot} shot(s) [{config_name}] is already completed!")
                print(f"   Checkpoint: {ckpt_path}")
                print(f"   Visualizations saved in: {os.path.join(exp_dir, 'visualizations')}")
                res = prev_data.get("results", {})
                res_entry = {
                    "config": config_name,
                    "params": str(config['params']),
                    "rank": config['r'],
                    "encoder": config['encoder'],
                    "position": config['position'],
                    "shots": shot,
                    "dataset": args.dataset,
                    "seed": args.seed,
                    "zero_shot_test_acc": res.get("zero_shot_test_acc", 0.0),
                    "final_train_acc": res.get("final_train_acc", 0.0),
                    "final_test_acc": res.get("final_test_acc", 0.0),
                    "training_time_seconds": res.get("training_time_seconds", 0.0),
                    "checkpoint": ckpt_path,
                    "output_dir": exp_dir,
                    "status": "Resumed (Cached)"
                }
                all_results.append(res_entry)
                print(f"   Zero-shot Acc: {res_entry['zero_shot_test_acc']:.2f}% | Final Test Acc: {res_entry['final_test_acc']:.2f}%\n")
                continue
            
            # Isolated checkpoint directory for this config to avoid collisions
            config_ckpt_dir = os.path.join(args.checkpoints_dir, config_name)
            
            # Build command line execution
            cmd = [
                sys.executable, main_script,
                '--dataset', str(args.dataset),
                '--shots', str(shot),
                '--batch_size', str(args.batch_size),
                '--seed', str(args.seed),
                '--backbone', str(args.backbone),
                '--lr', str(args.lr),
                '--n_iters', str(args.n_iters),
                '--position', str(config['position']),
                '--encoder', str(config['encoder']),
                '--r', str(config['r']),
                '--alpha', str(args.alpha),
                '--dropout_rate', str(args.dropout_rate),
                '--save_path', str(config_ckpt_dir),
                '--filename', 'lora_weights',
                '--clear_vis'
            ]
            
            if args.root_path:
                cmd.extend(['--root_path', str(args.root_path)])
                
            if config['params']:
                cmd.append('--params')
                cmd.extend(config['params'])
                
            print(f"▶️ Executing: {' '.join(cmd)}\n")
            
            start_time = datetime.now()
            process = subprocess.run(cmd, cwd=script_dir)
            duration = (datetime.now() - start_time).total_seconds()
            
            if process.returncode != 0:
                print(f"❌ Error: Experiment for {shot} shot(s) [{config_name}] failed with exit code {process.returncode}!")
                continue
                
            # Organize visualizations and output
            os.makedirs(exp_dir, exist_ok=True)
            dest_vis_dir = os.path.join(exp_dir, "visualizations")
            source_vis_dir = os.path.join(script_dir, "visualizations")
            
            if os.path.exists(source_vis_dir):
                if os.path.exists(dest_vis_dir):
                    shutil.rmtree(dest_vis_dir)
                shutil.copytree(source_vis_dir, dest_vis_dir)
                print(f"📁 Visualizations copied to: {dest_vis_dir}")
                
            summary_path = os.path.join(dest_vis_dir, "run_summary.json")
            res_data = {}
            if os.path.exists(summary_path):
                try:
                    with open(summary_path, 'r') as f:
                        res_data = json.load(f).get("results", {})
                except Exception as e:
                    print(f"⚠️ Warning: Could not read summary JSON: {e}")
                    
            res_entry = {
                "config": config_name,
                "params": str(config['params']),
                "rank": config['r'],
                "encoder": config['encoder'],
                "position": config['position'],
                "shots": shot,
                "dataset": args.dataset,
                "seed": args.seed,
                "zero_shot_test_acc": res_data.get("zero_shot_test_acc", 0.0),
                "final_train_acc": res_data.get("final_train_acc", 0.0),
                "final_test_acc": res_data.get("final_test_acc", 0.0),
                "training_time_seconds": res_data.get("training_time_seconds", duration),
                "checkpoint": ckpt_path,
                "output_dir": exp_dir,
                "status": "Completed"
            }
            all_results.append(res_entry)
            
            # Save individual experiment info
            with open(os.path.join(exp_dir, "experiment_info.json"), 'w') as f:
                json.dump(res_entry, f, indent=4)
                
            print(f"🎉 Shot {shot} for [{config_name}] finished successfully!")
            
    # Save overall summary files
    if all_results:
        summary_json_path = os.path.join(args.output_dir, "ablation_summary.json")
        summary_csv_path = os.path.join(args.output_dir, "ablation_summary.csv")
        summary_txt_path = os.path.join(args.output_dir, "ablation_summary.txt")
        
        with open(summary_json_path, 'w') as f:
            json.dump(all_results, f, indent=4)
            
        df = pd.DataFrame(all_results)
        df.to_csv(summary_csv_path, index=False)
        
        summary_str = get_summary_table_string(all_results)
        with open(summary_txt_path, 'w') as f:
            f.write(summary_str)
        
        print(summary_str)
        print(f"📑 Consolidated results saved to:")
        print(f"   JSON: {summary_json_path}")
        print(f"   CSV:  {summary_csv_path}")
        print(f"   TXT:  {summary_txt_path}\n")
    else:
        print("⚠️ No experiments were executed.")


if __name__ == '__main__':
    args = parse_args()
    run_experiments(args)
