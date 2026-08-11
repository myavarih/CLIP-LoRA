import os
import sys
import json
import shutil
import argparse
import subprocess
import pandas as pd
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Automated CLIP-LoRA Multi-Shot Experiment Runner with Resumption")
    parser.add_argument('--dataset', type=str, default='walnut', help='Dataset name (default: walnut)')
    parser.add_argument('--root_path', type=str, default='', help='Dataset root path (default: empty for auto-resolution)')
    parser.add_argument('--shots', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32], help='List of shots to run')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and evaluation')
    parser.add_argument('--seed', type=int, default=1, help='Fixed random seed')
    parser.add_argument('--backbone', type=str, default='ViT-B/16', help='CLIP backbone')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--n_iters', type=int, default=500, help='Number of iterations per shot')
    parser.add_argument('--position', type=str, default='all', choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'])
    parser.add_argument('--encoder', type=str, default='both', choices=['text', 'vision', 'both'])
    parser.add_argument('--params', type=str, nargs='+', default=['q', 'k', 'v'])
    parser.add_argument('--r', type=int, default=2, help='LoRA rank')
    parser.add_argument('--alpha', type=int, default=1, help='LoRA alpha scaling')
    parser.add_argument('--dropout_rate', type=float, default=0.25, help='LoRA dropout rate')
    parser.add_argument('--output_dir', type=str, default='experiments_output', help='Base directory to save results and visualizations')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='Base directory to save model checkpoints')
    parser.add_argument('--force', action='store_true', help='Force re-run even if saved experiment exists')
    return parser.parse_args()


def is_experiment_completed(checkpoints_dir, output_dir, backbone, dataset, shot, seed):
    """
    Checks if both the model checkpoint and the run summary JSON exist and are valid.
    """
    backbone_clean = backbone.replace('/', '').replace('-', '').lower()
    checkpoint_file = os.path.join(checkpoints_dir, backbone_clean, dataset, f"{shot}shots", f"seed{seed}", "lora_weights.pt")
    
    exp_dir = os.path.join(output_dir, f"{dataset}_{shot}shots_seed{seed}")
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


def print_summary_table(results):
    print("\n" + "=" * 80)
    print("                     📊 EXPERIMENT RESULTS SUMMARY 📊")
    print("=" * 80)
    header = f"{'Shot':<8} | {'Zero-Shot Acc':<15} | {'Final Train Acc':<16} | {'Final Test Acc':<15} | {'Train Time (s)':<15}"
    print(header)
    print("-" * 80)
    for r in results:
        shot_str = f"{r['shots']} shot"
        zs_acc = f"{r.get('zero_shot_test_acc', 0.0):.2f}%"
        train_acc = f"{r.get('final_train_acc', 0.0):.2f}%"
        test_acc = f"{r.get('final_test_acc', 0.0):.2f}%"
        train_time = f"{r.get('training_time_seconds', 0.0):.2f}s"
        print(f"{shot_str:<8} | {zs_acc:<15} | {train_acc:<16} | {test_acc:<15} | {train_time:<15}")
    print("=" * 80 + "\n")


def run_experiments(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_script = os.path.join(script_dir, "main.py")
    
    print(f"🚀 Starting CLIP-LoRA Experiments Suite")
    print(f"📁 Dataset: {args.dataset}")
    print(f"🎯 Target Shots: {args.shots}")
    print(f"📦 Batch Size: {args.batch_size}")
    print(f"🎲 Seed: {args.seed}")
    print(f"💾 Checkpoints Dir: {args.checkpoints_dir}")
    print(f"🖼️ Visualizations & Output Dir: {args.output_dir}\n")
    
    all_results = []
    
    for shot in args.shots:
        print(f"\n{'-'*60}")
        print(f"🔬 Processing Experiment: {args.dataset} | {shot} Shot(s) | Seed {args.seed}")
        print(f"{'-'*60}")
        
        completed, prev_data, ckpt_path, exp_dir = is_experiment_completed(
            args.checkpoints_dir, args.output_dir, args.backbone, args.dataset, shot, args.seed
        )
        
        if completed and not args.force:
            print(f"✅ Experiment for {shot} shot(s) is already completed!")
            print(f"   Checkpoint: {ckpt_path}")
            print(f"   Visualizations saved in: {os.path.join(exp_dir, 'visualizations')}")
            res = prev_data.get("results", {})
            res_entry = {
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
            '--position', str(args.position),
            '--encoder', str(args.encoder),
            '--r', str(args.r),
            '--alpha', str(args.alpha),
            '--dropout_rate', str(args.dropout_rate),
            '--save_path', str(args.checkpoints_dir),
            '--filename', 'lora_weights',
            '--clear_vis'
        ]
        
        if args.root_path:
            cmd.extend(['--root_path', str(args.root_path)])
            
        if args.params:
            cmd.append('--params')
            cmd.extend(args.params)
            
        print(f"▶️ Executing: {' '.join(cmd)}\n")
        
        start_time = datetime.now()
        process = subprocess.run(cmd, cwd=script_dir)
        duration = (datetime.now() - start_time).total_seconds()
        
        if process.returncode != 0:
            print(f"❌ Error: Experiment for {shot} shot(s) failed with exit code {process.returncode}!")
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
            
        print(f"🎉 Shot {shot} finished successfully!")
        
    # Save overall summary files
    if all_results:
        summary_json_path = os.path.join(args.output_dir, "all_shots_summary.json")
        summary_csv_path = os.path.join(args.output_dir, "results_summary.csv")
        
        with open(summary_json_path, 'w') as f:
            json.dump(all_results, f, indent=4)
            
        df = pd.DataFrame(all_results)
        df.to_csv(summary_csv_path, index=False)
        
        print_summary_table(all_results)
        print(f"📑 Consolidated results saved to:")
        print(f"   JSON: {summary_json_path}")
        print(f"   CSV:  {summary_csv_path}\n")
    else:
        print("⚠️ No experiments were executed.")


if __name__ == '__main__':
    args = parse_args()
    run_experiments(args)
