import os
import sys
import json
import shutil
import argparse
import itertools
import subprocess
import pandas as pd
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Automated CLIP-LoRA Multi-Shot & Grid Experiment Runner with Resumption")
    parser.add_argument('--dataset', type=str, default='walnut', help='Dataset name (default: walnut)')
    parser.add_argument('--root_path', type=str, default='', help='Dataset root path (default: empty for auto-resolution)')
    parser.add_argument('--shots', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32], help='List of shots to run')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and evaluation')
    parser.add_argument('--seed', type=int, nargs='+', default=[1], help='List of random seeds (e.g. 1 2 3)')
    parser.add_argument('--backbone', type=str, default='ViT-B/16', help='CLIP backbone')
    parser.add_argument('--resize_mode', type=str, default='pad', choices=['pad', 'crop', 'direct'], help="Image resize mode: 'pad' (Option 2: letterbox to square, zero cropping), 'crop' (legacy center crop), or 'direct' (stretch/squash)")
    parser.add_argument('--vflip', default=True, action=argparse.BooleanOptionalAction, help='Enable random vertical flip augmentation (default: True, auto-disabled for stanford_cars)')
    parser.add_argument('--shift_px', type=int, default=5, help='Maximum pixel translation movement in each direction (default: 5, 0 to disable)')
    parser.add_argument('--noise_std', type=float, default=0.015, help='Standard deviation of additive Gaussian noise (default: 0.015, 0 to disable)')
    parser.add_argument('--sp_noise_amount', type=float, default=0.005, help='Fraction of pixels corrupted by Salt and Pepper impulse noise (default: 0.005, 0 to disable)')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--n_iters', '--num_iter', '--num_iters', type=int, default=250, help='Number of iterations per shot (default: 250)')
    parser.add_argument('--position', type=str, default='all', choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'])
    parser.add_argument('--encoder', type=str, default='both', choices=['text', 'vision', 'both'])
    parser.add_argument('--params', type=str, nargs='+', default=['q', 'k', 'v'])
    parser.add_argument('--r', type=int, nargs='+', default=[2], help='List of LoRA ranks to run (e.g. 2 3)')
    parser.add_argument('--alpha', type=float, nargs='+', default=[1.0], help='List of LoRA alpha scalings to run (e.g. 1.5 2.0)')
    parser.add_argument('--dropout_rate', type=float, default=0.25, help='LoRA dropout rate')
    parser.add_argument('--output_dir', type=str, default='experiments_output', help='Base directory to save results and visualizations')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='Base directory to save model checkpoints')
    parser.add_argument('--force', action='store_true', help='Force re-run even if saved experiment exists')
    
    # Method & Prompt Tuning
    parser.add_argument('--method', type=str, default='coop_lora', choices=['lora', 'coop_lora', 'coop_only', 'rt_lora', 'csc_lora', 'coop_csc', 'res_cls_lora', 'plain_lora_res_cls'], help='Training method')
    parser.add_argument('--n_ctx', type=int, default=4, help='Number of learnable context tokens M')
    parser.add_argument('--csc', action='store_true', help='Use Class-Specific Context (CSC) where context is learned per class separately')
    parser.add_argument('--csc_iters', '--n_iters_csc', type=int, default=None, help='Number of iterations per shot to train CSC prompt before freezing it (default: full n_iters)')
    parser.add_argument('--prompt_pos', type=str, default='end', choices=['end', 'middle', 'front'])
    parser.add_argument('--ctx_init', type=str, default='')
    parser.add_argument('--learn_class_tokens', action='store_true', help='Make class tokens learnable')
    parser.add_argument('--class_token_mode', type=str, default='residual', choices=['residual', 'full'])
    parser.add_argument('--learn_template_tokens', action='store_true', default=True, help='Make template tokens learnable')
    parser.add_argument('--lr_class', type=float, default=1e-3, help='Learning rate for class residual tokens')
    parser.add_argument('--lr_template', type=float, default=1e-4, help='Learning rate for template residual tokens')
    
    # Loss flags
    parser.add_argument('--base_loss', type=str, default='ce', choices=['ce', 'contrastive'])
    parser.add_argument('--use_ordinal', action='store_true', help='Enable semantic / partial ordinal cost matrix loss')
    parser.add_argument('--ordinal_loss_type', type=str, default='expected_cost', choices=['expected_cost', 'soft_kl'])
    parser.add_argument('--lambda_ord', type=float, default=1.0, help='Weight for ordinal cost loss (default: 1.0)')
    parser.add_argument('--use_promptsrc', action='store_true', help='Enable PromptSRC self-consistency regularizer')
    parser.add_argument('--lambda_src', type=float, default=1.0)
    parser.add_argument('--src_temp', type=float, default=2.0)
    return parser.parse_args()


def is_experiment_completed(checkpoints_dir, output_dir, backbone, dataset, shot, seed, r=2, alpha=1.0, multi_config=False):
    """
    Checks if both the model checkpoint and the run summary JSON exist and are valid.
    Supports both multi-parameter nested paths and single-run legacy paths.
    """
    backbone_clean = backbone.replace('/', '').replace('-', '').lower()
    
    ckpt_path_config = os.path.join(checkpoints_dir, f"r{r}_a{alpha}", backbone_clean, dataset, f"{shot}shots", f"seed{seed}", "lora_weights.pt")
    ckpt_path_flat = os.path.join(checkpoints_dir, backbone_clean, dataset, f"{shot}shots", f"seed{seed}", "lora_weights.pt")
    
    if os.path.exists(ckpt_path_config) and os.path.getsize(ckpt_path_config) > 0:
        checkpoint_file = ckpt_path_config
    elif os.path.exists(ckpt_path_flat) and os.path.getsize(ckpt_path_flat) > 0:
        checkpoint_file = ckpt_path_flat
    else:
        checkpoint_file = ckpt_path_config if multi_config else ckpt_path_flat

    exp_dir_config = os.path.join(output_dir, f"{dataset}_r{r}_a{alpha}_{shot}shots_seed{seed}")
    exp_dir_flat = os.path.join(output_dir, f"{dataset}_{shot}shots_seed{seed}")
    
    if os.path.exists(exp_dir_config):
        exp_dir = exp_dir_config
    elif os.path.exists(exp_dir_flat) and not multi_config:
        exp_dir = exp_dir_flat
    else:
        exp_dir = exp_dir_config if multi_config else exp_dir_flat

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
    print("\n" + "=" * 95)
    print("                              📊 EXPERIMENT RESULTS SUMMARY 📊")
    print("=" * 95)
    header = f"{'Config (r, α, seed)':<24} | {'Shot':<6} | {'Zero-Shot Acc':<14} | {'Final Train Acc':<16} | {'Final Test Acc':<15} | {'Time':<10}"
    print(header)
    print("-" * 95)
    for r in results:
        cfg_str = f"r={r.get('r', 2)}, α={r.get('alpha', 1.0)}, s={r.get('seed', 1)}"
        shot_str = f"{r['shots']} shot"
        zs_acc = f"{r.get('zero_shot_test_acc', 0.0):.2f}%"
        train_acc = f"{r.get('final_train_acc', 0.0):.2f}%"
        test_acc = f"{r.get('final_test_acc', 0.0):.2f}%"
        train_time = f"{r.get('training_time_seconds', 0.0):.1f}s"
        print(f"{cfg_str:<24} | {shot_str:<6} | {zs_acc:<14} | {train_acc:<16} | {test_acc:<15} | {train_time:<10}")
    print("=" * 95 + "\n")


def run_experiments(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_script = os.path.join(script_dir, "main.py")
    
    # Determine if this is a multi-parameter grid
    multi_config = (len(args.r) > 1 or len(args.alpha) > 1 or len(args.seed) > 1)
    
    print(f"🚀 Starting CLIP-LoRA Multi-Shot Experiment Suite")
    print(f"📁 Dataset: {args.dataset}")
    print(f"🎯 Target Shots: {args.shots}")
    print(f"🔢 Ranks (r): {args.r}")
    print(f"📐 Alpha Scalings (α): {args.alpha}")
    print(f"🎲 Seeds: {args.seed}")
    print(f"📦 Batch Size: {args.batch_size}")
    print(f"🔄 Iterations per shot: {args.n_iters} (CSC iters: {args.csc_iters})")
    print(f"💾 Checkpoints Dir: {args.checkpoints_dir}")
    print(f"🖼️ Visualizations & Output Dir: {args.output_dir}\n")
    
    all_results = []
    
    # Iterate across grid: (r, alpha, seed, shot)
    for r_val in args.r:
        for alpha_val in args.alpha:
            for seed_val in args.seed:
                # Base save path for checkpoints for this config
                if multi_config:
                    ckpt_save_base = os.path.join(args.checkpoints_dir, f"r{r_val}_a{alpha_val}")
                else:
                    ckpt_save_base = args.checkpoints_dir

                for shot in args.shots:
                    config_tag = f"r={r_val}, α={alpha_val}, seed={seed_val}, shot={shot}"
                    print(f"\n{'-'*70}")
                    print(f"🔬 Processing Experiment: {args.dataset} | {config_tag}")
                    print(f"{'-'*70}")
                    
                    completed, prev_data, ckpt_path, exp_dir = is_experiment_completed(
                        args.checkpoints_dir, args.output_dir, args.backbone, args.dataset,
                        shot, seed_val, r=r_val, alpha=alpha_val, multi_config=multi_config
                    )
                    
                    if completed and not args.force:
                        print(f"✅ Experiment for [{config_tag}] is already completed!")
                        print(f"   Checkpoint: {ckpt_path}")
                        print(f"   Visualizations saved in: {os.path.join(exp_dir, 'visualizations')}")
                        res = prev_data.get("results", {})
                        res_entry = {
                            "dataset": args.dataset,
                            "r": r_val,
                            "alpha": alpha_val,
                            "seed": seed_val,
                            "shots": shot,
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
                        '--seed', str(seed_val),
                        '--backbone', str(args.backbone),
                        '--resize_mode', str(args.resize_mode),
                        '--shift_px', str(args.shift_px),
                        '--noise_std', str(args.noise_std),
                        '--sp_noise_amount', str(args.sp_noise_amount),
                        '--lr', str(args.lr),
                        '--n_iters', str(args.n_iters),
                        '--position', str(args.position),
                        '--encoder', str(args.encoder),
                        '--r', str(r_val),
                        '--alpha', str(alpha_val),
                        '--dropout_rate', str(args.dropout_rate),
                        '--save_path', str(ckpt_save_base),
                        '--filename', 'lora_weights',
                        '--clear_vis',
                        '--method', str(args.method),
                        '--n_ctx', str(args.n_ctx),
                        '--prompt_pos', str(args.prompt_pos),
                        '--class_token_mode', str(args.class_token_mode),
                        '--base_loss', str(args.base_loss),
                        '--ordinal_loss_type', str(args.ordinal_loss_type),
                        '--lambda_ord', str(args.lambda_ord),
                        '--lambda_src', str(args.lambda_src),
                        '--src_temp', str(args.src_temp),
                        '--lr_class', str(args.lr_class),
                        '--lr_template', str(args.lr_template)
                    ]
                    
                    if args.csc_iters is not None:
                        cmd.extend(['--csc_iters', str(args.csc_iters)])
                    if args.ctx_init:
                        cmd.extend(['--ctx_init', str(args.ctx_init)])
                    if args.csc or args.method in ['csc_lora', 'coop_csc']:
                        cmd.append('--csc')
                    if args.learn_class_tokens:
                        cmd.append('--learn_class_tokens')
                    if getattr(args, 'learn_template_tokens', True):
                        cmd.append('--learn_template_tokens')
                    if args.use_ordinal:
                        cmd.append('--use_ordinal')
                    if args.use_promptsrc:
                        cmd.append('--use_promptsrc')
                    if args.root_path:
                        cmd.extend(['--root_path', str(args.root_path)])
                    if args.vflip is not None:
                        cmd.append('--vflip' if args.vflip else '--no-vflip')
                        
                    if args.params:
                        cmd.append('--params')
                        cmd.extend(args.params)
                        
                    print(f"▶️ Executing: {' '.join(cmd)}\n")
                    
                    start_time = datetime.now()
                    process = subprocess.run(cmd, cwd=script_dir)
                    duration = (datetime.now() - start_time).total_seconds()
                    
                    if process.returncode != 0:
                        print(f"❌ Error: Experiment for [{config_tag}] failed with exit code {process.returncode}!")
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
                        "dataset": args.dataset,
                        "r": r_val,
                        "alpha": alpha_val,
                        "seed": seed_val,
                        "shots": shot,
                        "csc_iters": getattr(args, 'csc_iters', 100),
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
                        
                    print(f"🎉 [{config_tag}] finished successfully!")
        
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
