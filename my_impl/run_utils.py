
import random
import argparse  
import numpy as np 
import torch

    

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1, type=int)
    # Dataset arguments
    parser.add_argument('--root_path', type=str, default='')
    parser.add_argument('--dataset', type=str, default='dtd')
    parser.add_argument('--shots', default=16, type=int)
    # Model arguments
    parser.add_argument('--backbone', default='ViT-B/16', type=str)
    parser.add_argument('--resize_mode', default='pad', choices=['pad', 'crop', 'direct'], help="Image resize mode: 'pad' (Option 2: square letterbox padding, zero cropping), 'crop' (legacy center/random crop), or 'direct' (stretch/squash)")
    # Augmentation arguments
    parser.add_argument('--vflip', default=True, action=argparse.BooleanOptionalAction, help='Enable random vertical flip augmentation (default: True, auto-disabled for stanford_cars)')
    parser.add_argument('--shift_px', default=5, type=int, help='Maximum pixel translation movement in each direction (default: 5, 0 to disable)')
    parser.add_argument('--noise_std', default=0.015, type=float, help='Standard deviation of additive Gaussian noise (default: 0.015, 0 to disable)')
    parser.add_argument('--sp_noise_amount', default=0.005, type=float, help='Fraction of pixels corrupted by Salt and Pepper impulse noise (default: 0.005, 0 to disable)')
    # Training arguments
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--n_iters', '--num_iter', '--num_iters', default=250, type=int, help='Number of iterations per shot (default: 250)')
    parser.add_argument('--batch_size', default=32, type=int)
    # LoRA arguments
    parser.add_argument('--position', type=str, default='all', choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'], help='where to put the LoRA modules')
    parser.add_argument('--encoder', type=str, choices=['text', 'vision', 'both'], default='both')
    parser.add_argument('--params', metavar='N', type=str, nargs='+', default=['q', 'k', 'v'], help='list of attention matrices where putting a LoRA') 
    parser.add_argument('--r', default=2, type=int, help='the rank of the low-rank matrices')
    parser.add_argument('--alpha', default=1.0, type=float, help='scaling (see LoRA paper)')
    parser.add_argument('--dropout_rate', default=0.25, type=float, help='dropout rate applied before the LoRA module')
    
    parser.add_argument('--save_path', default=None, help='path to save the lora modules after training, not saved if None')
    parser.add_argument('--filename', default='lora_weights', help='file name to save the lora weights (.pt extension will be added)')
    
    parser.add_argument('--eval_only', default=False, action='store_true', help='only evaluate the LoRA modules (save_path should not be None)')
    parser.add_argument('--clear_vis', default=False, action='store_true', help='delete old visualizations folder before running')
    
    # Method Selection
    parser.add_argument('--method', type=str, default='coop_lora', choices=['lora', 'coop_lora', 'coop_only', 'rt_lora', 'csc_lora', 'coop_csc', 'res_cls_lora', 'plain_lora_res_cls'], help='Training method: legacy LoRA, CoOp Text + Vision LoRA, CoOp Only, Residual-Template Dual-LoRA (rt_lora), Class-Specific Context (csc_lora), or Plain LoRA with Residual Class Tokens (res_cls_lora)')
    
    # Prompt Tuning Arguments
    parser.add_argument('--n_ctx', type=int, default=4, help='Number of learnable context tokens M (default: 4)')
    parser.add_argument('--csc', default=False, action='store_true', help='Use Class-Specific Context (CSC) where each class learns its own context vectors separately')
    parser.add_argument('--csc_iters', '--n_iters_csc', type=int, default=None, help='Number of iterations per shot to train CSC prompt before freezing it (default: full n_iters)')
    parser.add_argument('--prompt_pos', type=str, default='end', choices=['end', 'middle', 'front'], help='Class token position in prompt')
    parser.add_argument('--ctx_init', type=str, default='', help='Context token initialization text (e.g. "a photo of a")')
    parser.add_argument('--learn_class_tokens', default=False, action='store_true', help='Make class descriptor tokens learnable')
    parser.add_argument('--class_token_mode', type=str, default='residual', choices=['residual', 'full'], help='Class token learning mode: residual shift or full trainable embedding')
    parser.add_argument('--learn_template_tokens', default=True, action='store_true', help='Make template token embeddings learnable via residual (rt_lora)')
    parser.add_argument('--lr_class', type=float, default=1e-3, help='Learning rate for class residual tokens (default: 1e-3)')
    parser.add_argument('--lr_template', type=float, default=1e-4, help='Learning rate for template residual tokens (default: 1e-4)')
    
    # Loss Arguments
    parser.add_argument('--base_loss', type=str, default='ce', choices=['ce', 'contrastive'], help='Base classification loss')
    parser.add_argument('--use_ordinal', default=False, action='store_true', help='Enable semantic / partial ordinal cost matrix loss')
    parser.add_argument('--ordinal_loss_type', type=str, default='expected_cost', choices=['expected_cost', 'soft_kl'], help='Ordinal loss type')
    parser.add_argument('--lambda_ord', type=float, default=1.0, help='Weight for ordinal cost loss (default: 1.0)')
    parser.add_argument('--use_promptsrc', default=False, action='store_true', help='Enable PromptSRC self-consistency and feature regularization')
    parser.add_argument('--lambda_src', type=float, default=1.0, help='Weight for PromptSRC loss')
    parser.add_argument('--src_temp', type=float, default=2.0, help='Distillation temperature for PromptSRC KL loss')
    
    args = parser.parse_args()

    return args
    

        
