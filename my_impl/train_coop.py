import os
import json
import time
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

from utils import Logger, cls_acc, clip_classifier, pre_load_features, compute_ordinal_metrics, save_run_info
from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, save_lora, load_lora
from prompt_learner import PromptLearner, CustomCoOpCLIP, ResidualTemplatePromptLearner, FeatureResidualPromptLearner
from mllm_prompts import get_mllm_prompts
from losses import CompositeCriterion
from vis_utils import plot_metrics, plot_confusion_matrix, plot_predictions, plot_embeddings, visualize_attention
from pytorch_grad_cam import EigenCAM
import clip


class CLIPVisionWrapper(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, x):
        device = next(self.clip_model.parameters()).device
        if device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                return self.clip_model.encode_image(x.type(self.clip_model.dtype))
        else:
            return self.clip_model.encode_image(x.type(self.clip_model.dtype))


def eigencam_reshape_transform(tensor, height=14, width=14):
    if tensor.ndim == 3 and tensor.shape[0] == height * width + 1:
        tensor = tensor.permute(1, 0, 2)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.permute(0, 3, 1, 2)
    return result.float()


def generate_eigencam_maps(clip_model, images_tensor, max_samples=20):
    if len(images_tensor) == 0:
        return []
    samples = images_tensor[:max_samples]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()
    wrapper = CLIPVisionWrapper(clip_model).to(device)
    wrapper.eval()
    
    if hasattr(clip_model, 'visual') and hasattr(clip_model.visual, 'transformer') and hasattr(clip_model.visual.transformer, 'resblocks'):
        target_layers = [clip_model.visual.transformer.resblocks[-1].ln_1]
        cam = EigenCAM(model=wrapper, target_layers=target_layers, reshape_transform=eigencam_reshape_transform)
        cams = []
        batch_size = 1
        for start_idx in range(0, len(samples), batch_size):
            batch = samples[start_idx:start_idx+batch_size].to(device)
            if device == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    batch_cam = cam(input_tensor=batch, targets=None)
            else:
                batch_cam = cam(input_tensor=batch, targets=None)
            cams.append(batch_cam)
        if device == "cuda":
            torch.cuda.empty_cache()
        return np.concatenate(cams, axis=0)
    return []


def get_prediction_indices(total_count, first_n=10, random_n=10):
    if total_count <= first_n:
        return np.arange(total_count)
    first_indices = list(range(first_n))
    remaining = list(range(first_n, total_count))
    if len(remaining) <= random_n:
        random_indices = remaining
    else:
        random_indices = list(np.random.choice(remaining, random_n, replace=False))
    return np.array(first_indices + random_indices)


def evaluate_coop(args, custom_model, loader, dataset):
    custom_model.eval()
    cost_matrix = getattr(dataset, 'cost_matrix', None)
    
    all_preds = []
    all_targets = []
    all_images = []
    all_probs = []
    
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            text_features = custom_model.encode_text()
            
        for images, target in loader:
            images = images.cuda()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                image_features = custom_model.encode_image(images)
                
            scale = custom_model.logit_scale.exp() if hasattr(custom_model, 'logit_scale') else 100.0
            cosine_similarity = scale * (image_features @ text_features.t())
            preds = cosine_similarity.argmax(dim=-1)
            probs = F.softmax(cosine_similarity, dim=-1)
            
            all_preds.append(preds.cpu())
            all_targets.append(target.cpu())
            all_images.append(images.cpu())
            all_probs.append(probs.cpu())
                
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_images = torch.cat(all_images)
    all_probs = torch.cat(all_probs)
    
    metrics = compute_ordinal_metrics(all_preds, all_targets, cost_matrix=cost_matrix)
    acc = metrics['acc']
    
    return acc, all_preds, all_targets, all_images, text_features, all_probs, metrics


def save_coop_checkpoint(args, list_lora_layers, prompt_learner, filename):
    if args.save_path is None:
        return
    os.makedirs(args.save_path, exist_ok=True)
    
    # 1. Save LoRA weights if vision LoRA is active
    if list_lora_layers is not None and len(list_lora_layers) > 0:
        save_lora(args, list_lora_layers, filename=filename)
        
    # 2. Save PromptLearner weights
    prompt_path = os.path.join(args.save_path, f"{filename}_prompt.pt")
    torch.save(prompt_learner.state_dict(), prompt_path)


def load_coop_checkpoint(args, list_lora_layers, prompt_learner, filename):
    if args.save_path is None:
        return
    # 1. Load LoRA weights
    if list_lora_layers is not None and len(list_lora_layers) > 0:
        load_lora(args, list_lora_layers, filename=filename)
        
    # 2. Load PromptLearner weights
    prompt_path = os.path.join(args.save_path, f"{filename}_prompt.pt")
    if os.path.exists(prompt_path):
        prompt_state = torch.load(prompt_path, map_location='cuda')
        prompt_learner.load_state_dict(prompt_state)


def run_coop_lora(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader, train_eval_loader=None):
    clip_model = clip_model.float()
    cost_matrix = getattr(dataset, 'cost_matrix', None)
    
    # 1. Compute and evaluate Zero-shot baseline
    Logger.step("Getting zero-shot textual features as baseline classifier.")
    use_mllm = getattr(args, 'use_mllm_prompts', False) or (args.method == 'mllm_feat_lora')
    mllm_dict = get_mllm_prompts(args.dataset, dataset.classnames) if use_mllm else None
    if mllm_dict is not None:
        Logger.step("Using MLLM Fine-Grained Prompt Ensemble (UniFGVC) for Zero-Shot and Anchor.")
        eval_template = mllm_dict
    else:
        eval_template = dataset.template
    zs_textual_features = clip_classifier(dataset.classnames, eval_template, clip_model)
    
    Logger.step("Loading visual features and labels from test set for zero-shot baseline.")
    test_features, test_labels = pre_load_features(clip_model, test_loader)
    test_features = test_features.cuda()
    test_labels = test_labels.cuda()
    
    zs_logits = logit_scale * test_features @ zs_textual_features
    zs_acc = cls_acc(zs_logits, test_labels)
    zs_preds = zs_logits.argmax(dim=-1).cpu()
    zs_metrics = compute_ordinal_metrics(zs_preds, test_labels.cpu(), cost_matrix=cost_matrix)
    
    Logger.success(f"Zero-shot CLIP's test accuracy: {zs_acc:.2f}% | Mean Cost: {zs_metrics.get('mean_cost', 0.0):.2f} | 1-Off Acc: {zs_metrics.get('acc_tolerance_1off', 0.0):.2f}%")
    test_features = test_features.cpu()
    test_labels = test_labels.cpu()
    torch.cuda.empty_cache()
    
    os.makedirs('visualizations', exist_ok=True)
    
    # 2. Setup PromptLearner / ResidualTemplatePromptLearner
    csc_flag = getattr(args, 'csc', False) or (args.method in ['csc_lora', 'coop_csc'])
    if args.method == 'mllm_feat_lora':
        Logger.step("Initializing FeatureResidualPromptLearner (MLLM Anchor + Learnable Feature Residual delta_w)...")
        zs_anchor = zs_textual_features.t().float().cuda() # [K, D]
        prompt_learner = FeatureResidualPromptLearner(base_features=zs_anchor).cuda()
    elif args.method in ['rt_lora', 'res_cls_lora', 'plain_lora_res_cls']:
        template_str = dataset.template[0] if hasattr(dataset, 'template') and dataset.template else "a photo of a {} walnut, a type of walnut."
        learn_template = getattr(args, 'learn_template_tokens', True) if args.method == 'rt_lora' else False
        learn_class = getattr(args, 'learn_class_tokens', True)
        desc = "RT-LoRA (Dual Residual Template)" if learn_template else "Plain LoRA + Residual Class Tokens"
        Logger.step(f"Initializing ResidualTemplatePromptLearner ({desc}) with template: '{template_str}' (learn_template={learn_template}, learn_class={learn_class})...")
        prompt_learner = ResidualTemplatePromptLearner(
            clip_model=clip_model,
            classnames=dataset.classnames,
            template=template_str,
            learn_template_tokens=learn_template,
            learn_class_tokens=learn_class
        ).cuda()
    else:
        csc_desc = f"CSC (Class-Specific Context, {len(dataset.classnames)}x{args.n_ctx} tokens)" if csc_flag else f"Unified Context ({args.n_ctx} tokens)"
        Logger.step(f"Initializing PromptLearner ({csc_desc}, learn_class_tokens={args.learn_class_tokens}, mode={args.class_token_mode})...")
        prompt_learner = PromptLearner(
            clip_model=clip_model,
            classnames=dataset.classnames,
            n_ctx=args.n_ctx,
            ctx_init=args.ctx_init,
            csc=csc_flag,
            class_token_position=args.prompt_pos,
            learn_class_tokens=args.learn_class_tokens,
            class_token_mode=args.class_token_mode
        ).cuda()
    
    # 3. Setup LoRA on Encoders
    list_lora_layers = []
    if args.method in ['rt_lora', 'res_cls_lora', 'plain_lora_res_cls']:
        encoder_target = getattr(args, 'encoder', 'both') or 'both'
        Logger.step(f"Applying LoRA to {encoder_target.upper()} Encoder(s) (r={args.r}, alpha={args.alpha}, params={args.params})...")
        args.encoder = encoder_target
        list_lora_layers = apply_lora(args, clip_model)
        mark_only_lora_as_trainable(clip_model)
    elif args.method in ['coop_lora', 'csc_lora', 'coop_csc', 'mllm_feat_lora']:
        encoder_target = getattr(args, 'encoder', 'vision') or 'vision'
        Logger.step(f"Applying LoRA to {encoder_target.upper()} Encoder(s) (r={args.r}, alpha={args.alpha}, params={args.params})...")
        args.encoder = encoder_target
        list_lora_layers = apply_lora(args, clip_model)
        mark_only_lora_as_trainable(clip_model)
    else:
        # All clip backbone weights frozen (e.g. coop_only)
        for p in clip_model.parameters():
            p.requires_grad = False

    clip_model = clip_model.cuda()
    custom_model = CustomCoOpCLIP(clip_model, prompt_learner).cuda()

    # Eval Only Mode
    if args.eval_only:
        Logger.step("Running Eval-Only mode with pre-trained checkpoint...")
        load_coop_checkpoint(args, list_lora_layers, prompt_learner, filename=args.filename)
        acc_test, test_preds, test_targets, test_images, text_features, test_probs, eval_metrics = evaluate_coop(args, custom_model, test_loader, dataset)
        Logger.success(f"Loaded Checkpoint Test Accuracy: {acc_test:.2f}% | Mean Cost: {eval_metrics.get('mean_cost', 0.0):.2f} | 1-Off Acc: {eval_metrics.get('acc_tolerance_1off', 0.0):.2f}%")
        return

    # 4. Setup Optimizer & Loss Function
    # Collect trainable parameters into structured param groups with differential learning rates:
    trainable_params = []
    if list_lora_layers:
        lora_params = get_lora_parameters(clip_model)
        if len(lora_params) > 0:
            trainable_params.append({'params': lora_params, 'lr': args.lr, 'weight_decay': 1e-2})

    # Standard CoOp context vectors (if present)
    ctx = getattr(prompt_learner, 'ctx', None)
    if ctx is not None:
        trainable_params.append({'params': [ctx], 'lr': args.lr, 'weight_decay': 0.0})

    # Class-specific residual tokens (High LR, no weight decay)
    class_deltas = getattr(prompt_learner, 'class_deltas', None)
    if class_deltas is not None:
        delta_params = list(class_deltas.parameters())
        if len(delta_params) > 0:
            lr_class = getattr(args, 'lr_class', None) or (args.lr * 5.0)
            trainable_params.append({'params': delta_params, 'lr': lr_class, 'weight_decay': 0.0})

    # Template token residuals (Lower LR, gentle L2 decay)
    template_deltas = getattr(prompt_learner, 'template_deltas', None)
    if template_deltas is not None:
        lr_template = getattr(args, 'lr_template', None) or (args.lr * 0.5)
        trainable_params.append({'params': [template_deltas], 'lr': lr_template, 'weight_decay': 1e-4})

    # Class full embeddings (if mode == 'full')
    class_embeds = getattr(prompt_learner, 'class_embeds', None)
    if class_embeds is not None:
        embed_params = list(class_embeds.parameters())
        if len(embed_params) > 0:
            trainable_params.append({'params': embed_params, 'lr': args.lr, 'weight_decay': 0.0})

    # Feature-space residual parameter (args.lr_prompt, default 1e-4)
    feat_deltas = getattr(prompt_learner, 'feature_deltas', None)
    if feat_deltas is not None:
        lr_prompt = getattr(args, 'lr_prompt', 1e-4)
        trainable_params.append({'params': [feat_deltas], 'lr': lr_prompt, 'weight_decay': 1e-2})
        Logger.info(f"Param Group: FeatureResidual deltas with lr={lr_prompt}")

    total_iters = args.n_iters * args.shots
    optimizer = torch.optim.AdamW(trainable_params, betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    # Staged Prompt Tuning (CSC freezing after csc_iters; defaults to full n_iters)
    csc_iters_arg = getattr(args, 'csc_iters', None)
    if csc_iters_arg is None:
        csc_iters_arg = getattr(args, 'n_iters_csc', None)
    if csc_iters_arg is None:
        csc_iters_arg = args.n_iters

    if csc_iters_arg is not None and csc_iters_arg < args.n_iters:
        total_csc_iters = csc_iters_arg * args.shots
    else:
        total_csc_iters = total_iters

    prompt_frozen = False
    if total_csc_iters <= 0:
        for p in prompt_learner.parameters():
            p.requires_grad = False
            p.grad = None
        prompt_frozen = True
        Logger.info("CSC Prompt Learner is frozen from start (0 iters). Only LoRA will train.")
    elif total_csc_iters < total_iters:
        Logger.info(f"Staged Training: CSC Prompt Learner will train for {total_csc_iters} iterations ({csc_iters_arg} iters/shot), then freeze for the remaining {total_iters - total_csc_iters} iterations (LoRA only).")

    criterion = CompositeCriterion(
        base_loss_type=args.base_loss,
        use_ordinal=args.use_ordinal,
        ordinal_type=args.ordinal_loss_type,
        cost_matrix=cost_matrix,
        lambda_ord=args.lambda_ord,
        use_promptsrc=args.use_promptsrc,
        lambda_src=args.lambda_src,
        src_temp=args.src_temp,
        use_kgcoop=getattr(args, 'use_kgcoop', False) or (args.method == 'mllm_feat_lora'),
        lambda_kg=getattr(args, 'lambda_kg', 2.0)
    )

    # Pre-compute fixed zero-shot text features for PromptSRC anchor if needed
    zs_text_features_norm = zs_textual_features.t().float().cuda() # [K, D]

    history_loss = []
    history_acc = []
    count_iters = 0
    start_train_time = time.time()

    Logger.step(f"Starting Training: Total Iters = {total_iters}, Base Loss = {args.base_loss}, Ordinal Loss = {args.use_ordinal} ({args.ordinal_loss_type}), PromptSRC = {args.use_promptsrc}...")

    # Reset train generator and global seeds to ensure identical epoch permutation ordering across all run methods
    if hasattr(train_loader, 'generator') and train_loader.generator is not None:
        train_loader.generator.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    while count_iters < total_iters:
        custom_model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.0

        for images, target in tqdm(train_loader):
            # Check for staged freezing of prompt learner
            if not prompt_frozen and count_iters >= total_csc_iters:
                for p in prompt_learner.parameters():
                    p.requires_grad = False
                    p.grad = None
                prompt_frozen = True
                Logger.step(f"Iter {count_iters}: Freezing CSC prompt learner parameters. Only LoRA modules will continue training for the remaining {total_iters - count_iters} iterations.")

            images, target = images.cuda(), target.cuda()

            # Zero-shot inference for PromptSRC regularization (frozen forward)
            zs_logits_batch = None
            zs_vis_feat_batch = None
            if args.use_promptsrc:
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        # Use unadapted base image features
                        zs_vis_feat_batch = clip_model.encode_image(images.type(clip_model.dtype))
                        zs_vis_feat_batch = zs_vis_feat_batch / zs_vis_feat_batch.norm(dim=-1, keepdim=True)
                        zs_logits_batch = (clip_model.logit_scale.exp()) * (zs_vis_feat_batch @ zs_text_features_norm.type(zs_vis_feat_batch.dtype).t())

            # Tuned Forward Pass
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits, image_features, text_features = custom_model(images)
                
                loss, loss_dict = criterion(
                    logits=logits,
                    targets=target,
                    image_features=image_features,
                    text_features=text_features,
                    logit_scale=clip_model.logit_scale,
                    zs_logits=zs_logits_batch,
                    zs_text_features=zs_text_features_norm,
                    zs_image_features=zs_vis_feat_batch
                )

            acc_train += cls_acc(logits, target) * target.shape[0]
            loss_epoch += loss.item() * target.shape[0]
            tot_samples += target.shape[0]

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()

            if scale_before <= scaler.get_scale():
                scheduler.step()

            count_iters += 1

            if count_iters == total_iters:
                break

        if tot_samples > 0:
            acc_train /= tot_samples
            loss_epoch /= tot_samples
            current_lr = scheduler.get_last_lr()[0]
            iters_left = total_iters - count_iters
            Logger.metric(f"Iters Left: {iters_left}, LR: {current_lr:.6f}, Train Acc: {acc_train:.2f}%, Train Loss: {loss_epoch:.4f}")
            history_loss.append(loss_epoch)
            history_acc.append(acc_train)

    train_time = time.time() - start_train_time
    Logger.info(f"Training completed in {train_time:.2f} seconds.")

    # Save final model checkpoint
    if args.save_path is not None:
        save_coop_checkpoint(args, list_lora_layers, prompt_learner, filename=args.filename)

    # 5. Final Evaluation & Visualizations
    Logger.step("Generating final evaluations and publication-quality visualizations...")
    plot_metrics(history_loss, history_acc, 'visualizations/training_metrics.png')

    # Train Split Visualizations
    acc_train_final = 0.0
    eval_loader_train = train_eval_loader if train_eval_loader is not None else train_loader
    if eval_loader_train is not None:
        acc_train_final, train_preds, train_targets, train_images, train_text_feats, train_probs, train_metrics = evaluate_coop(args, custom_model, eval_loader_train, dataset)
        train_indices = get_prediction_indices(len(train_images), first_n=10, random_n=10)
        plot_confusion_matrix(train_targets.numpy(), train_preds.numpy(), dataset.classnames, 'visualizations/final_train_confusion_matrix.png', split_name="Train")
        plot_predictions(train_images[train_indices], train_targets.numpy()[train_indices], train_preds.numpy()[train_indices], dataset.classnames, 'visualizations/final_train_predictions.png', probabilities=train_probs[train_indices], split_name="Train")
        
        if len(train_images) > 0:
            os.makedirs('visualizations/final_train_attention', exist_ok=True)
            num_vis = min(20, len(train_images))
            train_cam_maps = generate_eigencam_maps(clip_model, train_images[:num_vis])
            for idx in range(len(train_cam_maps)):
                pred_cls = dataset.classnames[train_preds[idx]]
                visualize_attention(train_images[idx], train_cam_maps[idx], f'visualizations/final_train_attention/img_{idx}.png', split_name="Train", class_name=pred_cls)

        trained_train_feats, trained_train_labs = pre_load_features(clip_model, eval_loader_train)
        plot_embeddings(trained_train_feats, train_text_feats.cpu(), trained_train_labs, dataset.classnames, 'visualizations/final_train_tsne.png', method='tsne', split_name="Train")
        plot_embeddings(trained_train_feats, train_text_feats.cpu(), trained_train_labs, dataset.classnames, 'visualizations/final_train_pca.png', method='pca', split_name="Train")

    # Test Split Visualizations
    acc_test, test_preds, test_targets, test_images, test_text_feats, test_probs, final_test_metrics = evaluate_coop(args, custom_model, test_loader, dataset)
    test_indices = get_prediction_indices(len(test_images), first_n=10, random_n=10)
    plot_confusion_matrix(test_targets.numpy(), test_preds.numpy(), dataset.classnames, 'visualizations/final_test_confusion_matrix.png', split_name="Test")
    plot_predictions(test_images[test_indices], test_targets.numpy()[test_indices], test_preds.numpy()[test_indices], dataset.classnames, 'visualizations/final_test_predictions.png', probabilities=test_probs[test_indices], split_name="Test")
    
    if len(test_images) > 0:
        os.makedirs('visualizations/final_test_attention', exist_ok=True)
        num_vis = min(20, len(test_images))
        test_cam_maps = generate_eigencam_maps(clip_model, test_images[:num_vis])
        for idx in range(len(test_cam_maps)):
            pred_cls = dataset.classnames[test_preds[idx]]
            visualize_attention(test_images[idx], test_cam_maps[idx], f'visualizations/final_test_attention/img_{idx}.png', split_name="Test", class_name=pred_cls)

    trained_test_feats, trained_test_labs = pre_load_features(clip_model, test_loader)
    plot_embeddings(trained_test_feats, test_text_feats.cpu(), trained_test_labs, dataset.classnames, 'visualizations/final_test_tsne.png', method='tsne', split_name="Test")
    plot_embeddings(trained_test_feats, test_text_feats.cpu(), trained_test_labs, dataset.classnames, 'visualizations/final_test_pca.png', method='pca', split_name="Test")

    # Save summary
    save_run_info(
        args, 
        zs_acc=zs_acc, 
        final_train_acc=acc_train_final, 
        final_test_acc=acc_test, 
        train_time=train_time,
        extra_metrics={
            "zero_shot_metrics": zs_metrics,
            "final_test_metrics": final_test_metrics
        }
    )

    Logger.success(f"Final Test Accuracy: {acc_test:.2f}% | Mean Cost: {final_test_metrics.get('mean_cost', 0.0):.2f} | 1-Off Tolerance Acc: {final_test_metrics.get('acc_tolerance_1off', 0.0):.2f}%")
    Logger.info("Visualizations and run summary saved to visualizations/ folder.")
    return
