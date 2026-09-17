import os
import json
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import clip
from tqdm import tqdm
from utils import cls_acc, clip_classifier, pre_load_features, Logger, save_run_info

from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, lora_state_dict, save_lora, load_lora
from loralib import layers as lora_layers
import time
from vis_utils import plot_metrics, plot_confusion_matrix, plot_predictions, plot_embeddings, visualize_attention
from pytorch_grad_cam import EigenCAM


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
    """
    Computes EigenCAM maps for the given batch of images (first max_samples) using pytorch-grad-cam.
    """
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


def evaluate_lora(args, clip_model, loader, dataset):
    clip_model.eval()
    
    with torch.no_grad():
        template = dataset.template[0] 
        texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
        text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)

    acc = 0.
    tot_samples = 0
    all_preds = []
    all_targets = []
    all_images = []
    all_probs = []
    
    logit_scale = clip_model.logit_scale.exp() if hasattr(clip_model, 'logit_scale') else 100.0
    
    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            images, target = images.cuda(), target.cuda()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                image_features = clip_model.encode_image(images)
                
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            cosine_similarity = image_features @ text_features.t()
            preds = cosine_similarity.argmax(dim=-1)
            probs = F.softmax(logit_scale * cosine_similarity, dim=-1)
            
            acc += cls_acc(cosine_similarity, target) * len(cosine_similarity)
            tot_samples += len(cosine_similarity)
            
            all_preds.append(preds.cpu())
            all_targets.append(target.cpu())
            all_images.append(images.cpu())
            all_probs.append(probs.cpu())
                
    acc /= tot_samples
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_images = torch.cat(all_images)
    all_probs = torch.cat(all_probs)
    
    return acc, all_preds, all_targets, all_images, text_features, all_probs


def run_lora(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader, train_eval_loader=None):
    clip_model = clip_model.float()
    VALIDATION = False
    
    # Textual features
    Logger.step("Getting textual features as CLIP's classifier.")
    textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Pre-load test features
    Logger.step("Loading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(clip_model, test_loader)
    
    test_features = test_features.cuda()
    test_labels = test_labels.cuda()
 
    # Zero-shot CLIP
    clip_logits = logit_scale * test_features @ textual_features
    zs_acc = cls_acc(clip_logits, test_labels)
    Logger.success("Zero-shot CLIP's test accuracy: {:.2f}%".format(zs_acc))
    test_features = test_features.cpu()
    test_labels = test_labels.cpu()
    torch.cuda.empty_cache()
    
    Logger.info("Generating Zero-shot visualizations...")
    os.makedirs('visualizations', exist_ok=True)
    
    # --- Zero-shot Train ---
    eval_loader_train = train_eval_loader if train_eval_loader is not None else train_loader
    if eval_loader_train is not None:
        zs_acc_train, zs_preds_train, zs_targets_train, zs_images_train, zs_text_features_train, zs_probs_train = evaluate_lora(args, clip_model, eval_loader_train, dataset)
        
        train_pred_indices = get_prediction_indices(len(zs_images_train), first_n=10, random_n=10)
        
        plot_confusion_matrix(zs_targets_train.numpy(), zs_preds_train.numpy(), dataset.classnames, 'visualizations/zs_train_confusion_matrix.png', split_name="Train")
        plot_predictions(zs_images_train[train_pred_indices], zs_targets_train.numpy()[train_pred_indices], zs_preds_train.numpy()[train_pred_indices], dataset.classnames, 'visualizations/zs_train_predictions.png', probabilities=zs_probs_train[train_pred_indices], split_name="Train")
        
        if len(zs_images_train) > 0:
            os.makedirs('visualizations/zs_train_attention', exist_ok=True)
            num_vis = min(20, len(zs_images_train))
            zs_cam_maps_train = generate_eigencam_maps(clip_model, zs_images_train[:num_vis])
            for idx in range(len(zs_cam_maps_train)):
                pred_cls = dataset.classnames[zs_preds_train[idx]]
                visualize_attention(zs_images_train[idx], zs_cam_maps_train[idx], f'visualizations/zs_train_attention/img_{idx}.png', split_name="Train", class_name=pred_cls)
                
        train_features, train_labels = pre_load_features(clip_model, eval_loader_train)
        plot_embeddings(train_features.cpu(), textual_features.t().cpu(), train_labels.cpu(), dataset.classnames, 'visualizations/zs_train_tsne.png', method='tsne', split_name="Train")
        plot_embeddings(train_features.cpu(), textual_features.t().cpu(), train_labels.cpu(), dataset.classnames, 'visualizations/zs_train_pca.png', method='pca', split_name="Train")

    # --- Zero-shot Test ---
    zs_acc_full, zs_preds, zs_targets, zs_images, zs_text_features, zs_probs = evaluate_lora(args, clip_model, test_loader, dataset)
    
    test_pred_indices = get_prediction_indices(len(zs_images), first_n=10, random_n=10)
    
    plot_confusion_matrix(zs_targets.numpy(), zs_preds.numpy(), dataset.classnames, 'visualizations/zs_test_confusion_matrix.png', split_name="Test")
    plot_predictions(zs_images[test_pred_indices], zs_targets.numpy()[test_pred_indices], zs_preds.numpy()[test_pred_indices], dataset.classnames, 'visualizations/zs_test_predictions.png', probabilities=zs_probs[test_pred_indices], split_name="Test")
    
    if len(zs_images) > 0:
        os.makedirs('visualizations/zs_test_attention', exist_ok=True)
        num_vis = min(20, len(zs_images))
        zs_cam_maps = generate_eigencam_maps(clip_model, zs_images[:num_vis])
        for idx in range(len(zs_cam_maps)):
            pred_cls = dataset.classnames[zs_preds[idx]]
            visualize_attention(zs_images[idx], zs_cam_maps[idx], f'visualizations/zs_test_attention/img_{idx}.png', split_name="Test", class_name=pred_cls)
        
    plot_embeddings(test_features.cpu(), textual_features.t().cpu(), test_labels.cpu(), dataset.classnames, 'visualizations/zs_test_tsne.png', method='tsne', split_name="Test")
    plot_embeddings(test_features.cpu(), textual_features.t().cpu(), test_labels.cpu(), dataset.classnames, 'visualizations/zs_test_pca.png', method='pca', split_name="Test")
    
    test_features = test_features.cpu()
    test_labels = test_labels.cpu()
    
    list_lora_layers = apply_lora(args, clip_model)
    clip_model = clip_model.cuda() 
    
    if args.eval_only:
        load_lora(args, list_lora_layers)
        
        start_eval = time.time()
        acc_test, test_preds, test_targets, test_images, text_features, test_probs = evaluate_lora(args, clip_model, test_loader, dataset)
        eval_time = time.time() - start_eval
        
        Logger.success("Test accuracy: {:.2f}% (Time: {:.2f}s)".format(acc_test, eval_time))
        
        # Plot evaluation results
        os.makedirs('visualizations', exist_ok=True)
        test_pred_indices = get_prediction_indices(len(test_images), first_n=10, random_n=10)
        plot_confusion_matrix(test_targets.numpy(), test_preds.numpy(), dataset.classnames, 'visualizations/test_confusion_matrix.png', split_name="Test")
        plot_predictions(test_images[test_pred_indices], test_targets.numpy()[test_pred_indices], test_preds.numpy()[test_pred_indices], dataset.classnames, 'visualizations/test_predictions.png', probabilities=test_probs[test_pred_indices], split_name="Test")
        
        if len(test_images) > 0:
            os.makedirs('visualizations/test_attention', exist_ok=True)
            num_vis = min(20, len(test_images))
            eval_cam_maps = generate_eigencam_maps(clip_model, test_images[:num_vis])
            for idx in range(len(eval_cam_maps)):
                pred_cls = dataset.classnames[test_preds[idx]]
                visualize_attention(test_images[idx], eval_cam_maps[idx], f'visualizations/test_attention/img_{idx}.png', split_name="Test", class_name=pred_cls)
                
        # We need test_features for embeddings, pre_load them
        test_features, _ = pre_load_features(clip_model, test_loader)
        plot_embeddings(test_features.cpu(), text_features.cpu(), test_labels.cpu(), dataset.classnames, 'visualizations/test_tsne.png', method='tsne', split_name="Test")
        plot_embeddings(test_features.cpu(), text_features.cpu(), test_labels.cpu(), dataset.classnames, 'visualizations/test_pca.png', method='pca', split_name="Test")
        
        Logger.info("All visualizations saved in the 'visualizations/' folder.")
        return

    mark_only_lora_as_trainable(clip_model)
    total_iters = args.n_iters * args.shots
    
    optimizer = torch.optim.AdamW(get_lora_parameters(clip_model), weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)
    
    # Tracking for visualizations
    history_loss = []
    history_acc = []
    
    # training LoRA
    scaler = torch.amp.GradScaler('cuda')
    count_iters = 0
    
    start_train_time = time.time()
    # Reset train generator and global seeds to ensure identical epoch permutation ordering across all run methods
    if hasattr(train_loader, 'generator') and train_loader.generator is not None:
        train_loader.generator.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    while count_iters < total_iters:
        clip_model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.
        if args.encoder == 'vision': 
            text_features = textual_features.t().half()
        for i, (images, target) in enumerate(tqdm(train_loader)):
            
            template = dataset.template[0]
            texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
            images, target = images.cuda(), target.cuda()
            if args.encoder == 'text' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    texts = clip.tokenize(texts).cuda()
                    class_embeddings = clip_model.encode_text(texts)
                text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)
                
            if args.encoder == 'vision' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = clip_model.encode_image(images)
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        image_features = clip_model.encode_image(images)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            
            cosine_similarity = logit_scale * image_features @ text_features.t()
            loss = F.cross_entropy(cosine_similarity, target)
            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
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
            Logger.metric('Iters Left: {}, LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}'.format(iters_left, current_lr, acc_train, loss_epoch))
            history_loss.append(loss_epoch)
            history_acc.append(acc_train)
            
        # Eval
        if VALIDATION:
            clip_model.eval()
            acc_val = evaluate_lora(args, clip_model, val_loader, dataset)[0]
            Logger.success("Val accuracy: {:.2f}%".format(acc_val))
        
    train_time = time.time() - start_train_time
    Logger.info(f"Training completed in {train_time:.2f} seconds.")
    
    # Save final model checkpoint
    if args.save_path is not None:
        save_lora(args, list_lora_layers, filename=args.filename)

    # Final Evaluation & Visualization
    Logger.step("Running final evaluation and generating plots...")
    os.makedirs('visualizations', exist_ok=True)
    
    # 1. Loss and Accuracy curve
    plot_metrics(history_loss, history_acc, 'visualizations/training_metrics.png')
    
    # 2. Final Evaluation (Train)
    acc_train_final = 0.0
    eval_loader_train = train_eval_loader if train_eval_loader is not None else train_loader
    if eval_loader_train is not None:
        Logger.info("Evaluating on Train set...")
        acc_train_final, train_preds, train_targets, final_train_images, final_train_text_features, train_probs = evaluate_lora(args, clip_model, eval_loader_train, dataset)
        Logger.success("Final Train accuracy: {:.2f}%".format(acc_train_final))
        
        train_pred_indices = get_prediction_indices(len(final_train_images), first_n=10, random_n=10)
        
        plot_confusion_matrix(train_targets.numpy(), train_preds.numpy(), dataset.classnames, 'visualizations/final_train_confusion_matrix.png', split_name="Train")
        plot_predictions(final_train_images[train_pred_indices], train_targets.numpy()[train_pred_indices], train_preds.numpy()[train_pred_indices], dataset.classnames, 'visualizations/final_train_predictions.png', probabilities=train_probs[train_pred_indices], split_name="Train")
        
        if len(final_train_images) > 0:
            os.makedirs('visualizations/final_train_attention', exist_ok=True)
            num_vis = min(20, len(final_train_images))
            train_cam_maps = generate_eigencam_maps(clip_model, final_train_images[:num_vis])
            for idx in range(len(train_cam_maps)):
                pred_cls = dataset.classnames[train_preds[idx]]
                visualize_attention(final_train_images[idx], train_cam_maps[idx], f'visualizations/final_train_attention/img_{idx}.png', split_name="Train", class_name=pred_cls)
                
        trained_train_features, trained_train_labels = pre_load_features(clip_model, eval_loader_train)
        plot_embeddings(trained_train_features, final_train_text_features.cpu(), trained_train_labels, dataset.classnames, 'visualizations/final_train_tsne.png', method='tsne', split_name="Train")
        plot_embeddings(trained_train_features, final_train_text_features.cpu(), trained_train_labels, dataset.classnames, 'visualizations/final_train_pca.png', method='pca', split_name="Train")

    # 3. Final Evaluation (Test)
    Logger.info("Evaluating on Test set...")
    acc_test, test_preds, test_targets, test_images, final_text_features, test_probs = evaluate_lora(args, clip_model, test_loader, dataset)
    Logger.success("Final Test accuracy: {:.2f}%".format(acc_test))
    
    test_pred_indices = get_prediction_indices(len(test_images), first_n=10, random_n=10)
    
    plot_confusion_matrix(test_targets.numpy(), test_preds.numpy(), dataset.classnames, 'visualizations/final_test_confusion_matrix.png', split_name="Test")
    plot_predictions(test_images[test_pred_indices], test_targets.numpy()[test_pred_indices], test_preds.numpy()[test_pred_indices], dataset.classnames, 'visualizations/final_test_predictions.png', probabilities=test_probs[test_pred_indices], split_name="Test")
    
    if len(test_images) > 0:
        os.makedirs('visualizations/final_test_attention', exist_ok=True)
        num_vis = min(20, len(test_images))
        test_cam_maps = generate_eigencam_maps(clip_model, test_images[:num_vis])
        for idx in range(len(test_cam_maps)):
            pred_cls = dataset.classnames[test_preds[idx]]
            visualize_attention(test_images[idx], test_cam_maps[idx], f'visualizations/final_test_attention/img_{idx}.png', split_name="Test", class_name=pred_cls)
    
    trained_test_features, trained_test_labels = pre_load_features(clip_model, test_loader)
    plot_embeddings(trained_test_features, final_text_features.cpu(), trained_test_labels, dataset.classnames, 'visualizations/final_test_tsne.png', method='tsne', split_name="Test")
    plot_embeddings(trained_test_features, final_text_features.cpu(), trained_test_labels, dataset.classnames, 'visualizations/final_test_pca.png', method='pca', split_name="Test")
    
    save_run_info(args, zs_acc, acc_train_final, acc_test, train_time)
    
    Logger.info("All visualizations and run summary saved in the 'visualizations/' folder.")
    return
            
    
            
