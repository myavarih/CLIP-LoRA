import os
import torch
import numpy as np
import torch.nn.functional as F

from utils import *

from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, lora_state_dict, save_lora, load_lora
from loralib import layers as lora_layers
import time
from vis_utils import plot_metrics, plot_confusion_matrix, plot_predictions, plot_embeddings, visualize_attention

def evaluate_lora(args, clip_model, loader, dataset, extract_attention=False):
    clip_model.eval()
    
    # Enable attention saving on the last vision transformer block if requested
    if extract_attention and hasattr(clip_model, 'visual') and hasattr(clip_model.visual, 'transformer'):
        clip_model.visual.transformer.resblocks[-1].save_attention = True
        
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
    attention_maps = []
    
    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            images, target = images.cuda(), target.cuda()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                image_features = clip_model.encode_image(images)
                
            if extract_attention and hasattr(clip_model, 'visual') and hasattr(clip_model.visual, 'transformer'):
                # Extract weights (batch_size, seq_len, seq_len)
                attn = clip_model.visual.transformer.resblocks[-1].last_attn_weights
                # Get CLS token attention to other patches (batch_size, 1, 1+num_patches)
                cls_attn = attn[:, 0, 1:] 
                # Assuming ViT-B/32 or similar where patches form a square grid
                grid_size = int(np.sqrt(cls_attn.shape[-1]))
                cls_attn = cls_attn.view(-1, grid_size, grid_size)
                attention_maps.append(cls_attn.cpu())
                    
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            cosine_similarity = image_features @ text_features.t()
            acc += cls_acc(cosine_similarity, target) * len(cosine_similarity)
            tot_samples += len(cosine_similarity)
            
            all_preds.append(cosine_similarity.argmax(dim=-1).cpu())
            all_targets.append(target.cpu())
            all_images.append(images.cpu())
                
    acc /= tot_samples
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_images = torch.cat(all_images)
    
    if extract_attention and len(attention_maps) > 0:
        attention_maps = torch.cat(attention_maps)
        # Disable attention saving
        clip_model.visual.transformer.resblocks[-1].save_attention = False
        return acc, all_preds, all_targets, all_images, text_features, attention_maps

    return acc, all_preds, all_targets, all_images, text_features


def run_lora(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader):
    
    VALIDATION = False
    
    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(clip_model, val_loader)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(clip_model, test_loader)
    
    test_features = test_features.cuda()
    test_labels = test_labels.cuda()
 
    # Zero-shot CLIP
    clip_logits = logit_scale * test_features @ textual_features
    zs_acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(zs_acc))
    
    print("Generating Zero-shot visualizations...")
    os.makedirs('visualizations', exist_ok=True)
    
    # --- Zero-shot Train ---
    zs_acc_train, zs_preds_train, zs_targets_train, zs_images_train, zs_text_features_train, zs_attn_maps_train = evaluate_lora(args, clip_model, train_loader, dataset, extract_attention=True)
    
    plot_confusion_matrix(zs_targets_train.numpy(), zs_preds_train.numpy(), dataset.classnames, 'visualizations/zs_train_confusion_matrix.png', split_name="Train")
    plot_predictions(zs_images_train, zs_targets_train.numpy(), zs_preds_train.numpy(), dataset.classnames, 'visualizations/zs_train_predictions.png', split_name="Train", num_images=100)
    
    if len(zs_attn_maps_train) > 0 and len(zs_images_train) > 0:
        os.makedirs('visualizations/zs_train_attention', exist_ok=True)
        for idx in range(min(len(zs_images_train), 100)):
            visualize_attention(zs_images_train[idx], zs_attn_maps_train[idx], f'visualizations/zs_train_attention/img_{idx}.png', split_name="Train")
            
    train_features, train_labels = pre_load_features(clip_model, train_loader)
    plot_embeddings(train_features.cpu(), textual_features.t().cpu(), train_labels.cpu(), dataset.classnames, 'visualizations/zs_train_tsne.png', method='tsne', split_name="Train")
    plot_embeddings(train_features.cpu(), textual_features.t().cpu(), train_labels.cpu(), dataset.classnames, 'visualizations/zs_train_pca.png', method='pca', split_name="Train")

    # --- Zero-shot Test ---
    zs_acc_full, zs_preds, zs_targets, zs_images, zs_text_features, zs_attn_maps = evaluate_lora(args, clip_model, test_loader, dataset, extract_attention=True)
    
    plot_confusion_matrix(zs_targets.numpy(), zs_preds.numpy(), dataset.classnames, 'visualizations/zs_test_confusion_matrix.png', split_name="Test")
    plot_predictions(zs_images, zs_targets.numpy(), zs_preds.numpy(), dataset.classnames, 'visualizations/zs_test_predictions.png', split_name="Test", num_images=100)
    
    if len(zs_attn_maps) > 0 and len(zs_images) > 0:
        os.makedirs('visualizations/zs_test_attention', exist_ok=True)
        for idx in range(min(len(zs_images), 100)):
            visualize_attention(zs_images[idx], zs_attn_maps[idx], f'visualizations/zs_test_attention/img_{idx}.png', split_name="Test")
        
    plot_embeddings(test_features.cpu(), textual_features.t().cpu(), test_labels.cpu(), dataset.classnames, 'visualizations/zs_test_tsne.png', method='tsne', split_name="Test")
    plot_embeddings(test_features.cpu(), textual_features.t().cpu(), test_labels.cpu(), dataset.classnames, 'visualizations/zs_test_pca.png', method='pca', split_name="Test")
    
    test_features = test_features.cpu()
    test_labels = test_labels.cpu()
    
    
    list_lora_layers = apply_lora(args, clip_model)
    clip_model = clip_model.cuda() 
    
    if args.eval_only:
        load_lora(args, list_lora_layers)
        
        start_eval = time.time()
        acc_test, test_preds, test_targets, test_images, text_features = evaluate_lora(args, clip_model, test_loader, dataset)
        eval_time = time.time() - start_eval
        
        print("**** Test accuracy: {:.2f} (Time: {:.2f}s). ****\n".format(acc_test, eval_time))
        
        # Plot evaluation results
        os.makedirs('visualizations', exist_ok=True)
        plot_confusion_matrix(test_targets.numpy(), test_preds.numpy(), dataset.classnames, 'visualizations/test_confusion_matrix.png')
        plot_predictions(test_images, test_targets.numpy(), test_preds.numpy(), dataset.classnames, 'visualizations/test_predictions.png')
        plot_embeddings(test_features, text_features, test_labels, dataset.classnames, 'visualizations/test_tsne.png', method='tsne')
        return

    mark_only_lora_as_trainable(clip_model)
    total_iters = args.n_iters * args.shots
    
    optimizer = torch.optim.AdamW(get_lora_parameters(clip_model), weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)
    
    best_acc_val, best_acc_test = 0., 0.
    best_epoch_val = 0
    
    best_epoch_val = 0
    
    # Tracking for visualizations
    history_loss = []
    history_acc = []
    
    # training LoRA
    scaler = torch.amp.GradScaler('cuda')
    count_iters = 0
    finish = False
    
    start_train_time = time.time()
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
            
        if count_iters < total_iters:
            acc_train /= tot_samples
            loss_epoch /= tot_samples
            current_lr = scheduler.get_last_lr()[0]
            print('LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}'.format(current_lr, acc_train, loss_epoch))
            history_loss.append(loss_epoch)
            history_acc.append(acc_train)
            
        # Eval
        if VALIDATION:
            clip_model.eval()
            acc_val = evaluate_lora(args, clip_model, val_loader, dataset)[0]
            print("**** Val accuracy: {:.2f}. ****\n".format(acc_val))
        
    
    acc_test = evaluate_lora(args, clip_model, test_loader, dataset)[0]
    print("**** Final test accuracy: {:.2f}. ****\n".format(acc_test))
    train_time = time.time() - start_train_time
    print(f"Training completed in {train_time:.2f} seconds.")
    
    if args.save_path != None:
        save_lora(args, list_lora_layers)

    # Final Evaluation & Visualization
    print("\nRunning final evaluation and generating plots...")
    os.makedirs('visualizations', exist_ok=True)
    
    # 1. Loss and Accuracy curve
    plot_metrics(history_loss, history_acc, 'visualizations/training_metrics.png')
    
    # 2. Final Evaluation (Train)
    print("Evaluating on Train set...")
    acc_train_final, train_preds, train_targets, final_train_images, final_train_text_features, attention_maps_train = evaluate_lora(args, clip_model, train_loader, dataset, extract_attention=True)
    print("**** Final Train accuracy: {:.2f}. ****\n".format(acc_train_final))
    
    plot_confusion_matrix(train_targets.numpy(), train_preds.numpy(), dataset.classnames, 'visualizations/final_train_confusion_matrix.png', split_name="Train")
    plot_predictions(final_train_images, train_targets.numpy(), train_preds.numpy(), dataset.classnames, 'visualizations/final_train_predictions.png', split_name="Train", num_images=100)
    
    if len(attention_maps_train) > 0 and len(final_train_images) > 0:
        os.makedirs('visualizations/final_train_attention', exist_ok=True)
        for idx in range(min(len(final_train_images), 100)):
            visualize_attention(final_train_images[idx], attention_maps_train[idx], f'visualizations/final_train_attention/img_{idx}.png', split_name="Train")
            
    trained_train_features, _ = pre_load_features(clip_model, train_loader)
    plot_embeddings(trained_train_features, final_train_text_features.cpu(), train_targets, dataset.classnames, 'visualizations/final_train_tsne.png', method='tsne', split_name="Train")
    plot_embeddings(trained_train_features, final_train_text_features.cpu(), train_targets, dataset.classnames, 'visualizations/final_train_pca.png', method='pca', split_name="Train")

    # 3. Final Evaluation (Test)
    print("Evaluating on Test set...")
    acc_test, test_preds, test_targets, test_images, final_text_features, attention_maps = evaluate_lora(args, clip_model, test_loader, dataset, extract_attention=True)
    print("**** Final Test accuracy: {:.2f}. ****\n".format(acc_test))
    
    plot_confusion_matrix(test_targets.numpy(), test_preds.numpy(), dataset.classnames, 'visualizations/final_test_confusion_matrix.png', split_name="Test")
    plot_predictions(test_images, test_targets.numpy(), test_preds.numpy(), dataset.classnames, 'visualizations/final_test_predictions.png', split_name="Test", num_images=100)
    
    if len(attention_maps) > 0 and len(test_images) > 0:
        os.makedirs('visualizations/final_test_attention', exist_ok=True)
        for idx in range(min(len(test_images), 100)):
            visualize_attention(test_images[idx], attention_maps[idx], f'visualizations/final_test_attention/img_{idx}.png', split_name="Test")
    
    trained_test_features, _ = pre_load_features(clip_model, test_loader)
    plot_embeddings(trained_test_features, final_text_features.cpu(), test_targets, dataset.classnames, 'visualizations/final_test_tsne.png', method='tsne', split_name="Test")
    plot_embeddings(trained_test_features, final_text_features.cpu(), test_targets, dataset.classnames, 'visualizations/final_test_pca.png', method='pca', split_name="Test")
    
    print("All visualizations saved in the 'visualizations/' folder.")
    return
            
    
            
