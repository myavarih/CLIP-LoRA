import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import cv2

# Define the global beautiful style
def apply_beautiful_style():
    import matplotlib.pyplot as plt
    plt.style.use('fivethirtyeight')

# Apply it when module is imported
apply_beautiful_style()

def plot_confusion_matrix(y_true, y_pred, classes, save_path, split_name=""):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', cbar_kws={'shrink': 0.8}, 
                xticklabels=classes, yticklabels=classes, linewidths=0.5, linecolor='white')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    title = f'{split_name} Confusion Matrix' if split_name else 'Confusion Matrix'
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_predictions(images, y_true, y_pred, classes, save_path, num_images=None, split_name=""):
    if num_images is None:
        num_images = len(images)
    num_images = min(num_images, len(images))
    
    batch_size = 64
    num_batches = (num_images + batch_size - 1) // batch_size
    
    for b in range(num_batches):
        start_idx = b * batch_size
        end_idx = min((b + 1) * batch_size, num_images)
        batch_count = end_idx - start_idx
        
        cols = 8
        rows = (batch_count + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        axes = axes.flatten() if batch_count > 1 else [axes]
        
        for i in range(batch_count):
            global_idx = start_idx + i
            # Denormalize image for visualization
            img = images[global_idx].cpu().numpy().transpose(1, 2, 0)
            mean = np.array([0.48145466, 0.4578275, 0.40821073])
            std = np.array([0.26862954, 0.26130258, 0.27577711])
            img = std * img + mean
            img = np.clip(img, 0, 1)
            
            true_label = classes[y_true[global_idx]]
            pred_label = classes[y_pred[global_idx]]
            color = 'green' if true_label == pred_label else 'red'
            
            axes[i].imshow(img)
            axes[i].axis('off')
            prefix = f"[{split_name}] " if split_name else ""
            axes[i].set_title(f"{prefix}True: {true_label}\nPred: {pred_label}", color=color)
            
        for i in range(batch_count, len(axes)):
            axes[i].axis('off')
            
        plt.tight_layout()
        if num_batches > 1:
            base, ext = os.path.splitext(save_path)
            batch_save_path = f"{base}_part{b+1}{ext}"
        else:
            batch_save_path = save_path
        plt.savefig(batch_save_path)
        plt.close()

def plot_metrics(losses, accuracies, save_path):
    epochs = range(1, len(losses) + 1)
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='C0')
    ax1.plot(epochs, losses, color='C0', label='Train Loss', marker='o', markersize=8, alpha=0.8)
    ax1.tick_params(axis='y', labelcolor='C0')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy (%)', color='C1')
    ax2.plot(epochs, accuracies, color='C1', label='Train Accuracy', marker='s', markersize=8, alpha=0.8)
    ax2.tick_params(axis='y', labelcolor='C1')

    # Adding legends for both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right', frameon=True, fancybox=True, shadow=True)

    fig.tight_layout()
    plt.title('Training Metrics Over Time', pad=15)
    plt.savefig(save_path)
    plt.close()

def plot_embeddings(image_features, text_features, image_labels, classes, save_path, method='tsne', split_name=""):
    # Convert to numpy
    img_feats = image_features.cpu().numpy()
    txt_feats = text_features.cpu().numpy()
    labels = image_labels.cpu().numpy()
    
    # Combine features to project them into the same space
    all_feats = np.vstack([img_feats, txt_feats])
    
    if method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=min(30, len(all_feats)-1), random_state=42)
    else:
        reducer = PCA(n_components=2)
        
    reduced_feats = reducer.fit_transform(all_feats)
    
    reduced_img = reduced_feats[:len(img_feats)]
    reduced_txt = reduced_feats[len(img_feats):]
    
    plt.figure(figsize=(12, 10))
    # Use a vibrant palette
    palette = sns.color_palette("Set2", len(classes))
    
    # Plot image features
    for i in range(len(classes)):
        idx = (labels == i)
        plt.scatter(reduced_img[idx, 0], reduced_img[idx, 1], color=palette[i], alpha=0.6, label=f'Image: {classes[i]}', s=20)
        
    # Plot text features
    for i in range(len(classes)):
        plt.scatter(reduced_txt[i, 0], reduced_txt[i, 1], color=palette[i], marker='*', edgecolor='black', linewidth=1.5, s=400, label=f'Text: {classes[i]}', zorder=5)
        
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=1.2, frameon=True, shadow=True)
    title = f'Embedding Space Visualization ({method.upper()})'
    if split_name:
        title = f'[{split_name}] {title}'
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_attention(image, attention_map, save_path, split_name=""):
    # image: 3x224x224 tensor
    # attention_map: HxW (e.g., 7x7) tensor
    img = image.cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std = np.array([0.26862954, 0.26130258, 0.27577711])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    
    attn = attention_map.cpu().numpy()
    # Resize attention map to image size
    attn = cv2.resize(attn, (img.shape[1], img.shape[0]))
    # Normalize attention to 0-1
    attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
    
    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * attn), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1] # BGR to RGB
    
    cam = heatmap * 0.5 + img * 0.5
    cam = np.clip(cam, 0, 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    prefix = f"[{split_name}] " if split_name else ""
    
    axes[0].imshow(img)
    axes[0].set_title(f'{prefix}Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(attn, cmap='jet')
    axes[1].set_title(f'{prefix}Attention Map')
    axes[1].axis('off')
    
    axes[2].imshow(cam)
    axes[2].set_title(f'{prefix}Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
