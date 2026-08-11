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

def plot_predictions(images, y_true, y_pred, classes, save_path, probabilities=None, num_images=None, split_name=""):
    if num_images is None:
        num_images = len(images)
    num_images = min(num_images, len(images))
    if num_images == 0:
        return
    
    # Clean grid layout: 5 columns with ample room to prevent text overlap
    cols = 5 if num_images >= 5 else max(1, num_images)
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(3.8 * cols, 4.2 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    else:
        axes = np.array(axes).flatten()
    
    for i in range(num_images):
        img = images[i].float().cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        true_idx = int(y_true[i])
        pred_idx = int(y_pred[i])
        true_label = classes[true_idx]
        pred_label = classes[pred_idx]
        is_correct = (true_idx == pred_idx)
        color = '#1b7a2b' if is_correct else '#c0292b'
        
        axes[i].imshow(img)
        axes[i].axis('off')
        
        if probabilities is not None and len(probabilities) > i:
            probs = probabilities[i]
            pred_conf = float(probs[pred_idx]) * 100
            if is_correct:
                title_text = f"True: {true_label}\nPred: {pred_label} ({pred_conf:.1f}%)"
            else:
                true_conf = float(probs[true_idx]) * 100
                title_text = f"True: {true_label} ({true_conf:.1f}%)\nPred: {pred_label} ({pred_conf:.1f}%)"
        else:
            title_text = f"True: {true_label}\nPred: {pred_label}"
            
        axes[i].set_title(title_text, color=color, fontsize=10, fontweight='semibold', pad=8)
        
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout(pad=2.0)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
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
    title = f'{split_name} Embedding Space Visualization ({method.upper()})' if split_name else f'Embedding Space Visualization ({method.upper()})'
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_attention(image, attention_map, save_path, split_name="", class_name=""):
    # image: 3x224x224 tensor
    # attention_map: HxW (e.g., 14x14 or 224x224) tensor or numpy array
    if torch.is_tensor(image):
        img = image.float().cpu().numpy().transpose(1, 2, 0)
    else:
        img = np.array(image, dtype=np.float32)
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
            
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    img = std * img + mean
    img = np.clip(img, 0, 1)
    
    if torch.is_tensor(attention_map):
        attn = attention_map.float().cpu().numpy()
    else:
        attn = np.array(attention_map, dtype=np.float32)
        
    # Resize attention/CAM map to image size if needed
    if attn.shape != (img.shape[0], img.shape[1]):
        attn = cv2.resize(attn, (img.shape[1], img.shape[0]))
        
    # Normalize CAM to 0-1
    denom = attn.max() - attn.min()
    if denom > 1e-8:
        attn = (attn - attn.min()) / denom
    else:
        attn = np.zeros_like(attn)
    
    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * attn), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1] # BGR to RGB
    
    cam = heatmap * 0.5 + img * 0.5
    cam = np.clip(cam, 0, 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(attn, cmap='jet')
    axes[1].set_title('EigenCAM Map', fontsize=12)
    axes[1].axis('off')
    
    axes[2].imshow(cam)
    axes[2].set_title('EigenCAM Overlay', fontsize=12)
    axes[2].axis('off')
    
    title_parts = []
    if split_name:
        title_parts.append(f"[{split_name}]")
    if class_name:
        title_parts.append(f"Predicted Class: {class_name}")
    if title_parts:
        plt.suptitle(" ".join(title_parts), fontsize=14, y=1.02)
        
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
