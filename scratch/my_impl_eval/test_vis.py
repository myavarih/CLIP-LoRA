import os
import torch
import numpy as np
import sys

# Make sure we can import vis_utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from vis_utils import plot_confusion_matrix, plot_predictions, plot_metrics, plot_embeddings, visualize_attention

# Path to artifact directory
out_dir = "/home/emmwhy/.gemini/antigravity-ide/brain/242f80bc-372a-441a-b253-56bd14112ed8"
os.makedirs(out_dir, exist_ok=True)

# 1. Confusion Matrix
classes = ['cat', 'dog', 'car', 'tree']
y_true = np.random.randint(0, 4, 100)
y_pred = y_true.copy()
# Add some noise for confusion
noise_idx = np.random.choice(100, 20, replace=False)
y_pred[noise_idx] = np.random.randint(0, 4, 20)
plot_confusion_matrix(y_true, y_pred, classes, os.path.join(out_dir, 'test_confusion_matrix.png'), split_name="Mock")

# 2. Predictions
images = torch.rand(16, 3, 224, 224)
# Denormalize expects values that will be clamped, so rand is fine
plot_predictions(images, y_true[:16], y_pred[:16], classes, os.path.join(out_dir, 'test_predictions.png'), split_name="Mock")

# 3. Metrics
losses = [0.8, 0.6, 0.4, 0.3, 0.25, 0.22, 0.2, 0.18, 0.17, 0.15]
accuracies = [50, 65, 75, 82, 85, 88, 90, 91, 92, 93]
plot_metrics(losses, accuracies, os.path.join(out_dir, 'test_metrics.png'))

# 4. Embeddings
image_features = torch.randn(100, 512)
text_features = torch.randn(4, 512)
image_labels = torch.tensor(y_true)
plot_embeddings(image_features, text_features, image_labels, classes, os.path.join(out_dir, 'test_tsne.png'), method='tsne', split_name="Mock")
plot_embeddings(image_features, text_features, image_labels, classes, os.path.join(out_dir, 'test_pca.png'), method='pca', split_name="Mock")

# 5. Attention
image = torch.rand(3, 224, 224)
attention_map = torch.rand(7, 7)
visualize_attention(image, attention_map, os.path.join(out_dir, 'test_attention.png'), split_name="Mock")

print("All test visualizations generated.")
