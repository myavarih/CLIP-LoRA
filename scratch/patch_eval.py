import os

lora_path = "scratch/my_impl_eval/lora.py"
vis_path = "scratch/my_impl_eval/vis_utils.py"
imgnet_path = "scratch/my_impl_eval/datasets/imagenet.py"

# 1. Patch imagenet.py
with open(imgnet_path, "r") as f:
    img_code = f.read()

img_code = img_code.replace(
    "os.path.join(os.path.join(self.dataset_dir, 'train'))", "root"
).replace(
    "os.path.join(os.path.join(self.dataset_dir, 'val'))", "root"
)
with open(imgnet_path, "w") as f:
    f.write(img_code)

# 2. Patch lora.py
with open(lora_path, "r") as f:
    lora_code = f.read()

lora_code = lora_code.replace(
    "all_images.append(images.cpu())",
    "if len(all_images) < 5: all_images.append(images.cpu())"
).replace(
    "def generate_eigencam_maps(clip_model, images_tensor, max_samples=20):",
    "def generate_eigencam_maps(clip_model, images_tensor, max_samples=20):\n    return []"
)
with open(lora_path, "w") as f:
    f.write(lora_code)

# 3. Patch vis_utils.py
with open(vis_path, "r") as f:
    vis_code = f.read()

funcs = [
    "def plot_confusion_matrix",
    "def plot_predictions",
    "def plot_embeddings",
    "def visualize_attention"
]
for func in funcs:
    vis_code = vis_code.replace(
        func + "(",
        func + "(*args, **kwargs):\n    return\ndef _old_" + func.replace("def ", "") + "("
    )
with open(vis_path, "w") as f:
    f.write(vis_code)
