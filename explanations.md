# Project Implementation Explanations

Here is a summary of the changes we've made to get the CLIP-LoRA setup running properly with the new dataset:

1. **Adding the Walnut dataset loader**
   - We needed to support the custom Walnut dataset: so we duplicated the repository into a separate `my_impl` directory to keep the original untouched. Then, we created a new dataset class in `my_impl/datasets/walnut.py` designed to parse the few-shot directories, and registered this new loader into `my_impl/datasets/__init__.py`.

2. **Fixing the CUDA Out Of Memory (OOM) error**
   - We encountered a GPU memory crash during the evaluation setup because the original `main.py` hardcoded a massive batch size of 256 for the validation and test datasets: so we changed those dataloaders to use the dynamically configurable `args.batch_size` (which defaults to 32), keeping memory usage safely within the limits of your 1.64 GiB GPU.

3. **Fixing the TypeError during accuracy calculation**
   - We encountered a crash `TypeError: only 0-dimensional arrays can be converted to Python scalars` during the zero-shot accuracy calculation in `utils.py`. This happened because newer versions of NumPy don't allow casting a 1-item array directly to a Python float using `float()`: so we fixed it by swapping out the clunky NumPy conversion for PyTorch's native `.item()` method, safely extracting the exact scalar value directly from the tensor.

4. **Fixing CUDA Out Of Memory (OOM) during training**
   - We encountered a GPU memory crash during the training phase because the default batch size of 32 was still too large for the available 1.64 GiB VRAM: so we lowered the default `--batch_size` argument in `run_utils.py` to `4` to ensure it can comfortably fit in memory. Note that this may result in noisier gradients and impact performance; if the accuracy drops, we can implement gradient accumulation as a future fix.

5. **Fixing PyTorch AMP and Scheduler Warnings**
   - We encountered deprecation and step-order warnings from PyTorch during training: so we updated the `GradScaler` initialization in `lora.py` to use the modern `torch.amp.GradScaler('cuda')` API, and wrapped the `scheduler.step()` call in a scale check to ensure it doesn't prematurely step when the optimizer skips due to inf/nan gradients.

6. **Restoring Original Batch Size**
   - We needed to restore the original batch size to match the original implementation: so we changed the default `--batch_size` argument in `run_utils.py` back to `32`.

7. **Styling and Visualization Upgrades**
   - We wanted to make the matplotlib plots more beautiful: so we adopted the FiveThirtyEight default style in `vis_utils.py`, switched to a classic sans-serif font, removed bolding from titles, adjusted color palettes, and added star markers for text features in embedding spaces.

8. **Fixing the NoneType crash in `--eval_only` mode**
   - We encountered a TypeError `NoneType object is not iterable` because `main.py` explicitly skips creating the `train_loader` to save time in eval-only mode, but `lora.py` was still trying to iterate over it: so we wrapped the train set evaluation blocks in `if train_loader is not None:` and fixed the `eval_only` logic to properly extract attention maps.

9. **Fixing Attention Weight Extraction in LoRA layers**
   - We encountered an issue where the final evaluation crashed because the attention weights were `None`: so we modified the custom `PlainMultiheadAttentionLoRA` forward method in `loralib/layers.py` to manually compute and return the Softmax attention weights when `need_weights=True` was passed.

10. **Fixing Float/Half precision mismatch in pre-loading features**
    - We encountered a `RuntimeError: mat1 and mat2 must have the same dtype` crash when trying to run `--eval_only` because the LoRA linear layers default to `Float32` but the CLIP inputs are `Float16`: so we added a `torch.amp.autocast` block inside the `pre_load_features` function in `utils.py` to handle the dtype conversion safely.

11. **Optimizing Initialization Time**
    - We noticed that the feature pre-loading step was taking a long time because it was pointlessly iterating over the validation set which was never used: so we removed the `val_features` extraction block from `lora.py` to speed up the startup sequence.

12. **Replacing Custom Tokenizer with Hugging Face Fast Rust Tokenizer**
    - We needed to accelerate text tokenization and eliminate the legacy slow pure-Python BPE implementation: so we updated `my_impl/clip/clip.py` to use `CLIPTokenizerFast` from `transformers` (achieving a ~23x speedup with exact numerical parity), removed `my_impl/clip/simple_tokenizer.py` and the 1.3 MB `my_impl/clip/bpe_simple_vocab_16e6.txt.gz` file, and registered `tokenizers` and `transformers` in `my_impl/requirements.txt`.

13. **Fixing Undefined BICUBIC Interpolation Constant**
    - We encountered a `NameError: name 'BICUBIC' is not defined` when setting up image preprocessing transforms in `my_impl/clip/clip.py` because the constant definition was omitted during tokenizer refactoring: so we restored the standard CLIP interpolation mode definition with `InterpolationMode.BICUBIC` from torchvision and a fallback to `Image.BICUBIC`.

14. **Customizing Attention Maps and Prediction Visualizations**
    - We needed to visualize the first 20 images for attention maps, display a mixture of 10 first + 10 random images for predictions, and eliminate redundant `[Test]` / `[Train]` bracket prefixes from subplot titles: so we implemented `get_prediction_indices` in `my_impl/lora.py`, configured attention map loops to iterate across `range(min(20, ...))`, and cleaned subplot titles in `my_impl/vis_utils.py`.

15. **Implementing Class Activation Maps (CAM) for Visual Explanations**
    - We encountered an issue where raw ViT `[CLS]` self-attention heatmaps suffered from attention sink artifacts (bright hotspots on empty background corners rather than the object of interest): so we implemented Class Activation Maps (CAM) by extracting dense spatial patch embeddings from `VisionTransformer`, projecting them to multimodal space, computing spatial cosine similarity against the predicted class text prompt embeddings, and rendering normalized CAM overlays in `my_impl/vis_utils.py`.

16. **Displaying Predictive Probabilities and Formatting Prediction Grid**
    - We needed to display prediction confidence probabilities alongside true/predicted class labels without text overlap: so we updated `evaluate_lora` in `my_impl/lora.py` to calculate softmax probabilities over logit-scaled cosine similarities, and refactored `plot_predictions` in `my_impl/vis_utils.py` to format percentage confidences, adopt a 5-column grid layout, and increase padding/figure sizing to prevent text clipping and overlapping.

17. **Switching to EigenCAM for Visual Explainability**
    - We encountered limitations and background sensitivity with spatial cosine CAM and raw attention maps: so we integrated `EigenCAM` from `pytorch-grad-cam` into `my_impl/lora.py` and `my_impl/vis_utils.py`, projecting the first principal component of post-norm 2D patch features at the final VisionTransformer block to accurately highlight the semantic subject in both Walnut and natural datasets while cleanly removing all legacy attention extraction hacks.

18. **Robust Path Resolution for Walnut Dataset Loader**
    - We encountered a `FileNotFoundError` when passing the direct dataset folder as `--root_path` because `Walnut` appended `Walnut_Color_Parvizi_3` unconditionally: so we updated `my_impl/datasets/walnut.py` to check if the root path already points to the dataset folder before joining subpaths.

19. **VRAM Optimization for EigenCAM on Low-Memory GPUs**
    - We encountered a CUDA Out Of Memory error during EigenCAM generation on the 1.64 GiB GPU because full evaluation batches and pre-loaded features remained allocated in VRAM: so we moved pre-loaded features to CPU immediately after zero-shot evaluation and chunked EigenCAM generation into mini-batches of 4.

20. **Fixing Syntax Error, Final Epoch Metric Logging, and Embedding Plot Label Alignment**
    - We encountered a syntax error in `get_prediction_indices` due to an accidental inline snippet paste, a metric loss logging omission on the final epoch, and scrambled labels in the post-training embedding plots when training with shuffled loaders: so we cleaned the accidental paste in [lora.py](file:///home/emmwhy/Projects/CV_Lab/CLIP-LoRA/my_impl/lora.py), updated metric recording to capture the final epoch when `tot_samples > 0`, and bound the true labels directly from `pre_load_features` into `plot_embeddings`.

21. **Automated Multi-Shot Experiment Suite and Resumption for Walnut Dataset on Kaggle**
    - We needed to run sequential experiments across shots 1 to 32 with a batch size of 32 and fixed seeds, saving checkpoints and copying visualizations to experiment folders while resuming seamlessly after previous runs: so we implemented `my_impl/run_experiments.py` with checkpoint and summary existence checking, per-shot output and visualization archiving, consolidated CSV/JSON metrics reporting, added missing visualization dependencies to `my_impl/requirements.txt`, and packed the clean release into `kaggle_upload.zip`.

22. **Fixing Half and Float Dtype Mismatch During Post-Training CAM Visualization**
    - We encountered a `RuntimeError: mat1 and mat2 must have the same dtype, but got Half and Float` during post-training visual explanations on CUDA because OpenAI CLIP's visual encoder operates in FP16 (Half) while LoRA projection weights and adapters initialized as standard FP32 (Float), causing matrix multiplications to crash during EigenCAM evaluation passes that ran outside of AMP autocast: so we updated `LinearLoRA`, `MergedLinear`, and the Conv LoRA layers in [layers.py](file:///home/emmwhy/Projects/CV_Lab/CLIP-LoRA/my_impl/loralib/layers.py) to dynamically match the input tensor's dtype to the layer's weights (and restore the original tensor dtype upon return), aligned device/dtype in `PlainMultiheadAttentionLoRA` during layer initialization, wrapped `CLIPVisionWrapper` and `generate_eigencam_maps` in [lora.py](file:///home/emmwhy/Projects/CV_Lab/CLIP-LoRA/my_impl/lora.py) with `torch.amp.autocast(device_type="cuda", dtype=torch.float16)`, and refreshed `kaggle_upload.zip`.

23. **Fixing ValueError: Attempting to unscale FP16 gradients**
    - We encountered a `ValueError: Attempting to unscale FP16 gradients.` error during training when `GradScaler.step(optimizer)` executed because OpenAI's `clip.load()` leaves weights in `float16` when loaded on CUDA, causing LoRA parameter matrices and optimizer gradients to be created as `torch.float16` which PyTorch's `GradScaler` rejects: so we converted `clip_model` to full precision using `clip_model = clip_model.float()` in [main.py](file:///home/emmwhy/Projects/CV_Lab/CLIP-LoRA/my_impl/main.py) and [lora.py](file:///home/emmwhy/Projects/CV_Lab/CLIP-LoRA/my_impl/lora.py) right after loading, ensuring LoRA parameters are stored and accumulated in `float32` master weights while forward computations run seamlessly in mixed-precision via `torch.amp.autocast(device_type="cuda", dtype=torch.float16)`.


\nwe needed to save a checkpoint every 100 iter and at the end test with all of them and report all accuracies but generate vis for the best one: so modified utils.py to allow custom filenames in save/load_lora, and modified lora.py to save every 100 iterations, evaluate all of them on the test set, and keep the best checkpoint for final visualizations.

24. **Fixing Missing JSON Import**
    - We encountered a `NameError: name 'json' is not defined` error when trying to dump the checkpoint accuracies: so we added `import json` to the top of `my_impl/lora.py`.
\nwe encountered a bug where all checkpoints evaluated to the identical accuracy because loralib permanently merged the first loaded weights when entering eval() mode: fixed by putting the model back into train() mode to unmerge weights before loading the next checkpoint. Also we needed to prefer the later checkpoint if accuracies were tied: so changed the > condition to >=.
\nwe needed to test multiple ablation configurations in a suite: so created a new run_ablation_suite.py that systematically tests different LoRA hyperparameters like rank, layers, injection matrices, and encoders
