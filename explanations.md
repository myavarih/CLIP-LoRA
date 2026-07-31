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
