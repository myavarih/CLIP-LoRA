#!/usr/bin/env python3
"""
download_kaggle_metrics.py
Extracts ONLY JSON, CSV, and summary evaluation metrics from specific Kaggle kernel versions
without keeping multi-hundred-megabyte checkpoint weights.

Usage examples:
    # 1. Download specific versions:
    python3 download_kaggle_metrics.py --kernel clip-lora --versions 21 22 23 24 25 --force

    # 2. Download from specific account without plots:
    python3 download_kaggle_metrics.py --owner emmwhy1 --kernel clip-lora --versions 18 19 20 21 22 23 --no-plots
"""

import os
import re
import sys
import time
import zlib
import json
import argparse
from kaggle.api.kaggle_api_extended import KaggleApi
import kagglesdk.kernels.types.kernels_api_service as svc

def parse_args():
    parser = argparse.ArgumentParser(description="Download only JSON and CSV metrics from Kaggle kernel versions.")
    parser.add_argument("--owner", type=str, default=None, help="Kaggle username/owner (defaults to logged-in user).")
    parser.add_argument("--kernel", type=str, default="clip-lora", help="Kernel slug name.")
    parser.add_argument("--versions", type=int, nargs="+", required=True, help="List of version numbers to download, e.g., --versions 21 22 23.")
    parser.add_argument("--output-dir", type=str, default="/home/emmwhy/Projects/CV_Lab/CLIP-LoRA/experiments_output", help="Directory to save extracted files.")
    parser.add_argument("--no-plots", action="store_true", help="Skip downloading summary PNG plots.")
    parser.add_argument("--force", "-f", action="store_true", default=True, help="Force overwrite existing local files (default: True).")
    parser.add_argument("--no-force", action="store_false", dest="force", help="Do not overwrite existing local files.")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retry attempts per version on network drops (default: 3).")
    return parser.parse_args()

def resolve_session_id(client, owner, kernel, version):
    """Resolves kernel session ID for a specific version number."""
    req = svc.ApiDownloadKernelOutputRequest()
    req.owner_slug = owner
    req.kernel_slug = kernel
    req.version_number = version
    req.file_path = "dummy"
    try:
        res = client.kernels.kernels_api_client.download_kernel_output(req)
        url = getattr(res, "url", "")
    except Exception as e:
        url = str(e)
    m = re.search(r"/kf/(\d+)/", url)
    if m:
        return int(m.group(1))
    return None

def extract_metrics_from_session(client, session_id, output_base, include_plots=True, force=True, max_retries=3):
    """Downloads kernel zip stream and decompresses only JSON/CSV/summary plots with real-time logging."""
    req = svc.ApiDownloadKernelOutputZipRequest()
    req.kernel_session_id = session_id
    
    data = b""
    for attempt in range(1, max_retries + 1):
        try:
            res = client.kernels.kernels_api_client.download_kernel_output_zip(req)
            data = b""
            if hasattr(res, "iter_content"):
                for chunk in res.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        data += chunk
            else:
                data = res.content
            if len(data) > 0:
                break
        except Exception as e:
            if attempt < max_retries and len(data) < 1024 * 1024:
                print(f"\n    [!] Network glitch on attempt {attempt}/{max_retries} ({e}). Retrying in 2s...", flush=True)
                time.sleep(2)
            else:
                # If we received partial data, still proceed to unpack what we got
                print(f"\n    [!] Stream closed ({e}). Unpacking {len(data) / (1024*1024):.1f} MB received data...", flush=True)
                break
    
    headers = [m.start() for m in re.finditer(b"PK\x03\x04", data)]
    extracted_files = []
    
    for h in headers:
        comp_method = int.from_bytes(data[h+8:h+10], "little")
        fn_len = int.from_bytes(data[h+26:h+28], "little")
        extra_len = int.from_bytes(data[h+28:h+30], "little")
        fn = data[h+30:h+30+fn_len].decode("utf-8", errors="ignore")
        payload_start = h + 30 + fn_len + extra_len
        
        # Filter strictly for metric files
        if fn.endswith("/"):
            continue
        is_json_or_csv = fn.endswith(".json") or fn.endswith(".csv") or fn.endswith(".log") or fn.endswith(".txt")
        is_summary_plot = include_plots and fn.endswith(".png") and any(
            k in fn for k in ["training_metrics", "confusion_matrix", "tsne", "pca"]
        )
        
        if not (is_json_or_csv or is_summary_plot):
            continue
        if "requirements.txt" in fn or "__pycache__" in fn or "split_zhou" in fn:
            continue
        
        clean_fn = fn
        if clean_fn.startswith("experiments_output/"):
            clean_fn = clean_fn[len("experiments_output/"):]
        
        dest_path = os.path.join(output_base, clean_fn)
        if os.path.exists(dest_path) and not force:
            print(f"    [-] Skipping (already exists): {clean_fn} -> {dest_path}", flush=True)
            continue
            
        try:
            if comp_method == 0:  # Stored (no compression)
                decompressed = data[payload_start:payload_start+1000000]
            elif comp_method == 8:  # Deflated
                d = zlib.decompressobj(-15)
                decompressed = d.decompress(data[payload_start:payload_start+50*1024*1024])
            else:
                continue
            
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, "wb") as out_f:
                out_f.write(decompressed)
            extracted_files.append(clean_fn)
            print(f"    [✓] Extracted: {clean_fn} -> {dest_path}", flush=True)
        except Exception:
            pass
    
    return extracted_files

def rebuild_all_shots_summary(output_base):
    """Rebuilds all_shots_summary.json in each experiment folder from experiment_info.json."""
    for item in sorted(os.listdir(output_base)):
        folder = os.path.join(output_base, item)
        if os.path.isdir(folder):
            all_infos = []
            for root, _, files in os.walk(folder):
                if "experiment_info.json" in files:
                    fp = os.path.join(root, "experiment_info.json")
                    try:
                        info = json.load(open(fp))
                        if isinstance(info, dict) and "shots" in info and "seed" in info and "final_test_acc" in info:
                            all_infos.append(info)
                    except Exception:
                        pass
            if all_infos:
                all_infos = sorted(all_infos, key=lambda x: (x.get("shots", 0), x.get("seed", 0)))
                out_summary = os.path.join(folder, "all_shots_summary.json")
                with open(out_summary, "w") as f:
                    json.dump(all_infos, f, indent=4)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    api = KaggleApi()
    api.authenticate()
    
    owner = args.owner if args.owner else api.get_config_value(api.CONFIG_NAME_USER)
    print(f"========================================================================")
    print(f" Extracting metrics from: {owner}/{args.kernel}")
    print(f" Target Versions        : {args.versions}")
    print(f" Force Overwrite        : {args.force}")
    print(f" Destination Directory  : {args.output_dir}")
    print(f"========================================================================\n")
    
    with api.build_kaggle_client() as client:
        for v in args.versions:
            print(f"[*] Resolving version {v}...", end=" ", flush=True)
            sid = resolve_session_id(client, owner, args.kernel, v)
            if not sid:
                print(f"FAILED (could not find session ID for v{v})")
                continue
            print(f"Session ID: {sid}.\n[*] Downloading & streaming version {v}...", flush=True)
            
            try:
                extracted = extract_metrics_from_session(
                    client, sid, args.output_dir,
                    include_plots=not args.no_plots,
                    force=args.force,
                    max_retries=args.max_retries
                )
                print(f"[*] Version {v} complete: {len(extracted)} files extracted.\n", flush=True)
            except Exception as e:
                print(f"    ERROR while downloading v{v}: {e}\n", flush=True)
    
    print("[*] Rebuilding summary matrices across all directories...")
    rebuild_all_shots_summary(args.output_dir)
    print("All tasks finished successfully!")

if __name__ == "__main__":
    main()
