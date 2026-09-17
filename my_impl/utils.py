from tqdm import tqdm
import numpy as np
import torch
import clip

def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = correct[: topk].reshape(-1).float().sum().item()
    acc = 100 * acc / target.shape[0]
    
    return acc


def clip_classifier(classnames, template, clip_model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            # Tokenize the prompts
            clean_name = classname.replace('_', ' ')
            if isinstance(template, dict):
                # Look up in dictionary by raw or clean name
                texts = template.get(classname, template.get(clean_name, None))
                if texts is None:
                    texts = [f"a photo of a {clean_name}"]
            elif isinstance(template, (list, tuple)):
                texts = [t.format(clean_name) for t in template]
            else:
                texts = [str(template).format(clean_name)]

            texts = clip.tokenize(texts, truncate=True).to(device)
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)
        clip_weights = torch.stack(clip_weights, dim=1).to(device)
        
    return clip_weights



def pre_load_features(clip_model, loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    features, labels = [], []
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(loader)):
            images, target = images.to(device), target.to(device)
            if device == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = clip_model.encode_image(images)
            else:
                image_features = clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            features.append(image_features.float().cpu())
            labels.append(target.cpu())
        features, labels = torch.cat(features), torch.cat(labels)
    return features, labels


class Logger:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def info(msg):
        print(f"{Logger.OKCYAN}ℹ️ {msg}{Logger.ENDC}")

    @staticmethod
    def success(msg):
        print(f"\n{Logger.OKGREEN}{Logger.BOLD}✅ {msg}{Logger.ENDC}\n")
        
    @staticmethod
    def step(msg):
        print(f"\n{Logger.OKBLUE}{Logger.BOLD}🚀 {msg}{Logger.ENDC}")
        
    @staticmethod
    def metric(msg):
        print(f"{Logger.OKCYAN}📊 {msg}{Logger.ENDC}")

def compute_ordinal_metrics(preds, targets, cost_matrix=None):
    """
    Computes exact accuracy, mean cost penalty, MAE, MSE, and 1-off tolerance accuracy.
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    exact_acc = 100.0 * np.mean(preds == targets)
    mae = float(np.mean(np.abs(preds - targets)))
    mse = float(np.mean((preds - targets) ** 2))
    
    metrics = {
        "acc": exact_acc,
        "mae": mae,
        "mse": mse
    }

    if cost_matrix is not None:
        if isinstance(cost_matrix, torch.Tensor):
            cost_matrix = cost_matrix.cpu().numpy()
        sample_costs = cost_matrix[targets, preds]
        mean_cost = float(np.mean(sample_costs))
        # Tolerance accuracy: prediction is exact or within mild cost <= 1.0
        acc_tol = 100.0 * float(np.mean(sample_costs <= 1.0))
        metrics["mean_cost"] = mean_cost
        metrics["acc_tolerance_1off"] = acc_tol
    else:
        # Default 1-off accuracy
        acc_1off = 100.0 * np.mean(np.abs(preds - targets) <= 1)
        metrics["acc_tolerance_1off"] = float(acc_1off)

    return metrics


def save_run_info(args, zs_acc, final_train_acc, final_test_acc, train_time, extra_metrics=None):
    import os, json, datetime
    os.makedirs('visualizations', exist_ok=True)
    results_dict = {
        "zero_shot_test_acc": zs_acc,
        "final_train_acc": final_train_acc,
        "final_test_acc": final_test_acc,
        "training_time_seconds": train_time
    }
    if extra_metrics is not None:
        results_dict.update(extra_metrics)

    info = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "arguments": vars(args),
        "results": results_dict
    }
    with open('visualizations/run_summary.json', 'w') as f:
        json.dump(info, f, indent=4)

