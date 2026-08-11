from tqdm import tqdm
import torch
import clip

def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = correct[: topk].reshape(-1).float().sum().item()
    acc = 100 * acc / target.shape[0]
    
    return acc


def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)
        clip_weights = torch.stack(clip_weights, dim=1).cuda()
        
    return clip_weights


def pre_load_features(clip_model, loader):
    features, labels = [], []
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(loader)):
            images, target = images.cuda(), target.cuda()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
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

def save_run_info(args, zs_acc, final_train_acc, final_test_acc, train_time):
    import os, json, datetime
    os.makedirs('visualizations', exist_ok=True)
    info = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "arguments": vars(args),
        "results": {
            "zero_shot_test_acc": zs_acc,
            "final_train_acc": final_train_acc,
            "final_test_acc": final_test_acc,
            "training_time_seconds": train_time
        }
    }
    with open('visualizations/run_summary.json', 'w') as f:
        json.dump(info, f, indent=4)
