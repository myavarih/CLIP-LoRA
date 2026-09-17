import torch
import torchvision.transforms as transforms
import clip
from datasets import build_dataset
from datasets.utils import build_data_loader

import os
import shutil
from utils import Logger
from run_utils import get_arguments, set_random_seed
from lora import run_lora
from train_coop import run_coop_lora


def main():

    # Load config file
    args = get_arguments()
    
    if args.clear_vis:
        if os.path.exists('visualizations'):
            Logger.info("Clearing old visualizations directory...")
            shutil.rmtree('visualizations')
    
    set_random_seed(args.seed)
    
    # CLIP
    clip_model, preprocess = clip.load(args.backbone, resize_mode=args.resize_mode)
    clip_model = clip_model.float()
    clip_model.eval()
    logit_scale = 100

    # Prepare dataset
    Logger.step("Preparing dataset.")
        
    dataset = build_dataset(args.dataset, args.root_path, args.shots, preprocess)
    
    if args.dataset == 'imagenet':
        val_loader = torch.utils.data.DataLoader(dataset.val, batch_size=args.batch_size, num_workers=8, shuffle=False, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=args.batch_size, num_workers=8, shuffle=False, pin_memory=True)
    else:
        val_loader = build_data_loader(data_source=dataset.val, batch_size=args.batch_size, is_train=False, tfm=preprocess, shuffle=False,  num_workers=8)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=args.batch_size, is_train=False, tfm=preprocess, shuffle=False,  num_workers=8)
        
    train_loader = None
    if not args.eval_only:
        if args.resize_mode == 'crop':
            train_tranform = transforms.Compose([
                transforms.RandomResizedCrop(size=224, scale=(0.08, 1), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            ])
        else:
            aug_list = [transforms.Lambda(lambda img: img.convert('RGB'))]
            if args.resize_mode == 'pad':
                aug_list.append(clip.SquarePad())
                aug_list.append(transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC))
            elif args.resize_mode == 'direct':
                aug_list.append(transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC))

            # Horizontal flip
            aug_list.append(transforms.RandomHorizontalFlip(p=0.5))

            # Vertical flip (natural for agricultural/unoriented objects, auto-skipped for cars)
            if getattr(args, 'vflip', True) and args.dataset not in ['stanford_cars', 'cars']:
                aug_list.append(transforms.RandomVerticalFlip(p=0.5))

            # Slight translation movement (~5px, no cropping)
            shift_px = getattr(args, 'shift_px', 5)
            if shift_px > 0:
                max_trans = shift_px / 224.0
                aug_list.append(transforms.RandomAffine(
                    degrees=0,
                    translate=(max_trans, max_trans),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    fill=0
                ))

            aug_list.append(transforms.ToTensor())

            # Subtle Gaussian noise
            noise_std = getattr(args, 'noise_std', 0.015)
            if noise_std > 0:
                aug_list.append(clip.AddGaussianNoise(std=noise_std, p=0.5))

            # Salt and Pepper impulse noise
            sp_amount = getattr(args, 'sp_noise_amount', 0.005)
            if sp_amount > 0:
                aug_list.append(clip.AddSaltAndPepperNoise(amount=sp_amount, p=0.5))

            aug_list.append(transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)))
            train_tranform = transforms.Compose(aug_list)
        
        train_generator = torch.Generator()
        train_generator.manual_seed(args.seed)

        if args.dataset == 'imagenet':
            train_loader = torch.utils.data.DataLoader(dataset.train_x, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True, generator=train_generator)
            train_eval_loader = torch.utils.data.DataLoader(dataset.train_x, batch_size=args.batch_size, num_workers=8, shuffle=False, pin_memory=True)
        else:
            train_loader = build_data_loader(data_source=dataset.train_x, batch_size=args.batch_size, tfm=train_tranform, is_train=True, shuffle=True, num_workers=8, generator=train_generator)
            train_eval_loader = build_data_loader(data_source=dataset.train_x, batch_size=args.batch_size, is_train=False, tfm=preprocess, shuffle=False, num_workers=8)

    if args.method in ['coop_lora', 'coop_only', 'rt_lora', 'csc_lora', 'coop_csc', 'res_cls_lora', 'plain_lora_res_cls', 'mllm_feat_lora']:
        run_coop_lora(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader, train_eval_loader=train_eval_loader)
    else:
        run_lora(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader, train_eval_loader=train_eval_loader)

if __name__ == '__main__':
    main()