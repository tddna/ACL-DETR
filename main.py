import argparse
import datetime
import json
import random
import time
from pathlib import Path
import copy
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from datasets.incremental import generate_cls_order
from engine import evaluate, train_one_epoch, train_one_epoch_incremental
from models import build_model
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch.distributed as dist
from engine import get_src_permutation_idx_public
from torch.utils.data import TensorDataset 
 
from util.report import save_evaluation_results

from torch.utils.tensorboard import SummaryWriter


torch.cuda.empty_cache()

import os
import torch

import torch
import os
import glob

def merge_epoch_cache(epoch, cache_dir="cache", world_size=None):
    os.makedirs(cache_dir, exist_ok=True)
    
    # 查找所有rank文件
    pattern = os.path.join(cache_dir, f"epoch_{epoch}_rank*.pth")
    rank_files = glob.glob(pattern)
    
    if not rank_files:
        raise FileNotFoundError(f"没有找到epoch {epoch}的缓存文件")
        
    # 合并所有rank文件
    all_features, all_targets = [], []
    for f in rank_files:
        d = torch.load(f, map_location="cpu")
        all_features.append(d["features"])
        all_targets.append(d["targets"])
    
    # 保存合并文件
    final_path = os.path.join(cache_dir, f"epoch_{epoch}_merged.pth")
    torch.save({
        "features": torch.cat(all_features, dim=0),
        "targets": torch.cat(all_targets, dim=0)
    }, final_path)
    
    # 删除原始rank文件
    for f in rank_files:
        os.remove(f)
        
    print(f"[✓] 合并完成epoch {epoch}缓存 -> {final_path}")




def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_balanced', default=10, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')


    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    parser.add_argument('--ref_cls_loss_coef', default=2, type=float)
    parser.add_argument('--ref_bbox_loss_coef', default=5, type=float)
    parser.add_argument('--ref_giou_loss_coef', default=2, type=float)
    parser.add_argument('--ref_loss_overall_coef', default=1, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default='./data/coco', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='./output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=64, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    # 在pycocotools.py中修改
    # 20 + 
    parser.add_argument('--num_of_phases', default=5, type=int)
    parser.add_argument('--cls_per_phase', default=10, type=int)
    parser.add_argument('--data_setting', default='tfh', choices=['tfs', 'tfh'])

    parser.add_argument('--seed_cls', default=123, type=int)
    parser.add_argument('--seed_data', default=123, type=int)
    parser.add_argument('--method', default='icarl', choices=['baseline', 'icarl'])
    parser.add_argument('--mem_rate', default=0.1, type=float)

    parser.add_argument('--debug_mode', default = False, action='store_true')
    parser.add_argument('--balanced_ft', default=False, action='store_true')
    parser.add_argument('--total_num_classes', default=91, type=int)
    
    parser.add_argument('--use_acil', default=False, action='store_true')
    parser.add_argument('--acil_cache', default=True, action='store_true')
    parser.add_argument('--buffer_size', default= 2048, type=int)
    parser.add_argument('--gamma', default=1e-3, type=float)
    parser.add_argument('--acil_device', default='cpu', type=str)
    # parser.add_argument('--base_epochs', default=1, type=int)
    # parser.add_argument('--realign_epochs', default=3, type=int)
    return parser

def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    
    writer = SummaryWriter(log_dir="./logs")


    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model

    cls_order = generate_cls_order(seed=args.seed_cls)   

    if args.data_setting=='tfs':
        total_phase_num = args.num_of_phases
    elif args.data_setting=='tfh':
        total_phase_num = args.num_of_phases
    else:
        raise ValueError('Please set the correct data setting.')
    
    print('pytorch model distributed...')
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    for phase_idx in range(total_phase_num):
        print('training phase '+ str(phase_idx) + '...')
        
        dataset_train = build_dataset(image_set='train', args=args, cls_order=cls_order, \
            phase_idx=phase_idx, incremental=True, incremental_val=False, val_each_phase=False)
        print(f"len(dataset_train): {len(dataset_train)}")
        dataset_val = build_dataset(image_set='val', args=args, cls_order=cls_order, \
            phase_idx=phase_idx, incremental=True, incremental_val=True, val_each_phase=False)

        if phase_idx >= 1:
            dataset_train_balanced = build_dataset(image_set='train', args=args, cls_order=cls_order, \
                phase_idx=phase_idx, incremental=True, incremental_val=False, val_each_phase=False, balanced_ft=True)
            dataset_val_old = build_dataset(image_set='val', args=args, cls_order=cls_order, \
                phase_idx=phase_idx-1, incremental=True, incremental_val=True, val_each_phase=False)
            dataset_val_new = build_dataset(image_set='val', args=args, cls_order=cls_order, \
                phase_idx=phase_idx, incremental=True, incremental_val=True, val_each_phase=True)

        # 分布式训练
        if args.distributed:
            if args.cache_mode:
                sampler_train = samplers.NodeDistributedSampler(dataset_train)
                sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
                if phase_idx >= 1:
                    sampler_train_balanced = samplers.NodeDistributedSampler(dataset_train_balanced)
                    sampler_val_old = samplers.NodeDistributedSampler(dataset_val_old, shuffle=False)
                    sampler_val_new = samplers.NodeDistributedSampler(dataset_val_new, shuffle=False)
            else:
                sampler_train = samplers.DistributedSampler(dataset_train)
                sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
                if phase_idx >= 1:
                    sampler_train_balanced = samplers.DistributedSampler(dataset_train_balanced)
                    sampler_val_old = samplers.DistributedSampler(dataset_val_old, shuffle=False)
                    sampler_val_new = samplers.DistributedSampler(dataset_val_new, shuffle=False)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            if phase_idx >= 1:
                sampler_train_balanced = torch.utils.data.RandomSampler(dataset_train_balanced)
                sampler_val_old = torch.utils.data.SequentialSampler(dataset_val_old)
                sampler_val_new = torch.utils.data.SequentialSampler(dataset_val_new)


        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, args.batch_size, drop_last=True)

        if phase_idx >= 1:        
            batch_sampler_train_balanced = torch.utils.data.BatchSampler(
                sampler_train_balanced, args.batch_size, drop_last=True)

        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                    collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                    pin_memory=True)
        data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                    drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                    pin_memory=True)
        
        if phase_idx >= 1:
            data_loader_train_balanced = DataLoader(dataset_train_balanced, batch_sampler=batch_sampler_train_balanced, collate_fn=utils.collate_fn, num_workers=args.num_workers, pin_memory=True)
            data_loader_val_old = DataLoader(dataset_val_old, args.batch_size, sampler=sampler_val_old, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers, pin_memory=True)
            data_loader_val_new = DataLoader(dataset_val_new, args.batch_size, sampler=sampler_val_new, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers, pin_memory=True)

        ## b.  匹配参数
        def match_name_keywords(n, name_keywords):
            out = False
            for b in name_keywords:
                if b in n:
                    out = True
                    break
            return out

        # for n, p in model_without_ddp.named_parameters():
        #     print(n)

        # 参数分组
        param_dicts = [
            {
                "params":
                    [p for n, p in model_without_ddp.named_parameters()
                    if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
                "lr": args.lr,
            },
            {
                "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
                "lr": args.lr_backbone,
            },
            {
                "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
                "lr": args.lr * args.lr_linear_proj_mult,
            }
        ]

        print('setting the optimizer...')
        
        if args.sgd:
            optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                        weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                        weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

        if phase_idx >= 1:
            if args.sgd:
                optimizer_balanced = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                            weight_decay=args.weight_decay)
            else:
                optimizer_balanced = torch.optim.AdamW(param_dicts, lr=args.lr,
                                            weight_decay=args.weight_decay)
            lr_scheduler_balanced = torch.optim.lr_scheduler.StepLR(optimizer_balanced, args.lr_drop_balanced)


        ## e. 评估coco api构建
        base_ds = get_coco_api_from_dataset(dataset_val)
        if phase_idx >= 1:
            base_ds_old = get_coco_api_from_dataset(dataset_val_old)
            base_ds_new = get_coco_api_from_dataset(dataset_val_new)

        # ## f. 加载预训练模型
        # if args.frozen_weights is not None:
        #     checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        #     model_without_ddp.detr.load_state_dict(checkpoint['model'])

        ## g. 输出目录
        this_phase_output_dir = args.output_dir + '/phase_'+str(phase_idx)
        output_dir = Path(this_phase_output_dir)
        Path(this_phase_output_dir).mkdir(parents=True, exist_ok=True)

        ## h. 训练
        print("start training")
        start_time = time.time()
        epoch = 0

        ### h1.base training e.g 在40上训练
        if phase_idx==0:
        #################################################################################################
            ckpt_path = './phase_0.pth'          
            checkpoint = torch.load(ckpt_path, map_location='cpu')

            missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
            unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
            if len(missing_keys) > 0:
                print('Missing Keys: {}'.format(missing_keys))
            if len(unexpected_keys) > 0:
                print('Unexpected Keys: {}'.format(unexpected_keys))
            
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            # print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1
        #################################################################################################
            # base training,前面部分应该更换为deformable detr的预训练模型
            # for epoch in range(args.epochs):
            #     if args.distributed:
            #         sampler_train.set_epoch(epoch)
            #     if epoch == args.epochs - 1:
            #         criterion.modify_acil_cache()
            #     train_stats = train_one_epoch(
            #         model, criterion,  data_loader_train, optimizer, device, epoch, args.clip_max_norm)
            #     dist.barrier()  # 所有进程等待 rank 0 合并完成

            # # 保存criterion
            # criterion.save_acil_cache(f'{args.output_dir}/acil_cache_{args.epochs}.pth')
            
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
            )
            print("Testing results for phase_0_base.")   
            save_evaluation_results(args.output_dir, phase_idx, test_stats, coco_evaluator, suffix='_base')
            torch.cuda.empty_cache()
            
            model_without_ddp.modify_acl_mode(True)
            # 加载acil cache
            if args.acil_cache:
                cache = torch.load(f'{args.output_dir}/acil_cache_{args.epochs}.pth')
                hs = torch.cat([f for f, _ in cache], dim=0)
                labels = torch.cat([l for _, l in cache], dim=0)
            else:
                hs,labels = criterion.get_acil_cache()
            dataset = TensorDataset(hs,labels)
            # for epoch in range(args.base_epochs):
            # for epoch in range(args.realign_epochs):
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=True
            )
            dataloader = DataLoader(dataset, batch_size=4, sampler=sampler)
            
            model.module.acl_fit(dataloader)
                
            print("acl_fit done")
                
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
            )
            print("Testing results for phase_0.")   
            save_evaluation_results(args.output_dir, phase_idx, test_stats, coco_evaluator, suffix=f'{args.buffer_size}_{args.gamma}')   
            
            model_path = output_dir / f'phase_{phase_idx}.pth'
            torch.save(model_without_ddp.state_dict(), model_path)
        
        else:
            state_dict = torch.load(f'{args.output_dir}/phase_{phase_idx-1}/phase_{phase_idx-1}.pth')
            model = model.module
            model.load_state_dict(state_dict)
            model.to(device)
            
            
            if args.distributed:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
                model_without_ddp = model.module            
            # for epoch in range(0, args.epochs):
            if args.distributed:
                sampler_train.set_epoch(epoch)
            train_stats = train_one_epoch_incremental(
                model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm)

            lr_scheduler.step()

            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
            )
            print("Testing results for all.")
            
            save_evaluation_results(args.output_dir, phase_idx, test_stats, coco_evaluator, suffix='_all')
            
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val_old, base_ds_old, device, args.output_dir
            )
            print("Testing results for old.")  
            save_evaluation_results(args.output_dir, phase_idx, test_stats, coco_evaluator, suffix='_old')

            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val_new, base_ds_new, device, args.output_dir
            )
            print("Testing results for new.")   
            save_evaluation_results(args.output_dir, phase_idx, test_stats, coco_evaluator, suffix='_new')
    
            if args.balanced_ft :
                for epoch in range(0, 1):
                    if args.distributed:
                        sampler_train_balanced.set_epoch(epoch)

                    train_stats = train_one_epoch(
                        model, criterion, data_loader_train_balanced, optimizer_balanced, device, epoch, args.clip_max_norm)
                    lr_scheduler_balanced.step()

                    if phase_idx >= 1:
                        test_stats, coco_evaluator = evaluate(
                            model, criterion, postprocessors, data_loader_val_old, base_ds_old, device, args.output_dir
                        )
                        print("Balanced FT - Testing results for old.")                    
                        test_stats, coco_evaluator = evaluate(
                            model, criterion, postprocessors, data_loader_val_new, base_ds_new, device, args.output_dir
                        )
                        print("Balanced FT - Testing results for new.")   
                        
                    test_stats, coco_evaluator = evaluate(model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir)
                    print("Balanced FT - Testing results for all.")                           
                save_evaluation_results(args.output_dir, phase_idx, test_stats, coco_evaluator, suffix='_F-all')
            
            model_path = output_dir / f'phase_{phase_idx}.pth'
            torch.save(model_without_ddp.state_dict(), model_path)

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('Training time {}'.format(total_time_str))

            # 在主训练循环中
            optimizer.zero_grad()
            torch.cuda.synchronize()
            dist.barrier()  # 添加额外同步点

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.debug_mode:
        args.epochs = 1
        args.cls_per_phase = 1
        args.batch_size = 2     

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
