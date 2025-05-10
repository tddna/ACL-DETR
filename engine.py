"""
Train and eval functions used in main.py
"""
# 标准库导入
import math
import os
import sys
from pathlib import Path
from typing import Iterable

# PyTorch相关导入
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset

# 自定义工具和评估器
import util.misc as utils
from util import box_ops
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher

# 模型相关导入
from models import build_model
from models.ACIL import get_src_permutation_idx_public

# 其他工具
import tqdm

def save_eval_results(coco_evaluator, result_txt):
    with open(result_txt, 'w') as f:
        if coco_evaluator is not None:
            for iou_type in coco_evaluator.coco_eval:
                stats = coco_evaluator.coco_eval[iou_type].stats
                f.write(f"IoU metric: {iou_type}\n")
                f.write(f" Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {stats[0]:.3f}\n")
                f.write(f" Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {stats[1]:.3f}\n")
                f.write(f" Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {stats[2]:.3f}\n")
                f.write(f" Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {stats[3]:.3f}\n")
                f.write(f" Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {stats[4]:.3f}\n")
                f.write(f" Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {stats[5]:.3f}\n")
                f.write(f" Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = {stats[6]:.3f}\n")
                f.write(f" Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = {stats[7]:.3f}\n")
                f.write(f" Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {stats[8]:.3f}\n")
                f.write(f" Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {stats[9]:.3f}\n")
                f.write(f" Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {stats[10]:.3f}\n")
                f.write(f" Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {stats[11]:.3f}\n")
                
def train_base(args,train_data_loader,val_data_loader,base_ds):
    model, criterion, postprocessors = build_model(args)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(args.device)
    
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
                [p for n, p in model.named_parameters()
                if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
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

    base_output_dir = (
        Path(args.output_dir) 
        / 'base'
        / f'base_{args.num_of_base}'
        / f'epoch_{args.epoch}'
    )
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    base_model_path = base_output_dir / 'base_best_model.pth'
    base_cache_path = base_output_dir / 'base_cache.pth'
    base_results_path = base_output_dir / 'base_results.txt'
    
    # eval
    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    
    best_ap = 0.0
    
    for epoch in range(args.epochs):
        # train
        prefetcher = data_prefetcher(train_data_loader, args.device, prefetch=True)
        samples, targets = prefetcher.next()
        pbar = tqdm(total=len(train_data_loader), desc=f'train:Epoch {epoch+1}/{args.epochs}')
        for _ in pbar:
            outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                        for k, v in loss_dict_reduced.items()}
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

            loss_value = losses_reduced_scaled.item()    

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            optimizer.zero_grad()
            losses.backward()
            if args.clip_max_norm > 0:
                grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
            else:
                grad_total_norm = utils.get_total_grad_norm(model.parameters(), args.clip_max_norm)
            optimizer.step()    
            
            samples, targets = prefetcher.next()
            
            # 更新tqdm进度条信息
            pbar.set_postfix({
                'loss': f"{loss_value:.4f}", 
                'class_error': f"{loss_dict_reduced['class_error']:.2f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
            })
            pbar.update(1)
        pbar.close()
        
        # eval
        coco_evaluator = CocoEvaluator(base_ds, iou_types)
        pbar = tqdm(total=len(val_data_loader), desc=f'eval:Epoch {epoch+1}/{args.epochs}')
        with torch.no_grad():
            for samples, targets in pbar:
                samples = samples.to(args.device)
                targets = [{k: v.to(args.device) for k, v in t.items()} for t in targets]
                if torch.cuda.device_count() > 1:
                    outputs = model.module.forward(samples)
                else:
                    outputs = model(samples)
                loss_dict = criterion(outputs, targets)
                weight_dict = criterion.weight_dict
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = utils.reduce_dict(loss_dict)
                loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                            for k, v in loss_dict_reduced.items()}
                loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                            for k, v in loss_dict_reduced.items() if k in weight_dict}
                losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

                loss_value = losses_reduced_scaled.item()   
                orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
                results = postprocessors['bbox'](outputs, orig_target_sizes)
                if 'segm' in postprocessors.keys():
                    target_sizes = torch.stack([t["size"] for t in targets], dim=0)
                    results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
                res = {target['image_id'].item(): output for target, output in zip(targets, results)}
                coco_evaluator.update(res)
                
                pbar.set_postfix({
                    'loss': f"{loss_value:.4f}", 
                    'class_error': f"{loss_dict_reduced['class_error']:.2f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
                })
                pbar.update(1)
        pbar.close()
        
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        
        stats = {k: meter.global_avg for k, meter in coco_evaluator.coco_eval['bbox'].stats.items()}
        if coco_evaluator is not None:
            if 'bbox' in postprocessors.keys():
                stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
            if 'segm' in postprocessors.keys():
                stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
            
        current_ap = coco_evaluator.coco_eval['bbox'].stats[0]  # AP@IoU=0.5:0.95

        if current_ap > best_ap:
            best_ap = current_ap
            print(f"save best model at epoch {epoch}")
            
            if torch.cuda.device_count() > 1:
                torch.save(model.module, base_model_path)
            else:
                torch.save(model, base_model_path)
            criterion.save_acil_cache(base_cache_path)
            save_eval_results(coco_evaluator,base_results_path)
            
        criterion.clear_acil_cache()

def train_acl(args,train_data_loader,val_data_loader,phase,base_ds):
    acl_output_dir = (
        Path(args.output_dir)
        / 'acl'
        / f'{args.num_of_base}'
        / f'epoch_{args.epoch}'
        / f'{args.buffer_size}'
        / f'{args.gamma}'
    )   
    acl_output_dir.mkdir(parents=True,exist_ok=True)
    
    acl_model_path = acl_output_dir / f'phase{phase}_model.pth'
    acl_result_path = acl_output_dir / f'phase{phase}_eval.txt'

    model, criterion, postprocessors = build_model(args)
    criterion.modify_acil_cache()
    
    if phase == 0:
        base_output_dir = Path(args.output_dir +'/base'+f'/base_{args.num_of_base}'+f'/epoch_{args.epoch}')
        base_output_dir.mkdir(parents=True, exist_ok=True)
        
        base_model_path = base_output_dir / 'base_best_model.pth'
        base_cache_path = base_output_dir / 'base_cache.pth'
        model = torch.load(base_model_path)
        
        model.modify_acl()    
    else :
        last_model_path = acl_output_dir / f'phase{phase-1}_model.pth'
        model = torch.load(last_model_path,map_location='cpu')
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(args.device)
    
    # train
    if phase == 0:
        cache = torch.load(base_cache_path,map_location='cpu')
        hs = torch.cat([f for f, _ in cache], dim=0)
        labels = torch.cat([l for _, l in cache], dim=0)
        dataset = TensorDataset(hs,labels)
            
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True
        )
        relign_loader = DataLoader(dataset, batch_size=4, sampler=sampler)
        if torch.cuda.device_count() > 1: 
            model.module.acl_fit(relign_loader)
        else:
            model.acl_fit(relign_loader) 
    else:
        hs_cache = []
        labels_cache = []
        prefetcher = data_prefetcher(train_data_loader, args.device, prefetch=True)
        samples, targets = prefetcher.next()
        for _ in tqdm(range(len(train_data_loader)),desc = "pre"):
            if torch.cuda.device_count > 1:
                outputs = model.module.forward(samples)
            else:
                outputs = model(samples)
            src_logits = outputs['pred_logits']
            
            indices = criterion.matcher(outputs, targets)
            
            idx = get_src_permutation_idx_public(indices)
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
            target_classes = torch.full(src_logits.shape[:2], criterion.num_classes,
                                        dtype=torch.int64, device=src_logits.device)
            target_classes[idx] = target_classes_o
            target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                                dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
            target_classes_onehot = target_classes_onehot[:,:,:-1]

            hs = outputs['dec_outputs']
            hs = hs.transpose(0,1).contiguous()
            
            hs_cache.append(hs)
            labels_cache.append(target_classes_onehot)
            samples,targets = prefetcher.next()
            
        hs = torch.cat(hs_cache,dim=0)
        labels = torch.cat(labels_cache,dim=0)
        
        dataset = TensorDataset(hs,labels)
            
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True
        )
        acl_loader = DataLoader(dataset, batch_size=4, sampler=sampler)
        if torch.cuda.device_count() > 1: 
            model.module.acl_fit(acl_loader)
        else:
            model.acl_fit(acl_loader)
    # eval
    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    pbar = tqdm(total=len(val_data_loader), desc=f'eval:Epoch {epoch+1}/{args.epochs}')
    with torch.no_grad():
        for samples, targets in pbar:
            samples = samples.to(args.device)
            targets = [{k: v.to(args.device) for k, v in t.items()} for t in targets]
            if torch.cuda.device_count() > 1:
                outputs = model.module.forward(samples)
            else:
                outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            
            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                        for k, v in loss_dict_reduced.items()}
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

            loss_value = losses_reduced_scaled.item()   
            
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessors['bbox'](outputs, orig_target_sizes)
            if 'segm' in postprocessors.keys():
                target_sizes = torch.stack([t["size"] for t in targets], dim=0)
                results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
            res = {target['image_id'].item(): output for target, output in zip(targets, results)}
            coco_evaluator.update(res)
            
            pbar.set_postfix({
                'loss': f"{loss_value:.4f}", 
                'class_error': f"{loss_dict_reduced['class_error']:.2f}"
            })
            pbar.update(1)
    pbar.close()
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    
    stats = {k: meter.global_avg for k, meter in coco_evaluator.coco_eval['bbox'].stats.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if torch.cuda.device_count() > 1:
        torch.save(model.module, acl_model_path)
    else:
        torch.save(model,acl_model_path)
    save_eval_results(coco_evaluator,acl_result_path)
    
    
