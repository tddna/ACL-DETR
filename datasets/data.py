import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional, List, Dict, Any

# 导入原有代码中的组件
from .coco import build_dataset
from .incremental import generate_cls_order
from . import samplers
from . import pycocotools
import util.misc as utils

class IncrementalDETRDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning数据模块，支持增量学习的DETR数据加载
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.cls_order = generate_cls_order(seed=args.seed_cls)
        self.phase_idx = 0
        
        # 各个数据集
        self.dataset_train = None
        self.dataset_val = None
        self.dataset_train_balanced = None
        self.dataset_val_old = None
        self.dataset_val_new = None
        
        # COCO评估API对象
        self.coco_api = None
        self.coco_api_old = None
        self.coco_api_new = None
        
        # 分布式训练相关
        self.distributed = args.distributed if hasattr(args, 'distributed') else False
        self.cache_mode = args.cache_mode if hasattr(args, 'cache_mode') else False
        
    def setup(self, stage: Optional[str] = None):
        """
        根据当前阶段初始化数据集
        
        Args:
            stage: 'fit', 'validate', 'test' 或 None
        """
        # 基本训练集和验证集
        self.dataset_train = build_dataset(
            image_set='train', 
            args=self.args, 
            cls_order=self.cls_order,
            phase_idx=self.phase_idx, 
            incremental=True, 
            incremental_val=False, 
            val_each_phase=False
        )
        
        self.dataset_val = build_dataset(
            image_set='val', 
            args=self.args, 
            cls_order=self.cls_order,
            phase_idx=self.phase_idx, 
            incremental=True, 
            incremental_val=True, 
            val_each_phase=False
        )
        
        # 从第1阶段开始需要额外的数据集
        if self.phase_idx >= 1:
            # 平衡训练集（旧类别和新类别样本平衡）
            self.dataset_train_balanced = build_dataset(
                image_set='train', 
                args=self.args, 
                cls_order=self.cls_order,
                phase_idx=self.phase_idx, 
                incremental=True, 
                incremental_val=False, 
                val_each_phase=False, 
                balanced_ft=True
            )
            
            # 旧类别验证集
            self.dataset_val_old = build_dataset(
                image_set='val', 
                args=self.args, 
                cls_order=self.cls_order,
                phase_idx=0,  # 始终是第0阶段的类别
                incremental=True, 
                incremental_val=True, 
                val_each_phase=False
            )
            
            # 新类别验证集
            self.dataset_val_new = build_dataset(
                image_set='val', 
                args=self.args, 
                cls_order=self.cls_order,
                phase_idx=1,  # 从第1阶段开始的类别
                incremental=True, 
                incremental_val=True, 
                val_each_phase=True
            )
        
        # 创建COCO API对象，用于评估
        self._setup_coco_api()
    
    def _setup_coco_api(self):
        """设置COCO API对象，用于评估"""
        from . import coco_eval
        self.coco_api = coco_eval.get_coco_api_from_dataset(self.dataset_val)
        
        if self.phase_idx >= 1:
            self.coco_api_old = coco_eval.get_coco_api_from_dataset(self.dataset_val_old)
            self.coco_api_new = coco_eval.get_coco_api_from_dataset(self.dataset_val_new)
    
    def _get_sampler(self, dataset, shuffle=False):
        """获取适当的采样器"""
        if self.distributed:
            if self.cache_mode:
                return samplers.NodeDistributedSampler(dataset, shuffle=shuffle)
            else:
                return samplers.DistributedSampler(dataset, shuffle=shuffle)
        else:
            if shuffle:
                return torch.utils.data.RandomSampler(dataset)
            else:
                return torch.utils.data.SequentialSampler(dataset)
    
    def train_dataloader(self):
        """返回训练数据加载器"""
        sampler = self._get_sampler(self.dataset_train, shuffle=True)
        
        if isinstance(sampler, torch.utils.data.RandomSampler):
            # 普通采样器，需要批量采样器
            batch_sampler = torch.utils.data.BatchSampler(
                sampler, self.args.batch_size, drop_last=True
            )
            loader = DataLoader(
                self.dataset_train,
                batch_sampler=batch_sampler,
                collate_fn=utils.collate_fn,
                num_workers=self.args.num_workers,
                pin_memory=True
            )
        else:
            # 分布式采样器
            loader = DataLoader(
                self.dataset_train,
                batch_size=self.args.batch_size,
                sampler=sampler,
                drop_last=True,
                collate_fn=utils.collate_fn,
                num_workers=self.args.num_workers,
                pin_memory=True
            )
        
        return loader
    
    def val_dataloader(self):
        """返回验证数据加载器"""
        sampler_val = self._get_sampler(self.dataset_val, shuffle=False)
        
        data_loaders = [
            DataLoader(
                self.dataset_val, 
                batch_size=self.args.batch_size,
                sampler=sampler_val,
                drop_last=False,
                collate_fn=utils.collate_fn,
                num_workers=self.args.num_workers,
                pin_memory=True
            )
        ]
        
        # 如果在第1阶段及以后，添加旧类别和新类别的验证集
        if self.phase_idx >= 1:
            sampler_val_old = self._get_sampler(self.dataset_val_old, shuffle=False)
            sampler_val_new = self._get_sampler(self.dataset_val_new, shuffle=False)
            
            data_loaders.extend([
                DataLoader(
                    self.dataset_val_old,
                    batch_size=self.args.batch_size,
                    sampler=sampler_val_old,
                    drop_last=False,
                    collate_fn=utils.collate_fn,
                    num_workers=self.args.num_workers,
                    pin_memory=True
                ),
                DataLoader(
                    self.dataset_val_new,
                    batch_size=self.args.batch_size,
                    sampler=sampler_val_new,
                    drop_last=False,
                    collate_fn=utils.collate_fn,
                    num_workers=self.args.num_workers,
                    pin_memory=True
                )
            ])
        
        return data_loaders
    
    def get_balanced_dataloader(self):
        """返回平衡微调的数据加载器"""
        if self.phase_idx < 1 or self.dataset_train_balanced is None:
            return None
        
        sampler = self._get_sampler(self.dataset_train_balanced, shuffle=True)
        
        if isinstance(sampler, torch.utils.data.RandomSampler):
            batch_sampler = torch.utils.data.BatchSampler(
                sampler, self.args.batch_size, drop_last=True
            )
            loader = DataLoader(
                self.dataset_train_balanced,
                batch_sampler=batch_sampler,
                collate_fn=utils.collate_fn,
                num_workers=self.args.num_workers,
                pin_memory=True
            )
        else:
            loader = DataLoader(
                self.dataset_train_balanced,
                batch_size=self.args.batch_size,
                sampler=sampler,
                drop_last=True,
                collate_fn=utils.collate_fn,
                num_workers=self.args.num_workers,
                pin_memory=True
            )
        
        return loader
    
    def set_phase(self, phase_idx):
        """
        设置当前阶段，并重新初始化数据集
        
        Args:
            phase_idx: 阶段索引
        """
        self.phase_idx = phase_idx
        self.setup()
    
    def get_coco_api(self, which='all'):
        """
        获取COCO API对象，用于评估
        
        Args:
            which: 'all', 'old', 或 'new'
        
        Returns:
            对应的COCO API对象
        """
        if which == 'all':
            return self.coco_api
        elif which == 'old':
            return self.coco_api_old
        elif which == 'new':
            return self.coco_api_new
        else:
            raise ValueError(f"无效的COCO API类型: {which}")
    
    def get_current_classes(self):
        """获取当前阶段的类别列表"""
        if self.args.data_setting == 'tfs':
            if self.phase_idx == 0:
                return self.cls_order[:self.args.cls_per_phase]
            else:
                return self.cls_order[:(self.phase_idx+1)*self.args.cls_per_phase]
        elif self.args.data_setting == 'tfh':
            if self.phase_idx == 0:
                return self.cls_order[:40]
            else:
                return self.cls_order[:(self.phase_idx)*self.args.cls_per_phase+40]
        else:
            raise ValueError(f"不支持的数据设置: {self.args.data_setting}")
