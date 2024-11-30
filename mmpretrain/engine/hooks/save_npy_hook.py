"""
@ Author       : jonnyzhang 71881972+jonnyzhang02@users.noreply.github.com
@ LastEditTime : 2024-11-29 21:28
@ FilePath     : /mmpretrain/mmpretrain/engine/hooks/save_npy_hook.py
@ 
@ coded by ZhangYang@BUPT, my email is zhangynag0706@bupt.edu.cn
"""
# Copyright (c) OpenMMLab. All rights reserved.
import math
import os.path as osp
from typing import Optional, Sequence

from mmengine.fileio import join_path
from mmengine.hooks import Hook
from mmengine.runner import EpochBasedTrainLoop, Runner
from mmengine.visualization import Visualizer

from mmpretrain.registry import HOOKS
from mmpretrain.structures import DataSample
from mmengine import MessageHub
message_hub = MessageHub.get_current_instance()
import pickle
import numpy



@HOOKS.register_module()
class SavenpyHook(Hook):
    def __init__(self,
                out_dir: Optional[str] = None,
                 **kwargs):
        self.out_dir = out_dir

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[DataSample]) -> None:
        out_path_npy = join_path(self.out_dir, f'iter_{batch_idx}.npy')
        out_path_pkl = join_path(self.out_dir, f'iter_{batch_idx}.pkl')
        pickle.dump(data_batch, open(out_path_pkl, 'wb'))
        numpy.save(out_path_npy, message_hub.get_info('outs_array'))
        

        
