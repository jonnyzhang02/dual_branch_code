# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmpretrain.evaluation.metrics import Accuracy
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from mmcv.cnn import build_activation_layer
from .cls_head import ClsHead


@MODELS.register_module()
class VigHeadOurs(ClsHead):
    """Classification head.

    Args:
        loss (dict): Config of classification loss. Defaults to
            ``dict(type='CrossEntropyLoss', loss_weight=1.0)``.
        topk (int | Tuple[int]): Top-k accuracy. Defaults to ``(1, )``.
        cal_acc (bool): Whether to calculate accuracy during training.
            If you use batch augmentations like Mixup and CutMix during
            training, it is pointless to calculate accuracy.
            Defaults to False.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to None.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 hidden_dim: int = 1024,
                 act_cfg: dict = dict(type='GELU'),
                 dropout: float = 0.,
                 **kwargs):
        super().__init__(**kwargs)

        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.act = build_activation_layer(act_cfg)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    # def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
    #     """The process before the final classification head.

    #     The input ``feats`` is a tuple of tensor, and each tensor is the
    #     feature of a backbone stage. In ``ClsHead``, we just obtain the feature
    #     of the last stage.
    #     """
    #     # The ClsHead doesn't have other module, just return after unpacking.
    #     return feats[-1]
    
    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of tensor, and each tensor is the
        feature of a stage_blocks stage. In ``VigClsHead``, we just obtain the
        feature of the last stage.
        """
        feats = self.fc1(feats)
        feats = self.bn(feats)
        feats = self.act(feats)
        feats = self.drop(feats)

        return feats

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        pre_logits = self.pre_logits(feats)
        # The final classification head.
        cls_score = self.fc2(pre_logits)
        return cls_score

    def loss(self, feats: Tuple[torch.Tensor], data_samples: List[DataSample],
             **kwargs) -> dict:
        """Calculate losses from the classification score.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[DataSample]): The annotation data of
                every samples.
            **kwargs: Other keyword arguments to forward the loss module.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # The part can be traced by torch.fx
        harris_loss = feats[0]
        theta_loss = feats[1]
        cos_loss = feats[2]
        step = feats[3]
        writer = feats[4]
        feats_t = feats[-2]
        feats = feats[-1]
        cls_score = self(feats)
        cls_score_t = self(feats_t)
        self.cal_acc = True

        # The part can not be traced by torch.fx
        losses = self._get_loss(cls_score, data_samples, **kwargs)
        losses_t = self._get_loss(cls_score_t, data_samples, **kwargs)

        # Add harris loss
        losses['loss'] += losses_t['loss']
        losses['loss'] += 0.1 * harris_loss
        losses['loss'] += 0.1 * theta_loss
        losses['loss'] += 0.1 * cos_loss

        # print('class_loss: ', losses['loss'].item(), 
        #       'harris_loss: ',  0.1 * harris_loss.item(), 
        #       'theta_loss: ',  0.1 * theta_loss.item())
        
        # writer.add_scalar('accuracy_top-1', losses['accuracy_top-1'][0].item(), global_step=step)
        # writer.add_scalar('accuracy_top-5', losses['accuracy_top-5'][0].item(), global_step=step)
        # writer.add_scalar('class_loss', losses['loss'].item(), global_step=step)
        # writer.add_scalar('class_loss_t', losses_t['loss'].item(), global_step=step)
        # writer.add_scalar('harris_loss', 0.1 * harris_loss.item(), global_step=step)
        # writer.add_scalar('theta_loss', 0.1 * theta_loss.item(), global_step=step)
        # writer.add_scalar('cos_loss', 0.1 * cos_loss.item(), global_step=step)

        return losses

    def _get_loss(self, cls_score: torch.Tensor,
                  data_samples: List[DataSample], **kwargs):
        """Unpack data samples and compute loss."""
        # Unpack data samples and pack targets
        if 'gt_score' in data_samples[0]:
            # Batch augmentation may convert labels to one-hot format scores.
            target = torch.stack([i.gt_score for i in data_samples])
        else:
            target = torch.cat([i.gt_label for i in data_samples])

        # compute loss
        losses = dict()
        # print(cls_score, target)
        loss = self.loss_module(
            cls_score, target, avg_factor=cls_score.size(0), **kwargs)
        losses['loss'] = loss

        # compute accuracy
        if self.cal_acc:
            assert target.ndim == 1, 'If you enable batch augmentation ' \
                'like mixup during training, `cal_acc` is pointless.'
            # acc = Accuracy.calculate(cls_score, target, topk=self.topk)
            # losses.update(
            #     {f'accuracy_top-{k}': a
            #      for k, a in zip(self.topk, acc)})

        return losses

    def predict(
        self,
        feats: Tuple[torch.Tensor],
        data_samples: Optional[List[Optional[DataSample]]] = None
    ) -> List[DataSample]:
        """Inference without augmentation.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[DataSample | None], optional): The annotation
                data of every samples. If not None, set ``pred_label`` of
                the input data samples. Defaults to None.

        Returns:
            List[DataSample]: A list of data samples which contains the
            predicted results.
        """
        # The part can be traced by torch.fx
        feats = feats[-1]
        cls_score = self(feats)

        # The part can not be traced by torch.fx
        predictions = self._get_predictions(cls_score, data_samples)
        return predictions

    def _get_predictions(self, cls_score, data_samples):
        """Post-process the output of head.

        Including softmax and set ``pred_label`` of data samples.
        """
        pred_scores = F.softmax(cls_score, dim=1)
        pred_labels = pred_scores.argmax(dim=1, keepdim=True).detach()

        out_data_samples = []
        if data_samples is None:
            data_samples = [None for _ in range(pred_scores.size(0))]

        for data_sample, score, label in zip(data_samples, pred_scores,
                                             pred_labels):
            if data_sample is None:
                data_sample = DataSample()

            data_sample.set_pred_score(score).set_pred_label(label)
            out_data_samples.append(data_sample)
        return out_data_samples
