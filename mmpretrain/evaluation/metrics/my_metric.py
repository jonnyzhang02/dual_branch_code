import torch
import numpy as np
from typing import List, Optional, Sequence, Union
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from mmpretrain.registry import METRICS

def label_to_onehot(label, num_classes):
    """Convert label to one-hot format."""
    onehot = torch.zeros(num_classes)
    onehot[label] = 1
    return onehot

def to_tensor(data):
    """Convert object to torch.Tensor."""
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, (list, tuple)):
        return torch.tensor(data)
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')

def _precision_recall_f1_support(pred, target, average=None):
    """Calculate precision, recall, f1-score and support for each class."""
    num_classes = pred.size(1)
    tp = (pred * target).sum(0)
    fp = pred.sum(0) - tp
    fn = target.sum(0) - tp
    
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1_score = 2 * precision * recall / (precision + recall + 1e-10)
    support = target.sum(0)
    
    if average == 'micro':
        tp = tp.sum()
        fp = fp.sum()
        fn = fn.sum()
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1_score = 2 * precision * recall / (precision + recall + 1e-10)
        support = support.sum()
    elif average == 'macro':
        precision = precision.mean()
        recall = recall.mean()
        f1_score = f1_score.mean()
        support = support.sum()
    
    return precision, recall, f1_score, support

@METRICS.register_module()
class MultiLabelMetricWithClasses(BaseMetric):
    """A collection of precision, recall, f1-score and support for multi-label tasks,
    including metrics for each individual class.

    Args:
        thr (float, optional): Predictions with scores under the threshold
            are considered as negative. Defaults to None.
        topk (int, optional): Predictions with the k-th highest scores are
            considered as positive. Defaults to None.
        items (Sequence[str]): The detailed metric items to evaluate.
            Defaults to ``('precision', 'recall', 'f1-score')``.
        average (str | None): How to calculate the final metrics.
            Defaults to "macro".
        collect_device (str): Device name used for collecting results.
            Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names. Defaults to None.
    """

    default_prefix: Optional[str] = 'multi-label'

    def __init__(self,
                 thr: Optional[float] = None,
                 topk: Optional[int] = None,
                 items: Sequence[str] = ('precision', 'recall', 'f1-score'),
                 average: Optional[str] = 'macro',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:

        super().__init__(collect_device=collect_device, prefix=prefix)
        
        logger = MMLogger.get_current_instance()
        if thr is None and topk is None:
            thr = 0.5
            logger.warning('Neither thr nor k is given, set thr as 0.5 by default.')
        elif thr is not None and topk is not None:
            logger.warning('Both thr and topk are given, use threshold in favor of top-k.')

        self.thr = thr
        self.topk = topk
        self.average = average

        for item in items:
            assert item in ['precision', 'recall', 'f1-score', 'support'], \
                f'The metric {item} is not supported, please choose from ' \
                '"precision", "recall", "f1-score" and "support".'
        self.items = tuple(items)

    def process(self, data_batch, data_samples: Sequence[dict]):
        """Process one batch of data samples.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            result = dict()
            result['pred_score'] = data_sample['pred_score'].clone()
            num_classes = result['pred_score'].size()[-1]

            if 'gt_score' in data_sample:
                result['gt_score'] = data_sample['gt_score'].clone()
            else:
                result['gt_score'] = label_to_onehot(data_sample['gt_label'], num_classes)

            self.results.append(result)

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        metrics = {}

        target = torch.stack([res['gt_score'] for res in results])
        pred = torch.stack([res['pred_score'] for res in results])

        thr = 0.5 if (self.thr is None and self.topk is None) else self.thr

        if self.thr is not None:
            pos_inds = (pred >= self.thr).long()
        else:
            _, topk_indices = pred.topk(self.topk)
            pos_inds = torch.zeros_like(pred).scatter_(1, topk_indices, 1)
            pos_inds = pos_inds.long()

        precision, recall, f1_score, support = _precision_recall_f1_support(pos_inds, target, average=None)
        
        num_classes = pred.size(1)
        for i in range(num_classes):
            class_metrics = {}
            if 'precision' in self.items:
                class_metrics[f'precision_class_{i}'] = precision[i].item()
            if 'recall' in self.items:
                class_metrics[f'recall_class_{i}'] = recall[i].item()
            if 'f1-score' in self.items:
                class_metrics[f'f1-score_class_{i}'] = f1_score[i].item()
            if 'support' in self.items:
                class_metrics[f'support_class_{i}'] = support[i].item()
            metrics.update(class_metrics)

        if self.average:
            precision, recall, f1_score, support = _precision_recall_f1_support(pos_inds, target, average=self.average)
            if 'precision' in self.items:
                metrics['precision'] = precision.item()
            if 'recall' in self.items:
                metrics['recall'] = recall.item()
            if 'f1-score' in self.items:
                metrics['f1-score'] = f1_score.item()
            if 'support' in self.items:
                metrics['support'] = support.item()

        if self.thr and self.thr != 0.5:
            metrics = {f'{k}_thr-{self.thr:.2f}': v for k, v in metrics.items()}
        elif self.topk:
            metrics = {f'{k}_top{self.topk}': v for k, v in metrics.items()}

        return metrics

    @staticmethod
    def calculate(pred: Union[torch.Tensor, np.ndarray, Sequence],
                  target: Union[torch.Tensor, np.ndarray, Sequence],
                  pred_indices: bool = False,
                  target_indices: bool = False,
                  average: Optional[str] = 'macro',
                  thr: Optional[float] = None,
                  topk: Optional[int] = None,
                  num_classes: Optional[int] = None) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Calculate the precision, recall, f1-score."""
        
        def _format_label(label, is_indices):
            if isinstance(label, np.ndarray):
                assert label.ndim == 2, 'The shape of array must be (N, num_classes).'
                label = torch.from_numpy(label)
            elif isinstance(label, torch.Tensor):
                assert label.ndim == 2, 'The shape of tensor must be (N, num_classes).'
            elif isinstance(label, Sequence):
                if is_indices:
                    assert num_classes is not None, 'For index-type labels, please specify `num_classes`.'
                    label = torch.stack([label_to_onehot(indices, num_classes) for indices in label])
                else:
                    label = torch.stack([to_tensor(onehot) for onehot in label])
            else:
                raise TypeError(f'Unsupported label type: {type(label)}.')
            return label

        pred = _format_label(pred, pred_indices)
        target = _format_label(target, target_indices).long()

        assert pred.shape == target.shape, f"Shape mismatch: pred {pred.shape}, target {target.shape}"

        if num_classes is not None:
            assert pred.size(1) == num_classes, f"num_classes mismatch: {pred.size(1)} vs {num_classes}"
        num_classes = pred.size(1)

        thr = 0.5 if (thr is None and topk is None) else thr

        if thr is not None:
            pos_inds = (pred >= thr).long()
        else:
            _, topk_indices = pred.topk(topk)
            pos_inds = torch.zeros_like(pred).scatter_(1, topk_indices, 1)
            pos_inds = pos_inds.long()

        return _precision_recall_f1_support(pos_inds, target, average)