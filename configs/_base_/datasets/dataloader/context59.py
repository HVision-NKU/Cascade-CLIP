import os.path as osp

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
import os.path as osp
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable
from torch.utils.data import Dataset

from mmseg.core import eval_metrics, intersect_and_union, pre_eval_to_metrics
from mmseg.utils import get_root_logger
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.pipelines import Compose, LoadAnnotations
from PIL import Image

@DATASETS.register_module()
class ZeroPascalVOCDataset59(CustomDataset):
    """PascalContext dataset.

    In segmentation map annotation for PascalContext, 0 stands for background,
    which is included in 60 categories. ``reduce_zero_label`` is fixed to
    False. The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is
    fixed to '.png'.

    Args:
        split (str): Split txt file for PascalContext.
    """

    CLASSES = ('aeroplane', 'bag', 'bed', 'bedclothes', 'bench',
               'bicycle', 'bird', 'boat', 'book', 'bottle', 'building', 'bus',
               'cabinet', 'car', 'cat', 'ceiling', 'chair', 'cloth',
               'computer', 'cow', 'cup', 'curtain', 'dog', 'door', 'fence',
               'floor', 'flower', 'food', 'grass', 'ground', 'horse',
               'keyboard', 'light', 'motorbike', 'mountain', 'mouse', 'person',
               'plate', 'platform', 'pottedplant', 'road', 'rock', 'sheep',
               'shelves', 'sidewalk', 'sign', 'sky', 'snow', 'sofa', 'table',
               'track', 'train', 'tree', 'truck', 'tvmonitor', 'wall', 'water',
               'window', 'wood')

    PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
               [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
               [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
               [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
               [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
               [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
               [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
               [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
               [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
               [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
               [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
               [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
               [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
               [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
               [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255]]


    def __init__(self, split, **kwargs):
        super(ZeroPascalVOCDataset59, self).__init__(
            img_suffix='.jpg', 
            seg_map_suffix='.png', 
            split=split, 
            reduce_zero_label=True,
            **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None


    def evaluate(self,
                    seen_idx,
                    unseen_idx,
                    results,
                    metric='mIoU',
                    logger=None,
                    gt_seg_maps=None,
                    **kwargs):
        """Evaluate the dataset.

        Args:
            results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
                    results or predict segmentation map for computing evaluation
                    metric.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU',
                'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            gt_seg_maps (generator[ndarray]): Custom gt seg maps as input,
                used in ConcatDataset

        Returns:
            dict[str, float]: Default metrics.
        """
        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))

        eval_results = {}
        # test a list of files
        if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(
                results, str):
            if gt_seg_maps is None:
                gt_seg_maps = self.get_gt_seg_maps()
            num_classes = len(self.CLASSES)
            ret_metrics = eval_metrics(
                results,
                gt_seg_maps,
                num_classes,
                self.ignore_index,
                metric,
                label_map=dict(),
                reduce_zero_label=self.reduce_zero_label)
        # test a list of pre_eval_results
        else:
            ret_metrics = pre_eval_to_metrics(results, metric)

        # Because dataset.CLASSES is required for per-eval.
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES

        seen_class_names = []
        for i in range(len(seen_idx)):
            seen_class_names.append(class_names[seen_idx[i]])
        seen_class_names = tuple(seen_class_names)

        unseen_class_names = []
        for i in range(len(unseen_idx)):
            unseen_class_names.append(class_names[unseen_idx[i]])
        unseen_class_names = tuple(unseen_class_names)

        # divide ret_metrics into seen and unseen part
        seen_ret_metrics = ret_metrics.copy()
        seen_ret_metrics['IoU'] = seen_ret_metrics['IoU'][seen_idx]
        seen_ret_metrics['Acc'] = seen_ret_metrics['Acc'][seen_idx]
        unseen_ret_metrics = ret_metrics.copy()
        unseen_ret_metrics['IoU'] = unseen_ret_metrics['IoU'][unseen_idx]
        unseen_ret_metrics['Acc'] = unseen_ret_metrics['Acc'][unseen_idx]

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        seen_ret_metrics_summary = OrderedDict({
            seen_ret_metric: np.round(np.nanmean(seen_ret_metric_value) * 100, 2)
            for seen_ret_metric, seen_ret_metric_value in seen_ret_metrics.items()
        })
        unseen_ret_metrics_summary = OrderedDict({
            unseen_ret_metric: np.round(np.nanmean(unseen_ret_metric_value) * 100, 2)
            for unseen_ret_metric, unseen_ret_metric_value in unseen_ret_metrics.items()
        })

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)

        seen_ret_metrics.pop('aAcc', None)
        seen_ret_metrics_class = OrderedDict({
            seen_ret_metric: np.round(seen_ret_metric_value * 100, 2)
            for seen_ret_metric, seen_ret_metric_value in seen_ret_metrics.items()
        })
        seen_ret_metrics_class.update({'Class': seen_class_names})
        seen_ret_metrics_class.move_to_end('Class', last=False)

        unseen_ret_metrics.pop('aAcc', None)
        unseen_ret_metrics_class = OrderedDict({
            unseen_ret_metric: np.round(unseen_ret_metric_value * 100, 2)
            for unseen_ret_metric, unseen_ret_metric_value in unseen_ret_metrics.items()
        })
        unseen_ret_metrics_class.update({'Class': unseen_class_names})
        unseen_ret_metrics_class.move_to_end('Class', last=False)

        # for logger
        print('\n' +  '+++++++++++ Total classes +++++++++++++')
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)
        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column('m' + key, [val])
        print_log('per class results:', logger)
        print_log(class_table_data.get_string(), logger=logger)
        print_log('Summary:', logger)
        print_log(summary_table_data.get_string(), logger=logger)


        print('\n' + '+++++++++++ Seen classes +++++++++++++')
        seen_class_table_data = PrettyTable()
        for key, val in seen_ret_metrics_class.items():
            seen_class_table_data.add_column(key, val)
        seen_summary_table_data = PrettyTable()
        for key, val in seen_ret_metrics_summary.items():
            if key == 'aAcc':
                seen_summary_table_data.add_column(key, [val])
            else:
                seen_summary_table_data.add_column('m' + key, [val])
        print_log('seen per class results:', logger)
        print_log(seen_class_table_data.get_string(), logger=logger)
        print_log('Seen Summary:', logger)
        print_log(seen_summary_table_data.get_string(), logger=logger)
        
        
        print('\n' + '+++++++++++ Unseen classes +++++++++++++')
        unseen_class_table_data = PrettyTable()
        for key, val in unseen_ret_metrics_class.items():
            unseen_class_table_data.add_column(key, val)
        unseen_summary_table_data = PrettyTable()
        for key, val in unseen_ret_metrics_summary.items():
            if key == 'aAcc':
                unseen_summary_table_data.add_column(key, [val])
            else:
                unseen_summary_table_data.add_column('m' + key, [val])
        print_log('unseen per class results:', logger)
        print_log(unseen_class_table_data.get_string(), logger=logger)
        print_log('Unseen Summary:', logger)
        print_log(unseen_summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key == 'aAcc':
                eval_results[key] = value / 100.0
            else:
                eval_results['m' + key] = value / 100.0

        ret_metrics_class.pop('Class', None)
        for key, value in ret_metrics_class.items():
            eval_results.update({
                key + '.' + str(name): value[idx] / 100.0
                for idx, name in enumerate(class_names)
            })

        return eval_results