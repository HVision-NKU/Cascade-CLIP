import argparse

from mmcv import Config
from mmcv.cnn import get_model_complexity_info
from mmseg.models import build_segmentor
import models

from fvcore.nn import FlopCountAnalysis

import torch
from numbers import Number
from typing import Any, Callable, List, Optional, Union
from numpy import prod
import numpy as np
from fvcore.nn import FlopCountAnalysis
from mmseg.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--fvcore',
        action='store_true', default=True)
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[512, 512],
        help='input image size')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
        
    datasets = [build_dataset(cfg.data.train)]
    
    if 'CLIP' in cfg.model.type:
        cfg.model.class_names = list(datasets[0].CLASSES)
    
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')).cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))
        
    if args.fvcore:
        print("#### Flops ####")
        with torch.no_grad():
            x = torch.randn(1, 3, input_shape[1], input_shape[1]).cuda()
            fca1 = FlopCountAnalysis(model, x)
            flops1 = fca1.total()
            print("#### GFLOPs: {:.1f}".format(flops1 / 1e9))

        print("#### Params. ####")
        print('Total:', sum(p.numel() for p in model.parameters()) / 1e6)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        print('Trainable', f'{n_parameters:.1f}')

    else:
        flops, params = get_model_complexity_info(model, input_shape)
        split_line = '=' * 30
        print('{0}\nInput shape: {1}\nFlops: {2}\nParams: {3}\n{0}'.format(
            split_line, input_shape, flops, params))
        print('!!!Please be cautious if you use the results in papers. '
              'You may need to check if all ops are supported and verify that the '
              'flops computation is correct.')
        

if __name__ == '__main__':
    main()
