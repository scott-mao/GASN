# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import torch


logger = logging.getLogger('global')


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    # filter 'num_batches_tracked'

    missing_keys = [x for x in missing_keys
                    if not x.endswith('num_batches_tracked')]
    if len(missing_keys) > 0:
        logger.info('[Warning] missing keys: {}'.format(missing_keys))
        logger.info('missing keys:{}'.format(len(missing_keys)))
    if len(unused_pretrained_keys) > 0:
        logger.info('[Warning] unused_pretrained_keys: {}'.format(
            unused_pretrained_keys))
        logger.info('unused checkpoint keys:{}'.format(
            len(unused_pretrained_keys)))
    logger.info('used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, \
        'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters
    share common prefix 'module.' '''
    logger.info('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x #得到state_dict是个OrderedDict 类型
    return {f(key): value for key, value in state_dict.items()} #把key和value分开


def load_pretrain(model, pretrained_path):
    logger.info('load pretrained model from {}'.format(pretrained_path)) #日志记录

    device = torch.cuda.current_device() #处理设备为GPU 0
    pretrained_dict = torch.load(pretrained_path,# 两个 OrderedDict 类型 中即使包含的 key-value 对完全相同，但只要它们的顺序不同，程序也依然会判断出两个 OrderedDict 是不相等的
        map_location=lambda storage, loc: storage.cuda(device))#把所有的张量加载到GPU 0


    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'],
                                        'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    #pretrained_dict.pop('rpn_head.rpn2.cls.head.3.bias')
    #pretrained_dict.pop('rpn_head.rpn2.cls.head.3.weight')
    #pretrained_dict.pop('rpn_head.rpn2.loc.head.3.bias')
    #pretrained_dict.pop('rpn_head.rpn2.loc.head.3.weight')
    #pretrained_dict.pop('rpn_head.rpn3.cls.head.3.bias')
    #pretrained_dict.pop('rpn_head.rpn3.cls.head.3.weight')
    #pretrained_dict.pop('rpn_head.rpn3.loc.head.3.bias')
    #pretrained_dict.pop('rpn_head.rpn3.loc.head.3.weight')
    #pretrained_dict.pop('rpn_head.rpn4.cls.head.3.bias')
    #pretrained_dict.pop('rpn_head.rpn4.cls.head.3.weight')
    #pretrained_dict.pop('rpn_head.rpn4.loc.head.3.bias')
    #pretrained_dict.pop('rpn_head.rpn4.loc.head.3.weight')
    try:
       check_keys(model, pretrained_dict)

       #!!!

       #

    except:
        logger.info('[Warning]: using pretrain as features.\
                Adding "features." as prefix')
        new_dict = {}
        for k, v in pretrained_dict.items():
            k = 'features.' + k
            new_dict[k] = v
        pretrained_dict = new_dict
        check_keys(model, pretrained_dict)
 #   pretrained_dict = {k: v for k, v in pretrained_dict if k in model}  # 去除一些不需要的参数
  #  model.update(pretrained_dict)  # 参数更新
   # model.load_state_dict(model)  # 加载
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def restore_from(model, optimizer, ckpt_path):
    device = torch.cuda.current_device()
    ckpt = torch.load(ckpt_path,
        map_location=lambda storage, loc: storage.cuda(device))
    epoch = ckpt['epoch']

    ckpt_model_dict = remove_prefix(ckpt['state_dict'], 'module.')
    check_keys(model, ckpt_model_dict)
    model.load_state_dict(ckpt_model_dict, strict=False)

    check_keys(optimizer, ckpt['optimizer'])
    optimizer.load_state_dict(ckpt['optimizer'])
    return model, optimizer, epoch
