# -*- coding: utf-8 -*-
# @Time    : 2023/8/15 18:33
# @Function:
import shutil

import torch
from datetime import datetime
import logging
import os
import sys
def info_output(loaders,cfg):
    test = loaders['test'].dataset.data
    label_bincount = torch.bincount(loaders['test'].dataset.data.y.squeeze().int())
    print(f'[INFO] label bincount:{label_bincount}')
    model_name = cfg.model.type
    data_name = cfg.data.data_name
    lr = cfg.train.graph_lr
    modelInfo = f'seed:{cfg.seed} | device:{cfg.device} | framework:{cfg.framework} | model:{model_name} | dataset:{data_name} | graph_lr:{lr }  '
    print(modelInfo)

def generat_res_path(cfg):
    # 配置日志记录
    log_path =os.path.join( cfg.resultPath , f'{cfg.model.type}-{cfg.data.loader_size}'
                                             f'-{cfg.train.graph_lr}-{cfg.train.epochs}'
                                             f'-{cfg.model.normal_coef}-{cfg.model.kl_1_coef}-{cfg.model.kl_2_coef}'
                                             f'-{cfg.model.GC_delta_coef}')
    # 判断路径是否存在
    count = 0
    for root, dirs, files in os.walk(log_path):
        for dir_name in dirs[:]:
            if dir_name != "model":# 使用切片复制dirs列表以便在循环中修改它
                dir_path = os.path.join(root, dir_name)
                best_res_log_path = os.path.join(dir_path, "best_res.log")
                # 如果子文件夹中不存在best.res.log文件，则删除该子文件夹
                if not os.path.exists(best_res_log_path):
                    print(f"Deleting directory: {dir_path}")
                    shutil.rmtree(dir_path)
                else:
                    count += 1
    if count+1 != cfg.seed:
        print("complish condition!")
        complish_flag = 1
    else:
        complish_flag = 0

    outdir = os.path.join(log_path, "sub_exp" + datetime.now().strftime('_%Y%m%d%H%M%S'))
    os.makedirs(outdir)
    return outdir,complish_flag

def setup_root_logger(log_file):
    # 获取根日志记录器
    root_logger = logging.getLogger()

    # 移除根日志记录器的所有处理器
    root_logger.handlers = []

    # 设置根日志记录器的日志级别
    root_logger.setLevel(logging.INFO)

    # 创建日志处理器（这里使用文件处理器）
    file_handler = logging.FileHandler(log_file)

    # 创建日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # 添加日志处理器到根日志记录器
    root_logger.addHandler(file_handler)