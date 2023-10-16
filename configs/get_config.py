# -*- coding: utf-8 -*-
# @Time    : 2023/8/15 16:43
# @Function:
import argparse
import yaml
from configs.yacs.get_init_cfg import get_init_cfg

def parse_args(args=None):
    parser = argparse.ArgumentParser(description='federatedscope',
                                     add_help=False)
    parser.add_argument('--cfg',
                        dest='cfg_file',
                        # default='gfl/baseline/fedavg_gnn_node_fullbatch_citation.yaml',
                        help='Config file path',
                        required=False,
                        type=str)

    parser.add_argument('opts',
                        help='See federatedscope/core/configs for all options',
                        default=None,
                        nargs=argparse.REMAINDER)
    parse_res = parser.parse_args(args)

    return parse_res

def get_cfg():
    args = parse_args()
    cfg = get_init_cfg()
    if args.cfg_file:
        cfg.merge_from_file(args.cfg_file)  # from yaml file
    cfg.merge_from_list(args.opts)  # from command lines
    return cfg
