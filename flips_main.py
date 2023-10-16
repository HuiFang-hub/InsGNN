import os
import logging
import torch
from configs.get_config import get_cfg
from lib.data_loader.get_data_loader import get_data_loaders
from log.info_output import info_output,generat_res_path,setup_root_logger
from src.structure.subg_extractor import InsGNN
from lib.result_comp import load_checkpoint, load_checkpoint2
from lib.visualization import  get_viz_idx,visualize_results,plot_loss,plot_auc
from log.save_results import save_res
from lib.baselines.gsat import GSAT
from lib.baselines.gat_explain import  GAT_static
from lib.baselines.gnnExplainer import GNNExplainer_static
from lib.baselines.pgeExplainer import PGExplainer_static
# torch.set_default_dtype(torch.Float)
import sys
############################

if __name__ == '__main__':
    # ------------ load parameters ----------------#
    cfg = get_cfg()
    # cfg.framework = 'GSAT'
    cfg.resultPath = os.path.join(cfg.resultPath, cfg.data.data_name, cfg.framework)
    if not os.path.exists(cfg.resultPath):
        os.makedirs(cfg.resultPath)
    cfg.device = torch.device("cuda:" + str(cfg.device) if cfg.use_gpu else "cpu")

    # ------------ load data ----------------#
    loaders, dataset, test_set, aux_info = \
        get_data_loaders(cfg,
                         splits={'train': 0.8, 'valid': 0.1, 'test': 0.1}, shapelet=cfg.timeseries['shapelet'],
                         mutag_x=cfg.data.mutag_x)

    # log config
    # info_output(loaders, cfg)
    # outdir, complish_flag = generat_res_path(cfg)
    # log_file = os.path.join(outdir, 'res.log')
    # setup_root_logger(log_file)
    # model_path = os.path.join(outdir, 'model')
    # os.makedirs(model_path)

    # save para
    # res_path = os.path.join(outdir, 'para.log')
    # save_res(cfg, res_path)
    model = None
    #train model
    if cfg.framework =='InsGNN':
        model = InsGNN(cfg,aux_info)

    elif cfg.framework =='GSAT':
        model = GSAT(cfg,aux_info)

    elif cfg.framework == 'GAT':
        model = GAT_static(cfg,aux_info)

    elif cfg.framework == 'GNNExplainer':
        model = GNNExplainer_static(cfg,aux_info)

    elif cfg.framework == 'PGExplainer':
        model = PGExplainer_static(cfg,aux_info)


    # load best model
    log_path = os.path.join(cfg.resultPath, f'{cfg.model.type}-{cfg.data.loader_size}'
                                            f'-{cfg.train.graph_lr}-{cfg.train.epochs}'
                                            f'-{cfg.model.normal_coef}-{cfg.model.kl_1_coef}-{cfg.model.kl_2_coef}'
                                            f'-{cfg.model.GC_delta_coef}')


    i=0
    res = {}
    for root, dirs, files in os.walk(log_path):
        for dir_name in dirs[:]:
            if dir_name == "model":
                best_epoch = 0
                model_path = os.path.join(root, dir_name)
                for filename in os.listdir(model_path):
                    if cfg.framework == 'InsGNN':
                        now_epoch = int(filename.split('_')[0])
                    else:
                        now_epoch = int(filename.split('.')[0])
                    if  now_epoch  > best_epoch:
                        best_epoch = now_epoch
                        file_path = os.path.join(model_path, filename)
                        model = load_checkpoint2(model, file_path)
                        maintain, decision_flips = model.flips_train(loaders,'train',cfg.data.r)
                        explain_res = {'maintain': maintain,
                                    'decision_flips': decision_flips}
                        res[i] = explain_res
                i += 1
    res_path = os.path.join(log_path, f'{cfg.data.r}_explain_res.log')
    save_res(res, res_path)




