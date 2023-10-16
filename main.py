import os
import logging
import torch
from configs.get_config import get_cfg
from lib.data_loader.get_data_loader import get_data_loaders
from log.info_output import info_output,generat_res_path,setup_root_logger
from src.structure.subg_extractor import InsGNN
from lib.result_comp import load_checkpoint
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
    info_output(loaders, cfg)
    outdir, complish_flag = generat_res_path(cfg)
    log_file = os.path.join(outdir, 'res.log')
    setup_root_logger(log_file)
    model_path = os.path.join(outdir, 'model')
    os.makedirs(model_path)

    # save para
    res_path = os.path.join(outdir, 'para.log')
    save_res(cfg, res_path)
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

    # elif cfg.framework == 'ProtGNN':
        # model = ProtGNN_static(cfg,aux_info)

    train_loss, best_edge_res, best_g_res = model.train(loaders, outdir, model_path)

    logging.info(f"best_edge_res: {best_edge_res}")
    logging.info(f"best_g_res: {best_g_res}")
    # figure visualization
    # loss
    if cfg.framework != 'PGExplainer':
        epochs = list(range(cfg.train.epochs))
        loss_path = os.path.join(outdir, "loss.png")
        plot_loss(epochs, train_loss,loss_path)

    # auc
    if "ogb" not in cfg.data.data_name:
        auc_path = os.path.join(outdir, "auc.png")
        plot_auc(best_edge_res['fpr'], best_edge_res['tpr'],auc_path)

    # subgraph
    if "ogb" not in cfg.data.data_name:
        best_epoch =  best_edge_res['best_edge_epoch']
        model = load_checkpoint(model, model_path, model_name=f'{best_epoch}_edge')
        assert aux_info['multi_label'] is False
        fig_path = os.path.join(outdir, "subgraph.png")
        all_viz_set = get_viz_idx(test_set, cfg.data.data_name, cfg.visual.num_viz_samples)
        visualize_results(model, all_viz_set, test_set, cfg.visual.num_viz_samples, cfg.data.data_name, cfg.shared_config.use_edge_attr,
                          fig_path, cfg)
    logging.info("finish!")


