
from configs.yacs.yacs  import CfgNode as CN
def get_init_cfg(cls=CN) :
    cfg = cls()
    cfg.use_gpu = True
    cfg.device = 2
    cfg.seed =  0
    cfg.framework = 'InsGNN'
    cfg.resultPath = 'result/results/'


    cfg.data= cls()
    cfg.data.root =  'data/'
    cfg.data.data_name = 'ba_2motifs'
    cfg.data.staticgraph = False
    cfg.data.mutag_x = False
    cfg.data.loader_size = 128
    cfg.data.splitter='rand'  # choices=['rand','fold10']
    cfg.data.data_split_ratio= 0.7
    cfg.data.fold_idx =  0  #'the index(<10) of fold in 10-fold validation.')
    cfg.data.shuffle = False
    cfg.data.one_hot = False
    cfg.data.ts_length = 921600
    cfg.data.hz = 128
    cfg.data.l_min = 128 * 30
    cfg.data.r = 0.2

    cfg.data.splits = cls()
    cfg.data.splits.train = 0.8
    cfg.data.splits.valid = 0.1
    cfg.data.splits.test = 0.1


    cfg.sampling = cls()
    cfg.sampling.hz = 128
    cfg.sampling.samplingFlag = True
    cfg.sampling.normalizeFlag = True
    cfg.sampling.step = 2
    cfg.sampling.ValueError_interrupt = True


    cfg.model= cls()
    cfg.model.type = 'GIN'
    cfg.model.data_batch_size = 3
    cfg.model.target_class = 2
    cfg.model.out_channels = 7
    cfg.model.model_save_path = 'result/model/'
    cfg.model.num_layers = 3
    cfg.model.num_mlp_layers = 2
    cfg.model.hidden_size = 64
    cfg.model.n_layers = 2
    cfg.model.dropout_p = 0.3
    cfg.model.pretrain_lr = 1.0e-3
    cfg.model.pretrain_epochs = 100
    cfg.model.from_scratch = True
    cfg.model.fix_r = False
    cfg.model.decay_interval = 60
    cfg.model.decay_r = 0.1
    cfg.model.final_r = 0.7

    cfg.model.normal_coef = 0.0
    cfg.model.kl_1_coef = 0.0
    cfg.model.kl_2_coef = 0.0
    cfg.model.GC_coef = 1.0
    cfg.model.GC_delta_coef = 0.0
    cfg.model.scalers = False
    cfg.model.aggregators = ['mean']

    cfg.gat = cls()
    cfg.gat.num_heads =  8
    cfg.gat.num_out_heads = 1
    cfg.gat.attn_drop = 0.8
    cfg.gat.negative_slope = 0.2
    cfg.gat.residual = False
    cfg.gat.final_dropout = 0.7
    cfg.gat.graph_pooling_type = "mean"

    cfg.pgexplainer= cls()
    cfg.pgexplainer.explainer_args= cls()
    cfg.pgexplainer.explainer_args.t0 = 5.0  # temperature denominator
    cfg.pgexplainer.explainer_args.t1 =1.0  # temperature numerator
    cfg.pgexplainer.explainer_args.coff_size = 0.01  # constrains on mask size
    cfg.pgexplainer.explainer_args.coff_ent = 5e-4

    cfg.pgexplainer.model_args = cls()
    cfg.pgexplainer.model_args.device_id = 0
    cfg.pgexplainer.model_args.model_name = 'devign'
    cfg.pgexplainer.model_args.checkpoint = './checkpoint'
    cfg.pgexplainer.model_args.concate = False  # whether to concate the gnn features before mlp
    cfg.pgexplainer.model_args.latent_dim = [128, 128, 128]  # the hidden units for each gnn layer
    cfg.pgexplainer.model_args.readout = 'max'  # the graph pooling method
    cfg.pgexplainer.model_args.mlp_hidden = []  # the hidden units for mlp classifier
    cfg.pgexplainer.model_args.gnn_dropout = 0.0  # the dropout after gnn layers
    cfg.pgexplainer.model_args.dropout = 0.5  # the dropout after mlp layers
    cfg.pgexplainer.model_args.adj_normlize = True  # the edge_weight normalization for gcn conv
    cfg.pgexplainer.model_args.emb_normlize = False  # the l2 normalization after gnn layer
    cfg.pgexplainer.model_args.model_path = ""  # default path to save the model
    cfg.pgexplainer.model_args.max_edge_types = 4

    cfg.timeseries = cls()
    cfg.timeseries.shapelet = True
    cfg.timeseries.epochs = 50
    cfg.timeseries.shaplet_method = 'shapelets_Its'
    cfg.timeseries.K = 5   # numbers of shapelets
    cfg.timeseries.R = 2
    cfg.timeseries.alpha = 0.01
    cfg.timeseries.lamda = 0.01
    cfg.timeseries.eta = 0.001
    cfg.timeseries.shapelet_initialization = 'segments_centroids'
    cfg.timeseries.patience = 20
    cfg.timeseries.shaplet_outsize = 30
    cfg.timeseries.shaplet_segment = 4


    cfg.train = cls()
    cfg.train.lr = 0.001
    cfg.train.weight_decay = 0.0
    cfg.train.graph_lr = 0.01
    cfg.train.optimizer = "Adam"
    cfg.train.final_dropout = 0.7
    cfg.train.learn_eps = "store_true"
    cfg.train.epochs = 150

    cfg.shared_config = cls()
    cfg.shared_config.use_edge_attr = True
    cfg.shared_config.learn_edge_att = True
    cfg.shared_config.learn_edge_att_m = False
    cfg.shared_config.precision_k = 5
    cfg.shared_config.num_viz_samples = 8
    cfg.shared_config.viz_interval = 10
    cfg.shared_config.viz_norm_att = True
    cfg.shared_config.extractor_dropout_p = 0.5

    cfg.visual = cls()
    cfg.visual.num_viz_samples = 10

    # cfg.GSAT_config = cls()
    # cfg.GSAT_config.method_name = 'GSAT'
    # cfg.GSAT_config.pred_loss_coef = 1
    # cfg.GSAT_config.info_loss_coef = 1
    # cfg.GSAT_config.epochs = 200
    # cfg.GSAT_config.lr = 3.0e-3
    # cfg.GSAT_config.weight_decay = 3.0e-6
    #
    # cfg.GSAT_config.from_scratch = True
    # cfg.GSAT_config.fix_r = False
    # cfg.GSAT_config.decay_interval = 60
    # cfg.GSAT_config.decay_r = 0.1
    # cfg.GSAT_config.final_r = 0.7

    return cfg