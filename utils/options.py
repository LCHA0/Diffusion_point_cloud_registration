import torch
from utils.attr_dict import AttrDict

option = AttrDict()
option.is_debug_mode = False
option.normalization_enabled = True
option.batch_size = 32
option.n_workers = 12
option.n_epoches = 30
option.lr = 0.001
option.n_input_feats = 0
option.seed = 1
option.is_completion = True
option.device = torch.device("cuda")

# model config
option.emb_nn = ["rpmnet_emb", "dgcnn", "pointnet"][0]
option.pointer = "transformer"
option.head = ["svd", "mlp"][0]
option.emb_dims = 96
option.n_blocks = 1
option.n_heads = 4
option.ff_dims = 256
option.dropout = 0.0
if option.emb_nn in ["rpmnet_emb"]:
    option.features = ['ppf', 'dxyz', 'xyz']
    option.feat_dim = 96
    option.num_neighbors = 64
    option.radius = 0.3
