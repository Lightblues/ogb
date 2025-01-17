from copy import copy
import argparse
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, ParameterDict, Parameter
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
from torch_geometric.data import NeighborSampler
# from torch_geometric.loader import NeighborSampler
from torch_geometric.utils.hetero import group_hetero_graph
from torch_geometric.nn import MessagePassing

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator, DglNodePropPredDataset

from logger import Logger
import os.path as osp
from dgl.data.utils import load_graphs, save_graphs, Subset


# C00: 增加了配置项!
dname = 'tiny'
# dname = 'ogbn-mag'
if dname=='tiny':
    meta_dict = {
        # 'dir_path': '/home/ec2-user/workspace/tab2graph/datasets/tiny/dgl_data_processed', 
        'dir_path': '/Users/frankshi/LProjects/github/tab2graph/datasets/tiny/dgl_data_processed',
        'num tasks': 1, 'task type': 'multi-class classification', 'eval metric': 'acc', 
        'num classes': 5, 'is hetero': 'True', 'binary': 'False',
        'add_inverse_edge': 'False', 'has_node_attr': 'True', 'has_edge_attr': 'False',
        'split': 'time', 'additional node files': 'None', 'additional edge files': 'None'
    }
    bs = 32
elif dname=='ogbn-mag':
    meta_dict = {
        'dir_path': '/home/ec2-user/workspace/tab2graph/datasets/ogbn-mag/dgl_data_processed', 
        'num tasks': 1, 'task type': 'multi-class classification', 'eval metric': 'acc', 
        'num classes': 349, 'is hetero': 'True', 'binary': 'False',
        'add_inverse_edge': 'False', 'has_node_attr': 'True', 'has_edge_attr': 'False',
        'split': 'time', 'additional node files': 'None', 'additional edge files': 'None'
    }
    bs = 1024

parser = argparse.ArgumentParser(description='OGBN-MAG (SAGE)')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--runs', type=int, default=10)
args = parser.parse_args()
print(args)

# dataset = PygNodePropPredDataset(name='ogbn-mag')
# dataset = DglNodePropPredDataset(name='ogbn-mag')

class CustomDglPropPredDataset(DglNodePropPredDataset):
    def __init__(self, name, meta_dict = None):
        self.name = name ## original name, e.g., ogbn-proteins

        self.dir_name = meta_dict['dir_path']
        self.original_root = ''
        self.root = meta_dict['dir_path']
        self.meta_info = meta_dict
        
        # self.download_name = self.meta_info['download_name'] ## name of downloaded file, e.g., tox21

        self.num_tasks = int(self.meta_info['num tasks'])
        self.task_type = self.meta_info['task type']
        self.eval_metric = self.meta_info['eval metric']
        self.num_classes = int(self.meta_info['num classes'])
        self.is_hetero = self.meta_info['is hetero'] == 'True'
        self.binary = self.meta_info['binary'] == 'True'
        
        super(DglNodePropPredDataset, self).__init__()
        
        self.pre_process()

    def pre_process(self):
        # processed_dir = osp.join(self.root, 'processed')
        # pre_processed_file_path = osp.join(processed_dir, 'dgl_data_processed')
        pre_processed_file_path = self.dir_name

        if osp.exists(pre_processed_file_path):
            self.graph, label_dict = load_graphs(pre_processed_file_path)
            if self.is_hetero:
                self.labels = label_dict
            else:
                self.labels = label_dict['labels']
        else:
            raise RuntimeError('Cannot find the pre-processed data. ')

    def get_idx_split(self, split_type = None):
        # n = self.graph[0].number_of_nodes()
        # num of nodes 'paper'
        n = self.graph[0].number_of_nodes('paper')
        idxs = torch.arange(n)
        idx_shuffled = idxs[torch.randperm(n)]
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        idx_train = idx_shuffled[:n_train]
        idx_val = idx_shuffled[n_train:(n_train + n_val)]
        idx_test = idx_shuffled[(n_train + n_val):]
        return {
            'train': {'paper': idx_train},
            'valid': {'paper': idx_val},
            'test': {'paper': idx_test}
        }

from dgl import DGLGraph
def add_properties_to_dgl(g:DGLGraph):
    """ 添加 edge_index_dict, num_nodes_dict, x_dict, y_dict 属性
    """
    graph = g.clone()

    # 在 convert_dglData.py 中, 预先将数据放进去
    graph.x_dict = graph.ndata['feat']
    graph.y_dict = graph.ndata['label']
    
    edge_index_dict = {}
    for edge_type in graph.canonical_etypes:
        src, dst = graph.all_edges(form='uv', etype=edge_type, order='srcdst')
        edge_index_dict[edge_type] = torch.stack([src, dst], dim=0)
    graph.edge_index_dict = edge_index_dict
    
    num_nodes_dict = {}
    for ntype in graph.ntypes:
        num_nodes_dict[ntype] = graph.number_of_nodes(ntype)
    graph.num_nodes_dict = num_nodes_dict
    
    return graph

import torch

def get_size_tensor(t:torch.Tensor):
    return t.element_size() * t.numel()

def get_size(x):
    if type(x) == torch.Tensor:
        return get_size_tensor(x)
    elif type(x) == list:
        return sum([get_size_tensor(t) for t in x])
    elif type(x) == dict:
        return sum([get_size_tensor(t) for t in x.values()])

# dataset = DglNodePropPredDataset(name='ogbn-mag')
# dname = 'ogbn-mag'
""" config file see ogb/nodeproppred/master.csv
,ogbn-proteins,ogbn-products,ogbn-arxiv,ogbn-mag,ogbn-papers100M
num tasks,112,1,1,1,1
num classes,2,47,40,349,172
eval metric,rocauc,acc,acc,acc,acc
task type,binary classification,multiclass classification,multiclass classification,multiclass classification,multiclass classification
download_name,proteins,products,arxiv,mag,papers100M-bin
version,1,1,1,2,1
url,http://snap.stanford.edu/ogb/data/nodeproppred/proteins.zip,http://snap.stanford.edu/ogb/data/nodeproppred/products.zip,http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip,http://snap.stanford.edu/ogb/data/nodeproppred/mag.zip,http://snap.stanford.edu/ogb/data/nodeproppred/papers100M-bin.zip
add_inverse_edge,True,True,False,False,False
has_node_attr,False,True,True,True,True
has_edge_attr,True,False,False,False,False
split,species,sales_ranking,time,time,time
additional node files,node_species,None,node_year,node_year,node_year
additional edge files,None,None,None,edge_reltype,None
is hetero,False,False,False,True,False
binary,False,False,False,False,True
"""
# C01: 定义为用DGL来存储数据! 
dataset = CustomDglPropPredDataset(name=dname, meta_dict=meta_dict)

split_idx = dataset.get_idx_split()
evaluator = Evaluator(name='ogbn-mag')
logger = Logger(args.runs, args)

# We do not consider those attributes for now.
# data.node_year_dict = None
# data.edge_reltype_dict = None

# C02: 需要为Graph添加两个属性
data = dataset[0][0]
print(data)
data = add_properties_to_dgl(data)

edge_index_dict = data.edge_index_dict

# # We need to add reverse edges to the heterogeneous graph.
# r, c = edge_index_dict[('author', 'affiliated_with', 'institution')]
# edge_index_dict[('institution', 'to', 'author')] = torch.stack([c, r])
# r, c = edge_index_dict[('author', 'writes', 'paper')]
# edge_index_dict[('paper', 'to', 'author')] = torch.stack([c, r])
# r, c = edge_index_dict[('paper', 'has_topic', 'field_of_study')]
# edge_index_dict[('field_of_study', 'to', 'paper')] = torch.stack([c, r])
# # Convert to undirected paper <-> paper relation.
# edge_index = to_undirected(edge_index_dict[('paper', 'cites', 'paper')])
# edge_index_dict[('paper', 'cites', 'paper')] = edge_index

# We convert the individual graphs into a single big one, so that sampling
# neighbors does not need to care about different edge types.
# This will return the following:
# * `edge_index`: The new global edge connectivity.
# * `edge_type`: The edge type for each edge.
# * `node_type`: The node type for each node.
# * `local_node_idx`: The original index for each node.
# * `local2global`: A dictionary mapping original (local) node indices of
#    type `key` to global ones.
# `key2int`: A dictionary that maps original keys to their new canonical type.
out = group_hetero_graph(data.edge_index_dict, data.num_nodes_dict)
edge_index, edge_type, node_type, local_node_idx, local2global, key2int = out

# Map informations to their canonical type.
x_dict = {}
for key, x in data.x_dict.items():
    x_dict[key2int[key]] = x.to(torch.float)        # 注意转为 torch.float32, 不然无法forward

num_nodes_dict = {}
for key, N in data.num_nodes_dict.items():
    num_nodes_dict[key2int[key]] = N

# Next, we create a train sampler that only iterates over the respective
# paper training nodes.
paper_idx = local2global['paper']
paper_train_idx = paper_idx[split_idx['train']['paper']]

# NOTE: 在新的图上, bs=1024 会OOM
train_loader = NeighborSampler(edge_index, node_idx=paper_train_idx,
                               sizes=[25, 20], batch_size=bs, shuffle=True,
                               num_workers=0)
                            #    num_workers=10)

full_loader = NeighborSampler(edge_index, node_idx=paper_idx,
                                sizes=[-1, -1], batch_size=bs, shuffle=False, num_workers=0)


class RGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_node_types,
                 num_edge_types):
        super(RGCNConv, self).__init__(aggr='mean')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        self.rel_lins = ModuleList([
            Linear(in_channels, out_channels, bias=False)
            for _ in range(num_edge_types)
        ])

        self.root_lins = ModuleList([
            Linear(in_channels, out_channels, bias=True)
            for _ in range(num_node_types)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.rel_lins:
            lin.reset_parameters()
        for lin in self.root_lins:
            lin.reset_parameters()

    def forward(self, x, edge_index, edge_type, target_node_type):
        x_src, x_target = x

        out = x_target.new_zeros(x_target.size(0), self.out_channels)

        for i in range(self.num_edge_types):
            mask = edge_type == i
            out.add_(self.propagate(edge_index[:, mask], x=x, edge_type=i))

        for i in range(self.num_node_types):
            mask = target_node_type == i
            out[mask] += self.root_lins[i](x_target[mask])

        return out

    def message(self, x_j, edge_type: int):
        return self.rel_lins[edge_type](x_j)


class RGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, num_nodes_dict, x_types, num_edge_types):
        super(RGCN, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout

        node_types = list(num_nodes_dict.keys())
        num_node_types = len(node_types)

        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        # C03: 去除了这里的embedding层
        # # Create embeddings for all node types that do not come with features.
        # self.emb_dict = ParameterDict({
        #     f'{key}': Parameter(torch.Tensor(num_nodes_dict[key], in_channels))
        #     for key in set(node_types).difference(set(x_types))
        # })
        # 改为, 其他的节点, 同一个类型就用一个embedding
        self.emb_dict = ParameterDict({ 
            f'{key}': Parameter(torch.Tensor(1, in_channels))
            for key in set(node_types).difference(set(x_types))
        })

        I, H, O = in_channels, hidden_channels, out_channels  # noqa

        # Create `num_layers` many message passing layers.
        self.convs = ModuleList()
        self.convs.append(RGCNConv(I, H, num_node_types, num_edge_types))
        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(H, H, num_node_types, num_edge_types))
        self.convs.append(RGCNConv(H, O, self.num_node_types, num_edge_types))

        self.reset_parameters()

    def reset_parameters(self):
        for emb in self.emb_dict.values():
            torch.nn.init.xavier_uniform_(emb)
        for conv in self.convs:
            conv.reset_parameters()

    def group_input(self, x_dict, node_type, local_node_idx, n_id=None):
        # Create global node feature matrix.
        if n_id is not None:
            node_type = node_type[n_id]
            local_node_idx = local_node_idx[n_id]

        h = torch.zeros((node_type.size(0), self.in_channels),
                        device=node_type.device)

        for key, x in x_dict.items():
            mask = node_type == key
            h[mask] = x[local_node_idx[mask]]

        # for key, emb in self.emb_dict.items():
        #     mask = node_type == int(key)
        #     h[mask] = emb[local_node_idx[mask]]
        for key, emb in self.emb_dict.items():
            mask = node_type == int(key)
            h[mask] = emb[0]

        return h

    def forward(self, n_id, x_dict, adjs, edge_type, node_type,
                local_node_idx):
        # TODO: 修改这里的函数!!! 从而去掉embedding层!
        x = self.group_input(x_dict, node_type, local_node_idx, n_id)
        node_type = node_type[n_id]

        for i, (edge_index, e_id, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target node embeddings.
            node_type = node_type[:size[1]]  # Target node types.
            conv = self.convs[i]
            x = conv((x, x_target), edge_index, edge_type[e_id], node_type)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)

        return x.log_softmax(dim=-1)

    def inference_fullgraph(self, x_dict, edge_index_dict, key2int):
        # We can perform full-batch inference on GPU.

        device = list(x_dict.values())[0].device

        x_dict = copy(x_dict)
        for key, emb in self.emb_dict.items():
            # x_dict[int(key)] = emb
            # 需要拓展 emb
            x_dict[int(key)] = emb[:].expand((node_type==int(key)).sum(), -1)

        adj_t_dict = {}
        for key, (row, col) in edge_index_dict.items():
            adj_t_dict[key] = SparseTensor(row=col, col=row).to(device)

        for i, conv in enumerate(self.convs):
            out_dict = {}

            for j, x in x_dict.items():
                out_dict[j] = conv.root_lins[j](x)

            for keys, adj_t in adj_t_dict.items():
                src_key, target_key = keys[0], keys[-1]
                out = out_dict[key2int[target_key]]
                tmp = adj_t.matmul(x_dict[key2int[src_key]], reduce='mean')
                out.add_(conv.rel_lins[key2int[keys]](tmp))

            if i != self.num_layers - 1:
                for j in range(self.num_node_types):
                    F.relu_(out_dict[j])

            x_dict = out_dict

        return x_dict

    # C04: 修改为采用minibatch来进行推理
    def inference(self, x_dict, edge_index_dict, key2int):
        res_paper = torch.zeros((num_nodes_dict[key2int['paper']], self.out_channels))
        for batch_size, n_id, adjs in full_loader:
            n_id = n_id.to(device)
            adjs = [adj.to(device) for adj in adjs]
            out = model(n_id, x_dict, adjs, edge_type, node_type, local_node_idx).cpu()
            local_paper_idx = n_id[:batch_size] - local2global['paper'][0]
            res_paper[local_paper_idx] = out
        return {key2int['paper']: res_paper}



device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
# NOTE: 对于tiny, 构造的feat的维度是5 而非128
in_channels = 5 if dname == 'tiny' else 128
model = RGCN(in_channels, args.hidden_channels, dataset.num_classes, args.num_layers,
             args.dropout, num_nodes_dict, list(x_dict.keys()),
             len(edge_index_dict.keys())).to(device)

# Create global label vector.
y_global = node_type.new_full((node_type.size(0), 1), -1)
y_global[local2global['paper']] = data.y_dict['paper']


for name in 'x_dict edge_type node_type local_node_idx y_global'.split():
    x = eval(name)
    size_GB = get_size(x) / 1024**3
    print(f'{name}: {size_GB:.2f} GB')

# Move everything to the GPU.
x_dict = {k: v.to(device) for k, v in x_dict.items()}
edge_type = edge_type.to(device)
node_type = node_type.to(device)
local_node_idx = local_node_idx.to(device)
y_global = y_global.to(device)


def train(epoch):
    model.train()

    pbar = tqdm(total=paper_train_idx.size(0))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = 0
    for batch_size, n_id, adjs in train_loader:
        n_id = n_id.to(device)
        adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()
        # NOTE: 减小 n_id
        # 不能这么写!!! n_id = n_id[:batch_size]
        out = model(n_id, x_dict, adjs, edge_type, node_type, local_node_idx)
        y = y_global[n_id][:batch_size].squeeze()
        loss = F.nll_loss(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        pbar.update(batch_size)

    pbar.close()

    loss = total_loss / paper_train_idx.size(0)

    return loss


@torch.no_grad()
def test():
    model.eval()

    out = model.inference(x_dict, edge_index_dict, key2int)
    out = out[key2int['paper']]

    y_pred = out.argmax(dim=-1, keepdim=True).cpu()
    y_true = data.y_dict['paper']

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']['paper']],
        'y_pred': y_pred[split_idx['train']['paper']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']['paper']],
        'y_pred': y_pred[split_idx['valid']['paper']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']['paper']],
        'y_pred': y_pred[split_idx['test']['paper']],
    })['acc']

    return train_acc, valid_acc, test_acc


test()  # Test if inference on GPU succeeds.
for run in range(args.runs):
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, 1 + args.epochs):
        loss = train(epoch)
        result = test()
        logger.add_result(run, result)
        train_acc, valid_acc, test_acc = result
        print(f'Run: {run + 1:02d}, '
              f'Epoch: {epoch:02d}, '
              f'Loss: {loss:.4f}, '
              f'Train: {100 * train_acc:.2f}%, '
              f'Valid: {100 * valid_acc:.2f}%, '
              f'Test: {100 * test_acc:.2f}%')
    logger.print_statistics(run)
logger.print_statistics()
