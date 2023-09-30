
## tasks

### 实现 DGL版本

总体目标: 提供DGL版本的数据, 放到 sampler.py 中可以直接使用. 

```python
data.edge_index_dict: {('author','affiliated_with','institution'): Tensor(2,M)}
data.num_nodes_dict: {'paper': 736389}
# N=736389 paper数量
data.x_dict: 只包含 {'paper': Tensor(N,features)}
data.y_dict: 只包含 {'paper': Tensor(N)}
```

### 对齐 tab2graph 和OGB任务设置

这里想要去掉RGCN中的embedding层 (同 tab2graph), 需要看OGB中的训练过程和模型处理. 

```python
    # 模型输入
    for batch_size, n_id, adjs in train_loader:
        out = model(n_id, x_dict, adjs, edge_type, node_type, local_node_idx)

    # 模型处理, 见 [RGCN模型]
```


## OGB

* OGB github: https://github.com/snap-stanford/ogb
* Leaderboard for Node Property Prediction: https://ogb.stanford.edu/docs/leader_nodeprop/


### Quick start

包括了 1] 封装 PyTorch Geometric and DGL data loaders; 2] Evaluators

```python
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader

# Download and process data at './dataset/ogbg_molhiv/'
dataset = PygGraphPropPredDataset(name = 'ogbg-molhiv')

split_idx = dataset.get_idx_split() 
train_loader = DataLoader(dataset[split_idx['train']], batch_size=32, shuffle=True)
valid_loader = DataLoader(dataset[split_idx['valid']], batch_size=32, shuffle=False)
test_loader = DataLoader(dataset[split_idx['test']], batch_size=32, shuffle=False)


from ogb.graphproppred import Evaluator

evaluator = Evaluator(name = 'ogbg-molhiv')
# You can learn the input and output format specification of the evaluator as follows.
# print(evaluator.expected_input_format) 
# print(evaluator.expected_output_format) 
input_dict = {'y_true': y_true, 'y_pred': y_pred}
result_dict = evaluator.eval(input_dict) # E.g., {'rocauc': 0.7321}
```


### OGB基本框架

```python
# examples/nodeproppred/mag/sampler.py

dataset = PygNodePropPredDataset(name='ogbn-mag')
data = dataset[0]       # data: tyg.Data
split_idx = dataset.get_idx_split()
evaluator = Evaluator(name='ogbn-mag')
logger = Logger(args.runs, args)

# 增加双向边!
# edge_index_dict: {edge_type: edge_index}; edge_index是from/to的index
# 例如key/edge_type可以是 ('author', 'affiliated_with', 'institution')
edge_index_dict = data.edge_index_dict

# 将异构图转为统一编码 (**这样可以进行采样!**) 具体见下 [采样过程]
# split_idx: 会对于图结构进行分割!!! 得到 train/valid/test 的index
edge_index, edge_type, node_type, local_node_idx, local2global, key2int = group_hetero_graph(data.edge_index_dict, data.num_nodes_dict)

# 调用pyg的 NeighborSampler. 需要用到 edge_index **所表示的图结构**. 通过制定 node_idx 来选择采样的目标?
train_loader = NeighborSampler(edge_index, node_idx=paper_train_idx,
                               sizes=[25, 20], batch_size=1024, shuffle=True,
                               num_workers=12)


""" 模型!
用到的数据结构: num_nodes_dict 不同类别的节点数量
"""
# def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_nodes_dict, x_types, num_edge_types):
model = RGCN(128, args.hidden_channels, dataset.num_classes, args.num_layers,
             args.dropout, num_nodes_dict, list(x_dict.keys()),
             len(edge_index_dict.keys())).to(device)

""" 训练流程. 调用 model, optimizer
用到的数据: paper_train_idx, x_dict, edge_type, node_type, local_node_idx, y_global
    y_global: 统一节点的标签 (193w) 
"""
def train(epoch):
    pbar = tqdm(total=paper_train_idx.size(0))
    for batch_size, n_id, adjs in train_loader:
        optimizer.zero_grad()
        out = model(n_id, x_dict, adjs, edge_type, node_type, local_node_idx)
        y = y_global[n_id][:batch_size].squeeze()
        loss = F.nll_loss(out, y)
        ...

""" 整体流程. 调用 model, train, test, logger
"""
for run in range(args.runs):
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, 1 + args.epochs):
        loss = train(epoch)
        result = test()
        logger.add_result(run, result)
        train_acc, valid_acc, test_acc = result
        ...
    logger.print_statistics(run)
logger.print_statistics()
```

### RGCN模型

注意到, 在这里先过了一个group_input得到所有节点的hdim表示, 然后分别过了多层的 RGCNConv

```python
# see: examples/nodeproppred/mag/sampler_dgl.py
# model = RGCN(128, args.hidden_channels, dataset.num_classes, args.num_layers,
#              args.dropout, num_nodes_dict, list(x_dict.keys()),
#              len(edge_index_dict.keys())).to(device)
class RGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_nodes_dict, x_types, num_edge_types):
        # 对于非x节点 (除了paper以外的节点), 定义了embedding
        self.emb_dict = ParameterDict({
            f'{key}': Parameter(torch.Tensor(num_nodes_dict[key], in_channels))
            for key in set(node_types).difference(set(x_types))
        })

        # 定义多层的消息传播
        I, H, O = in_channels, hidden_channels, out_channels  # noqa
        # Create `num_layers` many message passing layers.
        self.convs = ModuleList()
        self.convs.append(RGCNConv(I, H, num_node_types, num_edge_types))
        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(H, H, num_node_types, num_edge_types))
        self.convs.append(RGCNConv(H, O, self.num_node_types, num_edge_types))

    def forward(self, n_id, x_dict, adjs, edge_type, node_type, local_node_idx):
        """ 
        输入信息
            adjs: 两层hop的连接关系! 经过两层聚合之后, 得到bs=32的输出
            x_dict: {5: Tensor(N,features)} 其中5号节点是paper
        用到的其他信息

        """
        # 得到 n_id 所表示的
        x = self.group_input(x_dict, node_type, local_node_idx, n_id)
        node_type = node_type[n_id]
        
        # adjs: 两层hop的连接关系! 经过两层聚合之后, 得到bs=32的输出
        for i, (edge_index, e_id, size) in enumerate(adjs):
            """ edge_index, e_id: 采样得到的边! size=[in,out] 表示了两层中节点的数量!  """
            x_target = x[:size[1]]  # Target node embeddings.
            node_type = node_type[:size[1]]  # Target node types.
            conv = self.convs[i]
            x = conv((x, x_target), edge_index, edge_type[e_id], node_type)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)

        return x.log_softmax(dim=-1)

    def group_input(self, x_dict, node_type, local_node_idx, n_id=None):
        # Create global node feature matrix.
        # ... 根据 n_id 得到 node_type, local_node_idx 子集. 
        h = torch.zeros((node_type.size(0), self.in_channels))
        # 1] 对于 x_dict 内的节点, 直接用输入feat
        for key, x in x_dict.items():
            mask = node_type == key
            h[mask] = x[local_node_idx[mask]]
        # 2] 对于其他类型节点, 用embedding
        for key, emb in self.emb_dict.items():
            mask = node_type == int(key)
            h[mask] = emb[local_node_idx[mask]]
        return h
```

内部消息传播

```python
from torch_geometric.nn import MessagePassing
class RGCNConv(MessagePassing):
    def forward(self, x, edge_index, edge_type, target_node_type):
        _, x_target = x # 这里的 edge_index, edge_type 进行表达了图结构; 没用到 x_src 特征. 

        out = x_target.new_zeros(x_target.size(0), self.out_channels)
        # 消息传播. 利用了 MessagePassing 类实现的 propagate 函数
        for i in range(self.num_edge_types):
            mask = edge_type == i
            out.add_(self.propagate(edge_index[:, mask], x=x, edge_type=i))
        # 增加了root节点的信息
        for i in range(self.num_node_types):
            mask = target_node_type == i
            out[mask] += self.root_lins[i](x_target[mask])
```



### 采样过程

```python
from torch_geometric.utils.hetero import group_hetero_graph

""" 
原本 data (graph)
    {'author': 1134649, 'field_of_study': 59965, 'institution': 8740, 'paper': 736389}
得到的
    edge_type, edge_index: [42182144] 这里edge_type用的是global的全局节点编号
    node_type: [1939743] 合并了所有的节点! 
    local_node_idx: [1939743], 注意里面的编码是重复的 0,1,2...
    local2global: {'author': tensor([      0,    ... 1134648]), 'field_of_study': tensor([1134649, 113... 1194613]),} 转为统一的节点编码
    key2int: {key: int} 包括了节点和边!
        例如 ('author', 'affiliated_with', 'institution') -> 0 
        对于节点 'author' -> 0
 """
edge_index, edge_type, node_type, local_node_idx, local2global, key2int = group_hetero_graph(data.edge_index_dict, data.num_nodes_dict)


# pyg/loadder/neighbor_sampler.py
# 调用pyg的 NeighborSampler. 需要用到 edge_index, node_idx **所表示的图结构**
""" https://pytorch-geometric.readthedocs.io/en/latest/modules/sampler.html
来自 "Inductive Representation Learning on Large Graphs" <https://arxiv.org/abs/1706.02216> 
从而实现了 mini-batch training of GNNs on large-scale graphs where full-batch training is not feasible.
    sizes: [25, 20] 每一层的采样数量
    batch_size: 1024
    edge_index: 图结构
    node_idx: The nodes that should be considered for creating mini-batches. 这里指定了仅对于paper!!
 """
train_loader = NeighborSampler(edge_index, node_idx=paper_train_idx,
                               sizes=[25, 20], batch_size=1024, shuffle=True,
                               num_workers=12)
```


### Dataset


```python
# ogb/nodeproppred/dataset_dgl.py
""" 
data.edge_index_dict: 连边关系. 
"""
class DglNodePropPredDataset(object):
    def __init__(self, name, root = 'dataset', meta_dict = None):

    def pre_process(self):

    def get_idx_split(self, split_type = None):
        """ 返回 {split: Tensor} 分割. 
        可以保存为 `split_dict.pt` 直接读取. 或者根据同/异质图手动读取
         """
        # short-cut if split_dict.pt exists
        if os.path.isfile(os.path.join(path, 'split_dict.pt')):
            return torch.load(os.path.join(path, 'split_dict.pt'))
        if self.is_hetero:
            train_idx_dict, valid_idx_dict, test_idx_dict = read_nodesplitidx_split_hetero(path)
        else:
            train_idx = torch.from_numpy(pd.read_csv(osp.join(path, 'train.csv.gz'), compression='gzip', header = None).values.T[0]).to(torch.long)
            ...
            return {'train': train_idx, 'valid': valid_idx, 'test': test_idx}

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph[idx], self.labels
```

```python
# ogb/nodeproppred/dataset_pyg.py
class PygNodePropPredDataset(InMemoryDataset):
    def __init__(self, name, root = 'dataset', transform=None, pre_transform=None, meta_dict = None):

```



