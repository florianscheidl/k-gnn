import os
import os.path as osp
import sys

sys.path.append(os.getcwd())


import wandb
import argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
import torch_geometric.transforms as T
from sklearn.model_selection import train_test_split

from k_gnn import DataLoader, GraphConv
import k_gnn.pool as pool
from pyg_loader import load_pyg, load_ogb
from k_gnn import TwoMalkin, ConnectedThreeMalkin
from get_parser import get_parser

# Specify all the arguments (that wandb will use later) here.
parser = get_parser()
args = parser.parse_args()

# Set the seed for everything
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# TODO: We omit the pre_filter!
class MyFilter(object):
    def __call__(self, data):
        return data.num_nodes <= 70

class MyPreTransformNoFeatures(object):

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, data):
        data.x = torch.zeros((data.num_nodes, 1), dtype=torch.float)
        data = TwoMalkin()(data)
        data = ConnectedThreeMalkin()(data)
        data.x = degree(data.edge_index[0], data.num_nodes, dtype=torch.long) # use degree instead of one-hot encoding of degree.
        # print("NoFeatureTransform, data.x: ", data.x)
        data.x = F.one_hot(data.x, num_classes=self.num_classes).to(torch.float)
        setattr(data, '_real_num_node_features', self.num_classes)
        assert(data._real_num_node_features != 0.0)
        return data

# class PROTEINS_Filter(object): # TODO: This was provided by the authors of k-GNN, needs to be investigated.
#     def __call__(self, data):
#         return not (data.num_nodes == 7 and data.num_edges == 12) and \
#             data.num_nodes < 450

BATCH = args.batch_size
path = osp.join(
    osp.dirname(osp.realpath(__file__)), '..', 'data', '1_2_3_gnn', args.dataset)

# load and transform dataset

if args.dataset.startswith('TU_REDDIT'):
    pre_transform = MyPreTransformNoFeatures(num_classes=8000)
elif args.dataset.startswith('TU_IMDB'):
    pre_transform = MyPreTransformNoFeatures(num_classes=352)
else:
    pre_transform=T.Compose([TwoMalkin(), ConnectedThreeMalkin()])

if args.data_format == 'PyG':
    dataset = load_pyg(dataset_dir=path, name=args.dataset, pre_transform=pre_transform)
elif args.data_format == 'ogb':
    dataset = load_ogb(dataset_dir=path, name=args.dataset, pre_transform=pre_transform)
else:
    raise ValueError('Unknown data format: {}'.format(args.data_format))

if not hasattr(dataset.data,'_real_num_node_features'):
    print("lost num featuers after pre_transform.")

perm = torch.randperm(len(dataset), dtype=torch.long)
dataset = dataset[perm]

dataset.data.iso_type_2 = torch.unique(dataset.data.iso_type_2, True, True)[1]
num_i_2 = dataset.data.iso_type_2.max().item() + 1
dataset.data.iso_type_2 = F.one_hot(
    dataset.data.iso_type_2, num_classes=num_i_2).to(torch.float)

dataset.data.iso_type_3 = torch.unique(dataset.data.iso_type_3, True, True)[1]
num_i_3 = dataset.data.iso_type_3.max().item() + 1
dataset.data.iso_type_3 = F.one_hot(
    dataset.data.iso_type_3, num_classes=num_i_3).to(torch.float)

num_i = [args.initial_emb_dim, num_i_2+args.emb_dim, num_i_3+args.emb_dim]
# print("num_i", num_i)


class Net(torch.nn.Module):
    def __init__(self, in_dataset):
        super(Net, self).__init__()

        # initial layer
        if hasattr(in_dataset.data, '_real_num_node_features'):
            setattr(in_dataset,'num_node_features', in_dataset.data._real_num_node_features)
        else:
            print("No _real_num_node_features attribute found in in_dataset.data.")
        assert (in_dataset.data.num_node_features != 0.0)
        setattr(self,
                'conv_initial',
                GraphConv(in_dataset.data.num_node_features, args.initial_emb_dim))

        # args.num_layers_per_dim layers per dimension j
        for j in range(args.max_k):
            setattr(self,
                    'conv{}_{}'.format(j, 0),
                    GraphConv(num_i[j], args.emb_dim))
            for i in range(1,args.num_layers_per_dim):
                setattr(self,
                        'conv{}_{}'.format(j, i),
                        GraphConv(args.emb_dim, args.emb_dim))

        # final_classification layers
        setattr(self, 'fc0', torch.nn.Linear(args.max_k*args.emb_dim, args.emb_dim))
        for l in range(1, args.num_linear_layers-1):
            setattr(self, 'fc{}'.format(l), torch.nn.Linear(int(args.emb_dim/(2**(l-1))), int(args.emb_dim/(2**l))))
        if args.num_linear_layers > 1:
            setattr(self, 'fc{}'.format(args.num_linear_layers-1), torch.nn.Linear(int(args.emb_dim/(2**(args.num_linear_layers-2))), in_dataset.num_classes))

    def reset_parameters(self):
        for (name, module) in self._modules.items():
            module.reset_parameters()

    def forward(self, data):
        non_linearity = getattr(F, args.nonlinearity)
        x_per_dim = []

        # initial layer
        data.x = non_linearity(getattr(self, 'conv_initial')(data.x, data.edge_index))

        for j in range(args.max_k):
            for i in range(args.num_layers_per_dim):
                if j==0:
                    data.x = non_linearity(getattr(self, 'conv{}_{}'.format(j, i))(data.x, data.edge_index))
                else:
                    data.x = non_linearity(getattr(self, 'conv{}_{}'.format(j, i))(data.x, getattr(data, 'edge_index_{}'.format(j+1))))
            x = data.x
            if j==0:
                x_per_dim.append(scatter_mean(data.x, getattr(data, 'batch'), dim=0))
            else:
                x_per_dim.append(scatter_mean(data.x, getattr(data, f'batch_{j+1}'), dim=0))
            if j<args.max_k-1:
                pool_func = getattr(pool, f'{args.pool_func}')
                data.x = pool_func(x, getattr(data, f'assignment_index_{j+2}'))
                data.x = torch.cat([data.x, getattr(data, f'iso_type_{j+2}')], dim=1)
            else:
                x = torch.cat(x_per_dim, dim=1)

        if args.no_train:
            x = x.detach()

        for l in range(args.num_linear_layers-1):
            x = non_linearity(getattr(self, 'fc{}'.format(l))(x))
            if l<args.num_linear_layers-2:
                x = F.dropout(x, p=args.drop_rate, training=self.training)
        x = getattr(self, 'fc{}'.format(args.num_linear_layers-1))(x)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(in_dataset=dataset).to(device)


def train(epoch, loader, optimizer):
    model.train()
    loss_all = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(loader.dataset)


def val(loader):
    model.eval()
    loss_all = 0

    for data in loader:
        data = data.to(device)
        loss_all += F.nll_loss(model(data), data.y, reduction='sum').item()
    return loss_all / len(loader.dataset)


def test(loader):
    model.eval()
    correct = 0

    for data in loader:
        data = data.to(device)
        pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


acc = []
wandb.init()
for i in range(args.num_repeats):
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, gamma=args.lr_scheduler_decay_rate, step_size=args.lr_scheduler_decay_steps)

    if hasattr(dataset, 'test_mask'):
        test_mask = dataset.test_mask
        test_dataset = dataset[test_mask]
        split_train_dataset = dataset[~test_mask]
    elif hasattr(dataset, 'test_graph_index'):
        test_mask = dataset.test_graph_index
        test_dataset = dataset[test_mask]
        split_train_dataset = dataset[~test_mask]

    if hasattr(dataset, 'val_mask'):
        val_mask = dataset.val_mask
        val_dataset = split_train_dataset[val_mask]
        train_dataset = split_train_dataset[~val_mask]
    elif hasattr(dataset, 'val_graph_index'):
        val_mask = dataset.val_graph_index
        val_dataset = split_train_dataset[val_mask]
        train_dataset = split_train_dataset[~val_mask]

    if not ((hasattr(dataset, 'test_mask') or hasattr(dataset, 'test_graph_index')) and (hasattr(dataset, 'val_mask') or hasattr(dataset, 'val_graph_index'))):

        # define a random train, validation and test mask
        [train_ratio, val_ratio, test_ratio] = args.data_split

        print(f'No predefined split, creating random train/test/val split with ratio {train_ratio}/{test_ratio}/{val_ratio}')

        # create train, val and test mask
        test_val_mask = torch.full(size=(len(dataset),), fill_value=test_ratio+val_ratio, dtype=torch.float).bernoulli().bool()
        test_val_dataset = dataset[test_val_mask]
        test_mask = torch.full(size=(len(test_val_dataset),), fill_value=test_ratio/(test_ratio+val_ratio), dtype=torch.float).bernoulli().bool()
        test_dataset = test_val_dataset[test_mask]
        val_dataset = test_val_dataset[~test_mask]
        train_dataset = dataset[~test_val_mask]

        print("Length dataset:", len(dataset),"Length test_dataset:", len(test_dataset),"Length val_dataset:", len(val_dataset),"Length train_dataset:", len(train_dataset))
        # print("Size attributes dataset:", dataset.data.x.size(),"Size attributes test_dataset:", test_dataset.data.x.size(),"Size attributes val_dataset:", val_dataset.data.x.size(),"Size attributes train_dataset:", train_dataset.data.x.size())
        # this would be 10-fold CV:

        # test_mask = torch.zeros(len(dataset), dtype=torch.bool)
        # n = len(dataset) // 10
        # test_mask[i * n:(i + 1) * n] = 1

    val_loader = DataLoader(val_dataset, batch_size=BATCH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH)
    # print("INVESTIGATE TEST LOADER: ", test_loader.dataset.data.y, test_loader.dataset.data.x)
    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)

    print('---------------- Split {} ----------------'.format(i))

    best_val_loss, test_acc = 100, -1
    for epoch in range(1, args.epochs + 1):
        lr = scheduler.get_last_lr()[0]
        train_loss = train(epoch, train_loader, optimizer)
        val_loss = val(val_loader)
        scheduler.step(val_loss)
        if best_val_loss >= val_loss:
            # print("Computing test accuracy")
            test_acc = test(test_loader)
            best_val_loss = val_loss
        print('Epoch: {:03d}, LR: {:7f}, Train Loss: {:.7f}, '
              'Val Loss: {:.7f}, Test Acc: {:.7f}'.format(
                  epoch, lr, train_loss, val_loss, test_acc))
        wandb.log({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'test_acc': test_acc, 'lr': lr})
    acc.append(test_acc)
acc = torch.tensor(acc)
print('---------------- Final Result ----------------')
print('Mean: {:7f}, Std: {:7f}'.format(acc.mean(), acc.std()))
