import torch
from torch_geometric.nn import GCNConv
from torch_geometric.loader import NeighborLoader
import torch.nn.functional as F
import pickle
from tqdm import tqdm

#  定义图神经网络模型
class GCN(torch.nn.Module):
    def __init__(self, init_feature_dim, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(init_feature_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        # 第一层GCN
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        # 第二层GCN
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


with open('graph_data.pkl', 'rb') as f:
    data = pickle.load(f)
data.to(torch.device('cuda'))
train_loader = NeighborLoader(
    data,
    num_neighbors=[5],
    batch_size=200,
    input_nodes=data.train_mask,
    shuffle=True,
    #num_workers=4,           # 使用4个进程加载数据
    #persistent_workers=True  # 保持工作进程活动
)
test_loader = NeighborLoader(
    data,
    num_neighbors=[5],
    batch_size=200,
    input_nodes=data.test_mask,
    shuffle=True,
    #num_workers=4,           # 使用4个进程加载数据
    #persistent_workers=True  # 保持工作进程活动
)
model = GCN(init_feature_dim=data.num_features,
            hidden_channels=1000,
            num_classes=data.num_classes,
            )
model.to(torch.device('cuda'))

optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr)


def train():
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        criterion = torch.nn.BCEWithLogitsLoss()
        loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * batch.train_mask.sum()

    return total_loss / data.train_mask.sum()






def cumpute_ranks(y_pred,y_true):
    targets = torch.mul(y_pred, y_true)
    targets = torch.where(targets == 0, torch.tensor(-float('inf')), targets)
    targets ,_=targets.max(dim=1)
    ranks=(y_pred > targets.unsqueeze(1)).sum(dim=1)+1
    return ranks
def cumpute_metrics(y_pred,y_true,top_k=(1 ,3 ,10)):
    ranks=cumpute_ranks(y_pred,y_true)
    recalls = {}
    for k in top_k:
        TP=(ranks<=k).sum().item()
        recalls[f'recall@{k}'] = TP/(len(ranks))
    print(ranks.median().item())
    print(recalls)




def test():
    model.eval()
    all_outputs = []
    all_labels = []
    for batch in test_loader:
        out = model(batch.x, batch.edge_index)
        all_outputs.append(out[batch.test_mask])
        all_labels.append(batch.y[batch.test_mask])
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    cumpute_metrics(all_outputs,all_labels)


for epoch in tqdm(range(1600)):
    loss = train()
test()

