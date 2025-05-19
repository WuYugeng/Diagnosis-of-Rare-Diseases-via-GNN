import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import torch
from torch.nn import init
from torch import nn
from torch_geometric.nn import HeteroConv, MessagePassing,SAGEConv
from torch_geometric.loader import NeighborLoader
import torch.nn.functional as F
import pickle
from tqdm import tqdm
import json


#先优化性能，然后扩展到大图，然后更改kernel

class DiseasesClassifier(nn.Module):
    def __init__(self, num_classes, embed_dim):
        """输入类别数和嵌入维度，构建分类器"""
        super(DiseasesClassifier, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_dim))
        init.xavier_uniform_(self.weight)  # Xavier初始化

    def forward(self, embeds):
        """输入嵌入向量，输出分类得分"""
        scores = self.weight.mm(embeds.t())  # [num_classes, batch_size]
        return scores.t()  # [batch_size, num_classes]




class GraphSAGE(nn.Module):
    def __init__(self,num_layers,train_enc_dim,num_classes,kernel_params):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                SAGEConv(in_channels=train_enc_dim[i], out_channels=train_enc_dim[i + 1], **kernel_params)
            )
        self.classifier=DiseasesClassifier(embed_dim=train_enc_dim[-1], num_classes=num_classes)

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i != len(self.layers) - 1:  # 非最后一层加ReLU
                x = F.relu(x)

        #L2正则化(不加L2正则化模型表现更佳)
        #x=F.normalize(x, p=2, dim=1)
        scores = self.classifier(x)
        return scores



class DiseasesPredictor:
    def __init__(self,
                 data,
                 num_layers,
                 train_enc_dim,
                 num_neighbors,
                 cuda=True,
                 kernel="SAGE",
                 kernel_params=None,
            ):

        self.device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
        self.data = data
        self.num_neighbors = num_neighbors
        self.train_enc_dim=[self.data.num_features]+train_enc_dim


        if kernel == "SAGE":
            if kernel_params is None:
                kernel_params = {"aggr": "mean", "root_weight": True, "project": False}
            self.gnn=GraphSAGE(num_layers,self.train_enc_dim,data.num_classes,kernel_params)
        elif kernel == "GCN":
            pass

        self.gnn.to(self.device)

    def train(self, epoch=1000, batch_size=200, lr=0.3):
        self.gnn.train()
        #NeighborLoader：用于图神经网络的批处理（生成批次）
        #采样顺序为批次节点到多跳邻居，但GNN中的计算顺序仍为最远邻居到批次节点（顺序穿过GNN的计算层）
        train_loader = NeighborLoader(
            self.data,
            num_neighbors=self.num_neighbors,
            batch_size=batch_size,
            input_nodes=self.data.train_mask,
            shuffle=True,
            # num_workers=4,           # 使用4个进程加载数据
            # persistent_workers=True  # 保持工作进程活动
        )

        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,  self.gnn.parameters()), lr=lr)
        #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.gnn.parameters()), lr=lr,betas=(0.9, 0.99))
        criterion = torch.nn.BCEWithLogitsLoss()

        for epoch in tqdm(range(epoch)):
            if epoch % 100 == 0:
                self.test()
                self.gnn.train()
            for batch in train_loader:
                batch.to(self.device)
                optimizer.zero_grad()
                out = self.gnn(batch.x, batch.edge_index)
                loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
                loss.backward()
                optimizer.step()


    def test(self,top_k=(1, 3, 10),batch_size=200):
        self.gnn.eval()
        test_loader = NeighborLoader(
            self.data,
            num_neighbors=self.num_neighbors,
            batch_size=batch_size,
            input_nodes=self.data.test_mask,
            shuffle=True,
            # num_workers=4,           # 使用4个进程加载数据
            # persistent_workers=True  # 保持工作进程活动
        )
        all_outputs = []
        all_labels = []
        with torch.no_grad():
            for batch in test_loader:
                batch.to(self.device)
                out = self.gnn(batch.x, batch.edge_index)
                all_outputs.append(out[batch.test_mask])
                all_labels.append(batch.y[batch.test_mask])
        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        self.__cumpute_metrics__(all_outputs, all_labels, top_k)


    def __cumpute_ranks__(self,y_pred, y_true):
        targets = torch.mul(y_pred, y_true)
        targets = torch.where(targets == 0, torch.tensor(-float('inf')), targets)
        targets, _ = targets.max(dim=1)
        ranks = (y_pred > targets.unsqueeze(1)).sum(dim=1) + 1
        return ranks
    def __cumpute_metrics__(self,y_pred, y_true, top_k):
        ranks = self.__cumpute_ranks__(y_pred, y_true)
        recalls = {}
        for k in top_k:
            TP = (ranks <= k).sum().item()
            recalls[f'recall@{k}'] = TP / (len(ranks))
        print("中位排名：",ranks.median().item())
        #with open("result.json","w",encoding="utf-8") as f:
            #json.dump(recalls,f,ensure_ascii=False,indent=4)
        for key,value in recalls.items():
            print(f"{key}: {value}")


if __name__ == '__main__':
    with open('C:\internship\codes\my_gnn\Data\pkl_data\graph_data.pkl', 'rb') as f:
        data = pickle.load(f)
    #data.to(torch.device('cuda'))
    model=DiseasesPredictor(
        data=data,
        num_layers=3,
        train_enc_dim=[1000,1000,1000],
        num_neighbors=[5,5,5],
        cuda=True,
        kernel="SAGE",
        kernel_params=None,
    )
    model.train(epoch=2000, batch_size=200, lr=0.3)
    model.test()
