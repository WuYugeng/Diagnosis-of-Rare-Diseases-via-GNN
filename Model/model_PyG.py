import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch.nn.init import xavier_uniform_


class GraphSAGEClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dims, embed_dim, num_classes):
        super().__init__()

        # 构建4层GraphSAGE编码器
        self.conv_layers = nn.ModuleList()
        dims = [in_dim] + hidden_dims + [embed_dim]
        for i in range(len(dims) - 1):
            self.conv_layers.append(SAGEConv(dims[i], dims[i + 1]))

        # 分类层权重（参考DiseasesClassifier实现）
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_dim))
        xavier_uniform_(self.weight)  # Xavier初始化

        self.loss_fn = nn.BCEWithLogitsLoss()  # 保持与参考代码一致

    def forward(self, x, edge_index):
        # 多层图卷积
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index)
            if i != len(self.conv_layers) - 1:  # 最后一层不加激活
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)

        # L2归一化（最后一层输出）
        x = F.normalize(x, p=2, dim=-1)

        # 计算内积相似度（与参考代码的矩阵乘法一致）
        logits = torch.mm(x, self.weight.t())
        return logits

    def loss(self, logits, labels):
        return self.loss_fn(logits, labels.float())


class GraphSAGEPredictor:
    def __init__(self, data, hidden_dims=[256, 128, 64], embed_dim=32,
                 num_classes=5, lr=0.01, device='cuda'):
        self.data = data
        self.device = torch.device(device)

        # 初始化模型
        self.model = GraphSAGEClassifier(
            in_dim=data.num_features,
            hidden_dims=hidden_dims,
            embed_dim=embed_dim,
            num_classes=num_classes
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # 将数据移到设备
        self.data = self.data.to(self.device)

    def train_batch(self, batch_nodes):
        self.model.train()
        self.optimizer.zero_grad()

        # 全图前向传播
        logits = self.model(self.data.x, self.data.edge_index)

        # 仅计算batch节点的损失
        loss = self.model.loss(
            logits[self.data.train_mask][batch_nodes],
            self.data.y[self.data.train_mask][batch_nodes]
        )

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, mask):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.data.x, self.data.edge_index)
            probs = torch.sigmoid(logits)

            loss = self.model.loss(logits[mask], self.data.y[mask]).item()
            acc = ((probs[mask] > 0.5) == self.data.y[mask]).float().mean().item()
        return loss, acc

    def run(self, epochs=100, batch_size=512):
        train_nodes = torch.where(self.data.train_mask)[0].tolist()

        for epoch in range(epochs):
            # 随机打乱训练节点
            perm = torch.randperm(len(train_nodes))

            # Mini-batch训练
            total_loss = 0
            for i in range(0, len(train_nodes), batch_size):
                batch_idx = perm[i:i + batch_size]
                batch = [train_nodes[j] for j in batch_idx]

                loss = self.train_batch(batch)
                total_loss += loss

            # 评估
            train_loss, train_acc = self.evaluate(self.data.train_mask)
            test_loss, test_acc = self.evaluate(self.data.test_mask)

            print(f'Epoch: {epoch:03d} | Loss: {total_loss:.4f} | '
                  f'Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}')


# 使用示例
predictor = GraphSAGEPredictor(
    data=data,
    hidden_dims=[256, 128, 64],  # 对应4层结构（输入层 + 3 hidden + 输出层）
    embed_dim=32,
    num_classes=data.y.size(1),
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

predictor.run(epochs=100, batch_size=512)