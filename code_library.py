

















#生成map.pkl
import json
import pickle
# 患者
with open(r'C:\internship\codes\Disease-Prediction-via-GCN-main\data\PhenoBrain_data\merged_RD.json', 'r') as file1:
    cases = json.load(file1)
    # 在这里处理文件内容

all_the_disease=[]
all_the_HPO=[]
for HPO_list,disease_list in cases:
    all_the_disease+=disease_list
    all_the_HPO+=HPO_list

#疾病
all_the_disease=set(all_the_disease)
all_the_HPO=set(all_the_HPO)

with open(r'C:\internship\codes\timgroup_disease_diagnosis-main\codes\core\RD_to_hpo_dict.json', 'r') as file1:
    RD_to_HPO_dict = json.load(file1)

HPO_set=list()
for disease in all_the_disease:
	HPO_set+=RD_to_HPO_dict[disease]
HPO_set=set(HPO_set)


i=0
node_map=dict()
for case in cases:
    node_map[str(i)]=i
    i+=1

for disease in all_the_disease:
    node_map[disease]=i
    i+=1


for HPO in HPO_set|all_the_HPO:
    node_map[HPO]=i
    i+=1

print(node_map)
file_path=r".\graph.map.pkl"
with open(file_path, 'wb') as f:
    pickle.dump(node_map, f)  # 序列化并写入文件









#构造邻接字典
import json
import pickle
from collections import defaultdict
node_map = pickle.load(open( "graph.map.pkl", "rb"))
# 患者
with open(r'C:\internship\codes\Disease-Prediction-via-GCN-main\data\PhenoBrain_data\merged_RD.json', 'r') as file1:
    cases = json.load(file1)
edges=list()
i=0
for HPO_list,disease_list in cases:
    for HPO in HPO_list:
        edges.append((node_map[str(i)],node_map[HPO]))
        edges.append((node_map[HPO],node_map[str(i)]))
    i+=1


#疾病
all_the_disease=[]
for HPO_list,disease_list in cases:
    all_the_disease+=disease_list
all_the_disease=set(all_the_disease)
with open(r'C:\internship\codes\timgroup_disease_diagnosis-main\codes\core\RD_to_hpo_dict.json', 'r') as file1:
    RD_to_HPO_dict = json.load(file1)

for disease in all_the_disease:
    HPO_list=RD_to_HPO_dict[disease]
    for HPO in HPO_list:
        edges.append((node_map[disease],node_map[HPO]))
        edges.append((node_map[HPO],node_map[disease]))


graph = defaultdict(set)
for u, v in edges:
    graph[u].add(v)  # 直接追加，无需初始化空列表



file_path=r".\with_disease_graph.adj.pkl"
with open(file_path, 'wb') as f:
    pickle.dump(graph, f)  # 序列化并写入文件














#常规训练集转RD训练集
import json
file_path1 = r"C:\internship\codes\GNN\data\gnn_data\test_data\our_test_data.json"  # 替换为实际路径
with open(file_path1, 'r', encoding='utf-8') as f:
    cases = json.load(f)
file_path=r"C:\internship\codes\GNN\data\gnn_data\9260_rd_to_hpo_dict.json"
with open(file_path, 'r', encoding='utf-8') as f:
     dis_to_rd= json.load(f)

print(len(cases))
result_list=[]
for hpo_list,dis_list in cases:
    new_dis_list=[dis_to_rd[dis]for dis in dis_list]
    new_dis_list=list(set(new_dis_list))
    result_list.append([hpo_list,new_dis_list])




with open(file_path1, 'w', encoding='utf-8') as f:
    json.dump(result_list, f, ensure_ascii=False, indent=4)



















#划分训练集和测试集
import pickle

node_map = pickle.load(open( "graph.map.pkl", "rb"))
node_list = pickle.load(open( "graph.nodes.pkl", "rb"))
# 初始化分类结果
hp_strings = []  # 存储以 "HP:" 开头的字符串
numeric_strings = []  # 存储纯数字字符串
used_diseases = []  # 存储其他字符串

for s in node_list:
    if s.startswith("HP:"):
        hp_strings.append(s)
    elif s.isdigit():  # 判断是否为纯数字字符串
        numeric_strings.append(s)
    elif s.startswith("RD:"):
        used_diseases.append(s)


patient_node_int=[node_map[key] for key in numeric_strings]
diseases_node_int=[node_map[key] for key in used_diseases]

import random
def split_list_1_to_9(input_list):
    # 1. 打乱列表顺序（原地修改）
    shuffled_list = input_list.copy()  # 避免修改原列表
    random.shuffle(shuffled_list)

    # 2. 计算划分点（10% 的位置）
    split_idx = int(len(shuffled_list) * 0.2)

    # 3. 划分为两部分
    part1 = shuffled_list[:split_idx]  # 前 10%
    part2 = shuffled_list[split_idx:]  # 后 90%

    return part1, part2
test, train=split_list_1_to_9(patient_node_int)
train=train+diseases_node_int
print(len(train))
print(len(test))

file_path=r".\graph.train.pkl"
with open(file_path, 'wb') as f:
    pickle.dump(train, f)

file_path=r".\graph.test.pkl"
with open(file_path, 'wb') as f:
    pickle.dump(test, f)





#读取obo文件
import obonet

# 读取 .obo 文件
file_path = r"C:\internship\codes\timgroup_disease_diagnosis-main\codes\core\data\raw\HPO\2019\Ontology\hp.obo"  # 替换为实际路径
with open('file.txt', 'r', encoding='utf-8') as f:
    graph = obonet.read_obo(f)

# 打印图的基本信息
print(f"Number of nodes: {len(graph)}")  # 节点数量
print(f"Number of edges: {graph.number_of_edges()}")  # 边数量
















#生成9260_rd_to_hpo_dict.json
import json

default_prob=0.545

file_path = r"C:\internship\codes\GNN\data\gnn_data\9260_rd_to_hpo_dict.json"  # 替换为实际路径
with open(file_path, 'r', encoding='utf-8') as f:
    rd_to_hpo_dict = json.load(f)

file_path = r"C:\internship\codes\GNN\data\gnn_data\PhenoBrain\rd_dict.json"  # 替换为实际路径
with open(file_path, 'r', encoding='utf-8') as f:
    rd_dict = json.load(f)

file_path = r"C:\internship\codes\GNN\data\gnn_data\PhenoBrain\dis_to_hpo_prob_hpoa_None.json"  # 替换为实际路径
with open(file_path, 'r', encoding='utf-8') as f:
    dis_to_hpo_prob = json.load(f)

def get_prob(rd_code,hpo):
    source_codes = rd_dict[rd_code]["SOURCE_CODES"]
    prob_list = []
    for source_code in source_codes:
        if source_code in dis_to_hpo_prob:
            if hpo in dis_to_hpo_prob[source_code]:
                prob_list.append(dis_to_hpo_prob[source_code][hpo])
    filtered_list = [x for x in prob_list if x is not None]
    if len(filtered_list) == 0:
        return default_prob
    else:
        return min(filtered_list)

dis_tp_hpo_prob=dict()
for rd_code,HPO_list in rd_to_hpo_dict.items():
    dis_tp_hpo_prob[rd_code]=dict()
    for hpo in HPO_list:
        dis_tp_hpo_prob[rd_code][hpo]=get_prob(rd_code,hpo)
#print(dis_tp_hpo_prob)

file_path = r"C:\internship\codes\GNN\data\gnn_data\9260_rd_to_hpo_prob.json"  # 替换为实际路径
with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(dis_tp_hpo_prob, f, ensure_ascii=False, indent=4)











#常规训练集转RD训练集
import json
file_path1 = r"C:\internship\codes\GNN\data\gnn_data\train_data\our_train_data.json"  # 替换为实际路径
with open(file_path1, 'r', encoding='utf-8') as f:
    cases = json.load(f)
file_path=r"C:\internship\codes\GNN\data\gnn_data\dis_to_9260_RD_dict.json"
with open(file_path, 'r', encoding='utf-8') as f:
     dis_to_rd= json.load(f)

print(len(cases))
result_list=[]
for hpo_list,dis_list in cases:
    new_dis_list=[dis_to_rd[dis]for dis in dis_list]
    new_dis_list=list(set(new_dis_list))
    result_list.append([hpo_list,new_dis_list])




with open(file_path1, 'w', encoding='utf-8') as f:
    json.dump(result_list, f, ensure_ascii=False, indent=4)
















import torch
from Model.model_PyG import DiseasesPredictor
import time
import pickle
from torch_geometric.data import Data

file_path = "C:\internship\codes\Disease-Prediction-via-GCN-main\data\sample_data\sample_garph"
adj_lists = pickle.load(open(file_path + ".adj.pkl", "rb"))
labels = pickle.load(open(file_path + ".label.pkl", "rb"))
train = pickle.load(open(file_path + ".train.pkl", "rb"))
test = pickle.load(open(file_path + ".test.pkl", "rb"))
labels = torch.from_numpy(labels)
"""adj_lists = pickle.load(open("adj_list.pkl", "rb"))
labels = pickle.load(open("labels.pkl", "rb"))
train = pickle.load(open("train_data.pkl", "rb"))
test = pickle.load(open("test_data.pkl", "rb"))"""

# 构造 edge_index
edges = []
for src, neighbors in adj_lists.items():
    for dst in neighbors:
        edges.append([src, dst])
edge_index = torch.int.tensor(edges, dtype=torch.int).t().contiguous()

multi_class_num = 108
feature_dim = 10000
epochs = 8000
batch_size = 200
lr = 0.3
feature_data = torch.randn(10000, feature_dim)  # 10000个节点，每个节点feature_dim维特征
train_enc_dim = [1000, 1000, 1000, 1000]
t1 = time.time()

print(type(feature_data))
print(type(edge_index))
print(type(labels))
data = Data(
    x=feature_data,
    edge_index=edge_index,
    y=labels
)

# 初始化预测器
predictor = DiseasesPredictor(
    data=data,
    num_classes=multi_class_num,
    enc_dims=train_enc_dim,  # 2层GNN
    train_idx=train,
    test_idx=test,
    num_samples=[5,5],  # 每层采样数
    cuda=True,
    batch_size=batch_size,
)

# 训练和评估
predictor.train(epochs=epochs, lr=lr)
predictor.evaluate()















#生成并保存用于图神经网络训练的图数据
import pickle
from torch_geometric.data import Data
import torch
#from Model.model_PyG import DiseasesPredictor
from torch_geometric.loader import NeighborLoader

file_path = "C:\internship\codes\Disease-Prediction-via-GCN-main\data\sample_data\sample_garph"
adj_lists = pickle.load(open(file_path + ".adj.pkl", "rb"))
labels = pickle.load(open(file_path + ".label.pkl", "rb"))
train_ids = pickle.load(open(file_path + ".train.pkl", "rb"))
test_ids = pickle.load(open(file_path + ".test.pkl", "rb"))




"""
node_feature_matrix:float32 tensor
labels:float32 tensor
edge_index:int64 tensor
train_mask|test_mask:bool tensor
"""
edge_index = []
for node, adj_nodes in adj_lists.items():
    for adj_node in adj_nodes:
        edge_index.append([node, adj_node])  # 每条边格式 [src, dst]
edge_index = torch.tensor(edge_index, dtype=torch.int64).t().contiguous()

init_feature_dim = 10000
num_nodes=1008
node_feature_matrix = torch.rand((num_nodes, init_feature_dim), dtype=torch.float32)


labels=torch.tensor(labels, dtype=torch.float32)
num_new_rows = num_nodes - len(labels)
nan_padding = torch.full((num_new_rows, len(labels[0])), float('nan'), dtype=torch.float32)
labels = torch.cat([labels, nan_padding], dim=0)


train_mask = torch.isin(torch.arange(num_nodes), torch.tensor(train_ids))
test_mask = torch.isin(torch.arange(num_nodes), torch.tensor(test_ids))

data=Data(
    x=node_feature_matrix,
    edge_index=edge_index,
    y=labels,
    train_mask=train_mask,
    test_mask=test_mask,
    num_classes=108,
    num_nodes=num_nodes,
    num_features=init_feature_dim,
)

# 保存到文件
with open('graph_data.pkl', 'wb') as f:
    pickle.dump(data, f)















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

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
