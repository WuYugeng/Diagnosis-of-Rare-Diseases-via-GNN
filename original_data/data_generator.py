#生成模拟数据
import random
random.seed(42)
from tqdm import tqdm
import json
import os
from collections import defaultdict
import pickle
from hpo_reader import hpo_reader
current_dir = os.path.dirname(os.path.abspath(__file__))
from torch_geometric.data import Data
import torch


def generate_simulated_data(number_per_dis=10):
    # 获取文件所在目录
    file_path = os.path.join(current_dir,"9260_rd_to_hpo_prob.json")
    with open(file_path, 'r', encoding='utf-8') as f:
        rd_to_hpo_prob = json.load(f)

    simulated_data=[]
    for rd_code in tqdm(rd_to_hpo_prob.keys()):
        hpo_to_prob = rd_to_hpo_prob[rd_code]
        for i in range(number_per_dis):
            if len(hpo_to_prob) <=3:
                simulated_data.append([list(hpo_to_prob.keys()), [rd_code]])
            else:
                new_hpo_list = []
                while len(new_hpo_list)<3:
                    new_hpo_list = []
                    for hpo,prob in hpo_to_prob.items():
                        prob*=100
                        num=random.randint(0, 100)  # 包含0和100
                        if num<=prob:
                            new_hpo_list.append(hpo)
                simulated_data.append([new_hpo_list,[rd_code]])

    file_path = os.path.join(current_dir,"train_data","simulated_data.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(simulated_data, f, ensure_ascii=False, indent=4)




def merge_json_lists_from_folder(folder_path):
    """快速合并文件夹中所有JSON文件内的列表（假设所有文件格式正确）"""
    merged = []
    for file in os.listdir(folder_path):
        if file.endswith('.json'):
            with open(os.path.join(folder_path, file), 'r') as f:
                merged.extend(json.load(f))
    return merged
def generate_pkl_from_source_data(init_feature_dim):
    """通过test_data文件夹和train_data文件夹中的文件生成PyG的Data对象，
    该对象的结构如下：
    Data(
        x=node_feature_matrix,
        edge_index=edge_index,
        y=labels,
        train_mask=train_mask,
        test_mask=test_mask,
        num_classes=9260,
        num_nodes=num_nodes,
        num_features=init_feature_dim,
    )
    各成员的数据类型如下：
    node_feature_matrix:float32 tensor
    labels:float32 tensor
    edge_index:int64 tensor
    train_mask|test_mask:bool tensor
    others:int
    —"""
    train_data = merge_json_lists_from_folder(os.path.join(current_dir,"train_data"))
    test_data = merge_json_lists_from_folder(os.path.join(current_dir,"test_data"))
    with open("9260_rd_to_index.json", 'r', encoding="utf-8") as f:
        rd_to_index = json.load(f)
    reader = hpo_reader()
    hpo_adj_list = reader.get_adj_list()
    hpo_list = reader.get_hpo_list()

    hpo_to_index = dict()
    i = len(test_data) + len(train_data)
    for hpo in hpo_list:
        hpo_to_index[hpo] = i
        i += 1

    num_nodes=i
    adj_list = defaultdict(set)
    train_data_id = []
    test_data_id = []

    labels = torch.zeros(size=(num_nodes, 9260), dtype=torch.float32)
    node_feature_matrix = torch.rand((num_nodes, init_feature_dim), dtype=torch.float32)

    case_id = 0
    for hpo_list, rd_list in train_data:
        for rd in rd_list:
            labels[case_id][rd_to_index[rd]] = 1
        train_data_id.append(case_id)

        for hpo in hpo_list:
            adj_list[hpo_to_index[hpo]].add(case_id)
            adj_list[case_id].add(hpo_to_index[hpo])

        case_id += 1

    for hpo_list, rd_list in test_data:
        for rd in rd_list:
            labels[case_id][rd_to_index[rd]] = 1
        test_data_id.append(case_id)

        for hpo in hpo_list:
            adj_list[hpo_to_index[hpo]].add(case_id)
            adj_list[case_id].add(hpo_to_index[hpo])

        case_id += 1
    # labels为训练集和测试集的标签矩阵
    # train_data_id为训练样本行索引，test_data_id为测试样本行索引

    for hpo, hpo_list in hpo_adj_list.items():
        for father_hpo in hpo_list:
            pass
            #adj_list[hpo_to_index[father_hpo]].add(hpo_to_index[hpo])
            #adj_list[hpo_to_index[hpo]].add(hpo_to_index[father_hpo])


    edge_index = []
    for node, adj_nodes in adj_list.items():
        for adj_node in adj_nodes:
            edge_index.append([node, adj_node])  # 每条边格式 [src, dst]
    edge_index = torch.tensor(edge_index, dtype=torch.int64).t().contiguous()

    train_mask = torch.isin(torch.arange(num_nodes), torch.tensor(train_data_id))
    test_mask = torch.isin(torch.arange(num_nodes), torch.tensor(test_data_id))

    data = Data(
        x=node_feature_matrix,
        edge_index=edge_index,
        y=labels,
        train_mask=train_mask,
        test_mask=test_mask,
        num_classes=9260,
        num_nodes=num_nodes,
        num_features=init_feature_dim,
    )
    with open(os.path.join("pkl_data",'graph_data.pkl'), "wb") as f:  # 注意模式是二进制写入（'wb'）
        pickle.dump(data, f)










def generate_pkl_from_source_data_test(init_feature_dim):
    """通过test_data文件夹和train_data文件夹中的文件生成PyG的Data对象，
    该对象的结构如下：
    Data(
        x=node_feature_matrix,
        edge_index=edge_index,
        y=labels,
        train_mask=train_mask,
        test_mask=test_mask,
        num_classes=9260,
        num_nodes=num_nodes,
        num_features=init_feature_dim,
    )
    各成员的数据类型如下：
    node_feature_matrix:float32 tensor
    labels:float32 tensor
    edge_index:int64 tensor
    train_mask|test_mask:bool tensor
    others:int
    —"""
    train_data = merge_json_lists_from_folder(os.path.join(current_dir,"train_data"))
    test_data = merge_json_lists_from_folder(os.path.join(current_dir,"test_data"))
    train_data=train_data[:500]
    test_data = test_data[:500]
    with open("9260_rd_to_index.json", 'r', encoding="utf-8") as f:
        rd_to_index = json.load(f)
    reader = hpo_reader()
    hpo_adj_list = reader.get_adj_list()
    hpo_list = reader.get_hpo_list()

    hpo_to_index = dict()
    i = len(test_data) + len(train_data)
    for hpo in hpo_list:
        hpo_to_index[hpo] = i
        i += 1

    num_nodes=i
    num_classes=108
    adj_list = defaultdict(set)
    train_data_id = []
    test_data_id = []

    labels = torch.zeros(size=(num_nodes, num_classes), dtype=torch.float32)
    node_feature_matrix = torch.rand((num_nodes, init_feature_dim), dtype=torch.float32)

    case_id = 0
    for hpo_list, rd_list in train_data:
        for rd in rd_list:
            labels[case_id][rd_to_index[rd]] = 1
        train_data_id.append(case_id)

        for hpo in hpo_list:
            adj_list[hpo_to_index[hpo]].add(case_id)
            adj_list[case_id].add(hpo_to_index[hpo])

        case_id += 1

    for hpo_list, rd_list in test_data:
        for rd in rd_list:
            labels[case_id][rd_to_index[rd]] = 1
        test_data_id.append(case_id)

        for hpo in hpo_list:
            adj_list[hpo_to_index[hpo]].add(case_id)
            adj_list[case_id].add(hpo_to_index[hpo])

        case_id += 1
    # labels为训练集和测试集的标签矩阵
    # train_data_id为训练样本行索引，test_data_id为测试样本行索引

    for hpo, hpo_list in hpo_adj_list.items():
        for father_hpo in hpo_list:
            pass
            #adj_list[hpo_to_index[father_hpo]].add(hpo_to_index[hpo])
            #adj_list[hpo_to_index[hpo]].add(hpo_to_index[father_hpo])


    edge_index = []
    for node, adj_nodes in adj_list.items():
        for adj_node in adj_nodes:
            edge_index.append([node, adj_node])  # 每条边格式 [src, dst]
    edge_index = torch.tensor(edge_index, dtype=torch.int64).t().contiguous()

    train_mask = torch.isin(torch.arange(num_nodes), torch.tensor(train_data_id))
    test_mask = torch.isin(torch.arange(num_nodes), torch.tensor(test_data_id))

    data = Data(
        x=node_feature_matrix,
        edge_index=edge_index,
        y=labels,
        train_mask=train_mask,
        test_mask=test_mask,
        num_classes=num_classes,
        num_nodes=num_nodes,
        num_features=init_feature_dim,
    )
    with open(os.path.join("pkl_data",'graph_data.pkl'), "wb") as f:  # 注意模式是二进制写入（'wb'）
        pickle.dump(data, f)







import numpy as np
from typing import Tuple, List

def mixup(
        X: np.ndarray,
        y: np.ndarray,
        alpha: float = 0.2,
        num_augmented: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mixup数据增强：通过线性插值生成新样本。

    参数:
        X (np.ndarray): 原始特征矩阵 (num_samples, num_features)，多热编码的病症向量。
        y (np.ndarray): 原始标签 (num_samples,)，疾病类别编号。
        alpha (float): Beta分布的参数，控制插值强度（建议0.1~0.4）。
        num_augmented (int): 每个原始样本生成的新样本数量。

    返回:
        X_mixed (np.ndarray): 增强后的特征矩阵。
        y_mixed (np.ndarray): 增强后的标签（软标签或原始标签）。
    """
    X_mixed, y_mixed = [], []
    n_samples = X.shape[0]

    for _ in range(num_augmented):
        # 随机选择两个样本
        idx1, idx2 = np.random.choice(n_samples, 2, replace=False)

        # 从Beta分布生成插值系数lambda
        lam = np.random.beta(alpha, alpha)

        # 线性插值
        mixed_x = lam * X[idx1] + (1 - lam) * X[idx2]

        # 软标签（可选：直接使用原始标签时改为 y[idx1]）
        mixed_y = lam * y[idx1] + (1 - lam) * y[idx2]

        X_mixed.append(mixed_x)
        y_mixed.append(mixed_y)

    # 合并原始数据与增强数据
    X_mixed = np.concatenate([X, np.array(X_mixed)], axis=0)
    y_mixed = np.concatenate([y, np.array(y_mixed)], axis=0)

    return X_mixed, y_mixed


import numpy as np
from typing import Tuple, List, Dict


def random_perturbation(
        X: np.ndarray,
        y: np.ndarray,
        phenotype_hierarchy: Dict[int, List[int]],
        noise_prob: float = 0.3,
        ancestor_prob: float = 0.5,
        num_augmented: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Random Perturbation数据增强：通过替换部分病症为祖先节点或噪声生成新样本。

    参数:
        X (np.ndarray): 原始特征矩阵 (num_samples, num_features)，多热编码的病症向量。
        y (np.ndarray): 原始标签 (num_samples,)，疾病类别编号。
        phenotype_hierarchy (Dict[int, List[int]]): 病症层级关系，键为病症ID，值为祖先节点列表。
        noise_prob (float): 每条病症被替换的总概率（默认0.3）。
        ancestor_prob (float): 在替换时选择祖先节点（而非噪声）的概率（默认0.5）。
        num_augmented (int): 每个原始样本生成的新样本数量。

    返回:
        X_perturbed (np.ndarray): 增强后的特征矩阵。
        y_perturbed (np.ndarray): 增强后的标签（与原始标签相同）。
    """
    X_perturbed, y_perturbed = [], []
    n_samples, n_phenotypes = X.shape

    for _ in range(num_augmented):
        for i in range(n_samples):
            x_original = X[i].copy()
            perturbed_x = x_original.copy()

            # 找到当前样本中存在的病症（非零索引）
            present_phenotypes = np.where(x_original == 1)[0]

            for phen_idx in present_phenotypes:
                # 以noise_prob概率决定是否替换该病症
                if np.random.rand() < noise_prob:
                    # 以ancestor_prob概率选择替换为祖先节点，否则为噪声
                    if np.random.rand() < ancestor_prob and phen_idx in phenotype_hierarchy:
                        # 从祖先节点中随机选择一个
                        ancestors = phenotype_hierarchy[phen_idx]
                        if ancestors:
                            chosen_ancestor = np.random.choice(ancestors)
                            perturbed_x[phen_idx] = 0  # 移除原病症
                            perturbed_x[chosen_ancestor] = 1  # 添加祖先节点
                    else:
                        # 替换为无关噪声（随机一个不存在的病症）
                        absent_phenotypes = np.where(x_original == 0)[0]
                        if len(absent_phenotypes) > 0:
                            chosen_noise = np.random.choice(absent_phenotypes)
                            perturbed_x[phen_idx] = 0  # 移除原病症
                            perturbed_x[chosen_noise] = 1  # 添加噪声

            X_perturbed.append(perturbed_x)
            y_perturbed.append(y[i])

    # 合并原始数据与增强数据
    X_perturbed = np.concatenate([X, np.array(X_perturbed)], axis=0)
    y_perturbed = np.concatenate([y, np.array(y_perturbed)], axis=0)

    return X_perturbed, y_perturbed


if __name__ == '__main__':
    #generate_simulated_data(5)
    generate_pkl_from_source_data_test(10000)