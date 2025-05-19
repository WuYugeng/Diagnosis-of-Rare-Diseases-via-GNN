#读取obo文件
import obonet
from collections import defaultdict
import os

class hpo_reader():
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, "2019","Ontology","hp.obo")
        with open(file_path, 'r', encoding='utf-8') as f:
            self.graph = obonet.read_obo(f)
    def get_adj_list(self):
        """获取从子节点指向父节点的邻接矩阵"""
        adj_list = dict()
        # 获取节点（表型术语）信息
        for node_id, node_data in self.graph.nodes(data=True):
            if 'is_a' in node_data:
                adj_list[node_id] = node_data['is_a']
            else:
                adj_list[node_id] = []

        return adj_list

    def get_hpo_list(self):
        return [node_id for node_id in self.graph.nodes]




if __name__ == "__main__":
    reader=hpo_reader()
    a=reader.get_hpo_list()
    print(a)
    print(len(a))
