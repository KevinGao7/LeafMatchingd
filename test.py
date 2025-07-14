import os
from torch_geometric.datasets import CitationFull

# 指定文件夹路径
folder_path = './datasets'  # 替换为你的文件夹路径

# 创建文件夹（如果不存在）
os.makedirs(folder_path, exist_ok=True)

# 下载 Cora 数据集
dataset = CitationFull(root=folder_path, name='cora_ml', to_undirected=False)

# 查看数据集信息
data = dataset[0]
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Number of features: {data.num_node_features}')
print(f'Number of classes: {dataset.num_classes}')