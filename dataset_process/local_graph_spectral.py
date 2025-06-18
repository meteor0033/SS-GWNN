#节点及一阶邻居构成的局部图

import torch
from torch.utils.data import Dataset
import argparse
from pygod.utils import load_data
import copy
from torch_geometric.utils import add_self_loops, to_dense_adj
import numpy as np
from torch_geometric.data import Data
from pygod.utils.utility import check_parameter


# 归一化
def _normalize(x):
    x_min = x.min()
    x_max = x.max()
    x_norm = (x - x_min)/x_max
    return x_norm
#随机产生joint 异常
def gen_joint_structural_outlier(data, m, n, random_state=None):
    """
    We randomly select n nodes from the network which will be the anomalies 
    and for each node we select m nodes from the network. 
    We connect each of n nodes with the m other nodes.

    Parameters
    ----------
    data : PyTorch Geometric Data instance (torch_geometric.data.Data)
        The input data.
    m : int
        Number nodes in the outlier cliques.
    n : int
        Number of outlier cliques.
    p : int, optional
        Probability of edge drop in cliques. Default: ``0``.
    random_state : int, optional
        The seed to control the randomness, Default: ``None``.

    Returns
    -------
    data : PyTorch Geometric Data instance (torch_geometric.data.Data)
        The structural outlier graph with injected edges.
    y_outlier : torch.Tensor
        The outlier label tensor where 1 represents outliers and 0 represents
        regular nodes.
    """

    if not isinstance(data, Data):
        raise TypeError("data should be torch_geometric.data.Data")

    if isinstance(m, int):
        check_parameter(m, low=0, high=data.num_nodes, param_name='m')
    else:
        raise ValueError("m should be int, got %s" % m)

    if isinstance(n, int):
        check_parameter(n, low=0, high=data.num_nodes, param_name='n')
    else:
        raise ValueError("n should be int, got %s" % n)

    check_parameter(m * n, low=0, high=data.num_nodes, param_name='m*n')

    if random_state:
        np.random.seed(random_state)


    outlier_idx = np.random.choice(data.num_nodes, size=n, replace=False)
    all_nodes = [i for i in range(data.num_nodes)]
    rem_nodes = []
    
    for node in all_nodes:
        if node is not outlier_idx:
            rem_nodes.append(node)
    
    
    
    new_edges = []
    
    # connect all m nodes in each clique
    for i in range(0, n):
        other_idx = np.random.choice(data.num_nodes, size=m, replace=False)
        for j in other_idx:
            new_edges.append(torch.tensor([[i, j]], dtype=torch.long))
                    

    new_edges = torch.cat(new_edges)


    y_outlier = torch.zeros(data.x.shape[0], dtype=torch.long)
    y_outlier[outlier_idx] = 1

    data.edge_index = torch.cat([data.edge_index, new_edges.T], dim=1)

    return data, y_outlier

def get_neighbor(data, device):
    
    in_nodes = data.edge_index[0,:]
    out_nodes = data.edge_index[1,:]

    #不包含自身的邻居节点
    neighbor_dict = {}
    for in_node, out_node in zip(in_nodes, out_nodes):
        if in_node.item() not in neighbor_dict:
            neighbor_dict[in_node.item()] = []
        neighbor_dict[in_node.item()].append(out_node.item())
                       
    neighbor_num_list = []
    for i in neighbor_dict:
        neighbor_num_list.append(len(neighbor_dict[i]))
    
    neighbor_num_list = torch.tensor(neighbor_num_list).to(device)
    return neighbor_dict, neighbor_num_list

# 对neighbor_edge_spectral，中的向量进行截断或padding，是其长度为32
def truncate_and_pad(vector, length):
    """
    截断或填充向量到指定长度。 
    :param vector: 输入向量（列表）
    :param length: 目标长度
    :return: 处理后的向量
    """
    if len(vector) > length:
        # 截断
        return vector[:length]
    else:
        # 填充
        #print("vector:",vector,vector.shape,len(vector))
        padding_length = length - len(vector)
        return torch.from_numpy(np.pad(vector, (0, padding_length), 'constant'))
def neighbor_pad_or_truncate(vector_dict, k):
    processed_dict = {}
    for key, vector in vector_dict.items():
        
        processed_vector = truncate_and_pad(vector, k)
        processed_dict[key] = processed_vector
    return processed_dict

def neighbor_pad_or_truncate_list(vector_list, k):
    processed_dict = []
    for i, sim in enumerate(vector_list):
        temp = truncate_and_pad(np.array(sim), k)
        processed_dict.append(temp)
    return processed_dict

