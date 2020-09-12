import numpy as np
import torch


OPS = {'input': 0,
        'none': 1,
       'max_pool_3x3': 2,
       'avg_pool_3x3': 3,
       'skip_connect': 4,
       'sep_conv_3x3': 5,
       'sep_conv_5x5': 6,
       'dil_conv_3x3': 7,
       'dil_conv_5x5': 8,
       'concat': 9,
       'output': 10
       }


NUM_VERTICES = 15


def nasbench2graph2(data):
    matrix, ops = data[0], data[1]
    node_feature = torch.zeros(NUM_VERTICES, 11)
    edges = int(np.sum(matrix))
    edge_idx = torch.zeros(2, edges)
    counter = 0
    for i in range(NUM_VERTICES):
        idx = OPS[ops[i]]
        node_feature[i, idx] = 1
        for j in range(NUM_VERTICES):
            if matrix[i, j] == 1:
                edge_idx[0, counter] = i
                edge_idx[1, counter] = j
                counter += 1
    return edge_idx, node_feature

