import numpy as np
import nas_lib.nasbench101.model_spec as api
import torch
import math


OPS = {
    'input': 0,
    'conv3x3-bn-relu': 1,
    'conv1x1-bn-relu': 2,
    'maxpool3x3': 3,
    'output': 4,
    'isolate': 5
}

NUM_VERTICES = 7


def find_isolate_node(matrix):
    node_list = []
    for i in range(len(matrix)):
        if np.all(matrix[i, :] == 0) and np.all(matrix[:, i] == 0):
            if i == 0:
                continue
            matrix[0, i] = 1
            node_list.append(i)
    return node_list


def random_cell_gnn():
    """
    From the NASBench repository
    https://github.com/google-research/nasbench
    """
    INPUT = 'input'
    OUTPUT = 'output'
    CONV3X3 = 'conv3x3-bn-relu'
    CONV1X1 = 'conv1x1-bn-relu'
    MAXPOOL3X3 = 'maxpool3x3'
    OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]

    NUM_VERTICES = 7
    OP_SPOTS = NUM_VERTICES - 2
    MAX_EDGES = 9

    nasbench = api.NASBench('/home/albert/disk_a/datasets_train/nas_bench101/nasbench_only108.tfrecord')
    while True:
        matrix = np.random.choice(
            [0, 1], size=(NUM_VERTICES, NUM_VERTICES))
        matrix = np.triu(matrix, 1)
        isolate_nodes = find_isolate_node(matrix)
        ops = np.random.choice(OPS, size=NUM_VERTICES).tolist()
        ops[0] = INPUT
        ops[-1] = OUTPUT

        spec = api.ModelSpec(matrix=matrix, ops=ops)
        if nasbench.is_valid(spec):
            return {
                'matrix': matrix,
                'ops': ops,
                'isolate_node_idxs': isolate_nodes
            }


def nasbench2graph(data):
    matrix, ops, isolate_nodes = data['matrix'], data['ops'], data['isolate_node_idxs']
    node_feature = torch.zeros(NUM_VERTICES, 6)
    edges = int(np.sum(matrix))
    edge_idx = torch.zeros(2, edges)
    counter = 0
    for i in range(NUM_VERTICES):
        if i in isolate_nodes:
            node_feature[i, 5] = 1
        else:
            idx = OPS[ops[i]]
            node_feature[i, idx] = 1
        for j in range(NUM_VERTICES):
            if matrix[i, j] == 1:
                edge_idx[0, counter] = i
                edge_idx[1, counter] = j
                counter += 1
    return edge_idx, node_feature


def nasbench2graph2(data):
    matrix, ops = data[0], data[1]
    node_feature = torch.zeros(NUM_VERTICES, 6)
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


def nasbench2graph2direction2(data):
    matrix, ops = data[0], data[1]
    node_feature = torch.zeros(NUM_VERTICES, 6)
    edges = int(np.sum(matrix))
    edge_idx = torch.zeros(2, edges*2)
    counter = 0
    for i in range(NUM_VERTICES):
        idx = OPS[ops[i]]
        node_feature[i, idx] = 1
        for j in range(NUM_VERTICES):
            if matrix[i, j] == 1:
                edge_idx[0, counter] = i
                edge_idx[1, counter] = j
                counter += 1
                edge_idx[0, counter] = j
                edge_idx[1, counter] = i
                counter += 1
    return edge_idx, node_feature


def nasbench2graph2direction(data):
    matrix, ops, isolate_nodes = data['matrix'], data['ops'], data['isolate_node_idxs']
    node_feature = torch.zeros(NUM_VERTICES, 6)
    edges = int(np.sum(matrix))
    edge_idx = torch.zeros(2, edges*2)
    counter = 0
    for i in range(NUM_VERTICES):
        if i in isolate_nodes:
            node_feature[i, 5] = 1
        else:
            idx = OPS[ops[i]]
            node_feature[i, idx] = 1
        for j in range(NUM_VERTICES):
            if matrix[i, j] == 1:
                edge_idx[0, counter] = i
                edge_idx[1, counter] = j
                counter += 1
                edge_idx[0, counter] = j
                edge_idx[1, counter] = i
                counter += 1
    return edge_idx, node_feature


def nasbench2graphnnp(data, reverse=False):
    matrix, ops, isolate_nodes = data['matrix'], data['ops'], data['isolate_node_idxs']
    if reverse:
        matrix = matrix.T
    node_feature = torch.zeros(NUM_VERTICES, 6)
    edges = int(np.sum(matrix))
    edge_idx = torch.zeros(2, edges*2)
    counter = 0
    for i in range(NUM_VERTICES):
        if i in isolate_nodes:
            node_feature[i, 5] = 1
        else:
            idx = OPS[ops[i]]
            node_feature[i, idx] = 1
        for j in range(NUM_VERTICES):
            if matrix[i, j] == 1:
                # forware
                edge_idx[0, counter] = i
                edge_idx[1, counter] = j
                counter += 1
                # backward
                edge_idx[0, counter] = j
                edge_idx[1, counter] = i
                counter += 1
    return edge_idx, node_feature


def acc2binprob(val_accuracy, step=1, stretch_flag=False):
    max_val = np.max(val_accuracy)
    max_val = math.ceil(max_val) + 1
    if max_val < 20:
        max_val = 22
    max_val *= 10
    bins = list(range(0, max_val, step))
    bins = [v*0.1 for v in bins]
    prob, bins = np.histogram(val_accuracy, bins=bins, density=False)
    prob = np.array([prob[i]/np.sum(prob) for i in range(prob.shape[0])])
    if stretch_flag:
        prob = stretch(prob)
    idx = np.digitize(val_accuracy, bins, right=False)
    val_prob = prob[idx-1]
    return val_prob


def stretch(probs):
    max = np.max(probs)
    min = np.min(probs)
    probs = probs - min
    probs /= (max - min)
    return probs


def stretch_min(probs, lower=0.1):
    probs[probs == probs.min()] = lower
    probs[probs > 1.0] = 1.0
    return probs


if __name__ == '__main__':
    for _ in range(1000000):
        data = random_cell_gnn()
        edge_idx, node_f = nasbench2graph(data)
        print(data['matrix'])
        print(edge_idx)
        print(node_f)
        print(data['isolate_node_idxs'])
        print('####################################')
