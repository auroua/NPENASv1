import numpy as np
import copy
import nas_lib.nasbench101.model_spec as api
from nas_lib.utils.utils_data import find_isolate_node

INPUT = 'input'
OUTPUT = 'output'
SKIP = 'skip_connect'
CONV1X1 = 'nor_conv_1x1'
CONV3X3 = 'nor_conv_3x3'
AVGPOOL3X3 = 'avg_pool_3x3'
ISOLATE = 'isolate'
OPS = [SKIP, CONV1X1, CONV3X3, AVGPOOL3X3]

NUM_VERTICES = 8
OP_SPOTS = NUM_VERTICES - 2
# MAX_EDGES = 9


class Cell:
    def __init__(self, matrix, ops, isolate_node_idxs=None):
        self.matrix = matrix
        self.ops = ops
        self.isolate_node_idxs = isolate_node_idxs

    def serialize(self):
        return {
            'matrix': self.matrix,
            'ops': self.ops
        }

    def modelspec(self):
        return api.ModelSpec(matrix=self.matrix, ops=self.ops)

    @classmethod
    def random_cell(cls, nasbench):
        """
        From the NASBench repository
        https://github.com/google-research/nasbench
        """
        while True:
            matrix = np.random.choice(
                [0, 1], size=(NUM_VERTICES, NUM_VERTICES))
            matrix = np.triu(matrix, 1)
            ops = np.random.choice(OPS, size=NUM_VERTICES).tolist()
            ops[0] = INPUT
            ops[-1] = OUTPUT
            spec = api.ModelSpec(matrix=matrix, ops=ops)
            if nasbench.is_valid(spec):
                return {
                    'matrix': matrix,
                    'ops': ops
                }

    @classmethod
    def random_cell_gnn(cls, nasbench):
        """
        From the NASBench repository
        https://github.com/google-research/nasbench
        """
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

    def perturb(self, nasbench, edits=1):
        """
        create new perturbed cell
        inspird by https://github.com/google-research/nasbench
        """
        new_matrix = copy.deepcopy(self.matrix)
        new_ops = copy.deepcopy(self.ops)
        for _ in range(edits):
            while True:
                if np.random.random() < 0.5:
                    for src in range(0, NUM_VERTICES - 1):
                        for dst in range(src + 1, NUM_VERTICES):
                            new_matrix[src][dst] = 1 - new_matrix[src][dst]
                else:
                    for ind in range(1, NUM_VERTICES - 1):
                        available = [op for op in OPS if op != new_ops[ind]]
                        new_ops[ind] = np.random.choice(available)

                new_spec = api.ModelSpec(new_matrix, new_ops)
                if nasbench.is_valid(new_spec):
                    break
        return {
            'matrix': new_matrix,
            'ops': new_ops
        }

    def encode_cell2(self):
        """
        compute the "standard" encoding,
        i.e. adjacency matrix + op list encoding
        """
        OPS = [SKIP, CONV1X1, CONV3X3, AVGPOOL3X3, ISOLATE]
        encoding_length = (NUM_VERTICES ** 2 - NUM_VERTICES) // 2 + OP_SPOTS * len(OPS)
        encoding = np.zeros((encoding_length))
        n = 0
        for i in range(NUM_VERTICES - 1):
            for j in range(i + 1, NUM_VERTICES):
                encoding[n] = self.matrix[i][j]
                n += 1
        for i in range(1, NUM_VERTICES - 1):
            op_idx = OPS.index(self.ops[i])
            encoding[n + op_idx] = 1
            n += len(OPS)
        return tuple(encoding)

    def get_paths(self):
        """
        return all paths from input to output
        """
        paths = []
        for j in range(0, NUM_VERTICES):
            paths.append([[]]) if self.matrix[0][j] else paths.append([])

        # create paths sequentially
        for i in range(1, NUM_VERTICES - 1):
            for j in range(1, NUM_VERTICES):
                if self.matrix[i][j]:
                    for path in paths[i]:
                        paths[j].append([*path, self.ops[i]])
        return paths[-1]

    def get_path_indices(self):
        """
        compute the index of each path
        There are 3^0 + ... + 3^5 paths total.
        (Paths can be length 0 to 5, and for each path, for each node, there
        are three choices for the operation.)
        """
        paths = self.get_paths()
        # mapping = {CONV3X3: 0, CONV1X1: 1, MAXPOOL3X3: 2}
        mapping = {SKIP: 0, CONV1X1: 1, CONV3X3: 2, AVGPOOL3X3: 3}
        path_indices = []
        for path in paths:
            index = 0
            for i in range(NUM_VERTICES - 1):
                if i == len(path):
                    path_indices.append(index)
                    break
                else:
                    index += len(OPS) ** i * (mapping[path[i]] + 1)
        return tuple(path_indices)

    def encode_paths(self):
        """ output one-hot encoding of paths """
        num_paths = sum([len(OPS) ** i for i in range(OP_SPOTS + 1)])
        path_indices = self.get_path_indices()
        # path_encoding = np.zeros(num_paths, dtype=np.int8)
        path_encoding = np.zeros(num_paths)
        for index in path_indices:
            path_encoding[index] = 1
        return path_encoding

    def path_distance(self, other):
        """
        compute the distance between two architectures
        by comparing their path encodings
        """
        return np.sum(np.array(self.encode_paths() != np.array(other.encode_paths())))

    def edit_distance(self, other):
        """
        compute the distance between two architectures
        by comparing their adjacency matrices and op lists
        """
        graph_dist = np.sum(np.array(self.matrix) != np.array(other.matrix))
        ops_dist = np.sum(np.array(self.ops) != np.array(other.ops))
        return graph_dist + ops_dist