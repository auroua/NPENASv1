import numpy as np
import copy
import random
import nas_lib.nasbench101.model_spec as api
from nas_lib.utils.utils_data import find_isolate_node

INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]

NUM_VERTICES = 7
OP_SPOTS = NUM_VERTICES - 2
MAX_EDGES = 9


class CellM:
    def __init__(self, matrix, matrix_orig, ops, isolate_node_idxs=None):
        self.matrix = matrix
        self.matrix_orig = matrix_orig
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
            matrix_orig = matrix.copy()
            isolate_nodes = find_isolate_node(matrix)
            ops = np.random.choice(OPS, size=NUM_VERTICES).tolist()
            ops[0] = INPUT
            ops[-1] = OUTPUT

            spec = api.ModelSpec(matrix=matrix, ops=ops)
            if nasbench.is_valid(spec):
                return {
                    'matrix': matrix,
                    'matrix_orig': matrix_orig,
                    'ops': ops,
                    'isolate_node_idxs': isolate_nodes
                }

    def get_val_loss(self, nasbench, deterministic=1, patience=50):
        if not deterministic:
            # output one of the three validation accuracies at random
            return 100 * (1 - nasbench.query(api.ModelSpec(matrix=self.matrix, ops=self.ops))['validation_accuracy'])
        else:
            # query the api until we see all three accuracies, then average them
            # a few architectures only have two accuracies, so we use patience to avoid an infinite loop
            accs = []
            while len(accs) < 3 and patience > 0:
                patience -= 1
                acc = nasbench.query(api.ModelSpec(matrix=self.matrix, ops=self.ops))['validation_accuracy']
                if acc not in accs:
                    accs.append(acc)
            return round(100 * (1 - np.mean(accs)), 3)

    def get_test_loss(self, nasbench, patience=50):
        """
        query the api until we see all three accuracies, then average them
        a few architectures only have two accuracies, so we use patience to avoid an infinite loop
        """
        accs = []
        while len(accs) < 3 and patience > 0:
            patience -= 1
            acc = nasbench.query(api.ModelSpec(matrix=self.matrix, ops=self.ops))['test_accuracy']
            if acc not in accs:
                accs.append(acc)
        return round(100 * (1 - np.mean(accs)), 3)

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

    def mutate(self, nasbench, mutation_rate=1.0):
        """
        similar to perturb. A stochastic approach to perturbing the cell
        inspird by https://github.com/google-research/nasbench
        """
        while True:
            new_matrix = copy.deepcopy(self.matrix)
            new_ops = copy.deepcopy(self.ops)

            edge_mutation_prob = mutation_rate / NUM_VERTICES
            for src in range(0, NUM_VERTICES - 1):
                for dst in range(src + 1, NUM_VERTICES):
                    if random.random() < edge_mutation_prob:
                        new_matrix[src, dst] = 1 - new_matrix[src, dst]

            op_mutation_prob = mutation_rate / OP_SPOTS
            for ind in range(1, OP_SPOTS + 1):
                if random.random() < op_mutation_prob:
                    available = [o for o in OPS if o != new_ops[ind]]
                    new_ops[ind] = random.choice(available)

            new_spec = api.ModelSpec(new_matrix, new_ops)
            if nasbench.is_valid(new_spec):
                return {
                    'matrix': new_matrix,
                    'ops': new_ops
                }

    def mutate_gvae(self, nasbench, mutation_rate=1.0):
        """
        similar to perturb. A stochastic approach to perturbing the cell
        inspird by https://github.com/google-research/nasbench
        """
        while True:
            new_matrix = copy.deepcopy(self.matrix)
            new_ops = copy.deepcopy(self.ops)

            edge_mutation_prob = mutation_rate / NUM_VERTICES
            for src in range(0, NUM_VERTICES - 1):
                for dst in range(src + 1, NUM_VERTICES):
                    if random.random() < edge_mutation_prob:
                        new_matrix[src, dst] = 1 - new_matrix[src, dst]

            op_mutation_prob = mutation_rate / OP_SPOTS
            for ind in range(1, OP_SPOTS + 1):
                if random.random() < op_mutation_prob:
                    available = [o for o in OPS if o != new_ops[ind]]
                    new_ops[ind] = random.choice(available)

            isolate_nodes = find_isolate_node(new_matrix)
            new_spec = api.ModelSpec(new_matrix, new_ops)
            if nasbench.is_valid(new_spec):
                return {
                    'matrix': new_matrix,
                    'ops': new_ops,
                    'isolate_node_idxs': isolate_nodes
                }

    def encode_cell(self):
        """
        compute the "standard" encoding,
        i.e. adjacency matrix + op list encoding
        """
        encoding_length = (NUM_VERTICES ** 2 - NUM_VERTICES) // 2 + OP_SPOTS
        encoding = np.zeros((encoding_length))
        # dic = {CONV1X1: 0., CONV3X3: 0.5, MAXPOOL3X3: 1.0}
        dic = {CONV1X1: 0., CONV3X3: 0.3, MAXPOOL3X3: 0.6, 'isolate': 1.0}
        n = 0
        for i in range(NUM_VERTICES - 1):
            for j in range(i + 1, NUM_VERTICES):
                encoding[n] = self.matrix[i][j]
                n += 1
        for i in range(1, NUM_VERTICES - 1):
            encoding[-i] = dic[self.ops[i]]
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
        mapping = {CONV3X3: 0, CONV1X1: 1, MAXPOOL3X3: 2}
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