import numpy as np
import copy
import nas_lib.nasbench101.model_spec as api
from nas_lib.utils.utils_data import find_isolate_node
import random


INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
ISOLATE = 'isolate'
OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]

NUM_VERTICES = 7
OP_SPOTS = NUM_VERTICES - 2
MAX_EDGES = 9


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

    @classmethod
    def random_cell_both(cls, nasbench):
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
            return 100*(1 - nasbench.query(api.ModelSpec(matrix=self.matrix, ops=self.ops))['validation_accuracy'])
        else:
            # query the api until we see all three accuracies, then average them
            # a few architectures only have two accuracies, so we use patience to avoid an infinite loop
            accs = []
            while len(accs) < 3 and patience > 0:
                patience -= 1
                acc = nasbench.query(api.ModelSpec(matrix=self.matrix, ops=self.ops))['validation_accuracy']
                if acc not in accs:
                    accs.append(acc)
            return round(100*(1-np.mean(accs)), 3)

    def get_val_loss2(self, nasbench, deterministic=1, patience=50):
        if not deterministic:
            # output one of the three validation accuracies at random
            return 100*(nasbench.query(api.ModelSpec(matrix=self.matrix, ops=self.ops))['validation_accuracy'])
        else:
            # query the api until we see all three accuracies, then average them
            # a few architectures only have two accuracies, so we use patience to avoid an infinite loop
            accs = []
            while len(accs) < 3 and patience > 0:
                patience -= 1
                acc = nasbench.query(api.ModelSpec(matrix=self.matrix, ops=self.ops))['validation_accuracy']
                if acc not in accs:
                    accs.append(acc)
            return round(100*(np.mean(accs)), 3)

    def get_val_loss_nn_pred(self, nasbench, deterministic=1, patience=50):
        accs = []
        test_accs = []
        while len(accs) < 3 and patience > 0:
            patience -= 1
            acc = nasbench.query(api.ModelSpec(matrix=self.matrix, ops=self.ops))['validation_accuracy']
            test_acc = nasbench.query(api.ModelSpec(matrix=self.matrix, ops=self.ops))['test_accuracy']
            # if acc not in accs:
            accs.append(acc)
            # if test_acc not in test_accs:
            test_accs.append(test_acc)
        return 100*(np.mean(np.array(accs))), 100*(np.mean(np.array(test_accs)))

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
        return round(100*(1-np.mean(accs)), 3)

    def get_test_loss2(self, nasbench, patience=50):
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
        return 100*round((np.mean(accs)), 3)

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
                        for dst in range(src+1, NUM_VERTICES):
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

    def mutate2(self, nasbench, mutation_rate=1.0):
        """
        similar to perturb. A stochastic approach to perturbing the cell
        inspird by https://github.com/google-research/nasbench
        """
        iteration = 0
        while True:
            new_matrix = copy.deepcopy(self.matrix)
            new_ops = copy.deepcopy(self.ops)

            vertices = self.matrix.shape[0]
            op_spots = vertices - 2
            edge_mutation_prob = mutation_rate / vertices
            for src in range(0, vertices - 1):
                for dst in range(src + 1, vertices):
                    if random.random() < edge_mutation_prob:
                        new_matrix[src, dst] = 1 - new_matrix[src, dst]

            if op_spots != 0:
                op_mutation_prob = mutation_rate / op_spots
                for ind in range(1, op_spots + 1):
                    if random.random() < op_mutation_prob:
                        available = [o for o in OPS if o != new_ops[ind]]
                        new_ops[ind] = random.choice(available)

            new_spec = api.ModelSpec(new_matrix, new_ops)
            ops_idx = [-1] + [OPS.index(new_ops[idx]) for idx in range(1, len(new_ops)-1)] + [-2]
            iteration += 1
            if iteration == 500:
                ops_idx = [-1] + [OPS.index(self.ops[idx]) for idx in range(1, len(self.ops) - 1)] + [-2]
                return {
                    'matrix': copy.deepcopy(self.matrix),
                    'ops': copy.deepcopy(self.ops),
                    'ops_idx': ops_idx
                }
            if nasbench.is_valid(new_spec):
                return {
                    'matrix': new_matrix,
                    'ops': new_ops,
                    'ops_idx': ops_idx
                }

    def mutate_dist(self, nasbench, mutate_rate=1.0, chip_distribution=None):
        """
        similar to perturb. A stochastic approach to perturbing the cell
        inspird by https://github.com/google-research/nasbench
        """
        iteration = 0
        keys = [k for k in chip_distribution]
        sorted(keys)
        # print(keys)

        matrix_size = self.matrix.shape[0]
        allowed_nodes = list(range(1, matrix_size - 1))
        allowed_idxes = list(range(0, matrix_size))

        # print(allowed_nodes)

        mutate_iteration = 10
        mutate_rate = 0.4
        k_temp = []
        for k in keys:
            if 'isolate' in k:
                k_info = k.split('_')
                node = int(k_info[0])
                if node in allowed_nodes:
                    k_temp.append(k)
            else:
                info_k = k.split('_')
                node0, node1 = int(info_k[0]), int(info_k[1])
                if node0 in allowed_idxes and node1 in allowed_idxes:
                    k_temp.append(k)
        # print(k_temp)
        k_temp_exist = [k for k in k_temp if chip_distribution[k][2]==1]
        mean = [chip_distribution[d][0] for d in k_temp]
        std = [chip_distribution[d][1] for d in k_temp]

        mean_temp_exist = [chip_distribution[d][0] for d in k_temp_exist]
        std_temp_exist = [chip_distribution[d][1] for d in k_temp_exist]
        # for k in k_temp:
        #     print(k, chip_distribution[k][0], chip_distribution[k][1])
        while True:
            samples = np.random.normal(mean, std)
            sorted_indices = np.argsort(samples)

            samples_exist = np.random.normal(mean_temp_exist, std_temp_exist)
            sorted_indices_exist = np.argsort(samples_exist)

            new_matrix = copy.deepcopy(self.matrix)
            new_ops = copy.deepcopy(self.ops)

            # update_strategy = random.sample([0, 1], 1)[0]
            # if update_strategy == 1:
            for i in range(mutate_iteration):
                update_strategy = random.sample([0, 1], 1)[0]
                if update_strategy == 1:
                    if random.random() < mutate_rate:
                        k_min = k_temp[sorted_indices[i]]
                        info = k_min.split('_')
                        if 'isolate' in k_min:
                            if 'isolate' in self.ops:
                                node_idx = int(info[0])
                                new_matrix[0, node_idx] = 1
                                new_matrix[node_idx, allowed_idxes[-2]] = 1
                                new_ops.insert(node_idx, OPS[random.sample([0, 1, 2], 1)[0]])
                            else:
                                continue
                        else:
                            if '0' in k_min:
                                src, dst, op1 = int(info[0]), int(info[1]), info[2]
                                if new_matrix[src, dst] == 0:
                                    new_matrix[src, dst] = 1
                                if op1 != 'out':
                                    op1 = int(op1)
                                    new_ops[dst] = OPS[op1]
                            else:
                                src, dst, op1, op2 = int(info[0]), int(info[1]), info[2], info[3]
                                if new_matrix[src, dst] == 0:
                                    new_matrix[src, dst] = 1
                                if op1 == 'out':
                                    op2 = int(op2)
                                    new_ops[dst] = OPS[op2]
                                elif op2 == 'out':
                                    op1 = int(op1)
                                    new_ops[src] = OPS[op1]
                                else:
                                    op1, op2 = int(op1), int(op2)
                                    idx = random.randint(0, 1)
                                    if idx == 0:
                                        new_ops[src] = OPS[op1]
                                    else:
                                        new_ops[dst] = OPS[op2]
                else:
                    for i in range(mutate_iteration):
                        if random.random() < mutate_rate:
                            k_max = k_temp[sorted_indices_exist[(-1-i)]]
                            info = k_max.split('_')
                            if 'isolate' in k_max:
                                if 'isolate' in self.ops:
                                    node_idx = int(info[0])
                                    new_matrix[0, node_idx] = 1
                                    new_matrix[node_idx, allowed_idxes[-2]] = 1
                                    new_ops.insert(node_idx, OPS[random.sample([0, 1, 2], 1)[0]])
                                else:
                                    continue
                            else:
                                if '0' in k_max:
                                    src, dst, op1 = int(info[0]), int(info[1]), info[2]
                                    if new_matrix[src, dst] == 1:
                                        new_matrix[src, dst] = 0
                                    if op1 != 'out':
                                        op1 = int(op1)
                                        if new_ops[dst] == OPS[op1]:
                                            new_ops[dst] = OPS[random.sample(list(set((0, 1, 2))-set([op1])), 1)[0]]
                                else:
                                    src, dst, op1, op2 = int(info[0]), int(info[1]), info[2], info[3]
                                    if new_matrix[src, dst] == 1:
                                        new_matrix[src, dst] = 0
                                    if op1 == 'out':
                                        op2 = int(op2)
                                        new_ops[dst] = OPS[random.sample(list(set((0, 1, 2))-set([op2])), 1)[0]]
                                    elif op2 == 'out':
                                        op1 = int(op1)
                                        new_ops[src] = OPS[random.sample(list(set((0, 1, 2))-set([op1])), 1)[0]]
                                    else:
                                        op1, op2 = int(op1), int(op2)
                                        idx = random.randint(0, 1)
                                        if idx == 0:
                                            new_ops[src] = OPS[random.sample(list(set((0, 1, 2))-set([op1])), 1)[0]]
                                        else:
                                            new_ops[dst] = OPS[random.sample(list(set((0, 1, 2))-set([op2])), 1)[0]]

            if np.all(self.matrix == new_matrix) and self.ops == new_ops:
                continue
            new_spec = api.ModelSpec(new_matrix, new_ops)
            ops_idx = [-1] + [OPS.index(new_ops[idx]) for idx in range(1, len(new_ops)-1)] + [-2]
            iteration += 1
            if iteration == 500:
                ops_idx = [-1] + [OPS.index(self.ops[idx]) for idx in range(1, len(self.ops) - 1)] + [-2]
                return {
                    'matrix': copy.deepcopy(self.matrix),
                    'ops': copy.deepcopy(self.ops),
                    'ops_idx': ops_idx
                }
            if nasbench.is_valid(new_spec):
                return {
                    'matrix': new_matrix,
                    'ops': new_ops,
                    'ops_idx': ops_idx
                }

    def mutate_rates(self, nasbench, edge_rate, node_rate):
        """
        similar to perturb. A stochastic approach to perturbing the cell
        inspird by https://github.com/google-research/nasbench
        """
        while True:
            new_matrix = copy.deepcopy(self.matrix)
            new_ops = copy.deepcopy(self.ops)
            h, w = new_matrix.shape
            edge_mutation_prob = edge_rate
            for src in range(0, h - 1):
                for dst in range(src + 1, h):
                    if random.random() < edge_mutation_prob:
                        new_matrix[src, dst] = 1 - new_matrix[src, dst]

            op_mutation_prob = node_rate
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
        dic = {CONV1X1: 0., CONV3X3: 0.5, MAXPOOL3X3: 1.0}
        n = 0
        for i in range(NUM_VERTICES - 1):
            for j in range(i+1, NUM_VERTICES):
                encoding[n] = self.matrix[i][j]
                n += 1
        for i in range(1, NUM_VERTICES - 1):
            encoding[-i] = dic[self.ops[i]]
        return tuple(encoding)

    def encode_cell_3ops(self):
        """
        compute the "standard" encoding,
        i.e. adjacency matrix + op list encoding
        """
        OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]
        encoding_length = (NUM_VERTICES ** 2 - NUM_VERTICES) // 2 + OP_SPOTS * len(OPS)
        encoding = np.zeros((encoding_length))
        n = 0
        for i in range(NUM_VERTICES - 1):
            for j in range(i+1, NUM_VERTICES):
                encoding[n] = self.matrix[i][j]
                n += 1
        for i in range(1, NUM_VERTICES - 1):
            op_idx = OPS.index(self.ops[i])
            encoding[n+op_idx] = 1
            n += len(OPS)
        return tuple(encoding)

    def encode_cell2(self):
        """
        compute the "standard" encoding,
        i.e. adjacency matrix + op list encoding
        """
        OPS = [CONV3X3, CONV1X1, MAXPOOL3X3, ISOLATE]
        encoding_length = (NUM_VERTICES ** 2 - NUM_VERTICES) // 2 + OP_SPOTS * len(OPS)
        encoding = np.zeros((encoding_length))
        n = 0
        for i in range(NUM_VERTICES - 1):
            for j in range(i+1, NUM_VERTICES):
                encoding[n] = self.matrix[i][j]
                n += 1
        for i in range(1, NUM_VERTICES - 1):
            op_idx = OPS.index(self.ops[i])
            encoding[n+op_idx] = 1
            n += 4
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