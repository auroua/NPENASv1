import random
from nas_lib.nas_201_api.genotypes import Structure as CellStructure
from copy import deepcopy
import numpy as np
from nas_lib.data.cell_nasbench201 import Cell
import torch
import pickle
from nas_lib.configs import nas_bench_201_converted_base_path


NUM_VERTICES = 8
OPS = ['input', 'none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3', 'isolate', 'output']
OPS_GRAPH = ['input', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3', 'isolate', 'output']
OPS_LIST = ['avg_pool_3x3', 'nor_conv_1x1', 'nor_conv_3x3', 'none', 'skip_connect']
OP_SPOTS = 6


class NASBench201:
    def __init__(self, dataset):
        self.nas_bench = None
        self.max_nodes = 4
        self.op_names = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
        self.op_names_alphaX = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3', 'term']
        del self.nas_bench
        pkl_path = nas_bench_201_converted_base_path % dataset
        with open(pkl_path, 'rb') as f:
            self.total_archs = pickle.load(f)
            self.total_keys = pickle.load(f)

    def random_arch(self):
        genotypes = []
        for i in range(1, self.max_nodes):
            xlist = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                op_name = random.choice(self.op_names)
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return CellStructure(genotypes)

    def mutate(self, parent_arch):
        while True:
            child_arch = deepcopy(parent_arch)
            node_id = random.randint(0, len(child_arch.nodes) - 1)
            node_info = list(child_arch.nodes[node_id])
            snode_id = random.randint(0, len(node_info) - 1)
            xop = random.choice(self.op_names)
            while xop == node_info[snode_id][0]:
                xop = random.choice(self.op_names)
            node_info[snode_id] = (xop, node_info[snode_id][1])
            child_arch.nodes[node_id] = tuple(node_info)
            str_arch = child_arch.tostr()
            if str_arch in self.total_keys:
                break
        return child_arch, self.total_archs[str_arch]

    def get_candidates(self,
                       data,
                       num=100,
                       allow_isomorphisms=False,
                       patience_factor=5,
                       num_best_arches=10,
                       return_dist=False):
        """
        Creates a set of candidate architectures with mutated and/or random architectures
        """
        # test for isomorphisms using a hash map of path indices
        candidates = []
        dic = {}
        dist_list = []
        nums_list = []
        mutated_archs_list = []
        for d in data:
            arch = {'matrix': d[0][0], 'ops': d[0][1]}
            path_indices = Cell(**arch).get_path_indices()
            dic[path_indices] = 1

        best_arches = [CellStructure.str2structure(arch[6])
                       for arch in sorted(data, key=lambda i: i[4])[:num_best_arches * patience_factor]]
        best_arches_full = [arch for arch in sorted(data, key=lambda i: i[4])[:num_best_arches * patience_factor]]
        # stop when candidates is size num
        # use patience_factor instead of a while loop to avoid long or infinite runtime
        for idx, arch in enumerate(best_arches):
            if len(candidates) >= num:
                break
            nums = 0
            for i in range(num):
                _, mutated = self.mutate(arch)
                path_indices = Cell(matrix=mutated[0][0], ops=mutated[0][1]).get_path_indices()

                if allow_isomorphisms or path_indices not in dic:
                    dic[path_indices] = 1
                    candidates.append(mutated)
                    dist = adj_distance(arch.tostr(), mutated[6])
                    dist_list.append(dist)
                    nums += 1
            nums_list.append(nums)
            mutated_archs_list.append(best_arches_full[idx])
        if return_dist:
            return candidates[:num], dist_list[:num], 0, nums_list, mutated_archs_list
        else:
            return candidates[:num]

    def generate_random_dataset(self, num=10,
                                allow_isomorphisms=False,
                                deterministic_loss=False):
        data = []
        dic = {}
        key_list = []
        while True:
            k = random.sample(self.total_keys, 1)
            key_list.append(k[0])
            arch = self.total_archs[k[0]]
            cell_arch = Cell(matrix=arch[0][0], ops=arch[0][1], isolate_node_idxs=[])
            path_encoding1 = cell_arch.encode_paths()
            path_encoding2 = cell_arch.encode_cell2()
            path_indices = cell_arch.get_path_indices()
            if allow_isomorphisms or path_indices not in dic:
                dic[path_indices] = 1
                data.append(
                    (
                        arch[0],
                        arch[1],
                        arch[2],
                        path_encoding1,
                        arch[4],
                        arch[5],
                        arch[6],
                        path_encoding2
                    )
                )
            if len(data) == num:
                break
        return data

    def nasbench2graph2(self, data):
        matrix, ops = data[0], data[1]
        node_feature = torch.zeros(NUM_VERTICES, 8)
        edges = int(np.sum(matrix))
        edge_idx = torch.zeros(2, edges)
        counter = 0
        for i in range(NUM_VERTICES):
            idx = OPS.index(ops[i])
            node_feature[i, idx] = 1
            for j in range(NUM_VERTICES):
                if matrix[i, j] == 1:
                    edge_idx[0, counter] = i
                    edge_idx[1, counter] = j
                    counter += 1
        return edge_idx, node_feature

    def get_arch_list(self,
                      aux_file_path,
                      distance=None,
                      iteridx=0,
                      num_top_arches=5,
                      max_edits=20,
                      num_repeats=5,
                      random_encoding='adj',
                      verbose=0):
        # Method used for gp_bayesopt

        # load the list of architectures chosen by bayesopt so far
        base_arch_list = pickle.load(open(aux_file_path, 'rb'))
        top_arches = [archtuple[0] for archtuple in base_arch_list[:num_top_arches]]
        if verbose:
            top_5_loss = [archtuple[1][0] for archtuple in base_arch_list[:min(5, len(base_arch_list))]]
            print('top 5 val losses {}'.format(top_5_loss))

        # perturb the best k architectures
        dic = {}
        for archtuple in base_arch_list:
            arch = {'matrix': self.total_archs[archtuple[0]['string']][0][0],
                    'ops': self.total_archs[archtuple[0]['string']][0][1]}
            path_indices = Cell(**arch).get_path_indices()
            dic[path_indices] = 1

        new_arch_list = []
        for arch in top_arches:
            for edits in range(1, max_edits):
                for _ in range(num_repeats):
                    #perturbation = Cell(**arch).perturb(self.nasbench, edits)
                    perturbation = self.mutate_edit(arch['string'], mutation_rate=edits)
                    perturbation_arch = self.total_archs[perturbation][0]
                    path_indices = Cell(matrix=perturbation_arch[0], ops=perturbation_arch[1]).get_path_indices()
                    if path_indices not in dic:
                        dic[path_indices] = 1
                        new_arch_list.append({'string': perturbation})

        # make sure new_arch_list is not empty
        while len(new_arch_list) == 0:
            for _ in range(100):
                random_arch_str = self.random_arch().tostr()
                random_arch = self.total_archs[random_arch_str][0]
                path_indices = Cell(matrix=random_arch[0], ops=random_arch[1]).get_path_indices()
                if path_indices not in dic:
                    dic[path_indices] = 1
                    new_arch_list.append({'string': random_arch_str})

        return new_arch_list

    def mutate_edit(self, arch, mutation_rate=1.0):
        """
        code modified from project https://github.com/naszilla/naszilla
        """
        ops = get_op_list(arch)
        new_ops = []
        # keeping mutation_prob consistent with nasbench_101
        mutation_prob = mutation_rate / (OP_SPOTS - 2)

        for i, op in enumerate(ops):
            if random.random() < mutation_prob:
                available = [o for o in OPS_LIST if o != op]
                new_ops.append(random.choice(available))
            else:
                new_ops.append(op)
        return get_string_from_ops(new_ops)

    @classmethod
    def generate_distance_matrix(cls, arches_1, arches_2, distance):
        matrix = np.zeros([len(arches_1), len(arches_2)])
        for i, arch_1 in enumerate(arches_1):
            for j, arch_2 in enumerate(arches_2):
                if distance == 'adj':
                    matrix[i][j] = adj_distance(arch_1['string'], arch_2['string'])
                elif distance == 'nasbot':
                    matrix[i][j] = nasbot_distance(arch_1['string'], arch_2['string'])
                else:
                    raise ValueError(f'Distance {distance} does not support at present!')
        return matrix


def get_op_list(arch):
    """
    code modified from project https://github.com/naszilla/naszilla
    """
    # given a string, get the list of operations
    tokens = arch.split('|')
    ops = [t.split('~')[0] for i,t in enumerate(tokens) if i not in [0,2,5,9]]
    return ops

def get_string_from_ops(ops):
    """
    code modified from project https://github.com/naszilla/naszilla
    """
    # given a list of operations, get the string
    strings = ['|']
    nodes = [0, 0, 1, 0, 1, 2]
    for i, op in enumerate(ops):
        strings.append(op+'~{}|'.format(nodes[i]))
        if i < len(nodes) - 1 and nodes[i+1] == 0:
            strings.append('+|')
    return ''.join(strings)


def adj_distance(cell_1, cell_2):
    cell_1_ops = get_op_list(cell_1)
    cell_2_ops = get_op_list(cell_2)
    return np.sum([1 for i in range(len(cell_1_ops)) if cell_1_ops[i] != cell_2_ops[i]])


def nasbot_distance(cell_1, cell_2):
    # distance based on optimal transport between row sums, column sums, and ops

    cell_1_ops = get_op_list(cell_1)
    cell_2_ops = get_op_list(cell_2)

    cell_1_counts = [cell_1_ops.count(op) for op in OPS]
    cell_2_counts = [cell_2_ops.count(op) for op in OPS]
    ops_dist = np.sum(np.abs(np.subtract(cell_1_counts, cell_2_counts)))

    return ops_dist + adj_distance(cell_1, cell_2)


if __name__ == '__main__':
    search_space = NASBench201(dataset='cifar10-valid')
    # search_space = NASBench201(dataset='cifar100')
    # search_space = NASBench201(dataset='ImageNet16-120')
    total_val_list = []
    total_test_list = []
    for key in search_space.total_keys:
        total_val_list.append(search_space.total_archs[key][4])
        total_test_list.append(search_space.total_archs[key][5])
    total_val = np.array(total_val_list)
    total_test = np.array(total_test_list)
    idx = np.argmin(total_val)
    print(total_val[idx], total_test[idx], min(total_test_list))