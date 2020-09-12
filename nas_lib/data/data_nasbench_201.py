import random
from nas_lib.nas_201_api.genotypes import Structure as CellStructure
from copy import deepcopy
import numpy as np
from nas_lib.data.cell_nasbench201 import Cell
import torch
import pickle
from nas_lib.configs import nas_bench_201_converted_path

NUM_VERTICES = 8
OPS = ['input', 'none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3', 'isolate', 'output']
OPS_GRAPH = ['input', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3', 'isolate', 'output']


class NASBench201:
    def __init__(self):
        self.nas_bench = None
        self.max_nodes = 4
        self.op_names = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
        self.op_names_alphaX = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3', 'term']
        del self.nas_bench
        with open(nas_bench_201_converted_path, 'rb') as f:
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
                       num_best_arches=10):
        """
        Creates a set of candidate architectures with mutated and/or random architectures
        """
        # test for isomorphisms using a hash map of path indices
        candidates = []
        dic = {}
        for d in data:
            arch = {'matrix': d[0][0], 'ops': d[0][1]}
            path_indices = Cell(**arch).get_path_indices()
            dic[path_indices] = 1

        best_arches = [CellStructure.str2structure(arch[6])
                       for arch in sorted(data, key=lambda i: i[4])[:num_best_arches * patience_factor]]

        # stop when candidates is size num
        # use patience_factor instead of a while loop to avoid long or infinite runtime
        for idx, arch in enumerate(best_arches):
            if len(candidates) >= num:
                break
            for i in range(num):
                _, mutated = self.mutate(arch)
                path_indices = Cell(matrix=mutated[0][0], ops=mutated[0][1]).get_path_indices()

                if allow_isomorphisms or path_indices not in dic:
                    dic[path_indices] = 1
                    candidates.append(mutated)
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