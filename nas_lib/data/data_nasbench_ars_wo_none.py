import numpy as np
from .nasbench_asr.dataset import from_folder
from nas_lib.configs import nas_bench_asr_path
import torch
from collections import Counter
import random
from copy import deepcopy
from nas_lib.utils.utils_data import find_isolate_node
import pickle


OPS_LIST = ['linear', 'conv5', 'conv5d2', 'conv7', 'conv7d2', 'none']
OPS_CONN = ['none', 'skip_connect']
OPS = ['linear', 'conv5', 'conv5d2', 'conv7', 'conv7d2', 'none', 'skip_connect']
OPS_TOTAL = ['input', 'linear', 'conv5', 'conv5d2', 'conv7', 'conv7d2', 'skip_connect', 'isolate', 'output']
ADJ_NODES = 11
NUM_OPS = 9


class DataNasBenchASR_WO_None:
    def __init__(self):
        self.search_space = 'nasbench_asr'
        self.total_archs, self.all_datas_dict, self.all_datas_key_arch, self.all_datas_arch_key, \
        self.c_mapping, self.total_len = self.load_dates()
        self.total_keys = list(self.total_archs.keys())

    def generate_random_dataset(self, num, allow_isomorphisms, deterministic_loss=None):
        data = []
        dic = {}
        key_list = []
        while True:
            k = random.sample(self.total_keys, 1)
            key_list.append(k[0])
            arch = self.total_archs[k[0]]
            path_encoding1 = arch[3]
            path_encoding2 = arch[-1]
            path_indices = tuple(path_encoding1)
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

    def mutate(self, arch_str, mutate_rate=1):
        arch_list = self.all_datas_key_arch[arch_str]
        arch_list = list(map(list, arch_list))
        which_part = random.randint(0, 1)
        which_part_2 = random.randint(0, 1)
        while True:
            child_arch = deepcopy(arch_list)
            mutate_prob = float(mutate_rate) / (NUM_OPS - 5)
            # if mutate_rate == -1 or which_part == 0:
            # if np.random.rand() < mutate_prob:
            if True:
            # if which_part == 1 or which_part_2 == 1:
                block_op = random.randint(0, 2)
                xop = random.choice(OPS_LIST)
                xop_idx = OPS_LIST.index(xop)
                while xop_idx == child_arch[block_op][0]:
                    xop = random.choice(OPS_LIST)
                    xop_idx = OPS_LIST.index(xop)
                child_arch[block_op][0] = xop_idx

                block_conn = random.randint(0, 2)
                if block_conn == 0:
                    op_idx = random.randint(0, 1)
                    while op_idx == child_arch[0][1]:
                        op_idx = random.randint(0, 1)
                    child_arch[0][1] = op_idx
                elif block_conn == 1:
                    nested_block_idx = random.randint(1, 2)
                    op_idx = random.randint(0, 1)
                    while op_idx == child_arch[1][nested_block_idx]:
                        op_idx = random.randint(0, 1)
                    child_arch[1][nested_block_idx] = op_idx
                else:
                    nested_block_idx = random.randint(1, 3)
                    op_idx = random.randint(0, 1)
                    while op_idx == child_arch[2][nested_block_idx]:
                        op_idx = random.randint(0, 1)
                    child_arch[2][nested_block_idx] = op_idx
            # elif which_part == 1 and which_part_2 == 0:
            #     block_op = random.randint(0, 2)
            #     xop = random.choice(OPS_LIST)
            #     xop_idx = OPS_LIST.index(xop)
            #     while xop_idx == child_arch[block_op][0]:
            #         xop = random.choice(OPS_LIST)
            #         xop_idx = OPS_LIST.index(xop)
            #     child_arch[block_op][0] = xop_idx
            # elif which_part == 0 and which_part_2 == 1:
            #     block_conn = random.randint(0, 2)
            #     if block_conn == 0:
            #         op_idx = random.randint(0, 1)
            #         while op_idx == child_arch[0][1]:
            #             op_idx = random.randint(0, 1)
            #         child_arch[0][1] = op_idx
            #     elif block_conn == 1:
            #         nested_block_idx = random.randint(1, 2)
            #         op_idx = random.randint(0, 1)
            #         while op_idx == child_arch[1][nested_block_idx]:
            #             op_idx = random.randint(0, 1)
            #         child_arch[1][nested_block_idx] = op_idx
            #     else:
            #         nested_block_idx = random.randint(1, 3)
            #         op_idx = random.randint(0, 1)
            #         while op_idx == child_arch[2][nested_block_idx]:
            #             op_idx = random.randint(0, 1)
            #         child_arch[2][nested_block_idx] = op_idx
            # mutate_prob_p2 = float(mutate_rate) / (NUM_OPS - 5)
            # if mutate_rate == -1 or which_part == 1:
            # if np.random.rand() < mutate_prob:
            # if True:

            child_arch_tuple = tuple(list(map(tuple, child_arch)))
            if child_arch_tuple in self.all_datas_arch_key:
                return self.total_archs[self.all_datas_arch_key[child_arch_tuple]]

    def get_candidates(self,
                       data,
                       num=100,
                       allow_isomorphisms=False,
                       patience_factor=5,
                       num_best_arches=10,
                       mutation_rate=1,
                       return_dist=False):
        """
        Creates a set of candidate architectures with mutated and/or random architectures
        """
        # test for isomorphisms using a hash map of path indices
        candidates = []
        dist_list = []
        replicate_num = 0
        keys_list = []
        nums_list = []
        mutated_archs_list = []
        dic = {}
        for d in data:
            path_indices = self.get_path_indices(d[1], d[2])
            dic[path_indices] = 1
            keys_list.append(d[6])

        best_arches = [arch for arch in sorted(data,
                                               key=lambda i: i[4])[:num_best_arches * patience_factor]]

        # stop when candidates is size num
        # use patience_factor instead of a while loop to avoid long or infinite runtime
        for idx, arch in enumerate(best_arches):
            if len(candidates) >= num:
                break
            nums = 0
            for i in range(num):
                mutated = self.mutate(arch[6], mutate_rate=mutation_rate)
                path_indices = self.get_path_indices(mutated[1], mutated[2])
                if allow_isomorphisms or path_indices not in dic:
                    dic[path_indices] = 1
                    dist = adj_distance((0, arch[1], arch[2]), (0, mutated[1], mutated[2]))
                    candidates.append(mutated)
                    dist_list.append(dist)
                    if mutated[6] in keys_list:
                        replicate_num += 1
                    nums += 1
            nums_list.append(nums)
            mutated_archs_list.append(arch)
        if return_dist:
            return candidates[:num], dist_list[:num], replicate_num, nums_list, mutated_archs_list
        else:
            return candidates[:num]

    def get_arch_info_key(self, arch_info):
        return tuple(list(map(tuple, arch_info)))

    def load_dates(self):
        total_arch_data = {}
        total_val_data = []
        total_test_data = []
        all_data = from_folder(nas_bench_asr_path)
        all_datas = all_data.dbs
        all_datas_dict = {}
        all_datas_key_arch = {}
        all_datas_arch_key = {}
        for k, v in all_datas[0].items():
            if k not in all_datas_dict:
                all_datas_dict[k] = [[v[0][-1]], [v[-2]], v[-1]]
                all_datas_key_arch[k] = v[-1]
                all_datas_arch_key[self.get_arch_info_key(v[-1])] = k

        for data_list in all_datas[1:]:
            for k, v in data_list.items():
                all_datas_dict[k][0].append(v[0][-1])
                all_datas_dict[k][1].append(v[-2])

        for k, v in all_datas_key_arch.items():
            all_datas_dict[k][0] = np.mean(np.array(all_datas_dict[k][0]))
            all_datas_dict[k][1] = np.mean(np.array(all_datas_dict[k][1]))

        for k, v in all_datas_dict.items():
            adj_matrix, ops = self.arch2dat(v[-1])
            path_based_indices = self.get_path_indices(adj_matrix, ops)
            data = [
                [adj_matrix, ops],
                adj_matrix,  # store adjacency matrix
                ops,  # store ops list ['input', 'fc', 'output']
                path_based_indices,  # path based encoding
                v[0] * 100.,
                v[1] * 100.,
                k,
                self.adj_ops_encoding(adj_matrix, ops),  # store path encoding
            ]
            total_arch_data[k] = data
            total_val_data.append(v[0] * 100)
            total_test_data.append(v[1] * 100)
        total_arch_data, c_mapping, total_len = self.remapping_path_based_encoding(total_arch_data)
        min_validate_val = min(total_val_data)
        min_val_idx = total_val_data.index(min_validate_val)
        print(f'min val data value is {min_validate_val}, corr min test data is {total_test_data[min_val_idx]}, '
              f'and the min test val is {min(total_test_data)}')
        return total_arch_data, all_datas_dict, all_datas_key_arch, all_datas_arch_key, c_mapping, total_len

    def remapping_path_based_encoding(self, total_arch_data):
        all_idxs = [v[3] for k, v in total_arch_data.items()]
        all_idxs_00 = []
        for v in all_idxs:
            all_idxs_00.extend(v)
        c_0_dict = dict(Counter(all_idxs_00))
        c_0_mapping = {}
        for idx, k in enumerate(c_0_dict.keys()):
            c_0_mapping[k] = idx
        total_0 = len(c_0_mapping)
        for k, v in total_arch_data.items():
            path_idxs_0 = v[3]
            path_encoding = np.zeros((total_0), dtype=np.int16)
            for vp in path_idxs_0:
                path_encoding[c_0_mapping[vp]] = 1
            v[3] = path_encoding
            total_arch_data[k] = v
        return total_arch_data, c_0_mapping, total_0

    def arch2dat(self, arch_info):
        adj_matrix = np.zeros((ADJ_NODES, ADJ_NODES), dtype=np.int16)
        node_1_op = OPS_LIST[arch_info[0][0]]
        node_2_op = OPS_CONN[arch_info[0][1]]
        node_3_op = OPS_CONN[arch_info[1][1]]
        node_4_op = OPS_CONN[arch_info[2][1]]

        node_5_op = OPS_LIST[arch_info[1][0]]
        node_6_op = OPS_CONN[arch_info[1][2]]
        node_7_op = OPS_CONN[arch_info[2][2]]

        node_8_op = OPS_LIST[arch_info[2][0]]
        node_9_op = OPS_CONN[arch_info[2][3]]

        ops = ['input', node_1_op, node_2_op, node_3_op, node_4_op, node_5_op,
               node_6_op, node_7_op, node_8_op, node_9_op, 'output']

        adj_matrix[0, 1] = 1
        adj_matrix[0, 2] = 1
        adj_matrix[0, 3] = 1
        adj_matrix[0, 4] = 1

        adj_matrix[1, 5] = 1
        adj_matrix[1, 6] = 1
        adj_matrix[1, 7] = 1

        adj_matrix[2, 5] = 1
        adj_matrix[2, 6] = 1
        adj_matrix[2, 7] = 1

        adj_matrix[3, 8] = 1
        adj_matrix[5, 8] = 1
        adj_matrix[6, 8] = 1

        adj_matrix[3, 9] = 1
        adj_matrix[5, 9] = 1
        adj_matrix[6, 9] = 1

        adj_matrix[4, 10] = 1
        adj_matrix[7, 10] = 1
        adj_matrix[8, 10] = 1
        adj_matrix[9, 10] = 1

        if ops[1] == 'none':
            adj_matrix[0, 1] = 0
            adj_matrix[1, 5] = 0
            adj_matrix[1, 6] = 0
            adj_matrix[1, 7] = 0
        if ops[2] == 'none':
            adj_matrix[0, 2] = 0
            adj_matrix[2, 5] = 0
            adj_matrix[2, 6] = 0
            adj_matrix[2, 7] = 0
        if ops[3] == 'none':
            adj_matrix[0, 3] = 0
            adj_matrix[3, 8] = 0
            adj_matrix[3, 9] = 0
        if ops[4] == 'none':
            adj_matrix[0, 4] = 0
            adj_matrix[4, 10] = 0
        if ops[5] == 'none':
            adj_matrix[1, 5] = 0
            adj_matrix[2, 5] = 0
            adj_matrix[5, 8] = 0
            adj_matrix[5, 9] = 0
        if ops[6] == 'none':
            adj_matrix[1, 6] = 0
            adj_matrix[2, 6] = 0
            adj_matrix[6, 8] = 0
            adj_matrix[6, 9] = 0
        if ops[7] == 'none':
            adj_matrix[1, 7] = 0
            adj_matrix[2, 7] = 0
            adj_matrix[7, 10] = 0
        if ops[8] == 'none':
            adj_matrix[8, 10] = 0
            adj_matrix[3, 8] = 0
            adj_matrix[5, 8] = 0
            adj_matrix[6, 8] = 0
        if ops[9] == 'none':
            adj_matrix[3, 9] = 0
            adj_matrix[5, 9] = 0
            adj_matrix[6, 9] = 0
            adj_matrix[9, 10] = 0
        new_adj_matrix, new_ops, _ = self.remove_isolate_nodes(adj_matrix, ops)
        return new_adj_matrix, new_ops

    def remove_isolate_nodes(self, matrix, ops):
        isolate_indices = [idx for idx, op in enumerate(ops) if op == 'none']
        isolate_nodes_indices = find_isolate_node(matrix)
        assert isolate_indices == isolate_nodes_indices, 'The matrix and ops are not consistent.'
        new_ops = []
        for op in ops:
            if op == 'none':
                new_ops.append('isolate')
            else:
                new_ops.append(op)
        return matrix, new_ops, isolate_indices

    def get_paths(self, matrix, ops):
        """
        return all paths from input to output
        """
        paths = []
        for j in range(0, ADJ_NODES):
            paths.append([[]]) if matrix[0][j] else paths.append([])

        # create paths sequentially
        for i in range(1, ADJ_NODES - 1):
            for j in range(1, ADJ_NODES):
                if matrix[i][j]:
                    for path in paths[i]:
                        paths[j].append([*path, ops[i]])
        return paths[-1]

    def get_path_indices(self, matrix, ops):
        paths = self.get_paths(matrix, ops)
        mapping = {'linear': 0,
                   'conv5': 1,
                   'conv5d2': 2,
                   'conv7': 3,
                   'conv7d2': 4,
                   'skip_connect': 5
                   }
        path_indices = []
        for path in paths:
            index = 0
            for i in range(ADJ_NODES - 1):
                if i == len(path):
                    path_indices.append(index)
                    break
                else:
                    index += len(mapping) ** i * (mapping[path[i]] + 1)
        return tuple(path_indices)

    def path_based_encoding(self, adj_matrix, ops):
        path_indices = self.get_path_indices(adj_matrix, ops)
        path_encoding = np.zeros((self.total_len), dtype=np.int16)
        for index in path_indices:
            path_encoding[self.c_mapping[index]] = 1
        return path_encoding

    def adj_ops_encoding(self, adj_matrix, ops):
        encoding_length = (ADJ_NODES ** 2 - ADJ_NODES) // 2 + ADJ_NODES * len(OPS_TOTAL)
        encoding = np.zeros((encoding_length), dtype=np.int16)
        n = 0
        for i in range(ADJ_NODES - 1):
            for j in range(i + 1, ADJ_NODES):
                encoding[n] = adj_matrix[i][j]
                n += 1
        for i in range(ADJ_NODES):
            op_idx = OPS_TOTAL.index(ops[i])
            encoding[n + op_idx] = 1
            n += len(OPS_TOTAL)
        return tuple(encoding)

    def nasbench2graph2(self, data):
        matrix, ops = data[0], data[1]
        node_feature = torch.zeros(ADJ_NODES, len(OPS_TOTAL))
        edges = int(np.sum(matrix))
        edge_idx = torch.zeros(2, edges)
        counter = 0
        for i in range(ADJ_NODES):
            idx = OPS_TOTAL.index(ops[i])
            node_feature[i, idx] = 1
            for j in range(ADJ_NODES):
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
            arch_list = self.all_datas_key_arch[archtuple[0][0]]
            matrix, ops = self.arch2dat(arch_list)
            path_indices = self.get_path_indices(matrix, ops)
            dic[path_indices] = 1

        new_arch_list = []
        for arch in top_arches:
            for edits in range(1, max_edits):
                for _ in range(num_repeats):
                    #perturbation = Cell(**arch).perturb(self.nasbench, edits)
                    perturbation = self.mutate(arch[0], mutate_rate=edits)
                    path_indices = self.get_path_indices(matrix=perturbation[1], ops=perturbation[2])
                    if path_indices not in dic:
                        dic[path_indices] = 1
                        new_arch_list.append([perturbation[6], perturbation[1], perturbation[2]])

        # make sure new_arch_list is not empty
        while len(new_arch_list) == 0:
            for _ in range(100):
                random_arch = self.generate_random_dataset(num=1, allow_isomorphisms=False)[0]
                path_indices = self.get_path_indices(matrix=random_arch[1], ops=random_arch[2])
                if path_indices not in dic:
                    dic[path_indices] = 1
                    new_arch_list.append([random_arch[6], random_arch[1], random_arch[2]])

        return new_arch_list

    @classmethod
    def generate_distance_matrix(cls, arches_1, arches_2, distance):
        matrix = np.zeros([len(arches_1), len(arches_2)])
        for i, arch_1 in enumerate(arches_1):
            for j, arch_2 in enumerate(arches_2):
                if distance == 'adj':
                    matrix[i][j] = adj_distance(arch_1, arch_2)
                elif distance == 'nasbot':
                    matrix[i][j] = nasbot_distance(arch_1, arch_2)
                else:
                    raise ValueError(f'Distance {distance} does not support at present!')
        return matrix


def adj_distance(cell_1, cell_2):
    graph_dist = np.sum(cell_1[1] != cell_2[1])
    ops_dist = np.sum(cell_1[2] != cell_2[2])
    return graph_dist + ops_dist


def nasbot_distance(cell_1, cell_2):
    # distance based on optimal transport between row sums, column sums, and ops

    cell_1_matrix, cell_1_ops = cell_1[1], cell_1[2]
    cell_2_matrix, cell_2_ops = cell_2[1], cell_2[2]

    cell_1_row_sums = sorted(cell_1_matrix.sum(axis=0))
    cell_1_col_sums = sorted(cell_1_matrix.sum(axis=1))

    cell_2_row_sums = sorted(cell_2_matrix.sum(axis=0))
    cell_2_col_sums = sorted(cell_2_matrix.sum(axis=1))

    row_dist = np.sum(np.abs(np.subtract(cell_1_row_sums, cell_2_row_sums)))
    col_dist = np.sum(np.abs(np.subtract(cell_1_col_sums, cell_2_col_sums)))

    cell_1_counts = [cell_1_ops.count(op) for op in OPS]
    cell_2_counts = [cell_2_ops.count(op) for op in OPS]

    ops_dist = np.sum(np.abs(np.subtract(cell_1_counts, cell_2_counts)))

    return row_dist + col_dist + ops_dist