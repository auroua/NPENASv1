import os
import json
from nas_lib.configs import nas_bench_nlp_path
import networkx as nx
import numpy as np
from collections import defaultdict, Counter
from hashlib import sha256
import random
import pickle
import torch


INPUT_NODES = ['x', 'h_prev_0', 'h_prev_1', 'h_prev_2']
MIDDLE_NODES = ['i', 'i_act', 'j', 'j_act', 'o', 'o_act', 'f', 'f_act', 'r', 'r_act', 'rh', 'h_tilde', 'h_tilde_act',
                'z', 'z_act', 'h_new_1_part1', 'h_new_1_part2', 'node_0', 'node_1', 'node_2', 'node_3',
                'node_4', 'node_5', 'node_6', 'node_7', 'node_8', 'node_9', 'node_10', 'node_11', 'node_12',
                'node_13', 'node_14', 'node_15', 'node_16', 'node_17',  'node_18', 'node_19', 'node_20', 'node_21',
                'node_22', 'node_23']
OUTPUT_NODES = ['h_new_0', 'h_new_1', 'h_new_1_act', 'h_new_2']


OPS_LIST = ['input', 'activation_sigm', 'linear', 'elementwise_sum', 'activation_tanh', 'elementwise_prod',
            'activation_leaky_relu', 'blend', 'output', 'isolated']

OPS_LIST_ENCODING = ['linear', 'activation_sigm', 'activation_tanh', 'elementwise_sum', 'elementwise_prod',
                     'activation_leaky_relu', 'blend']
ADJ_SIZE = 26
OP_SPOTS = 24


class DataNasBenchNLP:
    def __init__(self, perf_type='best'):
        self.search_space = 'nasbench_nlp'
        self.perf_type = perf_type

        if perf_type == 'best':
            nas_nlp_save_path = os.path.join(nas_bench_nlp_path, 'nas_nlp_data_best.pkl')
            nas_nlp_adj_path = os.path.join(nas_bench_nlp_path, 'nas_nlp_adj_path_best.pkl')
        else:
            nas_nlp_save_path = os.path.join(nas_bench_nlp_path, 'nas_nlp_data_last.pkl')
            nas_nlp_adj_path = os.path.join(nas_bench_nlp_path, 'nas_nlp_adj_path_last.pkl')

        if os.path.exists(nas_nlp_save_path):
            with open(nas_nlp_save_path, 'rb') as f:
                self.total_archs, self.c_0_mapping, self.c_1_mapping, self.c_2_mapping, \
                self.c_3_mapping, self.total_0, self.total_1, self.total_2, self.total_len = pickle.load(f)
        else:
            self.total_archs, self.c_0_mapping, self.c_1_mapping, self.c_2_mapping, \
            self.c_3_mapping, self.total_0, self.total_1, self.total_2, self.total_len = self.load_dates()
            total_save_info = [self.total_archs, self.c_0_mapping, self.c_1_mapping, self.c_2_mapping,
                               self.c_3_mapping, self.total_0, self.total_1, self.total_2, self.total_len]
            with open(nas_nlp_save_path, 'wb') as f:
                pickle.dump(total_save_info, f)

        # c2mapping: 6643288, 89165658
        self.total_keys = list(self.total_archs.keys())
        self.arch_keys_dict = {v[6]: k for k, v in self.total_archs.items()}

        if os.path.exists(nas_nlp_adj_path):
            with open(nas_nlp_adj_path, 'rb') as f:
                self.distance_matrix = pickle.load(f)
        else:
            self.distance_matrix = self.distance_matrix_cal(self.total_archs, self.total_keys)
            with open(nas_nlp_adj_path, 'wb') as f:
                pickle.dump(self.distance_matrix, f)
        self.keys_idx_dict = {key: idx for idx, key in enumerate(self.total_keys)}

        self.total_val_data = total_val_data = [self.total_archs[k][4] for k in self.total_keys]
        self.total_test_data = total_test_data = [self.total_archs[k][5] for k in self.total_keys]

        min_validate_val = min(total_val_data)
        min_val_idx = total_val_data.index(min_validate_val)
        print(f'min val data value is {min_validate_val}, corr min test data is {total_test_data[min_val_idx]}, '
              f'and the min test val is {min(total_test_data)}')

    def generate_random_dataset(self, num, allow_isomorphisms, deterministic_loss=True):
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

    def mutate(self, arch_str, mutate_rate=1.0, by_distance=True, eps=0.1, arch2list=False):
        while True:
            mutate_flag = False
            arch_info = json.loads(arch_str)
            adj_matrix, ops, inputs, outputs = self.arch2dag(arch_info)
            adj_new_matrix, ops_new = self.add_dummy_nodes(adj_matrix, ops, inputs, outputs)
            key = self.make_keys(adj_new_matrix, ops_new)

            if arch2list and by_distance:
                key_idx = self.keys_idx_dict[key]
                self.distance_matrix[key_idx, key_idx] = 300
                distance_list = self.distance_matrix[key_idx, :].tolist()
                distance_list.sort()
                distance_set = list(set(distance_list))
                mutate_rate = int(mutate_rate)
                if mutate_rate in distance_set:
                    min_idxs = np.where(self.distance_matrix[key_idx, :] == mutate_rate)
                    min_idxs_list = min_idxs[0].tolist()
                    min_idx = random.choice(min_idxs_list)
                    closest_arch = self.total_archs[self.total_keys[min_idx]]
                    return closest_arch
                else:
                    min_list = [1000, 1000]
                    for idx, v in enumerate(distance_set):
                        dist = abs(mutate_rate - v)
                        if dist < min_list[1]:
                            min_list[0] = idx
                            min_list[1] = v
                    min_idxs = np.where(self.distance_matrix[key_idx, :] == min_list[1])
                    min_idxs_list = min_idxs[0].tolist()
                    min_idx = random.choice(min_idxs_list)
                    closest_arch = self.total_archs[self.total_keys[min_idx]]
                    return closest_arch
            elif by_distance:
                key_idx = self.keys_idx_dict[key]
                self.distance_matrix[key_idx, key_idx] = 300
                distance_list = self.distance_matrix[key_idx, :].tolist()
                distance_list.sort()
                distance_set = list(set(distance_list))
                if random.random() < eps:
                    min_val = random.choice(distance_set[0:20])
                else:
                    min_val = distance_set[0]
                min_idxs = np.where(self.distance_matrix[key_idx, :] == min_val)
                min_idxs_list = min_idxs[0].tolist()
                min_idx = random.choice(min_idxs_list)
                closest_arch = self.total_archs[self.total_keys[min_idx]]
                return closest_arch
            else:
                input_shape = adj_matrix.shape[0]
                edge_mutation_prob = mutate_rate / input_shape
                for src in range(0, input_shape - 1):
                    for dst in range(src + 1, input_shape):
                        if random.random() < edge_mutation_prob:
                            adj_matrix[src, dst] = 1 - adj_matrix[src, dst]
                            mutate_flag = True

                OP_SPOTS = input_shape - len(inputs) - len(outputs)
                op_mutation_prob = mutate_rate / OP_SPOTS
                for ind in range(1, OP_SPOTS + 1):
                    if random.random() < op_mutation_prob:
                        available = [o for o in OPS_LIST if o != ops[ind]]
                        ops[ind] = random.choice(available)
                        mutate_flag = True

                adj_new_matrix, ops_new = self.add_dummy_nodes(adj_matrix, ops, inputs, outputs)
                key = self.make_keys(adj_new_matrix, ops_new)
                if key in self.total_archs:
                    # print(mutate_flag)
                    return self.total_archs[key]

    def get_candidates(self,
                       data,
                       num=100,
                       allow_isomorphisms=False,
                       patience_factor=5,
                       num_best_arches=10,
                       mutation_rate=0.1,
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
            arch_info = json.loads(d[6])
            adj_matrix, ops, inputs, outputs = self.arch2dag(arch_info)
            adj_matrix_new, ops_new = self.add_dummy_nodes(adj_matrix, ops, inputs, outputs)
            path_dict = self.get_path_indices(adj_matrix_new, ops_new)
            path_indices = []
            for k, v in path_dict.items():
                temp_p = [k, *v]
                path_indices.extend(temp_p)
            path_indices = tuple(path_indices)
            dic[path_indices] = 1

        best_arches = [arch for arch in
                       sorted(data, key=lambda i: i[4])[:num_best_arches * patience_factor]]

        # stop when candidates is size num
        # use patience_factor instead of a while loop to avoid long or infinite runtime
        for idx, arch in enumerate(best_arches):
            if len(candidates) >= num:
                break
            nums = 0
            for i in range(num):
                mutated = self.mutate(arch[6], eps=mutation_rate)
                path_dict = self.get_path_indices(mutated[1], mutated[2])
                path_indices = []
                for k, v in path_dict.items():
                    temp_p = [k, *v]
                    path_indices.extend(temp_p)
                path_indices = tuple(path_indices)

                if allow_isomorphisms or path_indices not in dic:
                    dic[path_indices] = 1
                    candidates.append(mutated)
                    dist = adj_distance((0, arch[1], arch[2]), (0, mutated[1], mutated[2]))
                    dist_list.append(dist)
                    nums += 1
            nums_list.append(nums)
            mutated_archs_list.append(arch)
        if return_dist:
            return candidates[:num], dist_list[:num], 0, nums_list, mutated_archs_list
        else:
            return candidates[:num]

    def distance_matrix_cal(self, all_data, total_keys):
        distance_matrix = np.zeros((len(all_data), len(all_data)), dtype=np.int32)
        for i, k1 in enumerate(total_keys):
            for j, k2 in enumerate(total_keys):
                distance_matrix[i][j] = self.edit_distance(all_data[k1], all_data[k2])
        return distance_matrix

    def edit_distance(self, arch_1, arch_2):
        """
        compute the distance between two architectures
        by comparing their adjacency matrices and op lists
        """
        graph_dist = np.sum(arch_1[1] != arch_2[1])
        ops_dist = np.sum(arch_1[2] != arch_2[2])
        return graph_dist + ops_dist

    def load_dates(self):
        files = [os.path.join(nas_bench_nlp_path, f) for f in os.listdir(nas_bench_nlp_path) if f.endswith('json')]
        all_data = {}
        counter = 0
        for file in files:
            file_info = json.load(open(file, 'r'))
            if file_info['status'] != 'OK':
                continue
            counter += 1
            arch_info = json.loads(file_info['recepie'])
            adj_matrix, ops, inputs, outputs = self.arch2dag(arch_info)
            if self.perf_type == 'best':
                min_val_loss = min(file_info['val_losses'])
                min_test_loss = min(file_info['test_losses'])
            else:
                min_val_loss = file_info['val_losses'][-1]
                min_test_loss = file_info['test_losses'][-1]
            adj_new_matrix, ops_new = self.add_dummy_nodes(adj_matrix, ops, inputs, outputs)
            path_based_encoding = self.path_based_encoding(adj_new_matrix, ops_new)
            keys = self.make_keys(adj_new_matrix, ops_new)
            data = [
                [adj_matrix, ops],
                adj_new_matrix,  # store adjacency matrix
                ops_new,  # store ops list ['input', 'fc', 'output']
                path_based_encoding,  # path based encoding
                min_val_loss,
                min_test_loss,
                file_info['recepie'],
                self.adj_ops_encoding(adj_new_matrix, ops_new),  # store path encoding
            ]
            all_data[keys] = data
        all_data, c_0_mapping, c_1_mapping, c_2_mapping, c_3_mapping, \
        total_0, total_1, total_2, total_len = self.remapping_path_based_encoding(all_data)
        return all_data, c_0_mapping, c_1_mapping, c_2_mapping, c_3_mapping, total_0, total_1, total_2, total_len

    def make_keys(self, matrix, ops):
        matrix_list = matrix.tolist()
        matrix_list.extend(ops)
        return sha256(str(matrix_list).encode('utf-8')).hexdigest()

    def remapping_path_based_encoding(self, all_data):
        all_idxs_0 = {k: v[3][0] for k, v in all_data.items()}
        all_idxs_1 = {k: v[3][1] for k, v in all_data.items()}
        all_idxs_2 = {k: v[3][2] for k, v in all_data.items() if len(v[3]) >= 3}
        all_idxs_3 = {k: v[3][3] for k, v in all_data.items() if len(v[3]) == 4}
        all_idxs_00, all_idxs_11, all_idxs_22, all_idxs_33 = [], [], [], []

        for k, v in all_idxs_0.items():
            all_idxs_00.extend(v)
        for k, v in all_idxs_1.items():
            all_idxs_11.extend(v)
        for k, v in all_idxs_2.items():
            all_idxs_22.extend(v)
        for k, v in all_idxs_3.items():
            all_idxs_33.extend(v)

        c_0_dict = dict(Counter(all_idxs_00))
        c_1_dict = dict(Counter(all_idxs_11))
        c_2_dict = dict(Counter(all_idxs_22))
        c_3_dict = dict(Counter(all_idxs_33))

        c_0_mapping, c_1_mapping, c_2_mapping, c_3_mapping = {}, {}, {}, {}
        for idx, k in enumerate(c_0_dict.keys()):
            c_0_mapping[k] = idx
        for idx, k in enumerate(c_1_dict.keys()):
            c_1_mapping[k] = idx
        for idx, k in enumerate(c_2_dict.keys()):
            c_2_mapping[k] = idx
        for idx, k in enumerate(c_3_dict.keys()):
            c_3_mapping[k] = idx

        total_0 = len(c_0_mapping)
        total_1 = len(c_1_mapping)
        total_2 = len(c_2_mapping)
        # total_3 = len(c_3_mapping)
        total_len = total_0 + len(c_1_mapping) + len(c_2_mapping) + len(c_3_mapping)
        for k, v in all_data.items():
            path_idxs_0 = all_idxs_0[k]
            path_idxs_1 = all_idxs_1[k]
            if k in all_idxs_2:
                path_idxs_2 = all_idxs_2[k]
            if k in all_idxs_3:
                path_idxs_3 = all_idxs_3[k]
            path_encoding = np.zeros((total_len), dtype=np.int16)
            for vp in path_idxs_0:
                path_encoding[c_0_mapping[vp]] = 1
            for vp in path_idxs_1:
                path_encoding[total_0 + c_1_mapping[vp]] = 1
            if k in all_idxs_2:
                for vp in path_idxs_2:
                    path_encoding[total_0 + total_1 + c_2_mapping[vp]] = 1
            if k in all_idxs_3:
                for vp in path_idxs_3:
                    path_encoding[total_0 + total_1 + total_2 + c_3_mapping[vp]] = 1

            v[3] = path_encoding
            all_data[k] = v
        return all_data, c_0_mapping, c_1_mapping, c_2_mapping, c_3_mapping, total_0, total_1, total_2, total_len

    def add_dummy_nodes(self, adj_matrix, ops, input_nodes, output_nodes):
        node_shape = adj_matrix.shape[0]
        if node_shape == ADJ_SIZE:
            return adj_matrix.copy(), list(ops)
        input_num, output_num = len(input_nodes), len(output_nodes)
        prefix_counts = node_shape - output_num - 1
        output_indice = [prefix_counts+i for i in range(output_num)]
        padding_nums = ADJ_SIZE - node_shape
        adj_matrix_dummy = np.zeros(shape=(ADJ_SIZE, ADJ_SIZE), dtype=np.int32)
        ops_dummy = []
        adj_matrix_dummy[:prefix_counts, :prefix_counts] = adj_matrix[:prefix_counts, :prefix_counts]
        ops_dummy.extend(ops[:prefix_counts])
        for i in range(padding_nums):
            adj_matrix_dummy[0, i+prefix_counts] = 1
            ops_dummy.append('isolated')
        adj_dummy_idx = prefix_counts + padding_nums
        for idx, idx_idx in enumerate(output_indice):
            col_idx = adj_dummy_idx + idx
            for j in range(node_shape):
                if adj_matrix[j, idx_idx] == 1:
                    adj_matrix_dummy[j, col_idx] = 1
            ops_dummy.append(ops[idx_idx])
            adj_matrix_dummy[col_idx, -1] = 1
        ops_dummy.append('output')
        return adj_matrix_dummy, ops_dummy

    @classmethod
    def arch2dag(cls, recepie):
        G = nx.DiGraph()
        keys = cls.sort_keys(list(recepie.keys()))
        input_nodes, output_nodes = cls.get_inputs_outputs(recepie)
        total_keys = cls.get_total_inputs(recepie, output_nodes)
        op_list = []
        output_list = []
        # add nodes
        for k in total_keys:
            G.add_node(k)
            if k in input_nodes:
                op_list.append('input')
            elif k in output_nodes:
                output_list.append(recepie[k]['op'])
            else:
                op_list.append(recepie[k]['op'])
        G.add_node('output')
        op_list.extend(output_list)
        op_list.append('output')
        total_keys.append('output')

        # add link
        for k in keys:
            for input_k in recepie[k]['input']:
                G.add_edge(input_k, k, label=recepie[k]['op'])
            if k in output_nodes:
                G.add_edge(k, 'output')

        adj = np.array(nx.adjacency_matrix(G, nodelist=total_keys).todense(), dtype=np.int32)
        assert adj.shape[0] == len(op_list), 'The adj and operations are not consistence'
        return adj, op_list, input_nodes, output_nodes

    @classmethod
    def sort_keys(cls, input_keys, output_nodes=None):
        sorted_keys = []
        if output_nodes:
            for k in INPUT_NODES:
                if k in input_keys:
                    sorted_keys.append(k)
            for k in MIDDLE_NODES:
                if k in input_keys:
                    sorted_keys.append(k)
            for k in OUTPUT_NODES:
                if k in input_keys and k not in output_nodes:
                    sorted_keys.append(k)
            for k in OUTPUT_NODES:
                if k in input_keys and k in output_nodes:
                    sorted_keys.append(k)
        else:
            for k in INPUT_NODES:
                if k in input_keys:
                    sorted_keys.append(k)
            for k in MIDDLE_NODES:
                if k in input_keys:
                    sorted_keys.append(k)
            for k in OUTPUT_NODES:
                if k in input_keys:
                    sorted_keys.append(k)
        assert len(input_keys) == len(sorted_keys), 'inconsistent in the two keys size'
        return sorted_keys

    @classmethod
    def get_total_inputs(cls, recepie, output_nodes):
        total_nodes = []
        for k in recepie:
            total_nodes.append(k)
            total_nodes.extend(recepie[k]['input'])
        return cls.sort_keys(list(set(total_nodes)), output_nodes)

    @classmethod
    def get_inputs_outputs(cls, recepie):
        total_inputs = []
        op_keys = list(recepie.keys())
        inputs = []
        outputs = []
        for k in recepie:
            total_inputs.extend(recepie[k]['input'])
        for ko in recepie:
            if ko not in total_inputs:
                outputs.append(ko)
        for ki in total_inputs:
            if ki not in op_keys:
                inputs.append(ki)
        inputs_order = [ks for ks in INPUT_NODES if ks in inputs]
        outputs_order = [ks for ks in OUTPUT_NODES if ks in outputs]
        return inputs_order, outputs_order

    def exam_nodes(self, recepie):
        keys = list(recepie.keys())
        input_keys = []
        for _, v in recepie.items():
            input_node = v['input']
            input_keys.extend(input_node)
        for kr in input_keys:
            if kr in keys:
                keys.remove(kr)
        return input_keys

    def adj_ops_encoding(self, matrix, ops):
        encoding_length = (ADJ_SIZE ** 2 - ADJ_SIZE) // 2 + ADJ_SIZE * len(OPS_LIST)
        encoding = np.zeros((encoding_length), dtype=np.int32)
        n = 0
        for i in range(ADJ_SIZE - 1):
            for j in range(i+1, ADJ_SIZE):
                encoding[n] = matrix[i][j]
                n += 1
        for i in range(ADJ_SIZE):
            op_idx = OPS_LIST.index(ops[i])
            encoding[n+op_idx] = 1
            n += len(OPS_LIST)
        return tuple(encoding)

    def get_paths(self, matrix, ops):
        paths_dict = defaultdict(list)
        input_count = ops.count('input')
        for idx, op in enumerate(ops):
            if op == 'input':
                for j in range(0, ADJ_SIZE):
                    paths_dict[idx].append([['input']]) if matrix[idx][j] else paths_dict[idx].append([])

        # create paths sequentially
        for i in range(input_count, ADJ_SIZE-1):
            for j in range(input_count, ADJ_SIZE):
                if matrix[i][j]:
                    for k in paths_dict:
                        for path in paths_dict[k][i]:
                            paths_dict[k][j].append([*path, ops[i]])
        final_paths_dict = {k: v[-1] for k, v in paths_dict.items()}
        return final_paths_dict

    def get_path_indices(self, matrix, ops):
        paths = self.get_paths(matrix, ops)
        for k, v in paths.items():
            new_path = []
            for e2e in v:
                if 'input' in e2e:
                    e2e.remove('input')
                    new_path.append(e2e)
            paths[k] = new_path
        mapping = {'linear': 0,
                   'activation_sigm': 1,
                   'activation_tanh': 2,
                   'elementwise_sum': 3,
                   'elementwise_prod': 4,
                   'activation_leaky_relu': 5,
                   'blend': 6
                   }
        path_indices_dict = {}
        for k, v in paths.items():
            path_indices = []
            for path in v:
                index = 0
                for i in range(ADJ_SIZE - 1):
                    if i == len(path):
                        path_indices.append(index)
                        break
                    else:
                        index += len(OPS_LIST_ENCODING) ** i * (mapping[path[i]] + 1)
            path_indices.sort()
            path_indices_dict[k] = tuple(path_indices)
        return path_indices_dict

    def path_based_encoding(self, matrix, ops):
        path_indices = self.get_path_indices(matrix, ops)
        return path_indices

    def path_based_mapping_encoding(self, matrix, ops):
        path_indices = self.get_path_indices(matrix, ops)
        path_idxs_0 = path_indices[0]
        path_idxs_1 = path_indices[1]
        if len(path_indices) >= 3:
            path_idxs_2 = path_indices[2]
        if len(path_indices) >= 4:
            path_idxs_3 = path_indices[3]
        path_encoding = np.zeros((self.total_len), dtype=np.int16)
        for vp in path_idxs_0:
            path_encoding[self.c_0_mapping[vp]] = 1
        for vp in path_idxs_1:
            path_encoding[self.total_0 + self.c_1_mapping[vp]] = 1
        if len(path_indices) >= 3:
            for vp in path_idxs_2:
                if vp not in path_idxs_2:
                    print('-------------'*10)
                    print(list(self.c_2_mapping.keys()))
                    print(vp)
                    print('-------------' * 10)
                    continue
                total_2_idx = self.c_2_mapping[vp]
                path_encoding[self.total_0 + self.total_1 + total_2_idx] = 1
        if len(path_indices) >= 4:
            for vp in path_idxs_3:
                total_3_idx = self.c_3_mapping[vp]
                path_encoding[self.total_0 + self.total_1 + self.total_2 + total_3_idx] = 1
        return path_encoding

    def nasbench2graph2(self, data):
        matrix, ops = data[0], data[1]
        node_feature = torch.zeros(ADJ_SIZE, len(OPS_LIST))
        edges = int(np.sum(matrix))
        edge_idx = torch.zeros(2, edges)
        counter = 0
        for i in range(ADJ_SIZE):
            idx = OPS_LIST.index(ops[i])
            node_feature[i, idx] = 1
            for j in range(ADJ_SIZE):
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
            path_dict = self.path_based_encoding(matrix=archtuple[0][1], ops=archtuple[0][2])
            path_indices = []
            for k, v in path_dict.items():
                temp_p = [k, *v]
                path_indices.extend(temp_p)
            dic[tuple(path_indices)] = 1

        new_arch_list = []
        for arch in top_arches:
            for edits in range(1, max_edits):
                for _ in range(num_repeats):
                    #perturbation = Cell(**arch).perturb(self.nasbench, edits)
                    perturbation = self.mutate(arch[0], mutate_rate=edits, by_distance=True, arch2list=True)
                    path_dict = self.get_path_indices(matrix=perturbation[1], ops=perturbation[2])
                    path_indices = []
                    for k, v in path_dict.items():
                        temp_p = [k, *v]
                        path_indices.extend(temp_p)
                    path_indices = tuple(path_indices)
                    if path_indices not in dic:
                        dic[path_indices] = 1
                        new_arch_list.append([perturbation[6], perturbation[1], perturbation[2]])

        # make sure new_arch_list is not empty
        while len(new_arch_list) == 0:
            for _ in range(100):
                random_arch = self.generate_random_dataset(num=1, allow_isomorphisms=False)[0]
                path_dict = self.get_path_indices(matrix=random_arch[1], ops=random_arch[2])
                path_indices = []
                for k, v in path_dict.items():
                    temp_p = [k, *v]
                    path_indices.extend(temp_p)
                path_indices = tuple(path_indices)
                if  path_indices not in dic:
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

    cell_1_counts = [cell_1_ops.count(op) for op in OPS_LIST]
    cell_2_counts = [cell_2_ops.count(op) for op in OPS_LIST]

    ops_dist = np.sum(np.abs(np.subtract(cell_1_counts, cell_2_counts)))

    return row_dist + col_dist + ops_dist


def get_paths(matrix, ops):
    ADJ_SIZE = matrix.shape[0]
    paths_dict = defaultdict(list)
    input_count = ops.count('input')
    for idx, op in enumerate(ops):
        if op == 'input':
            for j in range(0, ADJ_SIZE):
                paths_dict[idx].append([['input']]) if matrix[idx][j] else paths_dict[idx].append([])

    # create paths sequentially
    for i in range(input_count, ADJ_SIZE-1):
        for j in range(input_count, ADJ_SIZE):
            if matrix[i][j]:
                for k in paths_dict:
                    for path in paths_dict[k][i]:
                        paths_dict[k][j].append([*path, ops[i]])
    final_paths_dict = {k: v[-1] for k, v in paths_dict.items()}
    return final_paths_dict


def add_dummy_nodes(adj_matrix, ops, input_nodes, output_nodes):
    node_shape = adj_matrix.shape[0]
    if node_shape == ADJ_SIZE:
        return adj_matrix.copy(), list(ops)
    input_num, output_num = len(input_nodes), len(output_nodes)
    prefix_counts = node_shape - output_num - 1
    output_indice = [prefix_counts+i for i in range(output_num)]
    padding_nums = ADJ_SIZE - node_shape
    adj_matrix_dummy = np.zeros(shape=(ADJ_SIZE, ADJ_SIZE), dtype=adj_matrix.dtype)
    ops_dummy = []
    adj_matrix_dummy[:prefix_counts, :prefix_counts] = adj_matrix[:prefix_counts, :prefix_counts]
    ops_dummy.extend(ops[:prefix_counts])
    for i in range(padding_nums):
        adj_matrix_dummy[0, i+prefix_counts] = 1
        ops_dummy.append('isolated')
    adj_dummy_idx = prefix_counts + padding_nums
    for idx, idx_idx in enumerate(output_indice):
        col_idx = adj_dummy_idx + idx
        for j in range(node_shape):
            if adj_matrix[j, idx_idx] == 1:
                adj_matrix_dummy[j, col_idx] = 1
        ops_dummy.append(ops[idx_idx])
        adj_matrix_dummy[col_idx, -1] = 1
    ops_dummy.append('output')
    return adj_matrix_dummy, ops_dummy


if __name__ == '__main__':
    # input_matrix = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 0],
    #                          [0, 0, 0, 1, 0, 1, 0, 0, 0],
    #                          [0, 0, 0, 1, 0, 1, 0, 0, 0],
    #                          [0, 0, 0, 0, 1, 0, 0, 0, 0],
    #                          [0, 0, 0, 0, 0, 0, 0, 1, 0],
    #                          [0, 0, 0, 0, 0, 0, 1, 0, 0],
    #                          [0, 0, 0, 0, 0, 0, 0, 0, 1],
    #                          [0, 0, 0, 0, 0, 0, 0, 0, 1],
    #                          [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    #                         )
    # ops = ['input', 'input', 'input', 'linear', 'activation_sigm', 'linear', 'activation_tanh', 'linear', 'output']
    # input_matrix = np.array([[0, 0, 0, 0, 1, 0, 0, 0],
    #                          [0, 0, 0, 1, 1, 0, 0, 0],
    #                          [0, 0, 0, 1, 0, 0, 0, 0],
    #                          [0, 0, 0, 0, 0, 0, 1, 0],
    #                          [0, 0, 0, 0, 0, 1, 0, 0],
    #                          [0, 0, 0, 0, 0, 0, 0, 1],
    #                          [0, 0, 0, 0, 0, 0, 0, 1],
    #                          [0, 0, 0, 0, 0, 0, 0, 0]]
    #                         )
    # ops = ['input', 'input', 'input', 'linear', 'linear', 'activation_sigm', 'activation_leaky_relu', 'output']

    # input_matrix = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                          [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    #                          [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                          [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    #                          [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    #                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #                          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    #                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    #                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    #                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    #                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    #                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    #                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    #                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    #                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    #                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    # ops = ['input', 'input', 'input', 'input', 'linear', 'linear', 'activation_sigm', 'linear', 'activation_sigm',
    #        'linear', 'linear', 'activation_leaky_relu', 'linear', 'activation_sigm', 'elementwise_sum',
    #        'activation_tanh', 'blend', 'output']
    # input_nodes = ['x', 'h_prev_0', 'h_prev_1', 'h_prev_2']
    # output_nodes = ['h_new_0', 'h_new_2']
    # paths = get_paths(input_matrix, ops)
    # for k, v in paths.items():
    #     print(v)
    # print('#######'*30)
    # dummy_matrix, dummy_ops = add_dummy_nodes(input_matrix, ops, input_nodes, output_nodes)
    # paths = get_paths(dummy_matrix, dummy_ops)
    # for k, v in paths.items():
    #     print(v)
    # print(len(dummy_ops))
    # print(dummy_ops)
    nasbench = DataNasBenchNLP(perf_type='last')
    nasbench_nlp = nasbench.total_archs
    total_matrix_shape_list = []
    total_ops_list = []
    # for k, v in nasbench_nlp.items():
    #     arch, ops = v[0][0], v[0][1]
    #     total_matrix_shape_list.append(arch.shape[0])
    #     total_ops_list.append(len(ops))
    # print(np.sum(np.array(total_matrix_shape_list) == 26))
    # print(np.sum(np.array(total_ops_list) == 26))
    # print('t')
    total_val_dataset = nasbench.total_val_data
    total_test_dataset = nasbench.total_test_data

    print(max(total_val_dataset))
    print(max(total_test_dataset))