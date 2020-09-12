from .arch_darts import ArchDarts
import numpy as np
from ..models_darts.datrs_neuralnet import DartsCifar10NeuralNet
from ..utils.utils_darts import convert_to_genotype, count_parameters_in_MB
from hashlib import sha256
import logging


OPS = ['none',
       'max_pool_3x3',
       'avg_pool_3x3',
       'skip_connect',
       'sep_conv_3x3',
       'sep_conv_5x5',
       'dil_conv_3x3',
       'dil_conv_5x5'
       ]


logger = logging.getLogger('nasbench_open_darts_cifar10')


class DataSetDarts:
    def __init__(self):
        self.search_space = 'darts'

    def get_type(self):
        return self.search_space

    def query_arch(self,
                   arch=None,
                   encode_paths=True):
        if arch is None:
            arch = ArchDarts.random_arch()
        if encode_paths:
            encoding = ArchDarts(arch).encode_paths()
        else:
            encoding = arch
        return (arch, encoding)

    def mutate_arch(self, arch, mutation_rate=1.0):
        return ArchDarts(arch).mutate(int(mutation_rate))

    def get_path_indices(self, arch):
        return ArchDarts(arch).get_path_indices()[0]

    def generate_random_dataset(self,
                                num=10,
                                encode_paths=True,
                                allow_isomorphisms=False
                                ):
        """
        create a dataset of randomly sampled architectues
        test for isomorphisms using a hash map of path indices
        use patience_factor to avoid infinite loops
        """
        data = []
        dic = {}
        while len(data) < num:
            archtuple = self.query_arch(encode_paths=encode_paths)
            path_indices = self.get_path_indices(archtuple[0])
            if allow_isomorphisms or path_indices not in dic:
                dic[path_indices] = 1
                data.append(archtuple)
        return data

    def get_candidates(self, macro_graph_dict, model_keys,
                       num=100,
                       acq_opt_type='mutation',
                       encode_paths=True,
                       allow_isomorphisms=False,
                       patience_factor=5,
                       num_best_arches=10):
        """
        Creates a set of candidate architectures with mutated and/or random architectures
        """
        data = [macro_graph_dict[k] for k in model_keys]
        # test for isomorphisms using a hash map of path indices
        candidates = []
        dic = {}
        for d in data:
            arch = d[0]
            path_indices = self.get_path_indices(arch)
            dic[path_indices] = 1

        if acq_opt_type in ['mutation', 'mutation_random']:
            # mutate architectures with the lowest validation error
            best_arches = [arch[0] for arch in sorted(data, key=lambda i: i[2])[:num_best_arches * patience_factor]]
            # stop when candidates is size num
            # use patience_factor instead of a while loop to avoid long or infinite runtime
            for arch in best_arches:
                if len(candidates) >= num:
                    break
                for i in range(num):
                    mutated = self.mutate_arch(arch)
                    archtuple = self.query_arch(mutated,
                                                encode_paths=encode_paths)
                    path_indices = self.get_path_indices(mutated)

                    if allow_isomorphisms or path_indices not in dic:
                        dic[path_indices] = 1
                        candidates.append(archtuple)
        return candidates

    def remove_duplicates(self, candidates, data):
        # input: two sets of architectues: candidates and data
        # output: candidates with arches from data removed

        dic = {}
        for d in data:
            dic[self.get_path_indices(d[0])] = 1
        unduplicated = []
        for candidate in candidates:
            if self.get_path_indices(candidate[0]) not in dic:
                dic[self.get_path_indices(candidate[0])] = 1
                unduplicated.append(candidate)
        return unduplicated

    def encode_data(self, dicts):
        # input: list of arch dictionary objects
        # output: xtrain (in binary path encoding), ytrain (val loss)
        data = []
        for dic in dicts:
            arch = dic['spec']
            encoding = ArchDarts(arch).encode_paths()
            data.append((arch, encoding, dic['val_loss_avg'], None))
        return data

    def assemble_graph(self, graph_dict, model_keys):
        train_data = []
        for k in model_keys:
            macro_matrix = np.zeros((30, 30), dtype=np.int8)
            arch_info = graph_dict[k][0]
            temp_matrix = []
            temp_ops = []
            for cell in arch_info:
                matrix, ops = self.assemble_matrix_ops(cell)
                temp_matrix.append(matrix)
                temp_ops.extend(ops)
            macro_matrix[0:15, 0:15] = temp_matrix[0]
            macro_matrix[15:, 15:] = temp_matrix[1]

            macro_matrix[14, 15] = 1
            macro_matrix[14, 16] = 1
            train_data.append([macro_matrix, temp_ops])
        return train_data

    def assemble_matrix_ops(self, normal_cell):
        normal_adjacency = np.zeros((15, 15), dtype=np.int8)
        normal_node_ops = ['input', 'input']
        for j, (idx, op) in enumerate(normal_cell):
            if j <= 1:
                normal_adjacency[idx, j + 2] = 1
                normal_node_ops.append(OPS[op])
                if j == 1:
                    normal_node_ops.append('concat')
            elif 1 < j <= 3:
                if idx == 2:
                    temp_idx = 4
                else:
                    temp_idx = idx
                normal_adjacency[temp_idx, j + 3] = 1
                normal_node_ops.append(OPS[op])
                if j == 3:
                    normal_node_ops.append('concat')
            elif 3 < j <= 5:
                if idx == 2:
                    temp_idx = 4
                elif idx == 3:
                    temp_idx = 7
                else:
                    temp_idx = idx
                normal_adjacency[temp_idx, j + 4] = 1
                normal_node_ops.append(OPS[op])
                if j == 5:
                    normal_node_ops.append('concat')
            else:
                if idx == 2:
                    temp_idx = 4
                elif idx == 3:
                    temp_idx = 7
                elif idx == 4:
                    temp_idx = 10
                else:
                    temp_idx = idx
                normal_adjacency[temp_idx, j + 5] = 1
                normal_node_ops.append(OPS[op])
                if j == 7:
                    normal_node_ops.append('concat')
        normal_adjacency[2, 4] = 1
        normal_adjacency[3, 4] = 1
        normal_adjacency[5, 7] = 1
        normal_adjacency[6, 7] = 1
        normal_adjacency[8, 10] = 1
        normal_adjacency[9, 10] = 1
        normal_adjacency[11, 13] = 1
        normal_adjacency[12, 13] = 1
        normal_adjacency[13, 14] = 1
        if not np.any(normal_adjacency[4]):
            normal_adjacency[4, 14] = 1
        if not np.any(normal_adjacency[7]):
            normal_adjacency[7, 14] = 1
        if not np.any(normal_adjacency[10, 13]):
            normal_adjacency[10, 14] = 1
        normal_node_ops.append('output')
        return normal_adjacency, normal_node_ops

    def assemble_cifar10_neural_net(self, data_dict):
        darts_neural_dict = {}
        parameters = {
            'init_channels': 16,
            'cifar_classed': 10,
            'layers': 8,
            'auxiliary': False,
            'stem_mult': 3
        }
        for data in data_dict:
            genotype = convert_to_genotype(data[0])
            k = sha256(str(genotype).encode('utf-8')).hexdigest()
            dart_neural = DartsCifar10NeuralNet(C=parameters['init_channels'],
                                                num_classes=parameters['cifar_classed'],
                                                layers=parameters['layers'],
                                                auxiliary=parameters['auxiliary'],
                                                genotype=genotype,
                                                key=k,
                                                stem_mult=parameters['stem_mult'])
            darts_neural_dict[k] = dart_neural
            logger.info(k)
            logger.info(count_parameters_in_MB(dart_neural))
            dart_neural.drop_path_prob = 0
        return darts_neural_dict



