from nasbench import api

from nas_lib.data.cell import Cell
import numpy as np
from nas_lib.utils.utils_data import find_isolate_node, NUM_VERTICES
import random
from nas_lib.configs import tf_records_path


class DataNasBenchDist:
    def __init__(self):
        self.nasbench = api.NASBench(tf_records_path)
        self.total_keys = self.get_total_keys()

    def get_total_keys(self):
        total_keys = []
        for k in self.nasbench.computed_statistics:
            total_keys.append(k)
        return total_keys

    def get_random_architectures(self):
        while True:
            k = random.sample(self.total_keys, 1)[0]
            final_validation_accuracy = []
            final_test_accuracy = []
            for results in self.nasbench.computed_statistics[k][108]:
                final_validation_accuracy.append(results['final_validation_accuracy'])
                final_test_accuracy.append(results['final_test_accuracy'])
            mean_val_accuracy = 100*(1-np.mean(final_validation_accuracy))
            mean_test_accuracy = 100*(1-np.mean(final_test_accuracy))
            o_matrix = self.nasbench.fixed_statistics[k]['module_adjacency']
            o_ops = self.nasbench.fixed_statistics[k]['module_operations']
            if o_matrix.shape[0] < 7:
                matrix, ops = self.matrix_dummy_nodes(o_matrix, o_ops)
                break
            else:
                matrix = o_matrix
                ops = o_ops
                break
        return {
            'matrix': matrix.astype(np.float32),
            'matrix_orig': o_matrix.astype(np.float32),
            'ops': ops,
            'isolate_node_idxs': [],
            'val_loss': mean_val_accuracy,
            'test_loss': mean_test_accuracy
        }

    def generate_random_dataset(self,
                                num=10,
                                train=True,
                                allow_isomorphisms=False,
                                deterministic_loss=True,
                                patience_factor=5):
        """
        create a dataset of randomly sampled architectues
        test for isomorphisms using a hash map of path indices
        use patience_factor to avoid infinite loops
        """
        data = []
        dic = {}
        tries_left = num * patience_factor
        while len(data) < num:
            tries_left -= 1
            if tries_left <= 0:
                break
            archtuple = self.query_arch(train=train)
            arch_temp = {
                'matrix': archtuple[0]['matrix'],
                'ops': archtuple[0]['ops']
            }
            path_indices = self.get_path_indices(arch_temp)

            if allow_isomorphisms or path_indices not in dic:
                dic[path_indices] = 1
                data.append(archtuple)
        return data

    def query_arch(self,
                   arch=None,
                   train=True):
        if arch is None:
            arch = self.get_random_architectures()
        arch_temp = {
            'matrix': arch['matrix'],
            'ops': arch['ops']
        }
        encoding = Cell(**arch_temp).encode_paths()

        if train:
            return arch, encoding, arch['val_loss'], arch['test_loss']
        else:
            return arch, encoding

    def get_path_indices(self, arch):
        return Cell(**arch).get_path_indices()

    def matrix_dummy_nodes(self, matrix_in, ops_in):
        # {2, 3, 4, 5, 6, 7}
        matrix = np.zeros((NUM_VERTICES, NUM_VERTICES))
        for i in range(matrix_in.shape[0]):
            idxs = np.where(matrix_in[i] == 1)
            for id in idxs[0]:
                if id == matrix_in.shape[0] - 1:
                    matrix[i, 6] = 1
                else:
                    matrix[i, id] = 1
        ops = ops_in[:(matrix_in.shape[0]-1)] + ['isolate'] * (7-matrix_in.shape[0]) + ops_in[-1:]
        find_isolate_node(matrix)
        return matrix, ops

    def get_all_path_encooding(self):
        total_arch_bananas_dict = {}
        total_keys = [k for k in self.nasbench.computed_statistics]
        for k in total_keys:
            arch_matrix = self.nasbench.fixed_statistics[k]['module_adjacency']
            arch_ops = self.nasbench.fixed_statistics[k]['module_operations']
            if arch_matrix.shape[0] < 7:
                matrix, ops = self.matrix_dummy_nodes(arch_matrix, arch_ops)
            else:
                matrix = arch_matrix
                ops = arch_ops
            spec = api.ModelSpec(matrix=arch_matrix, ops=arch_ops)
            if arch_matrix.shape[0] == 7:
                isolate_list = find_isolate_node(arch_matrix)
                if len(isolate_list) >= 1:
                    print(arch_matrix)
                    print(isolate_list)
            if not self.nasbench.is_valid(spec):
                continue

            arch = {
                'matrix': matrix,
                'ops': ops
            }
            encoding = Cell(**arch).encode_paths()
            total_arch_bananas_dict[k] = encoding
        return total_arch_bananas_dict