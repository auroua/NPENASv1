import copy

from nas_lib.configs import tf_records_path
from nasbench import api
from nas_lib.data.cell import Cell
import numpy as np
from nas_lib.utils.utils_data import find_isolate_node, NUM_VERTICES
import random
from hashlib import sha256
import pickle


OPS = {
    'input': 0,
    'conv3x3-bn-relu': 1,
    'conv1x1-bn-relu': 2,
    'maxpool3x3': 3,
    'output': 4,
    'isolate': 5
}

OPS_LIST_NEW = []
OPS_LIST = ['input', 'conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3', 'output', 'isolate']


class DataNasBenchNew:
    def __init__(self, search_space):
        self.search_space = search_space
        self.nasbench = api.NASBench(tf_records_path)
        self.total_archs, self.total_keys = self.get_clean_dummy_arch()
        self.ops_t = ['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3']

    def generate_random_dataset(self, num, encode_paths=True,
                                allow_isomorphisms=False,
                                deterministic_loss=True):
        data = []
        dic = {}
        key_list = []
        while True:
            k = random.sample(self.total_keys, 1)
            key_list.append(k)
            arch = self.total_archs[k[0]]
            if encode_paths:
                encoding = Cell(matrix=arch['matrix'], ops=arch['ops']).encode_paths()
            else:
                encoding = Cell(matrix=arch['matrix'], ops=arch['ops']).encode_cell2()
            path_indices = self.get_path_indices({'matrix': arch['matrix'], 'ops': arch['ops']})
            if allow_isomorphisms or path_indices not in dic:
                dic[path_indices] = 1
                data.append(((arch['matrix'], arch['ops'], []),
                             arch['o_matrix'],
                             arch['o_ops'],
                             encoding,
                             (1-arch['val'])*100,
                             (1-arch['test'])*100,
                             arch['key']))
            if len(data) == num:
                break
        return data

    def get_candidates(self,
                       data,
                       num=100,
                       acq_opt_type='mutation',
                       encode_paths=True,
                       allow_isomorphisms=False,
                       patience_factor=5,
                       deterministic_loss=True,
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
            path_indices = self.get_path_indices(arch)
            dic[path_indices] = 1

        mutate_arch_dict = {}
        if acq_opt_type in ['mutation', 'mutation_random']:
            # mutate architectures with the lowest validation error
            best_arches = [{'matrix': arch[1], 'ops': arch[2]}
                           for arch in sorted(data, key=lambda i: i[4])[:num_best_arches * patience_factor]]
            best_arch_datas = [d for d in sorted(data, key=lambda i: i[4])[:num_best_arches * patience_factor]]
            # stop when candidates is size num
            # use patience_factor instead of a while loop to avoid long or infinite runtime
            for idx, arch in enumerate(best_arches):
                if len(candidates) >= num:
                    break
                nums = 0
                mutate_arch_dict[idx] = 0
                for i in range(num):
                    mutated = self.mutate_arch(arch, encode_paths, require_distance=True)
                    path_indices = self.get_path_indices({'matrix': mutated[0][0],
                                                          'ops': mutated[0][1]})
                    if allow_isomorphisms or path_indices not in dic:
                        dic[path_indices] = 1
                        candidates.append(mutated)
                        mutate_arch_dict[idx] += 1
                        dist = adj_distance(arch, {'matrix': mutated[1], 'ops': mutated[2]})
                        dist_list.append(dist)
                        nums += 1
                nums_list.append(nums)
                mutated_archs_list.append(best_arch_datas[idx])
        if return_dist:
            return candidates[:num], dist_list[:num], 0, nums_list, mutated_archs_list
        else:
            return candidates[:num]

    def get_path_indices(self, arch):
        return Cell(**arch).get_path_indices()

    def mutate_arch(self, arch, encode_paths=True, mutation_rate=1.0, require_distance=False,
                    memory_array=None, distance_list=None):
        while True:
            arch_mutate = Cell(**{'matrix': arch['matrix'], 'ops': arch['ops']}).mutate2(self.nasbench, mutation_rate)
            matrix = arch_mutate['matrix']
            ops = arch_mutate['ops']
            results = self.query_arch(matrix=matrix,
                                      ops=ops,
                                      encode_paths=encode_paths,
                                      require_distance=require_distance)
            if results:
                break
        return results

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

    def query_arch(self,
                   matrix,
                   ops,
                   encode_paths=True,
                   deterministic=True,
                   require_distance=False):
        matrix = matrix.astype(np.int8)
        model_spec = api.ModelSpec(matrix=matrix, ops=ops)
        key = model_spec.hash_spec(self.ops_t)
        if not model_spec.valid_spec:
            return None

        o_matrix = model_spec.matrix
        o_ops = model_spec.ops

        if key in self.total_keys:
            matrix = self.total_archs[key]['matrix']
            ops = self.total_archs[key]['ops']
            val_loss = 100*(1-self.total_archs[key]['val'])
            test_loss = 100*(1-self.total_archs[key]['test'])
            arch = {
                'matrix': matrix,
                'ops': ops
            }
        else:
            matrix, ops = self.matrix_dummy_nodes(o_matrix, o_ops)
            arch = {
                'matrix': matrix,
                'ops': ops
            }
            val_loss = Cell(**arch).get_val_loss(self.nasbench, deterministic)
            test_loss = Cell(**arch).get_test_loss(self.nasbench)

        if encode_paths:
            encoding = Cell(**arch).encode_paths()
        else:
            encoding = Cell(**arch).encode_cell2()
        return [(matrix, ops, []),
                o_matrix,
                o_ops,
                encoding,
                val_loss,
                test_loss,
                key]

    def get_clean_dummy_arch(self):
        total_arch = {}
        total_keys = [k for k in self.nasbench.computed_statistics]

        best_key = None
        best_val = 0
        for k in total_keys:
            val_acc = []
            test_acc = []
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
            for i in range(3):
                val_acc.append(self.nasbench.computed_statistics[k][108][i]['final_validation_accuracy'])
                test_acc.append(self.nasbench.computed_statistics[k][108][i]['final_test_accuracy'])
            val_mean = float(np.mean(val_acc))
            test_mean = float(np.mean(test_acc))

            if best_val < val_mean:
                best_val = val_mean
                best_key = k

            total_arch[k] = {
                # 'o_matrix': arch_matrix.astype(np.float32),
                'o_matrix': arch_matrix,
                'o_ops': arch_ops,
                # 'matrix': matrix.astype(np.float32),
                'matrix': matrix,
                'ops': ops,
                'val': val_mean,
                'test': test_mean,
                'key': k
            }

        best_arch = total_arch[best_key]
        print(best_arch['val'], best_arch['test'])
        return total_arch, total_keys

    # The following method is used for prediction compare
    def generate_random_dataset_both(self,
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
            archtuple = self.query_arch_both(train=train,
                                             deterministic=deterministic_loss)
            arch_temp = {
                'matrix': archtuple[0]['matrix'],
                'ops': archtuple[0]['ops']
            }
            path_indices = self.get_path_indices(arch_temp)

            if allow_isomorphisms or path_indices not in dic:
                dic[path_indices] = 1
                data.append(archtuple)
        return data

    def query_arch_both(self,
                        arch=None,
                        train=True,
                        deterministic=True):
        if arch is None:
            arch = self.random_cell_gnn(self.nasbench)
        arch_temp = {
            'matrix': arch['matrix'],
            'ops': arch['ops']
        }
        encoding = Cell(**arch_temp).encode_paths()
        encoding_f = Cell(**arch_temp).encode_cell2()

        if train:
            return arch, encoding, encoding_f, arch['val_loss'], arch['test_loss']
        else:
            return arch, encoding, encoding_f

    def random_cell_gnn(self, nasbench):
        while True:
            key = random.sample(self.total_keys, 1)[0]
            architecture = self.total_archs[key]
            spec = api.ModelSpec(matrix=architecture['o_matrix'], ops=architecture['o_ops'])

            if nasbench.is_valid(spec):
                key = spec.hash_spec(self.ops_t)

                o_matrix, o_ops = spec.matrix, spec.ops
                if key in self.total_keys:
                    matrix = self.total_archs[key]['matrix']
                    ops = self.total_archs[key]['ops']
                    o_matrix = self.total_archs[key]['o_matrix']
                    val_loss = 100 * (1 - self.total_archs[key]['val'])
                    test_loss = 100 * (1 - self.total_archs[key]['test'])
                else:
                    matrix, ops = self.matrix_dummy_nodes(o_matrix, o_ops)
                    arch = {
                        'matrix': o_matrix,
                        'ops': o_ops
                    }
                    val_loss = Cell(**arch).get_val_loss(self.nasbench, True)
                    test_loss = Cell(**arch).get_test_loss(self.nasbench)
                return {
                    'matrix': matrix,
                    'matrix_orig': o_matrix,
                    'ops': ops,
                    'isolate_node_idxs': [],
                    'val_loss': val_loss,
                    'test_loss': test_loss
                }

    def remove_duplicates_both(self, candidates, data):
        # input: two sets of architectues: candidates and data
        # output: candidates with arches from data removed
        keys = []
        for d in data:
            k = sha256(str(d[1].tolist()).encode('utf-8')).hexdigest()
            keys.append(k)
        unduplicated = []
        for candidate in candidates:
            k_c = sha256(str(candidate[1].tolist()).encode('utf-8')).hexdigest()
            if k_c in keys:
                continue
            else:
                unduplicated.append(candidate)
        return unduplicated

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
            matrix = archtuple[0]['matrix']
            ops = archtuple[0]['ops']
            if matrix.shape[0] < NUM_VERTICES:
                matrix, ops = matrix_dummy_nodes(matrix, ops)
            path_indices = Cell(matrix=matrix, ops=ops).get_path_indices()
            dic[path_indices] = 1

        new_arch_list = []
        for arch in top_arches:
            for edits in range(1, max_edits):
                for _ in range(num_repeats):
                    #perturbation = Cell(**arch).perturb(self.nasbench, edits)
                    arch_mutate = Cell(**{'matrix': arch['matrix'], 'ops': arch['ops']}).mutate2(self.nasbench,
                                                                                                 mutation_rate=edits)
                    matrix = arch_mutate['matrix']
                    ops = arch_mutate['ops']
                    if matrix.shape[0] < NUM_VERTICES:
                        matrix, ops = matrix_dummy_nodes(matrix, ops)

                    path_indices = Cell(matrix=matrix, ops=ops).get_path_indices()
                    if path_indices not in dic:
                        dic[path_indices] = 1
                        new_arch_list.append(arch_mutate)

        # make sure new_arch_list is not empty
        while len(new_arch_list) == 0:
            for _ in range(100):
                arch = Cell.random_cell(self.nasbench)
                matrix = arch['matrix']
                ops = arch['ops']
                if matrix.shape[0] < NUM_VERTICES:
                    matrix, ops = matrix_dummy_nodes(matrix, ops)
                path_indices = Cell(matrix=matrix, ops=ops).get_path_indices()
                if path_indices not in dic:
                    dic[path_indices] = 1
                    new_arch_list.append(arch)

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


def matrix_dummy_nodes(matrix_in, ops_in):
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


def adj_distance(cell_1, cell_2):
    """
    compute the distance between two architectures
    by comparing their adjacency matrices and op lists
    (edit distance)
    """
    c1_matrix, c1_ops = matrix_dummy_nodes(cell_1['matrix'], cell_1['ops'])
    c2_matrix, c2_ops = matrix_dummy_nodes(cell_2['matrix'], cell_2['ops'])

    graph_dist = np.sum(c1_matrix != c2_matrix)
    ops_dist = np.sum(c1_ops != c2_ops)
    return graph_dist + ops_dist


def nasbot_distance(cell_1, cell_2):
#     # distance based on optimal transport between row sums, column sums, and ops
    c1_matrix, c1_ops = matrix_dummy_nodes(cell_1['matrix'], cell_1['ops'])
    c2_matrix, c2_ops = matrix_dummy_nodes(cell_2['matrix'], cell_2['ops'])

    cell_1_row_sums = sorted(c1_matrix.sum(axis=0))
    cell_1_col_sums = sorted(c1_matrix.sum(axis=1))

    cell_2_row_sums = sorted(c2_matrix.sum(axis=0))
    cell_2_col_sums = sorted(c2_matrix.sum(axis=1))

    row_dist = np.sum(np.abs(np.subtract(cell_1_row_sums, cell_2_row_sums)))
    col_dist = np.sum(np.abs(np.subtract(cell_1_col_sums, cell_2_col_sums)))

    cell_1_counts = [c1_ops.count(op) for op in OPS]
    cell_2_counts = [c2_ops.count(op) for op in OPS]

    ops_dist = np.sum(np.abs(np.subtract(cell_1_counts, cell_2_counts)))

    return row_dist + col_dist + ops_dist