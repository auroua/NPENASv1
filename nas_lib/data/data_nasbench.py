from nasbench import api

from nas_lib.configs import tf_records_path
from nas_lib.utils.utils_data import find_isolate_node
from .cell import Cell
from .cell_both import CellM


class DataNasBench:
    def __init__(self, search_space):
        self.search_space = search_space
        self.nasbench = api.NASBench(tf_records_path)
        self.total_keys = self.get_total_keys()

    def get_total_keys(self):
        total_keys = []
        for k in self.nasbench.computed_statistics:
            total_keys.append(k)
        return total_keys

    def get_type(self):
        return self.search_space

    def query_arch(self,
                   arch=None,
                   train=True,
                   encode_paths=True,
                   deterministic=True,
                   isnnp=False):
        if arch is None:
            arch = Cell.random_cell(self.nasbench)
        if encode_paths:
            encoding = Cell(**arch).encode_paths()
        else:
            encoding = Cell(**arch).encode_cell_3ops()

        if train:
            if isnnp:
                val_loss = Cell(**arch).get_val_loss2(self.nasbench, deterministic)
                test_loss = Cell(**arch).get_test_loss2(self.nasbench)
            else:
                val_loss = Cell(**arch).get_val_loss(self.nasbench, deterministic)
                test_loss = Cell(**arch).get_test_loss(self.nasbench)
            return arch, encoding, val_loss, test_loss
        else:
            return arch, encoding

    def query_arch_gin(self,
                       arch=None,
                       train=True,
                       encode_paths=True,
                       deterministic=True,
                       isnnp=False):
        if arch is None:
            arch = Cell.random_cell_gnn(self.nasbench)
        if encode_paths:
            encoding = Cell(**arch).encode_paths()
        else:
            encoding = Cell(**arch).encode_cell_3ops()

        if train:
            if isnnp:
                val_loss = Cell(**arch).get_val_loss2(self.nasbench, deterministic)
                test_loss = Cell(**arch).get_test_loss2(self.nasbench)
            else:
                val_loss = Cell(**arch).get_val_loss(self.nasbench, deterministic)
                test_loss = Cell(**arch).get_test_loss(self.nasbench)
            return arch, encoding, val_loss, test_loss
        else:
            return arch, encoding

    def generate_random_dataset_gin(self,
                                    num=10,
                                    train=True,
                                    encode_paths=True,
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
            archtuple = self.query_arch_gin(train=train,
                                            encode_paths=encode_paths,
                                            deterministic=deterministic_loss)
            path_indices = self.get_path_indices(archtuple[0])

            if allow_isomorphisms or path_indices not in dic:
                dic[path_indices] = 1
                data.append(archtuple)
        return data

    def generate_random_dataset(self,
                                num=10,
                                train=True,
                                encode_paths=True,
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
            archtuple = self.query_arch(train=train,
                                        encode_paths=encode_paths,
                                        deterministic=deterministic_loss)
            path_indices = self.get_path_indices(archtuple[0])

            if allow_isomorphisms or path_indices not in dic:
                dic[path_indices] = 1
                data.append(archtuple)
        return data

    def get_path_indices(self, arch):
        return Cell(**arch).get_path_indices()

    def mutate_arch(self, arch, mutation_rate=1.0):
        return Cell(**arch).mutate(self.nasbench, mutation_rate)

    def get_candidates(self, data,
                       num=100,
                       acq_opt_type='mutation',
                       encode_paths=True,
                       allow_isomorphisms=False,
                       patience_factor=5,
                       deterministic_loss=True,
                       num_best_arches=10):
        """
        Creates a set of candidate architectures with mutated and/or random architectures
        """
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
                                                train=False,
                                                deterministic=deterministic_loss,
                                                encode_paths=encode_paths)
                    path_indices = self.get_path_indices(mutated)

                    if allow_isomorphisms or path_indices not in dic:
                        dic[path_indices] = 1
                        candidates.append(archtuple)

        if acq_opt_type in ['random', 'mutation_random']:
            for _ in range(num * patience_factor):
                if len(candidates) >= 2 * num:
                    break
                archtuple = self.query_arch(train=False,
                                            deterministic=deterministic_loss,
                                            encode_paths=encode_paths)
                path_indices = self.get_path_indices(archtuple[0])
                if allow_isomorphisms or path_indices not in dic:
                    dic[path_indices] = 1
                    candidates.append(archtuple)
        return candidates[:num]

    def get_candidates_gin(self, data,
                           num=100,
                           acq_opt_type='mutation',
                           encode_paths=True,
                           allow_isomorphisms=False,
                           patience_factor=5,
                           deterministic_loss=True,
                           num_best_arches=10):
        """
        Creates a set of candidate architectures with mutated and/or random architectures
        """
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
                    mutated_matrix, mutated_ops = mutated['matrix'], mutated['ops']
                    isolate_nodes = find_isolate_node(mutated_matrix)
                    mutated['matrix'] = mutated_matrix
                    mutated['isolate_node_idxs'] = isolate_nodes

                    archtuple = self.query_arch_gin(mutated,
                                                    train=False,
                                                    deterministic=deterministic_loss,
                                                    encode_paths=encode_paths)
                    path_indices = self.get_path_indices(mutated)

                    if allow_isomorphisms or path_indices not in dic:
                        dic[path_indices] = 1
                        candidates.append(archtuple)

        if acq_opt_type in ['random', 'mutation_random']:
            for _ in range(num * patience_factor):
                if len(candidates) >= 2 * num:
                    break
                archtuple = self.query_arch(train=False,
                                            deterministic=deterministic_loss,
                                            encode_paths=encode_paths)
                path_indices = self.get_path_indices(archtuple[0])
                if allow_isomorphisms or path_indices not in dic:
                    dic[path_indices] = 1
                    candidates.append(archtuple)
        return candidates[:num]

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
            path_indices = self.get_path_indices_both(archtuple[0])

            if allow_isomorphisms or path_indices not in dic:
                dic[path_indices] = 1
                data.append(archtuple)
        return data

    def query_arch_both(self,
                        arch=None,
                        train=True,
                        deterministic=True):
        if self.search_space == 'nasbench_case1':
            if arch is None:
                arch = CellM.random_cell_gnn(self.nasbench)
            encoding = CellM(**arch).encode_paths()
            encoding_f = CellM(**arch).encode_cell()
            if train:
                val_loss = CellM(**arch).get_val_loss(self.nasbench, deterministic)
                test_loss = CellM(**arch).get_test_loss(self.nasbench)
                return arch, encoding, encoding_f, val_loss, test_loss
            else:
                return arch, encoding, encoding_f

    def get_path_indices_both(self, arch):
        return CellM(**arch).get_path_indices()

    def remove_duplicates_both(self, candidates, data):
        # input: two sets of architectues: candidates and data
        # output: candidates with arches from data removed
        dic = {}
        for d in data:
            dic[self.get_path_indices_both(d[0])] = 1
        unduplicated = []
        for candidate in candidates:
            if self.get_path_indices_both(candidate[0]) not in dic:
                dic[self.get_path_indices_both(candidate[0])] = 1
                unduplicated.append(candidate)
        return unduplicated

    def get_candidates_fixed_nums_single_arch(self, data,
                                              num=100,
                                              encode_paths=True,
                                              allow_isomorphisms=False,
                                              deterministic_loss=True,
                                              mutation_rate=1.0):
        """
        Creates a set of candidate architectures with mutated and/or random architectures
        """
        # test for isomorphisms using a hash map of path indices
        candidates = []
        dic = {}

        arch = data
        path_indices = self.get_path_indices(arch)
        dic[path_indices] = 1

        while len(candidates) <= num:
            mutated = self.mutate_arch(arch, mutation_rate=mutation_rate)
            archtuple = self.query_arch(mutated,
                                        train=True,
                                        deterministic=deterministic_loss,
                                        encode_paths=encode_paths)
            path_indices = self.get_path_indices(mutated)

            if allow_isomorphisms or path_indices not in dic:
                dic[path_indices] = 1
                candidates.append(archtuple)

        return candidates[:num]