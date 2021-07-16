# Copyright (c) XiDian University and Xi'an University of Posts&Telecommunication. All Rights Reserved

from nas_lib.nas_201_api import NASBench201API as API
from nas_lib.configs import nas_bench_201_path
from nas_lib.nas_201_api.genotypes import Structure as CellStructure
from nas_lib.data.cell_nasbench201 import Cell
import numpy as np
from nas_lib.utils.utils_data import find_isolate_node
import pickle


NUM_VERTICES = 8


def exchange_nodes_edges(genetype_data):
    ops = ['input']
    data_list = []
    for k in genetype_data:
        data_list.append(k)
    ops.append(data_list[0][0][0])  # 0--->1
    ops.append(data_list[1][0][0])  # 0--->2
    ops.append(data_list[2][0][0])  # 0--->3
    ops.append(data_list[1][1][0])  # 1--->4
    ops.append(data_list[2][1][0])  # 1--->5
    ops.append(data_list[2][2][0])  # 2--->6
    ops.append('output')

    adjacency_matrix = np.zeros((8, 8))
    adjacency_matrix[0, 1] = 1
    adjacency_matrix[0, 2] = 1
    adjacency_matrix[0, 3] = 1
    adjacency_matrix[1, 4] = 1
    adjacency_matrix[1, 5] = 1
    adjacency_matrix[2, 6] = 1
    adjacency_matrix[4, 6] = 1
    adjacency_matrix[3, 7] = 1
    adjacency_matrix[5, 7] = 1
    adjacency_matrix[6, 7] = 1

    del_idxs = [id for id, op in enumerate(ops) if op == 'none']
    ops = [op for op in ops if op != 'none']

    counter = 0
    for id in del_idxs:
        temp_id = id - counter
        adjacency_matrix = np.delete(adjacency_matrix, temp_id, axis=0)
        adjacency_matrix = np.delete(adjacency_matrix, temp_id, axis=1)
        counter += 1
    adjacency_matrix_dummy, ops_dummy = add_dummy_node(adjacency_matrix, ops)
    return adjacency_matrix, ops, adjacency_matrix_dummy, ops_dummy


def add_dummy_node(matrix_in, ops_in):
    # {1, 2, 3, 4, 5, 6, 7}
    matrix = np.zeros((NUM_VERTICES, NUM_VERTICES), dtype=np.int8)
    for i in range(matrix_in.shape[0]):
        idxs = np.where(matrix_in[i] == 1)
        for id in idxs[0]:
            if id == matrix_in.shape[0] - 1:
                matrix[i, NUM_VERTICES-1] = 1
            else:
                matrix[i, id] = 1
    ops = ops_in[:(matrix_in.shape[0] - 1)] + ['isolate'] * (NUM_VERTICES - matrix_in.shape[0]) + ops_in[-1:]
    find_isolate_node(matrix)
    return matrix, ops


def get_arch_acc_info(nas_bench, arch, dataname='cifar10-valid'):
    arch_index = nas_bench.query_index_by_arch(arch)
    assert arch_index >= 0, 'can not find this arch : {:}'.format(arch)
    arch_info = nas_bench.query_by_index(arch_index, dataname)
    if dataname == 'cifar10-valid':
        arch_info_test = nas_bench.query_by_index(arch_index, 'cifar10')
    else:
        arch_info_test = arch_info
    val_acc = []
    test_acc = []
    for k in arch_info:
        val_acc.append(arch_info[k].get_eval('x-valid')['accuracy'])
        test_acc.append(arch_info_test[k].get_eval('ori-test')['accuracy'])
    info = nas_bench.get_more_info(arch_index, dataname, None, use_12epochs_result=False, is_random=False)
    _, _, time_cost = info['test-accuracy'], info['valid-accuracy'], \
                                     info['train-all-time'] + info['valid-per-time']
    valid_acc = round(float(np.mean(val_acc)), 10)
    test_acc = round(float(np.mean(test_acc)), 4)
    return valid_acc, test_acc, time_cost


def generate_all_archs(nas_bench, dataset):
    total_archs = {}
    total_keys = []
    meta_archs = nas_bench.meta_archs
    for arch in meta_archs:
        val_acc, test_acc, time_cost = get_arch_acc_info(nas_bench, arch, dataname=dataset)
        structure = CellStructure.str2structure(arch)
        am, ops, am_dummy, ops_dummy = exchange_nodes_edges(structure)
        cell_arch = Cell(matrix=am_dummy, ops=ops_dummy, isolate_node_idxs=[])
        path_encoding1 = cell_arch.encode_paths()
        path_encoding2 = cell_arch.encode_cell2()
        total_archs[arch] = [
            (am_dummy, ops_dummy, []),
            am,
            ops,
            path_encoding1,
            100 - val_acc,
            100 - test_acc,
            arch,
            path_encoding2
        ]
        total_keys.append(arch)
    val_acc = [arch_info[4] for arch_info in total_archs]
    test_acc = [arch_info[5] for arch_info in total_archs]
    print(max(val_acc), min(val_acc), max(test_acc), min(test_acc))
    return total_archs, total_keys


def inti_nasbench_201(dataset, save_path):
    nas_bench = API(nas_bench_201_path)
    total_archs, total_keys = generate_all_archs(nas_bench, dataset)
    with open(save_path, 'wb') as f:
        pickle.dump(total_archs, f)
        pickle.dump(total_keys, f)


if __name__ == '__main__':
    # save_path = '/home/albert_wei/fdisk_a/train_output_2021/npenas/npenas_nasbench_ars_filter_none/arch_info.pkl'
    # inti_nasbench_201('cifar10-valid', save_path)  24.2200000016, 20.89   10.8413333455   8.07
    nas_bench = API(nas_bench_201_path)
    # arch = '|skip_connect~0|+|skip_connect~0|skip_connect~1|+|skip_connect~0|nor_conv_1x1~1|avg_pool_3x3~2|'
    # arch = '|skip_connect~0|+|nor_conv_1x1~0|nor_conv_1x1~1|+|nor_conv_3x3~0|skip_connect~1|avg_pool_3x3~2|'
    arch = '|nor_conv_3x3~0|+|skip_connect~0|nor_conv_3x3~1|+|nor_conv_1x1~0|nor_conv_1x1~1|nor_conv_3x3~2|'
    dataname = 'cifar10-valid'
    arch_index = nas_bench.query_index_by_arch(arch)
    arch_info = nas_bench.query_by_index(arch_index, dataname)
    arch_info_test = nas_bench.query_by_index(arch_index, 'cifar10')
    val_acc = []
    test_acc = []
    for k in arch_info:
        val_acc.append(arch_info[k].get_eval('x-valid')['accuracy'])
        test_acc.append(arch_info_test[k].get_eval('ori-test')['accuracy'])
    valid_acc = round(float(np.mean(val_acc)), 10)
    test_acc = round(float(np.mean(test_acc)), 4)
    print(100 - valid_acc, 100 - test_acc)