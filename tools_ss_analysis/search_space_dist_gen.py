import os
import argparse
from nas_lib.data.data import build_datasets
import numpy as np
import pickle
from nas_lib.data.data_nasbench2 import OPS_LIST as ops_nasbench_101
from nas_lib.data.data_nasbench_201 import OPS as ops_nasbench_201
from nas_lib.data.data_nasbench_nlp import OPS_LIST as ops_nasbench_nlp
from nas_lib.data.data_nasbench_asr import OPS_TOTAL as ops_nasbench_asr_n
from nas_lib.data.data_nasbench_ars_wo_none import OPS_TOTAL as ops_nasbench_asr_y


def edit_distance(adj_matrix_1, adj_matrix_2, ops_1, ops_2, OPS):
    """
    compute the distance between two architectures
    by comparing their adjacency matrices and op lists
    """
    graph_dist = np.sum(np.array(adj_matrix_1) != np.array(adj_matrix_2))
    ops_1 = [OPS.index(op) for op in ops_1]
    ops_2 = [OPS.index(op) for op in ops_2]
    ops_dist = np.sum(np.array(ops_1) != np.array(ops_2))
    return graph_dist + ops_dist


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args for search space analysis.')
    parser.add_argument('--search_space', type=str, default='nasbench_case2',
                        choices=['nasbench_case2', 'nasbench_201', 'nasbench_nlp', 'nasbench_asr'],
                        help='search space type.')
    parser.add_argument('--dataset', type=str, default='ImageNet16-120',
                        choices=['cifar10-valid', 'cifar100', 'ImageNet16-120'],
                        help='dataset name of nasbench-201.')
    parser.add_argument('--nasbench_nlp_type', type=str,
                        default='last', choices=['best', 'last'],
                        help='name of output files')
    parser.add_argument('--filter_none', type=str,
                        default='y', choices=['y', 'n'],
                        help='name of output files')
    parser.add_argument('--save_dir', type=str,
                        default='/home/albert_wei/fdisk_a/train_output_2021/npenas/search_space_analysis/',
                        help='output directory')
    args = parser.parse_args()

    ss = build_datasets(args.search_space, args.dataset, args.nasbench_nlp_type, args.filter_none)

    if args.search_space == 'nasbench_case2':
        save_dir = os.path.join(args.save_dir, f'nasbench_101.pkl')
        ops = ops_nasbench_101
    elif args.search_space == 'nasbench_201':
        save_dir = os.path.join(args.save_dir, f'{args.search_space}_{args.dataset}.pkl')
        ops = ops_nasbench_201
    elif args.search_space == 'nasbench_nlp':
        save_dir = os.path.join(args.save_dir, f'{args.search_space}_{args.nasbench_nlp_type}.pkl')
        ops = ops_nasbench_nlp
    elif args.search_space == 'nasbench_asr':
        save_dir = os.path.join(args.save_dir, f'{args.search_space}_{args.filter_none}.pkl')
        if args.filter_none == 'n':
            ops = ops_nasbench_asr_n
        else:
            ops = ops_nasbench_asr_y
    else:
        raise ValueError(f'Search Space {args.search_space} does not support at present!')

    val_list = []
    test_list = []
    if not args.search_space == 'nasbench_case2':
        dist_matrix = np.zeros((len(ss.total_keys), len(ss.total_keys)), dtype=np.int8)
        performance_matrix_val = np.zeros((len(ss.total_keys), len(ss.total_keys)), dtype=np.float16)
        performance_matrix_test = np.zeros((len(ss.total_keys), len(ss.total_keys)), dtype=np.float16)
    else:
        dist_matrix = np.zeros((10, 10), dtype=np.int8)
        performance_matrix_val = np.zeros((10, 10), dtype=np.float16)
        performance_matrix_test = np.zeros((10, 10), dtype=np.float16)
    for i, k1 in enumerate(ss.total_keys):
        arch_1 = ss.total_archs[k1]
        if args.search_space == 'nasbench_case2':
            val_list.append(arch_1['val'])
            test_list.append(arch_1['test'])
        else:
            val_list.append(arch_1[4])
            test_list.append(arch_1[5])
        if not args.search_space == 'nasbench_case2':
            for j, k2 in enumerate(ss.total_keys):
                arch_2 = ss.total_archs[k2]
                dist_matrix[i][j] = edit_distance(arch_1[1], arch_2[1], arch_1[2], arch_2[2], ops)
                performance_matrix_val[i][j] = abs(arch_1[4] - arch_2[4])
                performance_matrix_test[i][j] = abs(arch_1[5] - arch_2[5])

    with open(save_dir, 'wb') as f:
        pickle.dump(ss.total_keys, f)
        pickle.dump(dist_matrix, f)
        pickle.dump(performance_matrix_val, f)
        pickle.dump(performance_matrix_test, f)
        pickle.dump(val_list, f)
        pickle.dump(test_list, f)