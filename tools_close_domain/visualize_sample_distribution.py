import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from nas_lib.data.data import build_datasets
from nas_lib.utils.comm import set_random_seed
import os
import tensorflow as tf
import torch
import matplotlib.pyplot as plt
import argparse
import pickle

np.set_printoptions(threshold=sys.maxsize)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

device = torch.device('cuda:0')
plt.rcParams["font.family"] = "Times New Roman"


def kl_divergence(dis_a, dis_b):
    dis_a += 1e-6
    dis_b += 1e-6
    log_a = np.log(dis_a)
    log_b = np.log(dis_b)
    part1 = dis_a*log_a
    part2 = dis_a*log_b
    result = np.sum(part1-part2)
    print(result)


def path_encoding_distribution(path_encoding_list):
    encoding_list = [d[1] for d in path_encoding_list]
    x_idx = list(range(len(encoding_list[0])))
    np_encoding = np.sum(np.array(encoding_list), axis=0)/np.sum(np.array(encoding_list))
    plt.bar(x_idx, np_encoding)
    plt.show()
    return np_encoding, x_idx


def path_encoding_dict_distribution(path_encoding_list):
    encoding_list = [path_encoding_list[k] for k in path_encoding_list]
    x_idx = list(range(len(encoding_list[0])))
    np_encoding = np.sum(np.array(encoding_list), axis=0)/np.sum(np.array(encoding_list))
    plt.bar(x_idx, np_encoding)
    plt.show()
    return np_encoding


def meta_neural_net_experiment_pipeline1(samples):
    search_space = build_datasets('nasbench_case1')
    test_data_total = search_space.generate_random_dataset_gin(num=samples, allow_isomorphisms=False)
    encoding_2 = path_encoding_distribution(test_data_total)
    print(f'The number of samples is {len(test_data_total)}.')
    return encoding_2


def meta_neural_net_experiment_pipeline2(samples):
    search_space = build_datasets('nasbench_data_distribution')
    test_data_total = search_space.generate_random_dataset(num=samples, allow_isomorphisms=False)
    encoding_2 = path_encoding_distribution(test_data_total)
    print(f'The number of samples is {len(test_data_total)}')
    return encoding_2


def meta_neural_net_experiment_full():
    search_space = build_datasets('nasbench_data_distribution')
    total_encodings = search_space.get_all_path_encooding()
    distributions = path_encoding_dict_distribution(total_encodings)
    print(f'The number of samples is {len(total_encodings)}')
    return distributions


def plot_combine_graph(encoding_1, encoding_2, idx=None, encoding_all=None):
    fig, ax = plt.subplots()
    ax.plot(idx, encoding_1, label='default')
    ax.plot(idx, encoding_2, label='new')
    ax.plot(idx, encoding_all, label='ground truth')
    ax.set_xlabel('path index', fontsize=12)
    ax.set_ylabel('value', fontsize=12)
    plt.legend(loc='upper right', fontsize=12)
    fig.set_dpi(300.0)
    # ax.grid()
    plt.show()


def plot_combine_graph2_log_scale(encoding_1, encoding_2, idx=None, encoding_all=None):
    fig, ax = plt.subplots()
    idx_encoding1 = np.argwhere(encoding_1 != 0)
    ax.scatter(idx_encoding1, np.log10(encoding_1, where=(encoding_1 != 0))[idx_encoding1], label='default', marker='*',
               s=9)

    idx_encoding2 = np.argwhere(encoding_2 != 0)
    ax.scatter(idx_encoding2, np.log10(encoding_2, where=(encoding_2 != 0))[idx_encoding2], label='new', marker='o',
               s=9)

    idx_encoding_all = np.argwhere(encoding_all != 0)
    ax.scatter(idx_encoding_all, np.log10(encoding_all, where=(encoding_all != 0))[idx_encoding_all],
               label='ground truth', marker='1', s=9)
    # ax.set(xlabel='path index', ylabel='log value',
    #        title='Comparison of paths distribution of different sampling pipeline.')
    ax.set_xlabel('path index', fontsize=12)
    ax.set_ylabel('p', fontsize=12)
    plt.legend(loc='upper right', fontsize=12)
    # plt.legend(loc='upper right')
    fig.set_dpi(300.0)

    log_val = np.log10(encoding_2, where=(encoding_2 != 0))

    print('#######'*30)
    for i in range(len(log_val)):
        print(encoding_2[i], log_val[i])
    plt.show()


def main(args):
    set_random_seed(args.seed)
    encoding_sample_pipeline1, x_idx = meta_neural_net_experiment_pipeline1(samples=args.sample_nums)
    encoding_sample_pipeline2, _ = meta_neural_net_experiment_pipeline2(samples=args.sample_nums)
    encoding_full = meta_neural_net_experiment_full()
    # with open(args.save_dir, 'wb') as f:
    #     pickle.dump(encoding_sample_pipeline1, f)
    #     pickle.dump(encoding_sample_pipeline2, f)
    #     pickle.dump(encoding_full, f)
    #     pickle.dump(x_idx, f)
    print(encoding_sample_pipeline1)
    print(encoding_sample_pipeline2)
    print(encoding_full)
    plot_combine_graph(encoding_1=encoding_sample_pipeline1, encoding_2=encoding_sample_pipeline2, idx=x_idx,
                       encoding_all=encoding_full)
    plot_combine_graph2_log_scale(encoding_1=encoding_sample_pipeline1, encoding_2=encoding_sample_pipeline2,
                                  idx=x_idx, encoding_all=encoding_full)
    print('kl divergence ################')
    # kl_divergence(encoding_sample_pipeline1, encoding_full)
    # kl_divergence(encoding_sample_pipeline2, encoding_full)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args for algorithms compare.')
    parser.add_argument('--sample_nums', type=int, default=5000, help='Number of samples for two pipeline')
    parser.add_argument('--seed', type=int, default=99, help='Number of samples for two pipeline')
    parser.add_argument('--save_dir', type=str, default='./compare_diff.pkl',
                        help='Sample file save dir.')

    args = parser.parse_args()
    main(args)
