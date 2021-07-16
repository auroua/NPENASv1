import os
import matplotlib.pyplot as plt
import argparse
from tools_ss_analysis.search_space_analysis import get_all_data
from nas_lib.utils.corr import get_kendalltau_coorlection
import numpy as np
from collections import Counter


search_space_cifar10 = 'nasbench_201_cifar10-valid.pkl'
search_space_nlp = 'nasbench_nlp_last.pkl'
search_space_asr = 'nasbench_asr_y.pkl'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args for visualize darts architecture')
    parser.add_argument('--base_path', type=str,
                        default='/home/albert_wei/fdisk_a/train_output_2021/npenas/search_space_analysis/',
                        help='The folder of search space files.')
    args = parser.parse_args()


    nasbench_201_cifar_10_path = os.path.join(args.base_path, search_space_cifar10)
    nasbench_nlp = os.path.join(args.base_path, search_space_nlp)
    nasbench_asr = os.path.join(args.base_path, search_space_asr)

    label_lists = ['NASBench-201', 'NASBench-NLP', 'NASBench-ASR']
    idxs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100][::-1]
    total_kt_lists = []
    fig, ax = plt.subplots(1)
    for i, p in enumerate([nasbench_201_cifar_10_path, nasbench_nlp, nasbench_asr]):
    # for i, p in enumerate([nasbench_201_cifar_10_path]):
        total_keys, dist_matrix, val_matrix, test_matrix, _, _ = get_all_data(p)
        avg_dist = np.triu(dist_matrix, 1)
        all_ones = np.tril(np.ones_like(dist_matrix) * -1)
        final_avg_dist = avg_dist + all_ones
        final_list = final_avg_dist.reshape((1, -1)).tolist()[0]
        avg_counter = Counter(final_list)
        avg_dict = {k: v for k, v in avg_counter.items() if k > 0}
        key_list = sorted(list(avg_dict.keys()))
        val_list = [avg_dict[k] for k in key_list]
        total_element = sum(val_list) * 1.0
        val_list_normalize = [v / total_element for v in val_list]
        plt.plot(key_list, val_list_normalize, label=label_lists[i], marker='s', linewidth=1, ms=3)
    fig.set_dpi(600.0)
    # plt.xlim(max(idxs), min(idxs))
    plt.legend(loc='upper right')
    ax.set_xlabel('Distance', fontsize=12)
    ax.set_ylabel('Normalized Counts')

    plt.show()