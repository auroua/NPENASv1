import argparse
import pickle
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from nas_lib.utils.corr import get_kendalltau_coorlection


cm = 1/2.54

# # nasbench-nlp
# SEARCH_SPACE = 'NASBench-NLP'
# AX1_XLABEL = 'Distance'
# AX1_YLABEL = 'Normalize Count'
# AX2_XLABEL = 'Distance'
# AX2_YLABEL = 'Average Error Rate'
# AX3_XLABEL = 'Error Rate'
# AX3_YLABEL = 'Kendall Tau Correlation'


# nasbench-asr
SEARCH_SPACE = 'NASBench-ASR'
AX1_XLABEL = 'Distance'
AX1_YLABEL = 'Normalize Count'
AX2_XLABEL = 'Distance'
AX2_YLABEL = 'Average Phoneme Error Rate'
AX3_XLABEL = 'Phoneme Error Rate'
AX3_YLABEL = 'Kendall Tau Correlation'


def get_all_data(save_path):
    with open(save_path, 'rb') as f:
        total_keys = pickle.load(f)
        dist_matrix = pickle.load(f)
        val_matrix = pickle.load(f)
        test_matrix = pickle.load(f)
        val_list = pickle.load(f)
        test_list = pickle.load(f)
    return total_keys, dist_matrix, val_matrix, test_matrix, val_list, test_list


def get_focus_coefficient(dist_matrix, val_matrix, test_matrix, validate_list, testing_list):
    avg_dist = np.triu(dist_matrix, 1)
    all_ones = np.tril(np.ones_like(dist_matrix) * -1)
    final_avg_dist = avg_dist + all_ones
    final_list = final_avg_dist.reshape((1, -1)).tolist()[0]
    avg_counter = Counter(final_list)
    avg_dict = {k: v for k, v in avg_counter.items() if k > 0}
    key_list = sorted(list(avg_dict.keys()))
    val_list = [avg_dict[k] for k in key_list]
    total_element = sum(val_list) * 1.0
    val_list_normalize = [v/total_element for v in val_list]

    val_dist_list = []
    test_dist_list = []
    val_dist_mean_list = []
    test_dist_mean_list = []

    for k in key_list:
        mask = avg_dist == k
        val_dist = val_matrix[mask]
        test_dist = test_matrix[mask]
        val_dist_list.append(val_dist)
        test_dist_list.append(test_dist)
        val_dist_mean_list.append(np.mean(val_dist))
        test_dist_mean_list.append(np.mean(test_dist))
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16.256*cm, 25*cm))
    fig.set_dpi(600.0)
    ax1.set_xlabel(AX1_XLABEL)
    ax1.set_ylabel(AX1_YLABEL)
    ax1.set_title(SEARCH_SPACE)
    ax1.plot(key_list, val_list_normalize, 'o-')
    ax2.set_xlabel(AX2_XLABEL)
    ax2.set_ylabel(AX2_YLABEL)
    ax2.plot(key_list, val_dist_mean_list, '.-', label='val')
    ax2.plot(key_list, test_dist_mean_list, '*-', color='r', label='test')
    ax2.legend(loc='upper left')

    min_val, max_val = min(validate_list), max(validate_list)
    step_size = (max_val - min_val) / 10
    step_wise_list = [min_val + step_size*i for i in range(1, 10)][::-1]
    step_wise_list.insert(0, max_val)
    kt_list = []
    for thr in step_wise_list:
        thr_idx = [i for i, v in enumerate(validate_list) if v < thr]
        kt_list.append(get_kendalltau_coorlection([validate_list[i] for i in thr_idx], [testing_list[j] for j in thr_idx])[0])
    ax3.set_xlabel(AX3_XLABEL)
    ax3.set_ylabel(AX3_YLABEL)
    ax3.plot(step_wise_list, kt_list, 'o-')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args for search space analysis.')
    parser.add_argument('--search_space', type=str, default='nasbench_201',
                        choices=['nasbench_case2', 'nasbench_201', 'nasbench_nlp', 'nasbench_asr'],
                        help='search space type.')
    parser.add_argument('--dataset', type=str, default='cifar10-valid',
                        choices=['cifar10-valid', 'cifar100', 'ImageNet16-120'],
                        help='dataset name of nasbench-201.')
    parser.add_argument('--nasbench_nlp_type', type=str,
                        default='last', choices=['best', 'last'],
                        help='name of output files')
    parser.add_argument('--filter_none', type=str,
                        default='y', choices=['y', 'n'],
                        help='name of output files')
    parser.add_argument('--thr', type=float,
                        default=0.3, help='the thr of correlation.')
    parser.add_argument('--save_dir', type=str,
                        default='/home/albert_wei/fdisk_a/train_output_2021/npenas/search_space_analysis/nasbench_asr_y.pkl',
                        # default='/home/albert_wei/fdisk_a/train_output_2021/npenas/search_space_analysis/nasbench_201_cifar10-valid.pkl',
                        # default='/home/albert_wei/fdisk_a/train_output_2021/npenas/search_space_analysis/nasbench_nlp_last.pkl',
                        help='output directory')
    args = parser.parse_args()

    total_keys, dist_matrix, val_matrix, test_matrix, val_list, test_list = get_all_data(args.save_dir)

    kt_corr = get_kendalltau_coorlection(val_list, test_list)[0]
    bigger_thr = [i for i, v in enumerate(val_list) if v < args.thr]
    kt_corr_bigger = get_kendalltau_coorlection([val_list[i] for i in bigger_thr], [test_list[j] for j in bigger_thr])[0]
    print(f'the kendall tau correlation of validate and test accuracy is {kt_corr} and thr '
          f'correlation is {kt_corr_bigger}!')

    focus_coefficient = get_focus_coefficient(dist_matrix, val_matrix, test_matrix, val_list, test_list)