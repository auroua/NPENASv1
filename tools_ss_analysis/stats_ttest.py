import pickle
from scipy.stats import ttest_ind
import scipy.stats as stats
import numpy as np
import argparse


dataset_keys = ['Random', 'REA', 'BANANAS', 'BANANAS_F', 'NPENAS-BO', 'NPENAS-NP', 'NPENAS-NP-NEW', 'ORACLE']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args for visualize darts architecture')
    parser.add_argument('--file_path', type=str,
                        default='/home/albert_wei/fdisk_a/train_output_2021/npenas/final_results/npenas_results_pkl/nasbench_201_imagenet_gp.pkl',
                        help='The folder of search space files.')
    args = parser.parse_args()

    with open(args.file_path, 'rb') as f:
        dataset = pickle.load(f)
    if 'BANANAS-PE' not in dataset:
        bananas = dataset['BANANAS']
    else:
        bananas = dataset['BANANAS-PE']
    npenas_bo = dataset['NPENAS-BO']
    npenas_np = dataset['NPENAS-NP']
    # gp_adj = dataset['GP-ADJ']
    # gp_nasbot = dataset['GP-NASBOT']

    algo_1 = bananas
    algo_2 = npenas_np

    mean_1 = np.mean(algo_1, axis=0)
    mean_2 = np.mean(algo_2, axis=0)

    std_1 = np.std(algo_1, axis=0)
    std_2 = np.std(algo_2, axis=0)

    print('###'*10, ' mean ', '###'*10)
    print(mean_1)
    print(mean_2)

    print('###'*10, ' std ', '###'*10)

    print(std_1)
    print(std_2)

    for i in range(bananas.shape[1]):
        s, p = stats.levene(algo_1[:, i], algo_2[:, i])
        if p < 0.05:
            s, p = ttest_ind(algo_1[:, i], algo_2[:, i], equal_var=True,
                             alternative='two-sided')
            print(f's value is {s}, p value is {p}')
        else:
            s, p = ttest_ind(algo_1[:, i], algo_2[:, i], equal_var=False,
                             alternative='two-sided')
            print(f's value is {s}, p value is {p}')
        print('######')
