import os
import argparse
import matplotlib.pyplot as plt
from tools_ss_analysis.search_space_analysis import get_all_data
from nas_lib.utils.corr import get_kendalltau_coorlection

search_space_nasbench_101 = 'nasbench_101.pkl'
search_space_cifar10 = 'nasbench_201_cifar10-valid.pkl'
search_space_cifar100 = 'nasbench_201_cifar100.pkl'
search_space_imagenet = 'nasbench_201_ImageNet16-120.pkl'
search_space_nlp = 'nasbench_nlp.pkl'
search_space_asr = 'nasbench_asr.pkl'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args for visualize darts architecture')
    parser.add_argument('--search_space', type=str, default='others',
                        choices=['nasbench_nlp', 'others'],
                        help='The search space.')
    parser.add_argument('--base_path', type=str,
                        default='/media/albert_wei/HP_SSD/Backups/Papers_Backup#####/npenas/backups_npenas_paper_2021_7_14/result_files/search_space_analysis/',
                        help='The folder of search space files.')
    args = parser.parse_args()

    nasbench_101 = os.path.join(args.base_path, search_space_nasbench_101)
    nasbench_201_cifar_10_path = os.path.join(args.base_path, search_space_cifar10)
    nasbench_201_cifar_100_path = os.path.join(args.base_path, search_space_cifar100)
    nasbench_201_imagenet_path = os.path.join(args.base_path, search_space_imagenet)
    nasbench_nlp = os.path.join(args.base_path, search_space_nlp)
    nasbench_asr = os.path.join(args.base_path, search_space_asr)

    if args.search_space == 'others':
        label_lists = ['NASBench-101', 'NASBench-201-CIFAR-10', 'NASBench-201-CIFAR-100', 'NASBench-201-ImageNet',
                       'NASBench-ASR']
        idxs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100][::-1]
    elif args.search_space == 'nasbench_nlp':
        label_lists = ['NASBench-NLP']
        min_idx, max_idx = 0, 0
    else:
        raise NotImplementedError()
    total_kt_lists = []

    if args.search_space == 'others':
        for i, p in enumerate([nasbench_101, nasbench_201_cifar_10_path, nasbench_201_cifar_100_path, nasbench_201_imagenet_path,
                  nasbench_asr]):
            total_keys, _, _, _, validate_list, test_list = get_all_data(p)
            if i == 4:
                validate_list = [v*100 for v in validate_list]
                test_list = [t*100 for t in test_list]
            if i == 0:
                validate_list = [100*(1-v) for v in validate_list]
                test_list = [100*(1-t) for t in test_list]
            kt_list = []
            for thr in idxs:
                thr_idx = [i for i, v in enumerate(validate_list) if v < thr]
                kt_list.append(
                    get_kendalltau_coorlection([validate_list[i] for i in thr_idx], [test_list[j] for j in thr_idx])[0])
            total_kt_lists.append(kt_list)
        fig, ax = plt.subplots(1)

        for idx, kt_list in enumerate(total_kt_lists):
            m = label_lists[idx]
            plt.plot(idxs, kt_list, label=m, marker='s', linewidth=1, ms=3)
        fig.set_dpi(600.0)
        plt.xlim(max(idxs), min(idxs))
        plt.legend(loc='lower left')
        ax.set_xlabel('Validation Error Rate [%]', fontsize=12)
        ax.set_ylabel('Kendall Tau Correlation')
        plt.show()
    elif args.search_space == 'nasbench_nlp':
        for i, p in enumerate([nasbench_nlp]):
            total_keys, _, _, _, validate_list, test_list = get_all_data(p)
            min_val, max_val = min(validate_list), max(validate_list)
            min_idx = min_val
            max_idx = max_val
            step_size = (max_val - min_val) / 10
            step_wise_list = [min_val + step_size * i for i in range(1, 10)][::-1]
            step_wise_list.insert(0, max_val)
            kt_list = []
            for thr in step_wise_list:
                thr_idx = [i for i, v in enumerate(validate_list) if v < thr]
                kt_list.append(
                    get_kendalltau_coorlection([validate_list[i] for i in thr_idx], [test_list[j] for j in thr_idx])[0])
            total_kt_lists.append(kt_list)
        fig, ax = plt.subplots(1)

        m = label_lists[0]
        plt.plot(step_wise_list, total_kt_lists[0], label=m, marker='s', linewidth=1, ms=3)
        fig.set_dpi(600.0)
        plt.xlim(max_idx, min_idx)
        plt.legend(loc='lower left')
        ax.set_xlabel('Validation Log Perplexity', fontsize=12)
        ax.set_ylabel('Kendall Tau Correlation')
        plt.show()
    else:
        raise NotImplementedError()