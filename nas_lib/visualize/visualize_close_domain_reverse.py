import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"

model_lists_nasbench = ['Random', 'EA', 'BANANAS', 'BANANAS_F', 'NPENAS-BO', 'NPENAS-NP']
model_masks_nasbench = [True, True, True, True, True, True]


model_lists_scalar = ['scalar_10', 'scalar_30', 'scalar_50', 'scalar_70', 'scalar_100']
model_lists_scalar_mask = [True, True, True, True, True]


def convert2np(root_path, model_lists, end=None):
    total_dicts = dict()
    for m in model_lists:
        total_dicts[m] = []
    files = os.listdir(root_path)
    if end:
        files = list(files)[:end]
    else:
        files = list(files)
    for f in files:
        if 'log' in f:
            continue
        file_path = os.path.join(root_path, f)
        nested_dicts = dict()
        for m in model_lists:
            nested_dicts[m] = []
        with open(file_path, 'rb') as nf:
            try:
                algorithm_params, metann_params, results, walltimes = pickle.load(nf)
                print('######')
            except Exception as e:
                print(e)
                print(file_path)
            for i in range(len(results[0])):
                for idx, m in enumerate(model_lists):
                    nested_dicts[m].append(results[idx][i][1])
            for m in model_lists:
                total_dicts[m].append(nested_dicts[m])
    results_np = {m: np.array(total_dicts[m]) for m in model_lists}
    return results_np


def getmean(results_np, model_lists, category='mean'):
    if category == 'mean':
        results_mean = {m: np.mean(results_np[m], axis=0) for m in model_lists}
    elif category == 'medium':
        results_mean = {m: np.median(results_np[m], axis=0) for m in model_lists}
    elif category == 'percentile':
        results_mean = {m: np.percentile(results_np[m], 50, axis=0) for m in model_lists}
    else:
        raise ValueError('this type operation is not supported!')
    return results_mean


def get_quantile(results_np, model_lists, divider=30):
    results_quantile = {m: np.percentile(results_np[m], divider, axis=0) for m in model_lists}
    return results_quantile


def get_bounder(total_mean, quantile_30, quantile_70, model_lists, absolute=False):
    bound_dict = dict()
    for m in model_lists:
        bound_dict[m] = np.stack([(total_mean[m]-quantile_30[m]),
                                  (quantile_70[m]-total_mean[m])], axis=0)
    return bound_dict


def draw_plot_nasbench_101(root_path, model_lists, model_masks, draw_type='ERRORBAR', verbose=1):
    # draw_type  ERRORBAR, MEANERROR
    np_datas_dict = convert2np(root_path, model_lists=model_lists, end=None)
    # EA_reuslt = np_datas_dict['EA']
    # print(EA_reuslt.shape)
    # print(np.max(EA_reuslt, axis=0))
    # print(np.min(EA_reuslt, axis=0))
    np_mean_dict = getmean(np_datas_dict, model_lists=model_lists)
    np_quantile_30 = get_quantile(np_datas_dict, model_lists=model_lists, divider=30)
    np_quantile_70 = get_quantile(np_datas_dict, model_lists=model_lists, divider=70)

    if verbose:
        for k, v in np_mean_dict.items():
            print(k)
            print('30 quantile value')
            print(np_quantile_30[k])
            print('mean')
            print(v)
            print('70 quantile value')
            print(np_quantile_70[k])
            print('###############')
    np_bounds = get_bounder(np_mean_dict, np_quantile_30, np_quantile_70, model_lists=model_lists, absolute=True)
    # get data mean
    idx = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150])
    fig, ax = plt.subplots(1)
    upperlimits = [True] * 15
    lowerlimits = [True] * 15
    if draw_type == 'ERRORBAR':
        for j, m in enumerate(model_lists):
            if model_masks[j]:
                # plt.errorbar(idx, np_mean_dict[m], yerr=np_bounds[m], uplims=upperlimits, lolims=lowerlimits,
                #              label=m, capthick=2)
                plt.errorbar(idx, np_mean_dict[m], yerr=np_bounds[m], label=m, capsize=3, capthick=2)
    elif draw_type == 'MEANERROR':
        for j, m in enumerate(model_lists):
            if model_masks[j]:
                plt.plot(idx, np_mean_dict[m], label=m, marker='s', linewidth=1, ms=3)     # fmt='o',
    # ax.set_yticks(np.arange(92.5, 94.4, 0.2))
    ax.set_yticks(np.arange(5.8, 7.4, 0.2))
    # ax.grid(True)
    fig.set_dpi(300.0)
    ax.set_xlabel('number of samples')
    ax.set_ylabel('testing error of best neural net')
    plt.legend(loc='upper right')
    # plt.grid(b=True, which='major', color='#666699', linestyle='--')
    plt.show()


def draw_plot_nasbench_101_diff_training(root_path, model_lists, model_masks, search_strategy='gin_predictor',
                                         draw_type='ERRORBAR', verbose=1):
    # draw_type  ERRORBAR, MEANERROR
    np_datas_dict = convert2np(root_path, model_lists=model_lists, end=None)
    np_mean_dict = getmean(np_datas_dict, model_lists=model_lists)
    np_quantile_30 = get_quantile(np_datas_dict, model_lists=model_lists, divider=30)
    np_quantile_70 = get_quantile(np_datas_dict, model_lists=model_lists, divider=70)

    if verbose:
        for k, v in np_mean_dict.items():
            print(k)
            print('30 quantile value')
            print(np_quantile_30[k])
            print('mean')
            print(v)
            print('70 quantile value')
            print(np_quantile_70[k])
            print('###############')
    np_bounds = get_bounder(np_mean_dict, np_quantile_30, np_quantile_70, model_lists=model_lists, absolute=True)
    # get data mean
    idx = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150])
    fig, ax = plt.subplots(1)
    if draw_type == 'ERRORBAR':
        for j, m in enumerate(model_lists):
            if model_masks[j]:
                plt.errorbar(idx, np_mean_dict[m], yerr=np_bounds[m], label=m, capsize=3, capthick=2)
    elif draw_type == 'MEANERROR':
        for j, m in enumerate(model_lists):
            if model_masks[j]:
                plt.plot(idx, np_mean_dict[m], label=m, marker='s', linewidth=1, ms=3)     # fmt='o',
    # ax.set_yticks(np.arange(92.5, 94.4, 0.2))
    ax.set_yticks(np.arange(5.8, 7.4, 0.2))
    # ax.grid(True)
    fig.set_dpi(300.0)
    ax.set_title(search_strategy)
    ax.set_xlabel('number of samples')
    ax.set_ylabel('testing error of best neural net')
    plt.legend(loc='upper right')
    # plt.grid(b=True, which='major', color='#666699', linestyle='--')
    plt.show()


def draw_plot_nasbench_201(root_path, model_lists, model_masks, draw_type='ERRORBAR', verbose=1):
    # draw_type  ERRORBAR, MEANERROR
    np_datas_dict = convert2np(root_path, end=None, model_lists=model_lists)
    np_mean_dict = getmean(np_datas_dict, model_lists=model_lists)
    np_quantile_30 = get_quantile(np_datas_dict, model_lists=model_lists, divider=30)
    np_quantile_70 = get_quantile(np_datas_dict, model_lists=model_lists, divider=70)

    if verbose:
        for k, v in np_mean_dict.items():
            print(k)
            print('30 quantile value')
            print(np_quantile_30[k])
            print('mean')
            print(v)
            print('70 quantile value')
            print(np_quantile_70[k])
            print('###############')
    np_bounds = get_bounder(np_mean_dict, np_quantile_30, np_quantile_70, model_lists=model_lists, absolute=True)
    # get data mean
    idx = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    fig, ax = plt.subplots(1)
    upperlimits = [True] * 10
    lowerlimits = [True] * 10
    if draw_type == 'ERRORBAR':
        for j, m in enumerate(model_lists):
            if model_masks[j]:
                # plt.errorbar(idx, np_mean_dict[m], yerr=np_bounds[m], uplims=upperlimits, lolims=lowerlimits,
                #              label=m, capthick=2)
                plt.errorbar(idx, np_mean_dict[m], yerr=np_bounds[m], label=m, capsize=6, capthick=2)
    elif draw_type == 'MEANERROR':
        for j, m in enumerate(model_lists):
            if model_masks[j]:
                plt.plot(idx, np_mean_dict[m], label=m, marker='s', linewidth=1, ms=3)     # fmt='o',
    ax.set_yticks(np.arange(8.9, 10.9, 0.2))
    # ax.grid(True)
    fig.set_dpi(300.0)
    ax.set_xlabel('number of samples')
    ax.set_ylabel('testing error of best neural net')
    plt.legend(loc='upper right')
    # plt.grid(b=True, which='major', color='#666699', linestyle='--')
    plt.show()


def draw_plot_priori_scalar(root_path, model_lists, model_masks, draw_type='ERRORBAR', verbose=1):
    # draw_type  ERRORBAR, MEANERROR
    np_datas_dict = convert2np(root_path, end=None, model_lists=model_lists)
    np_mean_dict = getmean(np_datas_dict, model_lists=model_lists)
    np_quantile_30 = get_quantile(np_datas_dict, model_lists=model_lists, divider=30)
    np_quantile_70 = get_quantile(np_datas_dict, model_lists=model_lists, divider=70)

    if verbose:
        for k, v in np_mean_dict.items():
            print(k)
            print('30 quantile value')
            print(np_quantile_30[k])
            print('mean')
            print(v)
            print('70 quantile value')
            print(np_quantile_70[k])
            print('###############')
    np_bounds = get_bounder(np_mean_dict, np_quantile_30, np_quantile_70, model_lists=model_lists, absolute=True)
    # get data mean
    idx = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150])
    fig, ax = plt.subplots(1)
    upperlimits = [True] * 15
    lowerlimits = [True] * 15
    if draw_type == 'ERRORBAR':
        for j, m in enumerate(model_lists):
            if model_masks[j]:
                # plt.errorbar(idx, np_mean_dict[m], yerr=np_bounds[m], uplims=upperlimits, lolims=lowerlimits,
                #              label=m, capthick=2)
                plt.errorbar(idx, np_mean_dict[m], yerr=np_bounds[m], label=m, capsize=3, capthick=2)
    elif draw_type == 'MEANERROR':
        for j, m in enumerate(model_lists):
            if model_masks[j]:
                plt.plot(idx, np_mean_dict[m], label=m, marker='s', linewidth=1, ms=3)     # fmt='o',
    ax.set_yticks(np.arange(5.6, 7.4, 0.2))
    # ax.grid(True)
    fig.set_dpi(300.0)
    ax.set_xlabel('number of samples', fontsize=11)
    ax.set_ylabel('test error of best neural net', fontsize=11)
    plt.legend(loc='upper right', fontsize=11)
    # plt.grid(b=True, which='major', color='#666699', linestyle='--')
    plt.show()


def draw_plot_evaluation_compare(root_path, model_lists, model_masks, draw_type='ERRORBAR', verbose=1):
    # draw_type  ERRORBAR, MEANERROR
    np_datas_dict = convert2np(root_path, end=None, model_lists=model_lists)
    np_mean_dict = getmean(np_datas_dict, model_lists=model_lists)
    np_quantile_30 = get_quantile(np_datas_dict, model_lists=model_lists, divider=30)
    np_quantile_70 = get_quantile(np_datas_dict, model_lists=model_lists, divider=70)

    if verbose:
        for k, v in np_mean_dict.items():
            print(k)
            print('30 quantile value')
            print(np_quantile_30[k])
            print('mean')
            print(v)
            print('70 quantile value')
            print(np_quantile_70[k])
            print('###############')
    np_bounds = get_bounder(np_mean_dict, np_quantile_30, np_quantile_70, model_lists=model_lists, absolute=True)
    # get data mean
    idx = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150])
    fig, ax = plt.subplots(1)
    upperlimits = [True] * 15
    lowerlimits = [True] * 15
    if draw_type == 'ERRORBAR':
        for j, m in enumerate(model_lists):
            if model_masks[j]:
                # plt.errorbar(idx, np_mean_dict[m], yerr=np_bounds[m], uplims=upperlimits, lolims=lowerlimits,
                #              label=m, capthick=2)
                plt.errorbar(idx, np_mean_dict[m], yerr=np_bounds[m], label=m, capsize=3, capthick=2)
    elif draw_type == 'MEANERROR':
        for j, m in enumerate(model_lists):
            if model_masks[j]:
                plt.plot(idx, np_mean_dict[m], label=m, marker='s', linewidth=1, ms=3)     # fmt='o',
    ax.set_yticks(np.arange(5.7, 6.9, 0.2))
    # ax.grid(True)
    fig.set_dpi(300.0)
    ax.set_xlabel('Number of architectures in the pool', fontsize=12)
    ax.set_ylabel('Test error of best architecture (%)', fontsize=12)
    plt.legend(loc='upper right', fontsize=12)
    # plt.grid(b=True, which='major', color='#666699', linestyle='--')
    plt.show()


if __name__ == '__main__':
    root_path = '/home/albert_wei/Disk_A/train_output_npenas/close_domain_case1/'
    draw_plot_nasbench_101(root_path, draw_type='MEANERROR', model_lists=model_lists_nasbench,
                           model_masks=model_masks_nasbench)