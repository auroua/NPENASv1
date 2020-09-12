import copy
import sys
from .predictor import gin_predictor_train_num_constrict_case1, gin_uncertainty_predictor_train_num_constrict_case1, \
    gin_predictor_train_num_constrict_case2, gin_uncertainty_train_num_constrict_case2
from .bananas import bananas_diff_training_nums_case1, bananas_training_num_diff_case2
from .algo_compare import compute_best_test_losses_case1, compute_best_test_losses_case2


def run_diff_training_architectures_num_case1(algo_params, metann_params, search_space, gpu=None, logger=None,
                                              search_strategy='gin_predictor'):
    if search_strategy == 'gin_uncertainty_predictor':
        return run_diff_gin_uncertainty_predictor_case1(algo_params=algo_params, search_space=search_space,
                                                        gpu=gpu, logger=logger)
    elif search_strategy == 'gin_predictor':
        return run_diff_gin_predictor_case1(algo_params=algo_params, search_space=search_space, gpu=gpu, logger=logger)
    elif search_strategy == 'bananas':
        return run_diff_bananas_predictor_case1(algo_params=algo_params, metann_params=metann_params,
                                                search_space=search_space, gpu=gpu, logger=logger)
    elif search_strategy == 'bananas_f':
        return run_diff_bananas_predictor_case1(algo_params=algo_params, metann_params=metann_params,
                                                search_space=search_space, gpu=gpu, logger=logger)
    else:
        print('invalid algorithm name')
        sys.exit()


def run_diff_gin_predictor_case1(algo_params, search_space, gpu, logger):
    ps = copy.deepcopy(algo_params)
    ps.pop('algo_name')
    data = gin_predictor_train_num_constrict_case1(search_space, gpu=gpu, logger=logger, **ps)
    k = 10
    if 'k' in ps:
        k = ps['k']
    return compute_best_test_losses_case1(data, k, ps['total_queries'])


def run_diff_gin_uncertainty_predictor_case1(algo_params, search_space, gpu, logger):
    ps = copy.deepcopy(algo_params)
    ps.pop('algo_name')
    data = gin_uncertainty_predictor_train_num_constrict_case1(search_space, gpu=gpu, logger=logger, **ps)
    k = 10
    if 'k' in ps:
        k = ps['k']
    return compute_best_test_losses_case1(data, k, ps['total_queries'])


def run_diff_bananas_predictor_case1(algo_params, metann_params, search_space, gpu=None, logger=None):
    mp = copy.deepcopy(metann_params)
    ps = copy.deepcopy(algo_params)
    mp.pop('search_space')
    ps.pop('algo_name')
    data = bananas_diff_training_nums_case1(search_space, mp, gpu=gpu, logger=logger, **ps)
    k = 10
    if 'k' in ps:
        k = ps['k']
    return compute_best_test_losses_case1(data, k, ps['total_queries'])


def run_diff_training_architectures_num_case2(algo_params, metann_params, search_space, gpu=None, logger=None,
                                              search_strategy='gin_predictor'):
    if search_strategy == 'gin_uncertainty_predictor':
        return run_diff_gin_uncertainty_predictor_case2(algo_params=algo_params, search_space=search_space,
                                                        gpu=gpu, logger=logger)
    elif search_strategy == 'gin_predictor':
        return run_diff_gin_predictor_case2(algo_params=algo_params, search_space=search_space, gpu=gpu, logger=logger)
    elif search_strategy == 'bananas':
        return run_diff_bananas_predictor_case2(algo_params=algo_params, metann_params=metann_params,
                                                search_space=search_space, gpu=gpu, logger=logger)
    elif search_strategy == 'bananas_f':
        return run_diff_bananas_predictor_case2(algo_params=algo_params, metann_params=metann_params,
                                                search_space=search_space, gpu=gpu, logger=logger)
    else:
        print('invalid algorithm name')
        sys.exit()


def run_diff_gin_predictor_case2(algo_params, search_space, gpu, logger):
    ps = copy.deepcopy(algo_params)
    ps.pop('algo_name')
    data = gin_predictor_train_num_constrict_case2(search_space, gpu=gpu, logger=logger, **ps)
    k = 10
    if 'k' in ps:
        k = ps['k']
    return compute_best_test_losses_case2(data, k, ps['total_queries'])


def run_diff_gin_uncertainty_predictor_case2(algo_params, search_space, gpu, logger):
    ps = copy.deepcopy(algo_params)
    ps.pop('algo_name')
    data = gin_uncertainty_train_num_constrict_case2(search_space, gpu=gpu, logger=logger, **ps)
    k = 10
    if 'k' in ps:
        k = ps['k']
    return compute_best_test_losses_case2(data, k, ps['total_queries'])


def run_diff_bananas_predictor_case2(algo_params, metann_params, search_space, gpu=None, logger=None):
    mp = copy.deepcopy(metann_params)
    ps = copy.deepcopy(algo_params)
    mp.pop('search_space')
    ps.pop('algo_name')
    data = bananas_training_num_diff_case2(search_space, mp, gpu=gpu, logger=logger, **ps)
    k = 10
    if 'k' in ps:
        k = ps['k']
    return compute_best_test_losses_case2(data, k, ps['total_queries'])
