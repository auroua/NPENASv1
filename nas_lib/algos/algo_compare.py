# Copyright (c) XiDian University and Xi'an University of Posts&Telecommunication. All Rights Reserved

import copy
import sys
from .random import random_search_case1, random_search_case2, random_search_nasbench_201
from .evolution import evolution_search_case1, evolution_search_case2, evolution_search_nasbench201, \
    evolution_search_compare_case1, evolution_search_compare_case2, evolution_search_nasbench_nlp, \
    evolution_search_nasbench_asr
from .bananas import bananas_case1, bananas_case2, bananas_nasbench_201, bananas_nasbench_nlp, bananas_nasbench_asr
from .predictor import gin_uncertainty_case1, gin_uncertainty_case2, gin_predictor_case1, gin_predictor_case2, \
    gin_uncertainty_predictor_nasbench_201, gin_predictor_nasbench_201, gin_predictor_scalar_case2, \
    gin_uncertainty_predictor_nasbench_nlp, gin_predictor_nasbench_nlp, gin_uncertainty_predictor_nasbench_asr, \
    gin_predictor_nasbench_asr
from .oracle import oracle_nasbench_101_case2, oracle_nasbench_201, oracle_nasbench_asr, oracle_nasbench_nlp, \
    oracle_nasbench_101_case1
from .gp_bayesopt import gp_bayesopt_nasbench_101, gp_bayesopt_nasbench_201, gp_bayesopt_nasbench_nlp, \
    gp_bayesopt_nasbench_asr, gp_bayesopt_nasbench_101_case1
from .predictor_new import gin_predictor_new_nasbench_101, gin_predictor_new_nasbench_201, \
    gin_predictor_new_nasbench_nlp, gin_predictor_new_nasbench_asr
import numpy as np


def run_nas_algos_case1(algo_params, metann_params, search_space, gpu=None, logger=None, with_details='F',
                        record_kt='F', record_mutation='F'):
    mp = copy.deepcopy(metann_params)
    ps = copy.deepcopy(algo_params)
    algo_name = ps.pop('algo_name')
    if algo_name == 'random':
        data = random_search_case1(search_space, logger=logger, **ps)
    elif algo_name == 'evolution':
        data = evolution_search_case1(search_space, logger=logger, **ps)
    elif algo_name == 'bananas':
        mp.pop('search_space')
        data = bananas_case1(search_space, mp, gpu=gpu, logger=logger, **ps)
    elif algo_name == 'bananas_f':
        mp.pop('search_space')
        data = bananas_case1(search_space, mp, gpu=gpu, logger=logger, **ps)
    elif algo_name == 'gin_uncertainty_predictor':
        data = gin_uncertainty_case1(search_space, gpu=gpu, logger=logger, **ps)
    elif algo_name == 'gin_predictor':
        data = gin_predictor_case1(search_space, gpu=gpu, logger=logger, **ps)
    elif algo_name == 'oracle':
        data, full_data = oracle_nasbench_101_case1(search_space, logger=logger, record_mutation=record_mutation, **ps)
    elif algo_name == 'gp_bayesopt':
        data = gp_bayesopt_nasbench_101_case1(search_space, logger=logger, tmpdir=f'./temp_nasbench_101_case1_{gpu}', **ps)
    else:
        print('invalid algorithm name')
        sys.exit()
    k = 10
    if 'k' in ps:
        k = ps['k']
    if with_details == 'T':
        return compute_best_test_losses_case1_with_details(data, k, ps['total_queries'])
    else:
        return compute_best_test_losses_case1(data, k, ps['total_queries'])


def run_nas_algos_case2(algo_params, metann_params, search_space, gpu=None, logger=None, with_details='F',
                        record_kt='F', record_mutation='F'):
    mp = copy.deepcopy(metann_params)
    ps = copy.deepcopy(algo_params)
    algo_name = ps.pop('algo_name')
    full_data = None
    if algo_name == 'random':
        data = random_search_case2(search_space, logger=logger, **ps)
    elif algo_name == 'evolution':
        data, full_data = evolution_search_case2(search_space, logger=logger, record_kt=record_kt,
                                                 record_mutation=record_mutation, **ps)
    elif algo_name == 'bananas':
        mp.pop('search_space')
        data, full_data = bananas_case2(search_space, mp, gpu=gpu, logger=logger, record_kt=record_kt,
                                        record_mutation=record_mutation, **ps)
    elif algo_name == 'bananas_f':
        mp.pop('search_space')
        data, full_data = bananas_case2(search_space, mp, gpu=gpu, logger=logger, record_kt=record_kt,
                                        record_mutation=record_mutation, **ps)
    elif algo_name == 'gin_uncertainty_predictor':
        data, full_data = gin_uncertainty_case2(search_space, gpu=gpu, logger=logger, record_kt=record_kt,
                                                record_mutation=record_mutation, **ps)
    elif algo_name == 'gin_predictor':
        data, full_data = gin_predictor_case2(search_space, gpu=gpu, logger=logger, record_kt=record_kt,
                                              record_mutation=record_mutation, **ps)
    elif algo_name == 'gin_predictor_new':
        data, full_data = gin_predictor_new_nasbench_101(search_space, gpu=gpu, logger=logger, record_kt=record_kt,
                                                         record_mutation=record_mutation, **ps)
    elif algo_name == 'oracle':
        data, full_data = oracle_nasbench_101_case2(search_space, logger=logger, record_mutation=record_mutation, **ps)
    elif algo_name == 'gp_bayesopt':
        data = gp_bayesopt_nasbench_101(search_space, logger=logger, tmpdir=f'./temp_nasbench_101_{gpu}', **ps)
    else:
        print('invalid algorithm name')
        sys.exit()
    k = 10
    if 'k' in ps:
        k = ps['k']
    # if with_details == 'T':
    #     return compute_best_test_losses_case2_with_details(data, k, ps['total_queries'])
    # else:
    #     return compute_best_test_losses_case2(data, k, ps['total_queries'])
    if full_data is not None:
        return compute_best_test_losses_case2(data, k, ps['total_queries']), full_data
    else:
        return compute_best_test_losses_case2(data, k, ps['total_queries'])


def run_nas_algos_nasbench_201(algo_params, metann_params, search_space, dataname='cifar100', gpu=None, logger=None,
                               record_kt='F', record_mutation='F'):
    mp = copy.deepcopy(metann_params)
    ps = copy.deepcopy(algo_params)
    algo_name = ps.pop('algo_name')
    full_data = None
    if algo_name == 'random':
        data = random_search_nasbench_201(search_space, logger=logger, **ps)
    elif algo_name == 'evolution':
        data, full_data = evolution_search_nasbench201(search_space, logger=logger, record_kt=record_kt,
                                               record_mutation=record_mutation, **ps)
    elif algo_name == 'bananas':
        mp.pop('search_space')
        data, full_data = bananas_nasbench_201(search_space, mp, gpu=gpu, logger=logger, record_kt=record_kt,
                                               record_mutation=record_mutation, **ps)
    elif algo_name == 'bananas_f':
        mp.pop('search_space')
        mp['layer_width'] = 20
        data, full_data = bananas_nasbench_201(search_space, mp, gpu=gpu, logger=logger, record_kt=record_kt,
                                               record_mutation=record_mutation, **ps)
    elif algo_name == 'gin_uncertainty_predictor':
        data, full_data = gin_uncertainty_predictor_nasbench_201(search_space, gpu=gpu, logger=logger,
                                                                 record_kt=record_kt,
                                                                 record_mutation=record_mutation, **ps)
    elif algo_name == 'gin_predictor':
        data, full_data = gin_predictor_nasbench_201(search_space, dataname=dataname, gpu=gpu, logger=logger,
                                                     record_kt=record_kt, record_mutation=record_mutation, **ps)
    elif algo_name == 'gin_predictor_new':
        data, full_data = gin_predictor_new_nasbench_201(search_space, dataname=dataname, gpu=gpu, logger=logger,
                                                         record_kt=record_kt, record_mutation=record_mutation, **ps)
    # elif algo_name == 'gin_predictor_new_2':
    #     data, full_data = gin_predictor_new_2_nasbench_201(search_space, dataname=dataname, gpu=gpu, logger=logger,
    #                                                      record_kt=record_kt, record_mutation=record_mutation, **ps)
    elif algo_name == 'oracle':
        data, full_data = oracle_nasbench_201(search_space, logger=logger, record_mutation=record_mutation, **ps)
    elif algo_name == 'gp_bayesopt':
        data = gp_bayesopt_nasbench_201(search_space, logger=logger, tmpdir=f'./temp_nasbench_201_{gpu}', **ps)
    else:
        print('invalid algorithm name')
        sys.exit()
    k = 10
    if 'k' in ps:
        k = ps['k']
    if full_data is not None:
        return compute_best_test_losses_case2(data, k, ps['total_queries']), full_data
    else:
        return compute_best_test_losses_case2(data, k, ps['total_queries'])


def run_nas_algos_nasbench_nlp(algo_params, metann_params, search_space, gpu=None, logger=None, record_kt='F',
                               record_mutation='F'):
    mp = copy.deepcopy(metann_params)
    ps = copy.deepcopy(algo_params)
    algo_name = ps.pop('algo_name')
    full_data = None
    if algo_name == 'random':
        data = random_search_nasbench_201(search_space, logger=logger, **ps)
    elif algo_name == 'evolution':
        data, full_data = evolution_search_nasbench_nlp(search_space, logger=logger, record_kt=record_kt,
                                                        record_mutation=record_mutation, **ps)
    elif algo_name == 'bananas':
        mp.pop('search_space')
        data, full_data = bananas_nasbench_nlp(search_space, mp, gpu=gpu, logger=logger, record_kt=record_kt,
                                               record_mutation=record_mutation, **ps)
    elif algo_name == 'bananas_f':
        mp.pop('search_space')
        mp['layer_width'] = 20
        data, full_data = bananas_nasbench_nlp(search_space, mp, gpu=gpu, logger=logger, record_kt=record_kt,
                                               record_mutation=record_mutation, **ps)
    elif algo_name == 'gin_uncertainty_predictor':
        data, full_data = gin_uncertainty_predictor_nasbench_nlp(search_space, gpu=gpu, logger=logger,
                                                                 record_kt=record_kt,
                                                                 record_mutation=record_mutation, **ps)
    elif algo_name == 'gin_predictor':
        data, full_data = gin_predictor_nasbench_nlp(search_space, gpu=gpu, logger=logger, record_kt=record_kt,
                                                     record_mutation=record_mutation, **ps)
    elif algo_name == 'gin_predictor_new':
        data, full_data = gin_predictor_new_nasbench_nlp(search_space, gpu=gpu, logger=logger, record_kt=record_kt,
                                                         record_mutation=record_mutation, **ps)
    elif algo_name == 'oracle':
        data, full_data = oracle_nasbench_nlp(search_space, logger=logger, record_mutation=record_mutation, **ps)
    elif algo_name == 'gp_bayesopt':
        data = gp_bayesopt_nasbench_nlp(search_space, logger=logger, tmpdir=f'./temp_nasbench_nlp_{gpu}', **ps)
    else:
        print('invalid algorithm name')
        sys.exit()
    k = 10
    if 'k' in ps:
        k = ps['k']
    if full_data is not None:
        return compute_best_test_losses_case2(data, k, ps['total_queries']), full_data
    else:
        return compute_best_test_losses_case2(data, k, ps['total_queries'])


def run_nas_algos_nasbench_asr(algo_params, metann_params, search_space, gpu=None, logger=None,
                               record_kt='F', record_mutation='F'):
    mp = copy.deepcopy(metann_params)
    ps = copy.deepcopy(algo_params)
    algo_name = ps.pop('algo_name')
    full_data = None
    if algo_name == 'random':
        data = random_search_nasbench_201(search_space, logger=logger, **ps)
    elif algo_name == 'evolution':
        data, full_data = evolution_search_nasbench_asr(search_space, logger=logger, record_kt=record_kt,
                                                        record_mutation=record_mutation, **ps)
    elif algo_name == 'bananas':
        mp.pop('search_space')
        data, full_data = bananas_nasbench_asr(search_space, mp, gpu=gpu, logger=logger, record_kt=record_kt,
                                               record_mutation=record_mutation, **ps)
    elif algo_name == 'bananas_f':
        mp.pop('search_space')
        mp['layer_width'] = 20
        data, full_data = bananas_nasbench_asr(search_space, mp, gpu=gpu, logger=logger, record_kt=record_kt,
                                               record_mutation=record_mutation, **ps)
    elif algo_name == 'gin_uncertainty_predictor':
        data, full_data = gin_uncertainty_predictor_nasbench_asr(search_space, gpu=gpu, logger=logger,
                                                                 record_kt=record_kt,
                                                                 record_mutation=record_mutation, **ps)
    elif algo_name == 'gin_predictor':
        data, full_data = gin_predictor_nasbench_asr(search_space, gpu=gpu, logger=logger, record_kt=record_kt,
                                                     record_mutation=record_mutation, **ps)
    elif algo_name == 'gin_predictor_new':
        data, full_data = gin_predictor_new_nasbench_asr(search_space, gpu=gpu, logger=logger, record_kt=record_kt,
                                                         record_mutation=record_mutation, **ps)
    elif algo_name == 'oracle':
        data, full_data = oracle_nasbench_asr(search_space, logger=logger, record_mutation=record_mutation, **ps)
    elif algo_name == 'gp_bayesopt':
        data = gp_bayesopt_nasbench_asr(search_space, logger=logger, record_kt=record_kt,
                                        record_mutation=record_mutation, tmpdir=f'./temp_nasbench_asr_{gpu}', **ps)
    else:
        print('invalid algorithm name')
        sys.exit()
    k = 10
    if 'k' in ps:
        k = ps['k']
    if full_data is not None:
        return compute_best_test_losses_case2(data, k, ps['total_queries']), full_data
    else:
        return compute_best_test_losses_case2(data, k, ps['total_queries'])


def run_nas_scalar_prior(algo_params, search_space, gpu=None, logger=None):
    ps = copy.deepcopy(algo_params)
    algo_name = ps.pop('algo_name')
    if algo_name == 'gin_predictor_scalar_10':
        data = gin_predictor_scalar_case2(search_space, gpu=gpu, logger=logger, **ps)
    elif algo_name == 'gin_predictor_scalar_30':
        data = gin_predictor_scalar_case2(search_space, gpu=gpu, logger=logger, **ps)
    elif algo_name == 'gin_predictor_scalar_50':
        data = gin_predictor_scalar_case2(search_space, gpu=gpu, logger=logger, **ps)
    elif algo_name == 'gin_predictor_scalar_70':
        data = gin_predictor_scalar_case2(search_space, gpu=gpu, logger=logger, **ps)
    elif algo_name == 'gin_predictor_scalar_100':
        data = gin_predictor_scalar_case2(search_space, gpu=gpu, logger=logger, **ps)
    else:
        print('invalid algorithm name')
        sys.exit()
    k = 10
    if 'k' in ps:
        k = ps['k']
    return compute_best_test_losses_case2(data, k, ps['total_queries'])


def run_box_compare_case1(algo_params, metann_params, search_space, gpu=None, logger=None, with_details='T'):
    mp = copy.deepcopy(metann_params)
    ps = copy.deepcopy(algo_params)
    algo_name = ps.pop('algo_name')
    if algo_name == 'bananas':
        mp.pop('search_space')
        data = bananas_case1(search_space, mp, gpu=gpu, logger=logger, **ps)
    elif algo_name == 'bananas_f':
        mp.pop('search_space')
        data = bananas_case1(search_space, mp, gpu=gpu, logger=logger, **ps)
    elif algo_name == 'gin_uncertainty_predictor':
        data = gin_uncertainty_case1(search_space, gpu=gpu, logger=logger, **ps)
    elif algo_name == 'gin_predictor':
        data = gin_predictor_case1(search_space, gpu=gpu, logger=logger, **ps)
    else:
        print('invalid algorithm name')
        sys.exit()
    k = 10
    if 'k' in ps:
        k = ps['k']
    return compute_best_test_losses_case1_with_details(data, k, ps['total_queries'])


def run_box_compare_case2(algo_params, metann_params, search_space, gpu=None, logger=None, with_details='T'):
    mp = copy.deepcopy(metann_params)
    ps = copy.deepcopy(algo_params)
    algo_name = ps.pop('algo_name')
    if algo_name == 'bananas':
        mp.pop('search_space')
        data = bananas_case2(search_space, mp, gpu=gpu, logger=logger, **ps)
    elif algo_name == 'bananas_f':
        mp.pop('search_space')
        data = bananas_case2(search_space, mp, gpu=gpu, logger=logger, **ps)
    elif algo_name == 'gin_uncertainty_predictor':
        data = gin_uncertainty_case2(search_space, gpu=gpu, logger=logger, **ps)
    elif algo_name == 'gin_predictor':
        data = gin_predictor_case2(search_space, gpu=gpu, logger=logger, **ps)
    else:
        print('invalid algorithm name')
        sys.exit()
    k = 10
    if 'k' in ps:
        k = ps['k']
    return compute_best_test_losses_case2_with_details(data, k, ps['total_queries'])


def run_evolutionary_compare(algo_params, search_space, logger=None):
    """
        Evolutionary case 1 represent mutate one child each time.
        Evolutionary case 2 represent mutate many children each time.
    """
    ps = copy.deepcopy(algo_params)
    algo_name = ps.pop('algo_name')
    if algo_name == 'evolutionary_case1':
        data = evolution_search_compare_case1(search_space, logger=logger, **ps)
    elif 'evolutionary_case2' in algo_name:
        data = evolution_search_compare_case2(search_space, logger=logger, **ps)
    else:
        print('invalid algorithm name')
        sys.exit()
    k = 10
    if 'k' in ps:
        k = ps['k']
    return compute_best_test_losses_case1(data, k, ps['total_queries'])


def compute_best_test_losses_case1(data, k, total_queries):
    """
    Given full data from a completed nas algorithm,
    output the test error of the arch with the best val error
    after every multiple of k
    """
    results = []
    for query in range(k, total_queries + k, k):
        best_arch = sorted(data[:query], key=lambda i: i[2])[0]
        test_error = best_arch[3]
        results.append((query, test_error))
    return results


def compute_best_test_losses_case2(data, k, total_queries):
    """
    Given full data from a completed nas algorithm,
    output the test error of the arch with the best val error
    after every multiple of k
    """
    results = []
    for query in range(k, total_queries + k, k):
        best_arch = sorted(data[:query], key=lambda i: i[4])[0]
        test_error = best_arch[5]
        results.append((query, test_error))
    return results


def compute_best_test_losses_case1_with_details(data, k, total_queries):
    """
    Given full data from a completed nas algorithm,
    output the test error of the arch with the best val error
    after every multiple of k
    """
    results = []
    for query in range(k, total_queries + k, k):
        best_arch = sorted(data[:query], key=lambda i: i[2])[0]
        test_error = best_arch[3]
        results.append((query, test_error))
    val_distribution = [d[2] for d in data]
    val_datas_np = np.array(val_distribution).reshape((-1, 10))
    test_distribution = [d[3] for d in data]
    test_datas_np = np.array(test_distribution).reshape((-1, 10))
    dist_results = {'val': val_datas_np,
                    'test': test_datas_np}
    return results, dist_results


def compute_best_test_losses_case2_with_details(data, k, total_queries):
    """
    Given full data from a completed nas algorithm,
    output the test error of the arch with the best val error
    after every multiple of k
    """
    results = []
    for query in range(k, total_queries + k, k):
        best_arch = sorted(data[:query], key=lambda i: i[4])[0]
        test_error = best_arch[5]
        results.append((query, test_error))
    val_distribution = [d[4] for d in data]
    val_datas_np = np.array(val_distribution).reshape((-1, 10))
    test_distribution = [d[5] for d in data]
    test_datas_np = np.array(test_distribution).reshape((-1, 10))
    dist_results = {'val': val_datas_np,
                    'test': test_datas_np}
    return results, dist_results