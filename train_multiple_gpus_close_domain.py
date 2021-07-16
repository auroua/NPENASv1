# Copyright (c) XiDian University and Xi'an University of Posts&Telecommunication. All Rights Reserved

import argparse
from nas_lib.algos.algo_compare import run_nas_algos_case1, run_nas_algos_case2, run_nas_algos_nasbench_201, \
    run_nas_scalar_prior, run_evolutionary_compare, run_box_compare_case1, run_box_compare_case2, \
    run_nas_algos_nasbench_nlp, run_nas_algos_nasbench_asr
from nas_lib.configs import meta_neuralnet_params
from nas_lib.configs import algo_params_close_domain as algo_params
from nas_lib.data.data import build_datasets
import tensorflow as tf
import psutil
from nas_lib.utils.comm import set_random_seed, random_id, setup_logger
import torch.multiprocessing as multiprocessing
from torch.multiprocessing import Process
import os
from torch.multiprocessing import Queue
import pickle
import numpy as np
import time


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)


def ansyc_multiple_process_train(args, save_dir):
    q = Queue(10)
    metann_params = meta_neuralnet_params(args.search_space)
    data_lists = [build_datasets(metann_params['search_space'],
                                 args.dataset, args.nasbench_nlp_type,
                                 args.filter_none) for _ in range(args.gpus)]

    p_producer = Process(target=data_producers, args=(args, q))
    p_consumers = [Process(target=data_consumers, args=(args, q, save_dir, i, data_lists[i])) for i in range(args.gpus)]

    p_producer.start()
    for p in p_consumers:
        p.start()

    p_producer.join()
    for p in p_consumers:
        p.join()


def data_producers(args, queue):
    trials = args.trials
    for i in range(trials):
        queue.put({
            'iterate': i
        })
    for _ in range(args.gpus):
        queue.put('done')


def data_consumers(args, q, save_dir, i, search_space):
    set_random_seed(int(str(time.time()).split('.')[0][::-1][:9]))
    file_name = 'log_%s_%d' % ('gpus', i)
    logger = setup_logger(file_name, save_dir, i, log_level='DEBUG',
                          filename='%s.txt' % file_name)
    while True:
        msg = q.get()
        if msg == 'done':
            logger.info('thread %d end' % i)
            break
        iterations = msg['iterate']
        run_experiments_bananas_paradigm(args, save_dir, i, iterations, logger, search_space)


def run_experiments_bananas_paradigm(args, save_dir, i, iterations, logger, search_space):
    out_file = args.output_filename + '_gpus_%d_' % i + 'iter_%d' % iterations
    metann_params = meta_neuralnet_params(args.search_space)
    algorithm_params = algo_params(args.algo_params, args.search_budget)
    num_algos = len(algorithm_params)
    results = []
    full_data_results = []
    result_dist = []
    walltimes = []
    for j in range(num_algos):
        logger.info(' * Running algorithm: {}'.format(algorithm_params[j]))
        logger.info(' * Loss type: {}'.format(args.loss_type))
        logger.info(' * Trials: {}, Free Memory available {}'.format(iterations,
                                                                     psutil.virtual_memory().free/(1024*1024)))
        starttime = time.time()
        if args.algo_params == 'nasbench101_case1':
            algo_result = run_nas_algos_case1(algorithm_params[j], metann_params, search_space, gpu=i,
                                              logger=logger, with_details=args.with_details)
        elif args.algo_params == 'nasbench101_case2':
            algo_result = run_nas_algos_case2(algorithm_params[j], metann_params, search_space, gpu=i,
                                              logger=logger, with_details=args.with_details, record_kt=args.record_kt,
                                                     record_mutation=args.record_mutation)
        elif args.algo_params == 'nasbench_201':
            algo_result = run_nas_algos_nasbench_201(algorithm_params[j], metann_params, search_space, gpu=i,
                                                     dataname=args.dataset, logger=logger, record_kt=args.record_kt,
                                                     record_mutation=args.record_mutation)
        elif args.algo_params == 'nasbench_nlp':
            algo_result = run_nas_algos_nasbench_nlp(algorithm_params[j], metann_params, search_space, gpu=i,
                                                     logger=logger, record_kt=args.record_kt,
                                                     record_mutation=args.record_mutation)
        elif args.algo_params == 'nasbench_asr':
            algo_result = run_nas_algos_nasbench_asr(algorithm_params[j], metann_params, search_space, gpu=i,
                                                     logger=logger, record_kt=args.record_kt,
                                                     record_mutation=args.record_mutation)
        elif args.algo_params == 'scalar_prior':
            algo_result = run_nas_scalar_prior(algorithm_params[j], search_space, gpu=i, logger=logger)
        elif args.algo_params == 'evaluation_compare':
            algo_result = run_evolutionary_compare(algorithm_params[j], search_space, logger=logger)
        elif args.algo_params == 'box_compare_case1':
            algo_result = run_box_compare_case1(algorithm_params[j], metann_params, search_space, gpu=i,
                                                logger=logger, with_details=args.with_details)
        elif args.algo_params == 'box_compare_case2':
            algo_result = run_box_compare_case2(algorithm_params[j], metann_params, search_space, gpu=i,
                                                logger=logger, with_details=args.with_details)
        else:
            raise NotImplementedError("This algorithm does not support!")
        if args.with_details == 'T':
            algo_result_1, algo_result_dist = algo_result
            algo_result = np.round(algo_result_1, 5)
            walltimes.append(time.time() - starttime)
            results.append(algo_result)
            result_dist.append(algo_result_dist)
        else:
            if len(algo_result) == 2:
                algo_result, full_data = algo_result
                algo_result = np.round(algo_result, 5)
                # add walltime and results
                walltimes.append(time.time() - starttime)
                results.append(algo_result)
                full_data_results.append(full_data)
            else:
                algo_result = np.round(algo_result, 5)
                # add walltime and results
                walltimes.append(time.time() - starttime)
                results.append(algo_result)

    filename = os.path.join(save_dir, '{}_{}.pkl'.format(out_file, i))
    filename_full_data = os.path.join(save_dir, 'full_data_{}_{}.pkl'.format(out_file, i))
    logger.info(' * Trial summary: (params, results, walltimes)')
    logger.info(algorithm_params)
    logger.info(metann_params)
    for k in range(results[0].shape[0]):
        length = len(results)
        results_line = []
        for j in range(length):
            if j == 0:
                results_line.append(int(results[j][k, 0]))
                results_line.append(results[j][k, 1])
            else:
                results_line.append(results[j][k, 1])
        results_str = '  '.join([str(k) for k in results_line])
        logger.info(results_str)
    logger.info(walltimes)
    logger.info(' * Saving to file {}'.format(filename))
    with open(filename, 'wb') as f:
        if args.with_details == 'T':
            pickle.dump([algorithm_params, metann_params, results, result_dist, walltimes], f)
        else:
            pickle.dump([algorithm_params, metann_params, results, walltimes], f)
    if len(full_data_results) > 0:
        with open(filename_full_data, 'wb') as f:
            pickle.dump([algorithm_params, metann_params, full_data_results, walltimes], f)
    logger.info('#######################################################  Trails %d End  '
                '#######################################################' % iterations)


def main(args):
    save_dir = args.save_dir
    if not save_dir:
        save_dir = args.algo_params + '/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    multiprocessing.set_start_method('spawn')
    ansyc_multiple_process_train(args, save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args for algorithms compare.')
    parser.add_argument('--trials', type=int, default=600, help='Number of trials')
    parser.add_argument('--search_budget', type=int, default=100,
                        help='Searching budget, NasBench-101 dataset 150, NasBench-201 100')
    parser.add_argument('--search_space', type=str, default='nasbench_nlp',
                        choices=['nasbench_case1', 'nasbench_case2', 'nasbench_201',
                                 'nasbench_nlp', 'nasbench_asr'], help='nasbench')
    parser.add_argument('--algo_params', type=str, default='nasbench_nlp',
                        choices=['nasbench101_case1', 'nasbench101_case2', 'nasbench_201', 'scalar_prior',
                                 'evaluation_compare', 'box_compare_case1', 'box_compare_case2', 'experiment',
                                 'nasbench_nlp', 'nasbench_asr'],
                        help='which algorithms to compare')
    parser.add_argument('--dataset', type=str, default='cifar10-valid',
                        choices=['cifar10-valid', 'cifar100', 'ImageNet16-120'], help='dataset name of nasbench-201.')
    parser.add_argument('--output_filename', type=str, default=random_id(64), help='name of output files')
    parser.add_argument('--nasbench_nlp_type', type=str,
                        default='last', choices=['best', 'last'],
                        help='name of output files')
    parser.add_argument('--filter_none', type=str,
                        default='y', choices=['y', 'n'],
                        help='name of output files')
    parser.add_argument('--gpus', type=int, default=2, help='The num of gpus used for search.')
    parser.add_argument('--loss_type', type=str, default="mae", help='Loss used to train architecture.')
    parser.add_argument('--with_details', type=str, default="F", help='Record detailed training procedure.')
    parser.add_argument('--record_kt', type=str, default="T", help='Record kendall tau corr.')
    parser.add_argument('--record_mutation', type=str, default="T", help='Record architecture mutation information.')
    parser.add_argument('--save_dir', type=str,
                        default='/home/albert_wei/fdisk_a/train_output_2021/npenas/nasbench_nlp_mutation_strategies/',
                        help='output directory')

    args = parser.parse_args()
    main(args)
