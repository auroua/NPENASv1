# Copyright (c) XiDian University and Xi'an University of Posts&Telecommunication. All Rights Reserved

import os
import argparse
import numpy as np
import time
import torch.multiprocessing as multiprocessing
from nas_lib.utils.comm import random_id, setup_logger
from nas_lib.utils.utils_darts import compute_best_test_losses, compute_darts_test_losses
from nas_lib.algos_darts.build_open_algos import build_open_algos
from nas_lib.data_darts.build_ss import build_open_search_space_dataset
from nas_lib.configs import algo_params_open_domain
import pickle


if __name__ == "__main__":
    loss_type = 'mae'
    parser = argparse.ArgumentParser(description='Args for darts search space.')
    parser.add_argument('--search_space', type=str, default='darts', help='darts')
    parser.add_argument('--gpus', type=int, default=1, help='Number of gpus')
    parser.add_argument('--algorithm', type=str, default='gin_predictor',
                        choices=['gin_uncertainty_predictor', 'gin_predictor'], help='which parameters to use')
    parser.add_argument('--output_filename', type=str, default=random_id(64), help='name of output files')
    parser.add_argument('--node_nums', type=int, default=4, help='cell num')
    parser.add_argument('--log_level', type=str, default='DEBUG', help='information logging level')
    parser.add_argument('--seed', type=int, default=22, help='seed')
    parser.add_argument('--budget', type=int, default=100, help='searching budget.')
    parser.add_argument('--save_dir', type=str,
                        default='/home/albert_wei/Disk_A/train_output_npenas/npenas_open_domain_darts_2/',
                        help='name of save directory')
    args = parser.parse_args()

    # make save directory
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(os.path.join(save_dir, 'model_pkl')):
        os.mkdir(os.path.join(save_dir, 'model_pkl'))
    if not os.path.exists(os.path.join(save_dir, 'results')):
        os.mkdir(os.path.join(save_dir, 'results'))
    if not os.path.exists(os.path.join(save_dir, 'pre_train_models')):
        os.mkdir(os.path.join(save_dir, 'pre_train_models'))
    # 2. build architecture training dataset
    arch_dataset = build_open_search_space_dataset(args.search_space)
    logger = setup_logger("nasbench_open_%s_cifar10" % args.search_space, args.save_dir, 0, log_level=args.log_level)
    algo_info = algo_params_open_domain(args.algorithm)
    algo_info['total_queries'] = args.budget
    starttime = time.time()
    multiprocessing.set_start_method('spawn')
    temp_k = 10
    file_name = save_dir + '/results/%s_%d.pkl' % (algo_info['algo_name'], algo_info['total_queries'])

    data = build_open_algos(algo_info['algo_name'])(search_space=arch_dataset,
                                                    algo_info=algo_info,
                                                    logger=logger,
                                                    gpus=args.gpus,
                                                    save_dir=save_dir,
                                                    seed=args.seed)
    if 'random' in algo_info['algo_name']:
        results, result_keys = compute_best_test_losses(data, temp_k, total_queries=algo_info['total_queries'])
        algo_result = np.round(results, 5)
    else:
        results, result_keys = compute_darts_test_losses(data, temp_k, total_queries=algo_info['total_queries'])
        algo_result = np.round(results, 5)
    print(algo_result)

    with open(file_name, 'wb') as f:
        pickle.dump(results, f)
        pickle.dump(result_keys, f)