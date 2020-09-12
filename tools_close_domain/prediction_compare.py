import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import tensorflow as tf
import torch
import argparse
import copy
import matplotlib.pyplot as plt
from nas_lib.utils.utils_data import nasbench2graph2
from nas_lib.eigen.trainer_predictor import NasBenchGinPredictorTrainer
from nas_lib.eigen.trainer_uncertainty_predictor import NasBenchGinGaussianTrainer
from nas_lib.data.data import build_datasets
from nas_lib.models.meta_neural_net import MetaNeuralnet
from nas_lib.utils.comm import set_random_seed
import pickle
import time
from keras import backend as K


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
device = torch.device('cuda:0')


def _data_split(train_data):
    data_path_encoding = []
    data_wo_path_endocing = []
    for d in train_data:
        path_encoding_tuple = list()
        path_wo_encoding_tuple = list()
        for idx, d_nest in enumerate(d):
            if idx == 0 or idx == 3 or idx == 4:
                path_encoding_tuple.append(d_nest)
                path_wo_encoding_tuple.append(d_nest)
            elif idx == 1:
                path_encoding_tuple.append(d_nest)
            elif idx == 2:
                path_wo_encoding_tuple.append(d_nest)
            else:
                raise ValueError('idx is not support!')
        data_path_encoding.append(tuple(path_encoding_tuple))
        data_wo_path_endocing.append(tuple(path_wo_encoding_tuple))
    return data_path_encoding, data_wo_path_endocing


def kl_divergence(dis_a, dis_b):
    dis_a += 1e-6
    dis_b += 1e-6
    log_a = np.log(dis_a)
    log_b = np.log(dis_b)
    part1 = dis_a*log_a
    part2 = dis_a*log_b
    result = np.sum(part1-part2)
    print(result)
    return result


def path_encoding_distribution(path_encoding_list):
    encoding_list = [d[1] for d in path_encoding_list]
    x_idx = list(range(len(encoding_list[0])))
    np_encoding = np.sum(np.array(encoding_list), axis=0)/np.sum(np.array(encoding_list))
    # plt.bar(x_idx, np_encoding)
    # plt.show()
    return np_encoding


def plot_meta_neuralnet(ytrain, train_pred, ytest, test_pred, max_disp=500, title=None, file_name=None):
    fig, ax = plt.subplots()
    ax.scatter(ytrain[:max_disp], train_pred[:max_disp], label='training data', alpha=0.7, s=64)
    ax.scatter(ytest[:max_disp], test_pred[:max_disp], label='test data', alpha=0.7, marker='^')

    # axis limits
    plt.xlim((5, 15))
    plt.ylim((5, 15))
    ax_lim = np.array([np.min([plt.xlim()[0], plt.ylim()[0]]),
                       np.max([plt.xlim()[1], plt.ylim()[1]])])
    plt.xlim(ax_lim)
    plt.ylim(ax_lim)

    # 45-degree line
    ax.plot(ax_lim, ax_lim, 'k:')
    fig.set_dpi(300.0)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title)
    plt.legend(loc='best')
    plt.xlabel('true percent error')
    plt.ylabel('predicted percent error')
    plt.show()


def run_bananas(params,
                train_data,
                test_data,
                num_ensemble=3,
                encode_paths=True,
                n=10,
                gpu=1):
    xtrain = np.array([d[1] for d in train_data])
    ytrain = np.array([d[2] for d in train_data])

    xtest = np.array([d[1] for d in test_data])
    ytest = np.array([d[2] for d in test_data])

    train_errors = []
    test_errors = []
    train_preds = []
    test_preds = []
    for _ in range(num_ensemble):
        meta_neuralnet = MetaNeuralnet(gpu=gpu)
        meta_neuralnet.fit(xtrain, ytrain, **params)
        train_pred = np.squeeze(meta_neuralnet.predict(xtrain))
        train_error = np.mean(abs(train_pred - ytrain))
        train_errors.append(train_error)
        test_pred = np.squeeze(meta_neuralnet.predict(xtest))
        test_error = np.mean(abs(test_pred - ytest))
        test_errors.append(test_error)

        train_preds.append(train_pred)
        test_preds.append(test_pred)
        K.clear_session()
        tf.reset_default_graph()
        del meta_neuralnet

    train_error = np.round(np.mean(train_errors, axis=0), 3)
    test_error = np.round(np.mean(test_errors, axis=0), 3)

    train_preds = np.round(np.mean(train_preds, axis=0), 3)
    test_pred = np.round(np.mean(test_preds, axis=0), 3)

    print('Meta neuralnet training size: {}, encode paths: {}'.format(n, encode_paths))
    print('Train error: {}, test validation error: {}'.format(train_error, test_error))
    if encode_paths:
        # title = 'Path encoding, training set size {}, test error {}'.format(n, test_error)
        title = 'Path encoding, training set size {}'.format(n)
        file_all = 'bananas_path_based_samples_%d_epoch_%d.png' % (n, params['epochs'])
    else:
        # title = 'Adjacency list encoding, training set size {}, test error {}'.format(n, test_error)
        title = 'Adjacency list encoding, training set size {}'.format(n)
        file_all = 'bananas_adjacency_samples_%d_epoch_%d.png' % (n, params['epochs'])
    # plot_meta_neuralnet(ytrain, train_preds, ytest, test_pred, title=title, file_name=file_all)
    return train_error, test_error


def run_np_uncertainty(n, lr, train_data, test_data):
    arch_data = [d[0] for d in train_data]
    arch_graph = [[d[0]['matrix'], d[0]['ops']] for d in train_data]
    arch_data_edge_idx_list = []
    arch_data_node_f_list = []
    for arch in arch_graph:
        edge_index, node_f = nasbench2graph2(arch)
        arch_data_edge_idx_list.append(edge_index)
        arch_data_node_f_list.append(node_f)
    val_accuracy = np.array([d[2] for d in train_data])
    candiate_edge_list = []
    candiate_node_list = []
    candidate_arch_graph = [[d[0]['matrix'], d[0]['ops']] for d in test_data]
    for cand in candidate_arch_graph:
        edge_index, node_f = nasbench2graph2(cand)
        candiate_edge_list.append(edge_index)
        candiate_node_list.append(node_f)
    val_test_accuracy = np.array([d[2] for d in test_data])

    train_errors = []
    train_mean_errors = []
    val_pred_errors = []
    val_test_errors = []
    agent = NasBenchGinGaussianTrainer('gin_gaussian', lr=lr, device=device, epochs=1000,
                                       train_images=len(arch_data), batch_size=16)
    agent.fit(arch_data_edge_idx_list, arch_data_node_f_list, val_accuracy, logger=None)

    pred_train, mean_train, _ = agent.pred(arch_data_edge_idx_list, arch_data_node_f_list)
    predictions, mean_test_pred, _ = agent.pred(candiate_edge_list, candiate_node_list)

    pred_train = pred_train.cpu().numpy()
    mean_train = mean_train.cpu().numpy()
    predictions = predictions.cpu().numpy()
    mean_test_pred = mean_test_pred.cpu().numpy()
    train_error = np.mean(np.abs(pred_train - val_accuracy))
    train_error_mean = np.mean(np.abs(mean_train - val_accuracy))
    train_errors.append(train_error)
    train_mean_errors.append(train_error_mean)

    val_mean_error = np.mean(np.abs(val_test_accuracy - mean_test_pred))
    val_pred_error = np.mean(np.abs(val_test_accuracy - predictions))
    val_test_errors.append(val_mean_error)
    val_pred_errors.append(val_pred_error)

    train_error = np.round(np.mean(train_errors, axis=0), 3)
    train_mean_error = np.round(np.mean(train_mean_errors, axis=0), 3)
    val_error = np.round(np.mean(val_pred_errors, axis=0), 3)
    val_mean_error = np.round(np.mean(val_test_errors, axis=0), 3)

    print('Neural predictor with uncertainty estimation and predictor training size: {}'.format(n))
    print('Train predicted error: {}, Train mean error: {}, Test pred error: {}, Test mean error: {}'.format(
        train_error, train_mean_error, val_mean_error, val_error))

    title = 'Neural predictor Uncertainty, training set size {}'.format(n)
    file_all = 'neural_predictor_w.u._samples_%d_epoch_%d.png' % (n, 1000)
    # plot_meta_neuralnet(val_accuracy, mean_train, val_test_accuracy, mean_test_pred, title=title, file_name=file_all)
    return train_mean_error, val_error


def run_np(n, lr, train_data, test_data):
    arch_data = [d[0] for d in train_data]
    arch_graph = [[d[0]['matrix'], d[0]['ops']] for d in train_data]
    arch_data_edge_idx_list = []
    arch_data_node_f_list = []
    for arch in arch_graph:
        edge_index, node_f = nasbench2graph2(arch)
        arch_data_edge_idx_list.append(edge_index)
        arch_data_node_f_list.append(node_f)
    val_accuracy = np.array([d[2] for d in train_data])
    candiate_edge_list = []
    candiate_node_list = []
    candidate_arch_graph = [[d[0]['matrix'], d[0]['ops']] for d in test_data]
    for cand in candidate_arch_graph:
        edge_index, node_f = nasbench2graph2(cand)
        candiate_edge_list.append(edge_index)
        candiate_node_list.append(node_f)
    val_test_accuracy = np.array([d[2] for d in test_data])

    train_mean_errors = []
    val_test_errors = []
    agent = NasBenchGinPredictorTrainer('nnp', lr=lr, device=device, epochs=300,
                                        train_images=len(arch_data), batch_size=16, rate=20)
    agent.fit(arch_data_edge_idx_list, arch_data_node_f_list, val_accuracy, logger=None)

    mean_train = agent.pred(arch_data_edge_idx_list, arch_data_node_f_list)
    mean_test_pred = agent.pred(candiate_edge_list, candiate_node_list)

    mean_train = mean_train.cpu().numpy()
    mean_test_pred = mean_test_pred.cpu().numpy()

    train_error_mean = np.mean(np.abs(mean_train - val_accuracy))
    train_mean_errors.append(train_error_mean)

    val_mean_error = np.mean(np.abs(val_test_accuracy - mean_test_pred))
    val_test_errors.append(val_mean_error)

    train_mean_error = np.round(np.mean(train_mean_errors, axis=0), 3)
    val_mean_error = np.round(np.mean(val_test_errors, axis=0), 3)

    print('Neural predictor training size: {}'.format(n))
    print('Train mean error: {}, Test mean error: {}'.format(train_mean_error, val_mean_error))
    title = 'Neural predictor, training set size {}'.format(n)
    file_all = 'neural_predictor_samples_%d_epoch_%d.png' % (n, 300)
    # plot_meta_neuralnet(val_accuracy, mean_train, val_test_accuracy, mean_test_pred, title=title, file_name=file_all)
    return train_mean_error, val_mean_error


def meta_neural_net_experiment(i, lr, algo_dict, ns=(100, 500), search_space=None):
    gpu = 0
    meta_neural_net_params = {'loss': 'mae', 'num_layers': 10, 'layer_width': 20, 'epochs': 200,
                              'batch_size': 32, 'lr': .01, 'regularization': 0, 'verbose': 0}
    print('#########################  Epoch %d base lr %.4f #######################' % (i, lr))
    test_data_total_origin = search_space.generate_random_dataset_both(num=1000, allow_isomorphisms=False)
    for n in ns:
        test_data_total = copy.deepcopy(test_data_total_origin)
        test_size = 500
        train_data_total = search_space.generate_random_dataset_both(num=n, allow_isomorphisms=False)
        encoding_1 = path_encoding_distribution(train_data_total)
        encoding_2 = path_encoding_distribution(test_data_total)

        print(f'Training dataset size {n} kl divergence ################>>>>')
        kl_dist = kl_divergence(encoding_1, encoding_2)
        test_data_total = search_space.remove_duplicates_both(test_data_total, train_data_total)
        test_data_total = test_data_total[:int(len(test_data_total) - len(test_data_total) % 64)][:test_size]
        train_data_pe, train_data_wo_pe = _data_split(train_data_total)
        test_data_pe, test_data_wo_pe = _data_split(test_data_total)
        print('#########################  Real test data size is %d #######################' % (len(test_data_total)))
        algos_list = list(algo_dict.keys())
        for algos in algos_list:
            if algos == 'bananas_false':
                train_error, test_error = run_bananas(meta_neural_net_params,
                                                      train_data=train_data_wo_pe,
                                                      test_data=test_data_wo_pe,
                                                      n=n,
                                                      num_ensemble=5,
                                                      encode_paths=False,
                                                      gpu=gpu
                                                      )
                algo_dict['bananas_false'].append((train_error, test_error, kl_dist))
            elif algos == 'bananas_true':
                train_error, test_error = run_bananas(meta_neural_net_params,
                                                      train_data=train_data_pe,
                                                      test_data=test_data_pe,
                                                      n=n,
                                                      num_ensemble=5,
                                                      encode_paths=True,
                                                      gpu=gpu
                                                      )
                algo_dict['bananas_true'].append((train_error, test_error, kl_dist))
            elif algos == 'neural_predictor_uncertainty':
                train_error, test_error = run_np_uncertainty(n=n,
                                                             lr=lr,
                                                             train_data=train_data_pe,
                                                             test_data=test_data_pe
                                                             )
                algo_dict['neural_predictor_uncertainty'].append((train_error, test_error, kl_dist))
            elif algos == 'neural_predictor':
                train_error, test_error = run_np(n=n,
                                                 lr=lr,
                                                 train_data=train_data_pe,
                                                 test_data=test_data_pe
                                                 )
                algo_dict['neural_predictor'].append((train_error, test_error, kl_dist))
            else:
                raise ValueError('train error and test error should have value!')
            # train_error_list.append(train_error)
            # test_error_list.append(test_error)
        # key_train = 'train_error_%d' % n
        # key_test = 'test_error_%d' % n
        # result_dict[key_train] = train_error_list
        # result_dict_test[key_test] = test_error_list
    # return result_dict, result_dict_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args for algorithms compare.')
    parser.add_argument('--trials', type=int, default=300, help='Number of trials')
    parser.add_argument('--seed', type=int, default=434, help='Number of trials')
    parser.add_argument('--search_space', type=str, default='nasbench_case2',
                        choices=['nasbench_case1', 'nasbench_case2'], help='nas search space')
    parser.add_argument('--lr', type=float, default=0.005, help='Loss used to train architecture.')
    parser.add_argument('--save_path', type=str,
                        default='/home/albert_wei/Disk_A/train_output_npenas/prediction_compare/prediction_compare.pkl',
                        help='Loss used to train architecture.')

    args = parser.parse_args()

    algos_dict = dict()
    total_algos = ['bananas_true', 'bananas_false', 'neural_predictor_uncertainty', 'neural_predictor']
    for a in total_algos:
        algos_dict[a] = []
    # ns = [50, 100, 150]
    ns = [20]
    set_random_seed(args.seed)
    search_space = build_datasets(args.search_space)
    start = time.time()
    for i in range(args.trials):
        meta_neural_net_experiment(i, args.lr, algos_dict, ns, search_space)
        print(algos_dict)
    duration = time.time() - start
    print(duration)
    with open(args.save_path, 'wb') as f:
        pickle.dump(algos_dict, f)


