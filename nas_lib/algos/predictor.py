# Copyright (c) XiDian University and Xi'an University of Posts&Telecommunication. All Rights Reserved

import numpy as np
import torch
import copy
from ..utils.utils_data import nasbench2graph, nasbench2graph2
from .acquisition_functions import acq_fn
from ..eigen.trainer_predictor import NasBenchGinPredictorTrainer
from ..eigen.trainer_uncertainty_predictor import NasBenchGinGaussianTrainer
from nas_lib.utils.corr import get_kendalltau_coorlection


def gin_uncertainty_case1(search_space,
                          num_init=10,
                          k=10,
                          total_queries=150,
                          acq_opt_type='mutation',
                          allow_isomorphisms=False,
                          verbose=1,
                          agent=None,
                          logger=None,
                          gpu='0',
                          lr=0.01,
                          candidate_nums=100,
                          epochs=1000):
    """
    Bayesian optimization with a neural network model
    """
    device = torch.device('cuda:%d' % gpu)
    data = search_space.generate_random_dataset_gin(num=num_init,
                                                    encode_paths=True,
                                                    allow_isomorphisms=allow_isomorphisms,
                                                    deterministic_loss=True
                                                    )
    query = num_init + k
    search_agent = agent
    if len(data) <= 10:
        batch_size = 10
    else:
        batch_size = 16
    while query <= total_queries:
        agent = NasBenchGinGaussianTrainer(search_agent, lr=lr, device=device, epochs=epochs,
                                           train_images=len(data), batch_size=batch_size)
        val_accuracy = np.array([d[2] for d in data])
        arch_data_edge_idx_list = []
        arch_data_node_f_list = []
        for d in data:
            edge_index, node_f = nasbench2graph(d[0])
            arch_data_edge_idx_list.append(edge_index)
            arch_data_node_f_list.append(node_f)
        candidates = search_space.get_candidates_gin(data,
                                                     num=candidate_nums,
                                                     acq_opt_type=acq_opt_type,
                                                     encode_paths=True,
                                                     allow_isomorphisms=allow_isomorphisms,
                                                     deterministic_loss=True)
        candiate_edge_list = []
        candiate_node_list = []
        for cand in candidates:
            edge_index, node_f = nasbench2graph(cand[0])
            candiate_edge_list.append(edge_index)
            candiate_node_list.append(node_f)
        agent.fit(arch_data_edge_idx_list, arch_data_node_f_list, val_accuracy, logger=None)
        pred_train, mean_train, _ = agent.pred(arch_data_edge_idx_list, arch_data_node_f_list)
        predictions, _, _ = agent.pred(candiate_edge_list, candiate_node_list)
        sorted_indices = acq_fn(predictions, 'its_vae')
        for i in sorted_indices[:k]:
            archtuple = search_space.query_arch_gin(candidates[i][0],
                                                    encode_paths=True,
                                                    deterministic=True
                                                    )
            data.append(archtuple)
        if verbose:
            top_5_loss = sorted([d[2] for d in data])[:min(5, len(data))]
            logger.info('Query {}, training mean loss is  {}'.format(query,
                                                                     np.mean(np.abs(mean_train.cpu().numpy()-val_accuracy))))
            logger.info('Query {}, top 5 val losses {}'.format(query, top_5_loss))
        query += k
    return data


def gin_predictor_case1(search_space,
                        num_init=10,
                        k=10,
                        total_queries=150,
                        acq_opt_type='mutation',
                        allow_isomorphisms=False,
                        verbose=1,
                        agent=None,
                        logger=None,
                        gpu='0',
                        lr=0.01,
                        candidate_nums=100,
                        epochs=1000):
    """
    Bayesian optimization with a neural network model
    """
    device = torch.device('cuda:%d' % gpu)
    data = search_space.generate_random_dataset_gin(num=num_init,
                                                    allow_isomorphisms=allow_isomorphisms,
                                                    deterministic_loss=True)
    query = num_init + k
    search_agent = agent
    if len(data) <= 10:
        batch_size = 10
    else:
        batch_size = 16
    while query <= total_queries:
        agent = NasBenchGinPredictorTrainer(search_agent, lr=lr, device=device, epochs=epochs,
                                            train_images=len(data), batch_size=batch_size)
        val_accuracy = np.array([d[2] for d in data])
        arch_data_edge_idx_list = []
        arch_data_node_f_list = []
        for d in data:
            edge_index, node_f = nasbench2graph(d[0])
            arch_data_edge_idx_list.append(edge_index)
            arch_data_node_f_list.append(node_f)
        candidates = search_space.get_candidates_gin(data,
                                                     num=candidate_nums,
                                                     acq_opt_type=acq_opt_type,
                                                     allow_isomorphisms=allow_isomorphisms)
        candiate_edge_list = []
        candiate_node_list = []
        for cand in candidates:
            edge_index, node_f = nasbench2graph(cand[0])
            candiate_edge_list.append(edge_index)
            candiate_node_list.append(node_f)
        agent.fit(arch_data_edge_idx_list, arch_data_node_f_list, val_accuracy, logger=None)
        acc_train = agent.pred(arch_data_edge_idx_list, arch_data_node_f_list)
        acc_pred = agent.pred(candiate_edge_list, candiate_node_list)
        candidate_np = acc_pred.cpu().numpy()
        sorted_indices = np.argsort(candidate_np)
        for i in sorted_indices[:k]:
            archtuple = search_space.query_arch_gin(candidates[i][0],
                                                    encode_paths=True,
                                                    deterministic=True
                                                    )
            data.append(archtuple)
        if verbose:
            top_5_loss = sorted([d[2] for d in data])[:min(5, len(data))]
            logger.info('Query {}, training mean loss is  {}'.format(query,
                                                                     np.mean(np.abs(acc_train.cpu().numpy()-val_accuracy))))
            logger.info('Query {}, top 5 val losses {}'.format(query, top_5_loss))
        query += k
    return data


def gin_uncertainty_case2(search_space,
                          num_init=10,
                          k=10,
                          total_queries=150,
                          acq_opt_type='mutation',
                          allow_isomorphisms=False,
                          verbose=1,
                          agent=None,
                          logger=None,
                          gpu='0',
                          lr=0.01,
                          candidate_nums=100,
                          epochs=1000,
                          record_kt='F',
                          record_mutation='F',
                          activation_fn='celu'
                          ):
    """
    Bayesian optimization with a neural network model
    """
    device = torch.device('cuda:%d' % gpu)
    data = search_space.generate_random_dataset(num=num_init,
                                                allow_isomorphisms=allow_isomorphisms,
                                                deterministic_loss=True)
    query = num_init + k
    search_agent = agent
    kt_list = []
    kt_top_list = []
    mutate_list = []
    if len(data) <= 10:
        batch_size = 10
    else:
        batch_size = 16
    while query <= total_queries:
        arch_data = [d[0] for d in data]
        agent = NasBenchGinGaussianTrainer(search_agent, lr=lr, device=device, epochs=epochs,
                                           train_images=len(data), batch_size=batch_size, activation_fn=activation_fn)
        val_accuracy = np.array([d[4] for d in data])
        arch_data_edge_idx_list = []
        arch_data_node_f_list = []
        for arch in arch_data:
            edge_index, node_f = nasbench2graph2(arch)
            arch_data_edge_idx_list.append(edge_index)
            arch_data_node_f_list.append(node_f)
        if record_mutation == 'T':
            candidates, dist_list, replicate_num, mutated_nums_list, mutated_arch_list \
                = search_space.get_candidates(data,
                                              num=candidate_nums,
                                              acq_opt_type=acq_opt_type,
                                              allow_isomorphisms=allow_isomorphisms,
                                              return_dist=True)
            cand_val_list = [cand[4] for cand in candidates]
            mutate_list.append((dist_list, replicate_num, mutated_nums_list, mutated_arch_list, cand_val_list))
        else:
            candidates = search_space.get_candidates(data,
                                                     num=candidate_nums,
                                                     acq_opt_type=acq_opt_type,
                                                     allow_isomorphisms=allow_isomorphisms)
        candiate_edge_list = []
        candiate_node_list = []
        for cand in candidates:
            edge_index, node_f = nasbench2graph2(cand[0])
            candiate_edge_list.append(edge_index)
            candiate_node_list.append(node_f)
        agent.fit(arch_data_edge_idx_list, arch_data_node_f_list, val_accuracy, logger=None)
        pred_train, mean_train, _ = agent.pred(arch_data_edge_idx_list, arch_data_node_f_list)
        predictions, _, _ = agent.pred(candiate_edge_list, candiate_node_list)
        if record_kt == 'T':
            np_pred = predictions.cpu().numpy()
            candidates_gt = [can[4] for can in candidates]
            kt = get_kendalltau_coorlection(np_pred.tolist(), candidates_gt)[0]
            kt_list.append(kt)
        sorted_indices = acq_fn(predictions, 'its_vae')
        kt_top_pred_list = []
        kt_top_gt_list = []
        np_pred_temp = predictions.cpu().numpy()
        for i in sorted_indices[:k]:
            archtuple = search_space.query_arch(matrix=candidates[i][1],
                                                ops=candidates[i][2])
            data.append(archtuple)
            kt_top_pred_list.append(np_pred_temp[i])
            kt_top_gt_list.append(archtuple[4])
        kt_top_list.append(get_kendalltau_coorlection(kt_top_pred_list, kt_top_gt_list)[0])
        if verbose:
            top_5_loss = sorted([d[4] for d in data])[:min(5, len(data))]
            logger.info('Query {}, training mean loss is  {}'.format(query,
                                                                     np.mean(np.abs(mean_train.cpu().numpy()-val_accuracy))))
            logger.info('Query {}, top 5 val losses {}'.format(query, top_5_loss))
        query += k
    return data, {'type': 'gin_uncertainty', 'final_data': data, 'kt_list': kt_list, 'kt_top_list': kt_top_list,
                  'mutate_list': mutate_list}


def gin_predictor_case2(search_space,
                        num_init=10,
                        k=10,
                        total_queries=150,
                        acq_opt_type='mutation',
                        allow_isomorphisms=False,
                        verbose=1,
                        agent=None,
                        logger=None,
                        gpu='0',
                        lr=0.01,
                        candidate_nums=100,
                        epochs=1000,
                        record_kt='F',
                        record_mutation='F',
                        rate=None,
                        activation_fn='relu'
                        ):
    """
    Bayesian optimization with a neural network model
    """
    device = torch.device('cuda:%d' % gpu)
    data = search_space.generate_random_dataset(num=num_init,
                                                allow_isomorphisms=allow_isomorphisms,
                                                deterministic_loss=True)
    query = num_init + k
    search_agent = agent
    kt_list = []
    kt_top_list = []
    mutate_list = []
    if len(data) <= 10:
        batch_size = 10
    else:
        batch_size = 16
    if rate is None:
        rate = 10.
    else:
        rate = rate
    while query <= total_queries:
        arch_data = [d[0] for d in data]
        agent = NasBenchGinPredictorTrainer(search_agent, lr=lr, device=device, epochs=epochs,
                                            train_images=len(data), batch_size=batch_size, rate=rate,
                                            activation_fn=activation_fn)
        val_accuracy = np.array([d[4] for d in data])
        arch_data_edge_idx_list = []
        arch_data_node_f_list = []
        for arch in arch_data:
            edge_index, node_f = nasbench2graph2(arch)
            arch_data_edge_idx_list.append(edge_index)
            arch_data_node_f_list.append(node_f)
        if record_mutation == 'T':
            candidates, dist_list, replicate_num, mutated_nums_list, mutated_arch_list \
                = search_space.get_candidates(data,
                                              num=candidate_nums,
                                              acq_opt_type=acq_opt_type,
                                              allow_isomorphisms=allow_isomorphisms,
                                              return_dist=True)
            cand_val_list = [cand[4] for cand in candidates]
            mutate_list.append((dist_list, replicate_num, mutated_nums_list, mutated_arch_list, cand_val_list))
        else:
            candidates = search_space.get_candidates(data,
                                                     num=candidate_nums,
                                                     acq_opt_type=acq_opt_type,
                                                     allow_isomorphisms=allow_isomorphisms)
        candiate_edge_list = []
        candiate_node_list = []
        for cand in candidates:
            edge_index, node_f = nasbench2graph2(cand[0])
            candiate_edge_list.append(edge_index)
            candiate_node_list.append(node_f)
        agent.fit(arch_data_edge_idx_list, arch_data_node_f_list, val_accuracy, logger=None)
        acc_train = agent.pred(arch_data_edge_idx_list, arch_data_node_f_list)
        acc_pred = agent.pred(candiate_edge_list, candiate_node_list)
        candidate_np = acc_pred.cpu().numpy()
        if record_kt == 'T':
            candidates_gt = [can[4] for can in candidates]
            kt = get_kendalltau_coorlection(candidate_np.tolist(), candidates_gt)[0]
            kt_list.append(kt)
        sorted_indices = np.argsort(candidate_np)
        kt_top_pred_list = []
        kt_top_gt_list = []
        for i in sorted_indices[:k]:
            archtuple = search_space.query_arch(matrix=candidates[i][1],
                                                ops=candidates[i][2])
            data.append(archtuple)
            kt_top_pred_list.append(candidate_np[i])
            kt_top_gt_list.append(archtuple[4])
        kt_top_list.append(get_kendalltau_coorlection(kt_top_pred_list, kt_top_gt_list)[0])
        if verbose:
            top_5_loss = sorted([d[4] for d in data])[:min(5, len(data))]
            logger.info('Query {}, training mean loss is  {}'.format(query,
                                                                     np.mean(np.abs(acc_train.cpu().numpy()-val_accuracy))))
            logger.info('Query {}, top 5 val losses {}'.format(query, top_5_loss))
        query += k
    return data, {'type': 'gin_predictor', 'final_data': data, 'kt_top_list': kt_top_list,
                  'kt_list': kt_list, 'mutate_list': mutate_list}


def gin_uncertainty_predictor_nasbench_201(search_space,
                                           num_init=10,
                                           k=10,
                                           total_queries=150,
                                           acq_opt_type='mutation',
                                           allow_isomorphisms=False,
                                           verbose=1,
                                           agent=None,
                                           logger=None,
                                           gpu='0',
                                           lr=0.01,
                                           candidate_nums=100,
                                           epochs=1000,
                                           record_kt='F',
                                           record_mutation='F',
                                           activation_fn='celu'
                                           ):
    """
    Bayesian optimization with a neural network model
    """
    device = torch.device('cuda:%d' % gpu)
    data = search_space.generate_random_dataset(num=num_init,
                                                allow_isomorphisms=allow_isomorphisms,
                                                deterministic_loss=True)
    query = num_init + k
    search_agent = agent
    kt_list = []
    kt_top_list = []
    mutate_list = []
    if len(data) <= 10:
        batch_size = 10
    else:
        batch_size = 16
    while query <= total_queries:
        arch_data = [d[0] for d in data]
        agent = NasBenchGinGaussianTrainer(search_agent, lr=lr, device=device, epochs=epochs,
                                           train_images=len(data), batch_size=batch_size, input_dim=8,
                                           activation_fn=activation_fn)
        val_accuracy = np.array([d[4] for d in data])
        arch_data_edge_idx_list = []
        arch_data_node_f_list = []
        for arch in arch_data:
            edge_index, node_f = search_space.nasbench2graph2(arch)
            arch_data_edge_idx_list.append(edge_index)
            arch_data_node_f_list.append(node_f)
        if record_mutation == 'T':
            candidates, dist_list, replicate_num, mutated_nums_list, mutated_arch_list \
                = search_space.get_candidates(data,
                                              num=candidate_nums,
                                              allow_isomorphisms=allow_isomorphisms,
                                              return_dist=True
                                              )
            cand_val_list = [cand[4] for cand in candidates]
            mutate_list.append((dist_list, replicate_num, mutated_nums_list, mutated_arch_list, cand_val_list))
        else:
            candidates = search_space.get_candidates(data,
                                                     num=candidate_nums,
                                                     allow_isomorphisms=allow_isomorphisms
                                                     )
        candiate_edge_list = []
        candiate_node_list = []
        for cand in candidates:
            edge_index, node_f = search_space.nasbench2graph2(cand[0])
            candiate_edge_list.append(edge_index)
            candiate_node_list.append(node_f)
        agent.fit(arch_data_edge_idx_list, arch_data_node_f_list, val_accuracy, logger=None)
        pred_train, mean_train, _ = agent.pred(arch_data_edge_idx_list, arch_data_node_f_list)
        predictions, _, _ = agent.pred(candiate_edge_list, candiate_node_list)
        if record_kt == 'T':
            np_pred = predictions.cpu().numpy()
            candidates_gt = [can[4] for can in candidates]
            kt = get_kendalltau_coorlection(np_pred.tolist(), candidates_gt)[0]
            kt_list.append(kt)
        sorted_indices = acq_fn(predictions, 'its_vae')
        kt_top_pred_list = []
        kt_top_gt_list = []
        np_pred = predictions.cpu().numpy()
        for i in sorted_indices[:k]:
            archtuple = candidates[i]
            data.append(archtuple)
            kt_top_pred_list.append(np_pred[i])
            kt_top_gt_list.append(archtuple[4])
        kt_top_list.append(get_kendalltau_coorlection(kt_top_pred_list, kt_top_gt_list)[0])
        if verbose:
            top_5_loss = sorted([d[4] for d in data])[:min(5, len(data))]
            logger.info('Query {}, training mean loss is  {}'.format(query,
                                                                     np.mean(np.abs(mean_train.cpu().numpy()-val_accuracy))))
            logger.info('Query {}, top 5 val losses {}'.format(query, top_5_loss))
        query += k
    return data, {'type': 'gin_uncertainty', 'final_data': data, 'kt_list': kt_list, 'kt_top_list': kt_top_list,
                  'mutate_list': mutate_list}


def gin_predictor_nasbench_201(search_space,
                               dataname='cifar100',
                               num_init=10,
                               k=10,
                               total_queries=150,
                               acq_opt_type='mutation',
                               allow_isomorphisms=False,
                               verbose=1,
                               agent=None,
                               logger=None,
                               gpu='0',
                               lr=0.01,
                               candidate_nums=100,
                               epochs=1000,
                               record_kt='F',
                               record_mutation='F',
                               rate=None
                               ):
    """
    Bayesian optimization with a neural network model
    """
    if rate is None:
        if dataname == 'cifar10-valid':
            rate = 100.
        elif dataname == 'cifar100':
            rate = 100.
        elif dataname == 'ImageNet16-120':
            rate = 100.
        else:
            raise NotImplementedError()
    else:
        rate = rate
    kt_list = []
    kt_top_list = []
    mutate_list = []
    device = torch.device('cuda:%d' % gpu)
    data = search_space.generate_random_dataset(num=num_init,
                                                allow_isomorphisms=allow_isomorphisms,
                                                deterministic_loss=True)
    query = num_init + k
    search_agent = agent
    if len(data) <= 10:
        batch_size = 10
    else:
        batch_size = 16
    while query <= total_queries:
        arch_data = [d[0] for d in data]
        agent = NasBenchGinPredictorTrainer(search_agent, lr=lr, device=device, epochs=epochs,
                                            train_images=len(data), batch_size=batch_size, input_dim=8, rate=rate)
        val_accuracy = np.array([d[4] for d in data])
        arch_data_edge_idx_list = []
        arch_data_node_f_list = []
        for arch in arch_data:
            edge_index, node_f = search_space.nasbench2graph2(arch)
            arch_data_edge_idx_list.append(edge_index)
            arch_data_node_f_list.append(node_f)
        if record_mutation == 'T':
            candidates, dist_list, replicate_num, mutated_nums_list, mutated_arch_list \
                = search_space.get_candidates(data,
                                              num=candidate_nums,
                                              allow_isomorphisms=allow_isomorphisms,
                                              return_dist=True
                                              )
            cand_val_list = [cand[4] for cand in candidates]
            mutate_list.append((dist_list, replicate_num, mutated_nums_list, mutated_arch_list, cand_val_list))
        else:
            candidates = search_space.get_candidates(data,
                                                     num=candidate_nums,
                                                     allow_isomorphisms=allow_isomorphisms
                                                     )
        candiate_edge_list = []
        candiate_node_list = []
        for cand in candidates:
            edge_index, node_f = search_space.nasbench2graph2(cand[0])
            candiate_edge_list.append(edge_index)
            candiate_node_list.append(node_f)
        agent.fit(arch_data_edge_idx_list, arch_data_node_f_list, val_accuracy, logger=None)
        acc_train = agent.pred(arch_data_edge_idx_list, arch_data_node_f_list)
        acc_pred = agent.pred(candiate_edge_list, candiate_node_list)
        candidate_np = acc_pred.cpu().numpy()
        if record_kt == 'T':
            candidates_gt = [can[4] for can in candidates]
            kt = get_kendalltau_coorlection(candidate_np.tolist(), candidates_gt)[0]
            kt_list.append(kt)
        sorted_indices = np.argsort(candidate_np)
        kt_top_pred_list = []
        kt_top_gt_list = []
        for i in sorted_indices[:k]:
            archtuple = candidates[i]
            data.append(archtuple)
            kt_top_pred_list.append(candidate_np[i])
            kt_top_gt_list.append(archtuple[4])
        kt_top_list.append(get_kendalltau_coorlection(kt_top_pred_list, kt_top_gt_list)[0])
        if verbose:
            top_5_loss = sorted([d[4] for d in data])[:min(5, len(data))]
            logger.info('Query {}, training mean loss is  {}'.format(query,
                                                                     np.mean(np.abs(acc_train.cpu().numpy()-val_accuracy))))
            logger.info('Query {}, top 5 val losses {}'.format(query, top_5_loss))
        query += k
    return data, {'type': 'gin_predictor', 'final_data': data, 'kt_top_list': kt_top_list,
                  'kt_list': kt_list, 'mutate_list': mutate_list}


def gin_predictor_scalar_case2(search_space,
                               num_init=10,
                               k=10,
                               total_queries=150,
                               acq_opt_type='mutation',
                               allow_isomorphisms=False,
                               verbose=1,
                               agent=None,
                               logger=None,
                               gpu='0',
                               lr=0.01,
                               candidate_nums=100,
                               epochs=1000,
                               scalar=10,
                               ):
    """
    Bayesian optimization with a neural network model
    """
    device = torch.device('cuda:%d' % gpu)
    data = search_space.generate_random_dataset(num=num_init,
                                                allow_isomorphisms=allow_isomorphisms,
                                                deterministic_loss=True)
    query = num_init + k
    search_agent = agent
    if len(data) <= 10:
        batch_size = 10
    else:
        batch_size = 16
    while query <= total_queries:
        arch_data = [d[0] for d in data]
        agent = NasBenchGinPredictorTrainer(search_agent, lr=lr, device=device, epochs=epochs,
                                            train_images=len(data), batch_size=batch_size, rate=scalar)
        val_accuracy = np.array([d[4] for d in data])
        arch_data_edge_idx_list = []
        arch_data_node_f_list = []
        for arch in arch_data:
            edge_index, node_f = nasbench2graph2(arch)
            arch_data_edge_idx_list.append(edge_index)
            arch_data_node_f_list.append(node_f)
        candidates = search_space.get_candidates(data,
                                                 num=candidate_nums,
                                                 acq_opt_type=acq_opt_type,
                                                 allow_isomorphisms=allow_isomorphisms)
        candiate_edge_list = []
        candiate_node_list = []
        for cand in candidates:
            edge_index, node_f = nasbench2graph2(cand[0])
            candiate_edge_list.append(edge_index)
            candiate_node_list.append(node_f)
        agent.fit(arch_data_edge_idx_list, arch_data_node_f_list, val_accuracy, logger=None)
        acc_train = agent.pred(arch_data_edge_idx_list, arch_data_node_f_list)
        acc_pred = agent.pred(candiate_edge_list, candiate_node_list)
        candidate_np = acc_pred.cpu().numpy()
        sorted_indices = np.argsort(candidate_np)
        for i in sorted_indices[:k]:
            archtuple = search_space.query_arch(matrix=candidates[i][1],
                                                ops=candidates[i][2])
            data.append(archtuple)
        if verbose:
            top_5_loss = sorted([d[4] for d in data])[:min(5, len(data))]
            logger.info('Query {}, training mean loss is  {}'.format(query,
                                                                     np.mean(np.abs(acc_train.cpu().numpy()-val_accuracy))))
            logger.info('Query {}, top 5 val losses {}'.format(query, top_5_loss))
        query += k
    return data


def gin_predictor_train_num_constrict_case1(search_space,
                                            num_init=10,
                                            k=10,
                                            total_queries=150,
                                            acq_opt_type='mutation',
                                            allow_isomorphisms=False,
                                            verbose=1,
                                            agent=None,
                                            logger=None,
                                            gpu='0',
                                            lr=0.01,
                                            candidate_nums=100,
                                            epochs=1000,
                                            training_nums=150):
    """
    Bayesian optimization with a neural network model
    """
    device = torch.device('cuda:%d' % gpu)
    data = search_space.generate_random_dataset_gin(num=num_init,
                                                    allow_isomorphisms=allow_isomorphisms,
                                                    deterministic_loss=True)
    query = num_init + k
    search_agent = agent
    train_data = []
    train_flag = False
    while query <= total_queries:
        if len(train_data) < training_nums:
            train_data = copy.deepcopy(data)
            train_flag = True
        candidates = search_space.get_candidates_gin(data,
                                                     num=candidate_nums,
                                                     acq_opt_type=acq_opt_type,
                                                     allow_isomorphisms=allow_isomorphisms)
        candiate_edge_list = []
        candiate_node_list = []
        for cand in candidates:
            edge_index, node_f = nasbench2graph(cand[0])
            candiate_edge_list.append(edge_index)
            candiate_node_list.append(node_f)
        if train_flag:
            arch_data_edge_idx_list = []
            arch_data_node_f_list = []
            for d in train_data:
                edge_index, node_f = nasbench2graph(d[0])
                arch_data_edge_idx_list.append(edge_index)
                arch_data_node_f_list.append(node_f)
            val_accuracy = np.array([d[2] for d in train_data])
            batch_size = 10 if len(train_data) <= 10 else 16
            agent = NasBenchGinPredictorTrainer(search_agent, lr=lr, device=device, epochs=epochs,
                                                train_images=len(train_data), batch_size=batch_size)
            agent.fit(arch_data_edge_idx_list, arch_data_node_f_list, val_accuracy, logger=None)
            acc_train = agent.pred(arch_data_edge_idx_list, arch_data_node_f_list).cpu().numpy()
        acc_pred = agent.pred(candiate_edge_list, candiate_node_list)
        candidate_np = acc_pred.cpu().numpy()
        sorted_indices = np.argsort(candidate_np)
        for i in sorted_indices[:k]:
            archtuple = search_space.query_arch_gin(candidates[i][0],
                                                    encode_paths=True,
                                                    deterministic=True
                                                    )
            data.append(archtuple)
        if verbose:
            top_5_loss = sorted([d[2] for d in data])[:min(5, len(data))]
            logger.info('Query {}, training data nums {},  training mean loss is  {}'.format(query, len(train_data),
                                                                                             np.mean(np.abs(acc_train-val_accuracy))))
            logger.info('Query {}, top 5 val losses {}'.format(query, top_5_loss))
        query += k
        train_flag = False
    return data


def gin_uncertainty_predictor_train_num_constrict_case1(search_space,
                                                        num_init=10,
                                                        k=10,
                                                        total_queries=150,
                                                        acq_opt_type='mutation',
                                                        allow_isomorphisms=False,
                                                        verbose=1,
                                                        agent=None,
                                                        logger=None,
                                                        gpu='0',
                                                        lr=0.01,
                                                        candidate_nums=100,
                                                        epochs=1000,
                                                        training_nums=150):
    """
    Bayesian optimization with a neural network model
    """
    device = torch.device('cuda:%d' % gpu)
    data = search_space.generate_random_dataset_gin(num=num_init,
                                                    encode_paths=True,
                                                    allow_isomorphisms=allow_isomorphisms,
                                                    deterministic_loss=True
                                                    )
    query = num_init + k
    search_agent = agent
    train_data = []
    train_flag = False
    while query <= total_queries:
        if len(train_data) < training_nums:
            train_data = copy.deepcopy(data)
            train_flag = True
        candidates = search_space.get_candidates_gin(data,
                                                     num=candidate_nums,
                                                     acq_opt_type=acq_opt_type,
                                                     encode_paths=True,
                                                     allow_isomorphisms=allow_isomorphisms,
                                                     deterministic_loss=True)
        candiate_edge_list = []
        candiate_node_list = []
        for cand in candidates:
            edge_index, node_f = nasbench2graph(cand[0])
            candiate_edge_list.append(edge_index)
            candiate_node_list.append(node_f)

        if train_flag:
            batch_size = 10 if len(train_data) <= 10 else 16
            agent = NasBenchGinGaussianTrainer(search_agent, lr=lr, device=device, epochs=epochs,
                                               train_images=len(train_data), batch_size=batch_size)
            val_accuracy = np.array([d[2] for d in train_data])
            arch_data_edge_idx_list = []
            arch_data_node_f_list = []
            for d in train_data:
                edge_index, node_f = nasbench2graph(d[0])
                arch_data_edge_idx_list.append(edge_index)
                arch_data_node_f_list.append(node_f)
            agent.fit(arch_data_edge_idx_list, arch_data_node_f_list, val_accuracy, logger=None)
            pred_train, mean_train, _ = agent.pred(arch_data_edge_idx_list, arch_data_node_f_list)
            mean_train = mean_train.cpu().numpy()
        predictions, _, _ = agent.pred(candiate_edge_list, candiate_node_list)
        sorted_indices = acq_fn(predictions, 'its_vae')
        for i in sorted_indices[:k]:
            archtuple = search_space.query_arch_gin(candidates[i][0],
                                                    encode_paths=True,
                                                    deterministic=True
                                                    )
            data.append(archtuple)
        if verbose:
            top_5_loss = sorted([d[2] for d in data])[:min(5, len(data))]
            logger.info('Query {}, training data nums {},  training mean loss is  {}'.format(query, len(train_data),
                                                                     np.mean(np.abs(mean_train - val_accuracy))))
            logger.info('Query {}, top 5 val losses {}'.format(query, top_5_loss))
        query += k
        train_flag = False
    return data


def gin_predictor_train_num_constrict_case2(search_space,
                                            num_init=10,
                                            k=10,
                                            total_queries=150,
                                            acq_opt_type='mutation',
                                            allow_isomorphisms=False,
                                            verbose=1,
                                            agent=None,
                                            logger=None,
                                            gpu='0',
                                            lr=0.01,
                                            candidate_nums=100,
                                            epochs=1000,
                                            training_nums=150):
    """
    Bayesian optimization with a neural network model
    """
    device = torch.device('cuda:%d' % gpu)
    data = search_space.generate_random_dataset(num=num_init,
                                                allow_isomorphisms=allow_isomorphisms,
                                                deterministic_loss=True)
    query = num_init + k
    search_agent = agent
    train_data = []
    train_flag = False
    while query <= total_queries:
        if len(train_data) < training_nums:
            train_data = copy.deepcopy(data)
            train_flag = True
        batch_size = 10 if len(train_data) <= 10 else 16
        candidates = search_space.get_candidates(data,
                                                 num=candidate_nums,
                                                 acq_opt_type=acq_opt_type,
                                                 allow_isomorphisms=allow_isomorphisms)
        candiate_edge_list = []
        candiate_node_list = []
        for cand in candidates:
            edge_index, node_f = nasbench2graph2(cand[0])
            candiate_edge_list.append(edge_index)
            candiate_node_list.append(node_f)
        if train_flag:
            arch_data = [d[0] for d in train_data]
            val_accuracy = np.array([d[4] for d in train_data])
            agent = NasBenchGinPredictorTrainer(search_agent, lr=lr, device=device, epochs=epochs,
                                                train_images=len(train_data), batch_size=batch_size)
            arch_data_edge_idx_list = []
            arch_data_node_f_list = []
            for arch in arch_data:
                edge_index, node_f = nasbench2graph2(arch)
                arch_data_edge_idx_list.append(edge_index)
                arch_data_node_f_list.append(node_f)
            agent.fit(arch_data_edge_idx_list, arch_data_node_f_list, val_accuracy, logger=None)
            acc_train = agent.pred(arch_data_edge_idx_list, arch_data_node_f_list).cpu().numpy()
        acc_pred = agent.pred(candiate_edge_list, candiate_node_list)
        candidate_np = acc_pred.cpu().numpy()
        sorted_indices = np.argsort(candidate_np)
        for i in sorted_indices[:k]:
            archtuple = search_space.query_arch(matrix=candidates[i][1],
                                                ops=candidates[i][2])
            data.append(archtuple)
        if verbose:
            top_5_loss = sorted([d[4] for d in data])[:min(5, len(data))]
            logger.info('Query {}, training data nums {},  training mean loss is  {}'.format(query, len(train_data),
                                                                     np.mean(np.abs(acc_train-val_accuracy))))
            logger.info('Query {}, top 5 val losses {}'.format(query, top_5_loss))
        query += k
        train_flag = False
    return data


def gin_uncertainty_train_num_constrict_case2(search_space,
                                              num_init=10,
                                              k=10,
                                              total_queries=150,
                                              acq_opt_type='mutation',
                                              allow_isomorphisms=False,
                                              verbose=1,
                                              agent=None,
                                              logger=None,
                                              gpu='0',
                                              lr=0.01,
                                              candidate_nums=100,
                                              epochs=1000,
                                              training_nums=150):
    """
    Bayesian optimization with a neural network model
    """
    device = torch.device('cuda:%d' % gpu)
    data = search_space.generate_random_dataset(num=num_init,
                                                allow_isomorphisms=allow_isomorphisms,
                                                deterministic_loss=True)
    query = num_init + k
    search_agent = agent
    train_data = []
    train_flag = False
    while query <= total_queries:
        if len(train_data) < training_nums:
            train_data = copy.deepcopy(data)
            train_flag = True
        candidates = search_space.get_candidates(data,
                                                 num=candidate_nums,
                                                 acq_opt_type=acq_opt_type,
                                                 allow_isomorphisms=allow_isomorphisms)
        candiate_edge_list = []
        candiate_node_list = []
        for cand in candidates:
            edge_index, node_f = nasbench2graph2(cand[0])
            candiate_edge_list.append(edge_index)
            candiate_node_list.append(node_f)
        if train_flag:
            batch_size = 10 if len(train_data) <= 10 else 16
            agent = NasBenchGinGaussianTrainer(search_agent, lr=lr, device=device, epochs=epochs,
                                               train_images=len(train_data), batch_size=batch_size)
            arch_data = [d[0] for d in train_data]
            val_accuracy = np.array([d[4] for d in train_data])
            arch_data_edge_idx_list = []
            arch_data_node_f_list = []
            for arch in arch_data:
                edge_index, node_f = nasbench2graph2(arch)
                arch_data_edge_idx_list.append(edge_index)
                arch_data_node_f_list.append(node_f)
            agent.fit(arch_data_edge_idx_list, arch_data_node_f_list, val_accuracy, logger=None)
            pred_train, mean_train, _ = agent.pred(arch_data_edge_idx_list, arch_data_node_f_list)
            mean_train = mean_train.cpu().numpy()
        predictions, _, _ = agent.pred(candiate_edge_list, candiate_node_list)
        sorted_indices = acq_fn(predictions, 'its_vae')
        for i in sorted_indices[:k]:
            archtuple = search_space.query_arch(matrix=candidates[i][1],
                                                ops=candidates[i][2])
            data.append(archtuple)
        if verbose:
            top_5_loss = sorted([d[4] for d in data])[:min(5, len(data))]
            logger.info('Query {}, training data nums {},  training mean loss is  {}'.format(query, len(train_data),
                                                                                             np.mean(np.abs(mean_train-val_accuracy))))
            logger.info('Query {}, top 5 val losses {}'.format(query, top_5_loss))
        query += k
        train_flag = False
    return data


def gin_uncertainty_predictor_nasbench_nlp(search_space,
                                           num_init=10,
                                           k=10,
                                           total_queries=150,
                                           acq_opt_type='mutation',
                                           allow_isomorphisms=False,
                                           verbose=1,
                                           agent=None,
                                           logger=None,
                                           gpu='0',
                                           lr=0.01,
                                           candidate_nums=100,
                                           epochs=1000,
                                           mutation_rate=0.1,
                                           record_kt='F',
                                           record_mutation='F',
                                           activation_fn='celu'
                                           ):
    """
    Bayesian optimization with a neural network model
    """
    device = torch.device('cuda:%d' % gpu)
    data = search_space.generate_random_dataset(num=num_init,
                                                allow_isomorphisms=allow_isomorphisms,
                                                deterministic_loss=True)
    query = num_init + k
    search_agent = agent
    if len(data) <= 10:
        batch_size = 10
    else:
        batch_size = 16
    kt_list = []
    kt_top_list = []
    mutate_list = []
    while query <= total_queries:
        arch_data = [(d[1], d[2]) for d in data]
        agent = NasBenchGinGaussianTrainer(search_agent, lr=lr, device=device, epochs=epochs,
                                           train_images=len(data), batch_size=batch_size, input_dim=10,
                                           activation_fn=activation_fn)
        val_accuracy = np.array([d[4] for d in data])
        arch_data_edge_idx_list = []
        arch_data_node_f_list = []
        for arch in arch_data:
            edge_index, node_f = search_space.nasbench2graph2(arch)
            arch_data_edge_idx_list.append(edge_index)
            arch_data_node_f_list.append(node_f)
        if record_mutation == 'T':
            candidates, dist_list, replicate_num, mutated_nums_list, mutated_arch_list \
                = search_space.get_candidates(data,
                                              num=candidate_nums,
                                              allow_isomorphisms=allow_isomorphisms,
                                              mutation_rate=mutation_rate,
                                              return_dist=True
                                              )
            cand_val_list = [cand[4] for cand in candidates]
            mutate_list.append((dist_list, replicate_num, mutated_nums_list, mutated_arch_list, cand_val_list))
        else:
            candidates = search_space.get_candidates(data,
                                                     num=candidate_nums,
                                                     allow_isomorphisms=allow_isomorphisms,
                                                     mutation_rate=mutation_rate
                                                     )
        candiate_edge_list = []
        candiate_node_list = []
        for cand in candidates:
            edge_index, node_f = search_space.nasbench2graph2((cand[1], cand[2]))
            candiate_edge_list.append(edge_index)
            candiate_node_list.append(node_f)
        agent.fit(arch_data_edge_idx_list, arch_data_node_f_list, val_accuracy, logger=None)
        pred_train, mean_train, _ = agent.pred(arch_data_edge_idx_list, arch_data_node_f_list)
        predictions, _, _ = agent.pred(candiate_edge_list, candiate_node_list)
        if record_kt == 'T':
            np_pred = predictions.cpu().numpy()
            candidates_gt = [can[4] for can in candidates]
            kt = get_kendalltau_coorlection(np_pred.tolist(), candidates_gt)[0]
            kt_list.append(kt)
        sorted_indices = acq_fn(predictions, 'its_vae')
        kt_top_pred_list = []
        kt_top_gt_list = []
        np_pred = predictions.cpu().numpy()
        for i in sorted_indices[:k]:
            archtuple = candidates[i]
            data.append(archtuple)
            kt_top_pred_list.append(np_pred[i])
            kt_top_gt_list.append(archtuple[4])
        kt_top_list.append(get_kendalltau_coorlection(kt_top_pred_list, kt_top_gt_list)[0])
        if verbose:
            top_5_loss = sorted([d[4] for d in data])[:min(5, len(data))]
            logger.info('Query {}, training mean loss is  {}'.format(query,
                                                                     np.mean(np.abs(mean_train.cpu().numpy()-val_accuracy))))
            logger.info('Query {}, top 5 val losses {}'.format(query, top_5_loss))
        query += k
    return data, {'type': 'gin_uncertainty', 'final_data': data, 'kt_list': kt_list, 'kt_top_list': kt_top_list,
                  'mutate_list': mutate_list}


def gin_predictor_nasbench_nlp(search_space,
                               num_init=10,
                               k=10,
                               total_queries=150,
                               acq_opt_type='mutation',
                               allow_isomorphisms=False,
                               verbose=1,
                               agent=None,
                               logger=None,
                               gpu='0',
                               lr=0.01,
                               candidate_nums=100,
                               epochs=1000,
                               mutation_rate=0.1,
                               record_kt='F',
                               record_mutation='F',
                               rate=None
                               ):
    """
    Bayesian optimization with a neural network model
    """
    device = torch.device('cuda:%d' % gpu)
    data = search_space.generate_random_dataset(num=num_init,
                                                allow_isomorphisms=allow_isomorphisms,
                                                deterministic_loss=True)
    query = num_init + k
    search_agent = agent
    if len(data) <= 10:
        batch_size = 10
    else:
        batch_size = 16
    kt_list = []
    kt_top_list = []
    mutate_list = []
    if rate is not None:
        rate = rate
    else:
        rate = 5.
    while query <= total_queries:
        arch_data = [(d[1], d[2]) for d in data]
        agent = NasBenchGinPredictorTrainer(search_agent, lr=lr, device=device, epochs=epochs,
                                            train_images=len(data), batch_size=batch_size, input_dim=10, rate=rate)
        val_accuracy = np.array([d[4] for d in data])
        arch_data_edge_idx_list = []
        arch_data_node_f_list = []
        for arch in arch_data:
            edge_index, node_f = search_space.nasbench2graph2(arch)
            arch_data_edge_idx_list.append(edge_index)
            arch_data_node_f_list.append(node_f)
        if record_mutation == 'T':
            candidates, dist_list, replicate_num, mutated_nums_list, mutated_arch_list \
                = search_space.get_candidates(data,
                                              num=candidate_nums,
                                              allow_isomorphisms=allow_isomorphisms,
                                              mutation_rate=mutation_rate,
                                              return_dist=True
                                              )
            cand_val_list = [cand[4] for cand in candidates]
            mutate_list.append((dist_list, replicate_num, mutated_nums_list, mutated_arch_list, cand_val_list))
        else:
            candidates = search_space.get_candidates(data,
                                                     num=candidate_nums,
                                                     allow_isomorphisms=allow_isomorphisms,
                                                     mutation_rate=mutation_rate
                                                 )
        candiate_edge_list = []
        candiate_node_list = []
        for cand in candidates:
            edge_index, node_f = search_space.nasbench2graph2((cand[1], cand[2]))
            candiate_edge_list.append(edge_index)
            candiate_node_list.append(node_f)
        agent.fit(arch_data_edge_idx_list, arch_data_node_f_list, val_accuracy, logger=None)
        acc_train = agent.pred(arch_data_edge_idx_list, arch_data_node_f_list)
        acc_pred = agent.pred(candiate_edge_list, candiate_node_list)
        candidate_np = acc_pred.cpu().numpy()
        if record_kt == 'T':
            candidates_gt = [can[4] for can in candidates]
            kt = get_kendalltau_coorlection(candidate_np.tolist(), candidates_gt)[0]
            kt_list.append(kt)
        sorted_indices = np.argsort(candidate_np)
        kt_top_pred_list = []
        kt_top_gt_list = []
        for i in sorted_indices[:k]:
            archtuple = candidates[i]
            data.append(archtuple)
            kt_top_pred_list.append(candidate_np[i])
            kt_top_gt_list.append(archtuple[4])
        kt_top_list.append(get_kendalltau_coorlection(kt_top_pred_list, kt_top_gt_list)[0])
        if verbose:
            top_5_loss = sorted([d[4] for d in data])[:min(5, len(data))]
            logger.info('Query {}, training mean loss is  {}'.format(query,
                                                                     np.mean(np.abs(acc_train.cpu().numpy()-val_accuracy))))
            logger.info('Query {}, top 5 val losses {}'.format(query, top_5_loss))
        query += k
    return data, {'type': 'gin_predictor', 'final_data': data, 'kt_top_list': kt_top_list,
                  'kt_list': kt_list, 'mutate_list': mutate_list}


def gin_uncertainty_predictor_nasbench_asr(search_space,
                                           num_init=10,
                                           k=10,
                                           total_queries=150,
                                           acq_opt_type='mutation',
                                           allow_isomorphisms=False,
                                           verbose=1,
                                           agent=None,
                                           logger=None,
                                           gpu='0',
                                           lr=0.01,
                                           candidate_nums=100,
                                           epochs=1000,
                                           mutation_rate=1,
                                           record_kt='F',
                                           record_mutation='F',
                                           activation_fn='celu'
                                           ):
    """
    Bayesian optimization with a neural network model
    """
    device = torch.device('cuda:%d' % gpu)
    data = search_space.generate_random_dataset(num=num_init,
                                                allow_isomorphisms=allow_isomorphisms,
                                                deterministic_loss=True)
    query = num_init + k
    search_agent = agent
    if len(data) <= 10:
        batch_size = 10
    else:
        batch_size = 16
    kt_list = []
    kt_top_list = []
    mutate_list = []
    while query <= total_queries:
        arch_data = [(d[1], d[2]) for d in data]
        agent = NasBenchGinGaussianTrainer(search_agent, lr=lr, device=device, epochs=epochs,
                                           train_images=len(data), batch_size=batch_size, input_dim=9,
                                           activation_fn=activation_fn)
        val_accuracy = np.array([d[4] for d in data])
        arch_data_edge_idx_list = []
        arch_data_node_f_list = []
        for arch in arch_data:
            edge_index, node_f = search_space.nasbench2graph2(arch)
            arch_data_edge_idx_list.append(edge_index)
            arch_data_node_f_list.append(node_f)
        if record_mutation == 'T':
            candidates, dist_list, replicate_num, mutated_nums_list, mutated_arch_list \
                = search_space.get_candidates(data,
                                              num=candidate_nums,
                                              allow_isomorphisms=allow_isomorphisms,
                                              mutation_rate=mutation_rate,
                                              return_dist=True
                                              )
            cand_val_list = [cand[4] for cand in candidates]
            mutate_list.append((dist_list, replicate_num, mutated_nums_list, mutated_arch_list, cand_val_list))
        else:
            candidates = search_space.get_candidates(data,
                                                     num=candidate_nums,
                                                     allow_isomorphisms=allow_isomorphisms,
                                                     mutation_rate=mutation_rate
                                                 )
        candiate_edge_list = []
        candiate_node_list = []
        for cand in candidates:
            edge_index, node_f = search_space.nasbench2graph2((cand[1], cand[2]))
            candiate_edge_list.append(edge_index)
            candiate_node_list.append(node_f)
        agent.fit(arch_data_edge_idx_list, arch_data_node_f_list, val_accuracy, logger=None)
        pred_train, mean_train, _ = agent.pred(arch_data_edge_idx_list, arch_data_node_f_list)
        predictions, _, _ = agent.pred(candiate_edge_list, candiate_node_list)
        if record_kt == 'T':
            np_pred = predictions.cpu().numpy()
            candidates_gt = [can[4] for can in candidates]
            kt = get_kendalltau_coorlection(np_pred.tolist(), candidates_gt)[0]
            kt_list.append(kt)
        sorted_indices = acq_fn(predictions, 'its_vae')
        kt_top_pred_list = []
        kt_top_gt_list = []
        np_pred = predictions.cpu().numpy()
        for i in sorted_indices[:k]:
            archtuple = candidates[i]
            data.append(archtuple)
            kt_top_pred_list.append(np_pred[i])
            kt_top_gt_list.append(archtuple[4])
        kt_top_list.append(get_kendalltau_coorlection(kt_top_pred_list, kt_top_gt_list)[0])
        if verbose:
            top_5_loss = sorted([d[4] for d in data])[:min(5, len(data))]
            logger.info('Query {}, training mean loss is  {}'.format(query,
                                                                     np.mean(np.abs(mean_train.cpu().numpy()-val_accuracy))))
            logger.info('Query {}, top 5 val losses {}'.format(query, top_5_loss))
        query += k
    return data, {'type': 'gin_uncertainty', 'final_data': data, 'kt_list': kt_list, 'kt_top_list': kt_top_list,
                  'mutate_list': mutate_list}


def gin_predictor_nasbench_asr(search_space,
                               num_init=10,
                               k=10,
                               total_queries=150,
                               acq_opt_type='mutation',
                               allow_isomorphisms=False,
                               verbose=1,
                               agent=None,
                               logger=None,
                               gpu='0',
                               lr=0.01,
                               candidate_nums=100,
                               epochs=1000,
                               mutation_rate=1,
                               record_kt='F',
                               record_mutation='F',
                               rate=None
                               ):
    """
    Bayesian optimization with a neural network model
    """
    device = torch.device('cuda:%d' % gpu)
    data = search_space.generate_random_dataset(num=num_init,
                                                allow_isomorphisms=allow_isomorphisms,
                                                deterministic_loss=True)
    query = num_init + k
    search_agent = agent
    if len(data) <= 10:
        batch_size = 10
    else:
        batch_size = 16
    kt_list = []
    kt_top_list = []
    mutate_list = []
    if rate is None:
        rate = 100.
    else:
        rate = rate
    while query <= total_queries:
        arch_data = [(d[1], d[2]) for d in data]
        agent = NasBenchGinPredictorTrainer(search_agent, lr=lr, device=device, epochs=epochs,
                                            train_images=len(data), batch_size=batch_size, input_dim=9, rate=rate)
        val_accuracy = np.array([d[4] for d in data])
        arch_data_edge_idx_list = []
        arch_data_node_f_list = []
        for arch in arch_data:
            edge_index, node_f = search_space.nasbench2graph2(arch)
            arch_data_edge_idx_list.append(edge_index)
            arch_data_node_f_list.append(node_f)
        if record_mutation == 'T':
            candidates, dist_list, replicate_num, mutated_nums_list, mutated_arch_list \
                = search_space.get_candidates(data,
                                              num=candidate_nums,
                                              allow_isomorphisms=allow_isomorphisms,
                                              mutation_rate=mutation_rate,
                                              return_dist=True
                                              )
            cand_val_list = [cand[4] for cand in candidates]
            mutate_list.append((dist_list, replicate_num, mutated_nums_list, mutated_arch_list, cand_val_list))
        else:
            candidates = search_space.get_candidates(data,
                                                     num=candidate_nums,
                                                     allow_isomorphisms=allow_isomorphisms,
                                                     mutation_rate=mutation_rate
                                                 )
        candiate_edge_list = []
        candiate_node_list = []
        for cand in candidates:
            edge_index, node_f = search_space.nasbench2graph2((cand[1], cand[2]))
            candiate_edge_list.append(edge_index)
            candiate_node_list.append(node_f)
        agent.fit(arch_data_edge_idx_list, arch_data_node_f_list, val_accuracy, logger=None)
        acc_train = agent.pred(arch_data_edge_idx_list, arch_data_node_f_list)
        acc_pred = agent.pred(candiate_edge_list, candiate_node_list)
        candidate_np = acc_pred.cpu().numpy()
        if record_kt == 'T':
            candidates_gt = [can[4] for can in candidates]
            kt = get_kendalltau_coorlection(candidate_np.tolist(), candidates_gt)[0]
            kt_list.append(kt)
        sorted_indices = np.argsort(candidate_np)
        kt_top_pred_list = []
        kt_top_gt_list = []
        for i in sorted_indices[:k]:
            archtuple = candidates[i]
            data.append(archtuple)
            kt_top_pred_list.append(candidate_np[i])
            kt_top_gt_list.append(archtuple[4])
        kt_top_list.append(get_kendalltau_coorlection(kt_top_pred_list, kt_top_gt_list)[0])
        if verbose:
            top_5_loss = sorted([d[4] for d in data])[:min(5, len(data))]
            logger.info('Query {}, training mean loss is  {}'.format(query,
                                                                     np.mean(np.abs(acc_train.cpu().numpy()-val_accuracy))))
            logger.info('Query {}, top 5 val losses {}'.format(query, top_5_loss))
        query += k
    return data, {'type': 'gin_predictor', 'final_data': data, 'kt_top_list': kt_top_list,
                  'kt_list': kt_list, 'mutate_list': mutate_list}