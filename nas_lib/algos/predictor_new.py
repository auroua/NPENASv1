import numpy as np
import torch
import copy
from ..utils.utils_data import nasbench2graph, nasbench2graph2
from .acquisition_functions import acq_fn
from ..eigen.trainer_predictor import NasBenchGinPredictorTrainer
from ..eigen.trainer_uncertainty_predictor import NasBenchGinGaussianTrainer
from nas_lib.utils.corr import get_kendalltau_coorlection


def gin_predictor_new_nasbench_101(search_space,
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
                                   record_mutation='F'
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
        agent = NasBenchGinPredictorTrainer(search_agent, lr=lr, device=device, epochs=epochs,
                                            train_images=len(data), batch_size=batch_size)
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
        candidates_gt = [can[4] for can in candidates]
        if query == 20:
            candidate_np = np.array(candidates_gt)
        if record_kt == 'T':
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
    return data, {'type': 'gin_predictor_new', 'final_data': data,
                  'kt_list': kt_list, 'kt_top_list': kt_top_list,
                  'mutate_list': mutate_list}


def gin_predictor_new_nasbench_201(search_space,
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
                                   record_mutation='F'
                                   ):
    """
    Bayesian optimization with a neural network model
    """
    if dataname == 'cifar10-valid':
        rate = 100.
    elif dataname == 'cifar100':
        rate = 100.
    elif dataname == 'ImageNet16-120':
        rate = 100.
    else:
        raise NotImplementedError()
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
        candidates_gt = [can[4] for can in candidates]
        if query == 20:
            candidate_np = np.array(candidates_gt)
        if record_kt == 'T':
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
    return data, {'type': 'gin_predictor_new', 'final_data': data,
                  'kt_list': kt_list, 'kt_top_list': kt_top_list, 'mutate_list': mutate_list}


def gin_predictor_new_nasbench_nlp(search_space,
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
        candidates_gt = [can[4] for can in candidates]
        if query == 20:
            candidate_np = np.array(candidates_gt)
        if record_kt == 'T':
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
    return data, {'type': 'gin_predictor_new', 'final_data': data, 'kt_top_list': kt_top_list,
                  'kt_list': kt_list, 'mutate_list': mutate_list}


def gin_predictor_new_nasbench_asr(search_space,
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
        candidates_gt = [can[4] for can in candidates]
        if query == 20:
            candidate_np = np.array(candidates_gt)
        if record_kt == 'T':
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
    return data, {'type': 'gin_predictor_new', 'final_data': data, 'kt_top_list': kt_top_list,
                  'kt_list': kt_list, 'mutate_list': mutate_list}


def gin_predictor_new_2_nasbench_201(search_space,
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
                                     record_mutation='F'
                                     ):
    """
    Bayesian optimization with a neural network model
    """
    if dataname == 'cifar10-valid':
        rate = 100.
    elif dataname == 'cifar100':
        rate = 100.
    elif dataname == 'ImageNet16-120':
        rate = 100.
    else:
        raise NotImplementedError()
    kt_list = []
    mutate_list = []
    device = torch.device('cuda:%d' % gpu)
    data = search_space.generate_random_dataset(num=90,
                                                allow_isomorphisms=allow_isomorphisms,
                                                deterministic_loss=True)
    query = num_init + k
    search_agent = agent
    if len(data) <= 10:
        batch_size = 10
    else:
        batch_size = 16
    agent = NasBenchGinPredictorTrainer(search_agent, lr=lr, device=device, epochs=epochs,
                                        train_images=len(data), batch_size=batch_size, input_dim=8, rate=rate)
    while query <= total_queries:
        arch_data = [d[0] for d in data]
        val_accuracy = np.array([d[4] for d in data])
        arch_data_edge_idx_list = []
        arch_data_node_f_list = []
        for arch in arch_data:
            edge_index, node_f = search_space.nasbench2graph2(arch)
            arch_data_edge_idx_list.append(edge_index)
            arch_data_node_f_list.append(node_f)
        if query == 20:
            agent.fit(arch_data_edge_idx_list, arch_data_node_f_list, val_accuracy, logger=None)
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
        acc_train = agent.pred(arch_data_edge_idx_list, arch_data_node_f_list)
        acc_pred = agent.pred(candiate_edge_list, candiate_node_list)
        candidate_np = acc_pred.cpu().numpy()
        candidates_gt = [can[4] for can in candidates]
        if record_kt == 'T':
            kt = get_kendalltau_coorlection(candidate_np.tolist(), candidates_gt)[0]
            kt_list.append(kt)
        sorted_indices = np.argsort(candidate_np)
        for i in sorted_indices[:k]:
            archtuple = candidates[i]
            data.append(archtuple)
        if verbose:
            top_5_loss = sorted([d[4] for d in data])[:min(5, len(data))]
            logger.info('Query {}, training mean loss is  {}'.format(query,
                                                                     np.mean(np.abs(acc_train.cpu().numpy()-val_accuracy))))
            logger.info('Query {}, top 5 val losses {}'.format(query, top_5_loss))
        query += k
    return data, {'type': 'gin_predictor_new_2', 'final_data': data,
                  'kt_list': kt_list, 'mutate_list': mutate_list}