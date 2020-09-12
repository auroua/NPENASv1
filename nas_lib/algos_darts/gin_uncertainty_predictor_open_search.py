from nas_lib.utils.utils_darts import init_nasbench_macro_cifar10
from hashlib import sha256
from nas_lib.eigen.trainer_nasbench_open_darts_async import async_macro_model_train
from nas_lib.utils.utils_darts import convert_to_genotype
from nas_lib.models_darts.darts_graph import nasbench2graph2
import numpy as np
from nas_lib.eigen.trainer_uncertainty_predictor import NasBenchGinGaussianTrainer
from nas_lib.algos.acquisition_functions import acq_fn
import torch
import random
import torch.backends.cudnn as cudnn


def gin_uncertainty_predictor_search_open(search_space,
                                          algo_info,
                                          logger,
                                          gpus,
                                          save_dir,
                                          verbose=True,
                                          dataset='cifar10',
                                          seed=111222333):
    """
    regularized evolution
    """
    total_queries = algo_info['total_queries']
    num_init = algo_info['num_init']
    k_num = algo_info['k']
    epochs = algo_info['epochs']
    batch_size = algo_info['batch_size']
    lr = algo_info['lr']
    encode_path = algo_info['encode_path']
    candidate_nums = algo_info['candidate_nums']
    macro_graph_dict = {}
    model_keys = []
    init_nasbench_macro_cifar10(save_dir)

    # set seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.enabled = True
    cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

    data_dict = search_space.generate_random_dataset(num=num_init,
                                                     encode_paths=encode_path)
    data_dict_keys = [convert_to_genotype(d[0], verbose=False) for d in data_dict]
    data_dict_keys = [sha256(str(k).encode('utf-8')).hexdigest() for k in data_dict_keys]
    model_keys.extend(data_dict_keys)
    for i, d in enumerate(data_dict):
        macro_graph_dict[data_dict_keys[i]] = list(d)
    darts_neural_dict = search_space.assemble_cifar10_neural_net(data_dict)
    data = async_macro_model_train(model_data=darts_neural_dict,
                                   gpus=gpus,
                                   save_dir=save_dir,
                                   dataset=dataset)
    for k, v in data.items():
        if k not in macro_graph_dict:
            raise ValueError('model trained acc key should in macro_graph_dict')
        macro_graph_dict[k].extend(v)
    query = num_init + len(data_dict_keys)
    while query <= total_queries:
        train_data = search_space.assemble_graph(macro_graph_dict, model_keys)
        val_losses = np.array([macro_graph_dict[k][2] for k in model_keys])

        arch_data_edge_idx_list = []
        arch_data_node_f_list = []
        for arch in train_data:
            edge_index, node_f = nasbench2graph2(arch)
            arch_data_edge_idx_list.append(edge_index)
            arch_data_node_f_list.append(node_f)

        candidate_graph_dict = {}
        candidates = search_space.get_candidates(macro_graph_dict,
                                                 model_keys,
                                                 num=candidate_nums,
                                                 encode_paths=encode_path,
                                                 )
        candidate_dict_keys = [convert_to_genotype(d[0], verbose=False) for d in candidates]
        candidate_dict_keys = [sha256(str(k).encode('utf-8')).hexdigest() for k in candidate_dict_keys]
        for i, d in enumerate(candidates):
            candidate_graph_dict[candidate_dict_keys[i]] = list(d)
        xcandidates = search_space.assemble_graph(candidate_graph_dict, candidate_dict_keys)
        candiate_edge_list = []
        candiate_node_list = []
        for cand in xcandidates:
            edge_index, node_f = nasbench2graph2(cand)
            candiate_edge_list.append(edge_index)
            candiate_node_list.append(node_f)
        meta_neuralnet = NasBenchGinGaussianTrainer(lr=lr, epochs=epochs, train_images=len(arch_data_edge_idx_list),
                                                    batch_size=batch_size, input_dim=11, agent_type='gin_gaussian')
        meta_neuralnet.fit(arch_data_edge_idx_list, arch_data_node_f_list, val_losses, logger=logger)
        pred_train, mean_train, _ = meta_neuralnet.pred(arch_data_edge_idx_list, arch_data_node_f_list)
        predictions, _, _ = meta_neuralnet.pred(candiate_edge_list, candiate_node_list)

        sorted_indices = acq_fn(predictions, 'its_vae')
        temp_candidate_train_arch = []
        for j in sorted_indices[:k_num]:
            model_keys.append(candidate_dict_keys[j])
            macro_graph_dict[candidate_dict_keys[j]] = candidate_graph_dict[candidate_dict_keys[j]]
            temp_candidate_train_arch.append(candidate_graph_dict[candidate_dict_keys[j]])
        darts_candidate_neural_dict = search_space.assemble_cifar10_neural_net(temp_candidate_train_arch)
        darts_candidate_acc = async_macro_model_train(model_data=darts_candidate_neural_dict,
                                                      gpus=gpus,
                                                      save_dir=save_dir,
                                                      dataset=dataset)
        for k, v in darts_candidate_acc.items():
            if k not in macro_graph_dict:
                raise ValueError('model trained acc key should in macro_graph_dict')
            macro_graph_dict[k].extend(v)
        train_error = np.mean(np.abs(mean_train.cpu().numpy()-val_losses))
        val_losses = np.array([macro_graph_dict[k][2] for k in model_keys])
        if verbose:
            top_5_loss = sorted(val_losses)[:min(5, len(val_losses))]
            logger.info('Query {}, training mean loss is  {}'.format(query, train_error))
            logger.info('Query {}, top 5 val losses {}'.format(query, top_5_loss))
        query += len(temp_candidate_train_arch)
    return macro_graph_dict, model_keys