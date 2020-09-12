import sys


def meta_neuralnet_params(param_str):
    if param_str == 'nasbench_case1':
        params = {'search_space': 'nasbench_case1', 'loss': 'mae', 'num_layers': 10, 'layer_width': 20,
                  'epochs': 150, 'batch_size': 32, 'lr': .01, 'regularization': 0, 'verbose': 0}
    elif param_str == 'nasbench_case2':
        params = {'search_space': 'nasbench_case2', 'loss': 'mae', 'num_layers': 10, 'layer_width': 20,
                  'epochs': 150, 'batch_size': 32, 'lr': .01, 'regularization': 0, 'verbose': 0}
    elif param_str == 'nasbench_201':
        params = {'search_space': 'nasbench_201', 'loss': 'mape', 'num_layers': 10, 'layer_width': 200,
                  'epochs': 200, 'batch_size': 32, 'lr': .001, 'regularization': 0, 'verbose': 0}
    else:
        print('invalid meta neural net params')
        sys.exit()
    return params


def algo_params_close_domain(param_str, search_budget=100):
    """
      Return params list based on param_str.
      These are the parameters used to produce the figures in the paper
      For AlphaX and Reinforcement Learning, we used the corresponding github repos:
      https://github.com/linnanwang/AlphaX-NASBench101
      https://github.com/automl/nas_benchmarks
    """
    params = []

    if param_str == 'nasbench101_case1':
        params.append({'algo_name': 'random', 'total_queries': search_budget})
        params.append({'algo_name': 'evolution', 'total_queries': search_budget, 'population_size': 30, 'num_init': 10,
                       'k': 10, 'tournament_size': 10, 'mutation_rate': 1.0})
        params.append({'algo_name': 'bananas', 'total_queries': search_budget, 'num_ensemble': 5,
                       'allow_isomorphisms': False, 'acq_opt_type': 'mutation', 'candidate_nums': 100})
        params.append({'algo_name': 'bananas_f', 'total_queries': search_budget, 'num_ensemble': 5,
                       'encode_paths': False, 'allow_isomorphisms': False, 'acq_opt_type': 'mutation',
                       'candidate_nums': 100})
        params.append({'algo_name': 'gin_uncertainty_predictor', 'total_queries': search_budget,
                       'agent': 'gin_gaussian', 'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005,
                       'candidate_nums': 100, 'epochs': 1000})
        params.append({'algo_name': 'gin_predictor', 'total_queries': search_budget, 'agent': 'gin_predictor',
                       'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005, 'candidate_nums': 100, 'epochs': 300})
    elif param_str == 'nasbench101_case2':
        params.append({'algo_name': 'random', 'total_queries': search_budget})
        params.append({'algo_name': 'evolution', 'total_queries': search_budget, 'population_size': 30, 'num_init': 10, 'k': 10,
                       'tournament_size': 10, 'mutation_rate': 1.0})
        params.append({'algo_name': 'bananas', 'total_queries': search_budget, 'num_ensemble': 5, 'allow_isomorphisms': False,
                       'acq_opt_type': 'mutation', 'candidate_nums': 100, 'num_init': 10, 'k': 10,
                       'encode_paths': True})
        params.append({'algo_name': 'bananas_f', 'total_queries': search_budget, 'num_ensemble': 5,
                       'allow_isomorphisms': False, 'acq_opt_type': 'mutation', 'candidate_nums': 100, 'num_init': 10,
                       'k': 10, 'encode_paths': False})
        params.append({'algo_name': 'gin_uncertainty_predictor', 'total_queries': search_budget,
                       'agent': 'gin_gaussian', 'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005,
                       'candidate_nums': 100, 'epochs': 1000})
        params.append({'algo_name': 'gin_predictor', 'total_queries': search_budget, 'agent': 'gin_predictor',
                       'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005, 'candidate_nums': 100, 'epochs': 300})
    elif param_str == 'nasbench_201':
        params.append({'algo_name': 'random', 'total_queries': search_budget})
        params.append({'algo_name': 'evolution', 'total_queries': search_budget, 'population_size': 30, 'num_init': 10,
                       'k': 10, 'tournament_size': 10, 'mutation_rate': 1.0, 'allow_isomorphisms': False,
                       'deterministic': True})
        params.append({'algo_name': 'bananas', 'total_queries': search_budget, 'num_ensemble': 5,
                       'allow_isomorphisms': False, 'acq_opt_type': 'mutation', 'candidate_nums': 100, 'num_init': 10,
                       'k': 10, 'encode_paths': True, 'eva_new': False})
        params.append({'algo_name': 'bananas_f', 'total_queries': search_budget, 'num_ensemble': 5,
                       'allow_isomorphisms': False, 'acq_opt_type': 'mutation', 'candidate_nums': 100, 'num_init': 10,
                       'k': 10, 'encode_paths': False})
        params.append({'algo_name': 'gin_uncertainty_predictor', 'total_queries': search_budget,
                       'agent': 'gin_gaussian', 'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005,
                       'candidate_nums': 100, 'epochs': 1000})
        params.append({'algo_name': 'gin_predictor', 'total_queries': search_budget, 'agent': 'gin_predictor',
                       'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005, 'candidate_nums': 100, 'epochs': 300})
    elif param_str == 'scalar_prior':
        params.append({'algo_name': 'gin_predictor_scalar_10', 'total_queries': search_budget, 'agent': 'gin_predictor',
                       'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005, 'candidate_nums': 100, 'epochs': 300,
                       'scalar': 10})
        params.append({'algo_name': 'gin_predictor_scalar_30', 'total_queries': search_budget, 'agent': 'gin_predictor',
                       'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005, 'candidate_nums': 100, 'epochs': 300,
                       'scalar': 30})
        params.append({'algo_name': 'gin_predictor_scalar_50', 'total_queries': search_budget, 'agent': 'gin_predictor',
                       'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005, 'candidate_nums': 100, 'epochs': 300,
                       'scalar': 50})
        params.append({'algo_name': 'gin_predictor_scalar_70', 'total_queries': search_budget, 'agent': 'gin_predictor',
                       'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005, 'candidate_nums': 100, 'epochs': 300,
                       'scalar': 70})
        params.append({'algo_name': 'gin_predictor_scalar_100', 'total_queries': search_budget, 'agent': 'gin_predictor',
                       'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005, 'candidate_nums': 100, 'epochs': 300,
                       'scalar': 100})
    elif param_str == 'evaluation_compare':
        params.append({'algo_name': 'evolutionary_case1', 'total_queries': search_budget, 'population_size': 150,
                       'num_init': 10, 'k': 10, 'tournament_size': 2, 'mutation_rate': 1.0, 'candidate_num': 100})
        params.append({'algo_name': 'evolutionary_case2', 'total_queries': search_budget, 'population_size': 150,
                       'num_init': 10, 'k': 10, 'tournament_size': 2, 'mutation_rate': 1.0, 'candidate_num': 100,
                       'mutation_num': 10})
        params.append({'algo_name': 'evolutionary_case2_20', 'total_queries': search_budget, 'population_size': 150,
                       'num_init': 10, 'k': 10, 'tournament_size': 2, 'mutation_rate': 1.0, 'candidate_num': 100,
                       'mutation_num': 20})
        params.append({'algo_name': 'evolutionary_case2_30', 'total_queries': search_budget, 'population_size': 150,
                       'num_init': 10, 'k': 10, 'tournament_size': 2, 'mutation_rate': 1.0, 'candidate_num': 100,
                       'mutation_num': 30})
    elif param_str == 'box_compare_case1':
        params.append({'algo_name': 'bananas', 'total_queries': search_budget, 'num_ensemble': 5,
                       'allow_isomorphisms': False, 'acq_opt_type': 'mutation', 'candidate_nums': 100})
        params.append({'algo_name': 'bananas_f', 'total_queries': search_budget, 'num_ensemble': 5,
                       'encode_paths': False, 'allow_isomorphisms': False, 'acq_opt_type': 'mutation',
                       'candidate_nums': 100})
        params.append({'algo_name': 'gin_uncertainty_predictor', 'total_queries': search_budget,
                       'agent': 'gin_gaussian', 'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005,
                       'candidate_nums': 100, 'epochs': 1000})
        params.append({'algo_name': 'gin_predictor', 'total_queries': search_budget, 'agent': 'gin_predictor',
                       'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005, 'candidate_nums': 100, 'epochs': 300})
    elif param_str == 'box_compare_case2':
        params.append({'algo_name': 'bananas', 'total_queries': search_budget, 'num_ensemble': 5, 'allow_isomorphisms': False,
                       'acq_opt_type': 'mutation', 'candidate_nums': 100, 'num_init': 10, 'k': 10,
                       'encode_paths': True})
        params.append({'algo_name': 'bananas_f', 'total_queries': search_budget, 'num_ensemble': 5,
                       'allow_isomorphisms': False, 'acq_opt_type': 'mutation', 'candidate_nums': 100, 'num_init': 10,
                       'k': 10, 'encode_paths': False})
        params.append({'algo_name': 'gin_uncertainty_predictor', 'total_queries': search_budget,
                       'agent': 'gin_gaussian', 'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005,
                       'candidate_nums': 100, 'epochs': 1000})
        params.append({'algo_name': 'gin_predictor', 'total_queries': search_budget, 'agent': 'gin_predictor',
                       'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005, 'candidate_nums': 100, 'epochs': 300})
    elif param_str == 'experiment':
        pass
    else:
        print('invalid algorithm params')
        sys.exit()

    print('* Running experiment: ' + param_str)
    return params


def algo_params_close_domain_diff_training_nums(param_str, search_stratege, search_budget=100):
    params = []
    params.append([10, 20, 40, 60, 80, 100, 120, 140])
    if param_str == 'compare_diff_training_architectures_case1':
        if search_stratege == 'gin_predictor':
            params.append({'algo_name': 'gin_predictor', 'total_queries': search_budget, 'agent': 'gin_predictor',
                           'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005, 'candidate_nums': 100, 'epochs': 300})
        elif search_stratege == 'gin_uncertainty_predictor':
            params.append({'algo_name': 'gin_uncertainty_predictor', 'total_queries': search_budget,
                           'agent': 'gin_gaussian', 'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005,
                           'candidate_nums': 100, 'epochs': 1000})
        elif search_stratege == 'bananas':
            params.append({'algo_name': 'bananas', 'total_queries': search_budget, 'num_ensemble': 5,
                           'allow_isomorphisms': False, 'acq_opt_type': 'mutation', 'candidate_nums': 100})
        elif search_stratege == 'bananas_f':
            params.append({'algo_name': 'bananas_f', 'total_queries': search_budget, 'num_ensemble': 5,
                           'encode_paths': False, 'allow_isomorphisms': False, 'acq_opt_type': 'mutation',
                           'candidate_nums': 100})
        else:
            raise NotImplementedError()
    elif param_str == 'compare_diff_training_architectures_case2':
        if search_stratege == 'gin_predictor':
            params.append({'algo_name': 'gin_predictor', 'total_queries': search_budget, 'agent': 'gin_predictor',
                           'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005, 'candidate_nums': 100,
                           'epochs': 300})
        elif search_stratege == 'gin_uncertainty_predictor':
            params.append({'algo_name': 'gin_uncertainty_predictor', 'total_queries': search_budget,
                           'agent': 'gin_gaussian', 'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005,
                           'candidate_nums': 100, 'epochs': 1000})
        elif search_stratege == 'bananas':
            params.append(
                {'algo_name': 'bananas', 'total_queries': search_budget, 'num_ensemble': 5, 'allow_isomorphisms': False,
                 'acq_opt_type': 'mutation', 'candidate_nums': 100, 'num_init': 10, 'k': 10,
                 'encode_paths': True})
        elif search_stratege == 'bananas_f':
            params.append({'algo_name': 'bananas_f', 'total_queries': search_budget, 'num_ensemble': 5,
                           'allow_isomorphisms': False, 'acq_opt_type': 'mutation', 'candidate_nums': 100,
                           'num_init': 10,
                           'k': 10, 'encode_paths': False})
        else:
            raise NotImplementedError()
    else:
        print('invalid algorithm params')
        sys.exit()

    print('* Running experiment: ' + param_str)
    return params


def algo_params_open_domain(param_str):
    if param_str == 'gin_uncertainty_predictor':
        param = {'algo_name': 'gin_uncertainty_predictor', 'total_queries': 150, 'agent': 'gin_gaussian', 'num_init': 10,
                 'mutation_rate': 1.0, 'k': 10, 'epochs': 1000, 'batch_size': 32, 'lr': 0.005, 'encode_path': True,
                 'allow_isomorphisms': False,  'candidate_nums': 100}
    elif param_str == 'gin_predictor':
        param = {'algo_name': 'gin_predictor', 'total_queries': 100, 'agent': 'gin_predictor', 'num_init': 10,
                 'allow_isomorphisms': False, 'k': 10, 'epochs': 300, 'batch_size': 32, 'lr': 0.005, 'encode_path': True,
                 'candidate_nums': 100}
    else:
        raise NotImplementedError("This algorithm have not implement!")
    print('* Running experiment: ' + str(param))
    return param


cifar10_path = '/home/albert_wei/Disk_A/dataset_train/cifar10/'
tf_records_path = '/home/albert_wei/Disk_A/dataset_train/nas_bench101/nasbench_only108.tfrecord'
nas_bench_201_path = '/home/albert_wei/Disk_A/dataset_train/nas_bench_201/NAS-Bench-201-v1_1-096897.pth'
nas_bench_201_converted_path = '/home/albert_wei/Disk_A/dataset_train/nas_bench_201/arch_info.pkl'