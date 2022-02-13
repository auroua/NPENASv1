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
    elif param_str == 'nasbench_nlp':
        params = {'search_space': 'nasbench_nlp', 'loss': 'mape', 'num_layers': 10, 'layer_width': 200,
                  'epochs': 200, 'batch_size': 32, 'lr': .001, 'regularization': 0, 'verbose': 0}
    elif param_str == 'nasbench_asr':
        params = {'search_space': 'nasbench_asr', 'loss': 'mape', 'num_layers': 10, 'layer_width': 200,
                  'epochs': 200, 'batch_size': 32, 'lr': .001, 'regularization': 0, 'verbose': 0}
    else:
        print('invalid meta neural net params')
        sys.exit()
    return params


def algo_params_close_domain(param_str, search_budget=100, comparison_type="algorithms",
                             nasbench_201_dataset="cifar100",
                             relu_celu_comparison_algo_type="NPENAS_BO"):
    """
      Return params list based on param_str.
      These are the parameters used to produce the figures in the paper
      For AlphaX and Reinforcement Learning, we used the corresponding github repos:
      https://github.com/linnanwang/AlphaX-NASBench101
      https://github.com/automl/nas_benchmarks
    """
    params = []
    if comparison_type == "relu_celu":
        if param_str == 'nasbench101_case2' and relu_celu_comparison_algo_type == "NPENAS_NP":
            params.append({'algo_name': 'gin_predictor', 'total_queries': search_budget, 'agent': 'gin_predictor',
                           'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005, 'candidate_nums': 100, 'epochs': 300,
                           'rate': 10., 'activation_fn': 'relu'})
            params.append({'algo_name': 'gin_predictor', 'total_queries': search_budget, 'agent': 'gin_predictor',
                           'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005, 'candidate_nums': 100, 'epochs': 300,
                           'rate': 10., 'activation_fn': 'celu'})
            params.append({'algo_name': 'oracle', 'total_queries': search_budget, 'allow_isomorphisms': False,
                           'candidate_nums': 100, 'num_init': 10, 'k': 10, 'mutation_rate': -1})
        elif param_str == 'nasbench101_case2' and relu_celu_comparison_algo_type == "NPENAS_BO":
            params.append({'algo_name': 'gin_uncertainty_predictor', 'total_queries': search_budget,
                           'agent': 'gin_gaussian', 'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005,
                           'candidate_nums': 100, 'epochs': 1000, 'activation_fn': 'relu'})
            params.append({'algo_name': 'gin_uncertainty_predictor', 'total_queries': search_budget,
                           'agent': 'gin_gaussian', 'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005,
                           'candidate_nums': 100, 'epochs': 1000, 'activation_fn': 'celu'})
            params.append({'algo_name': 'oracle', 'total_queries': search_budget, 'allow_isomorphisms': False,
                           'candidate_nums': 100, 'num_init': 10, 'k': 10, 'mutation_rate': -1})
        elif param_str == 'nasbench_201':
            params.append({'algo_name': 'gin_uncertainty_predictor', 'total_queries': search_budget,
                           'agent': 'gin_gaussian', 'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005,
                           'candidate_nums': 100, 'epochs': 1000, 'activation_fn': 'relu'})
            params.append({'algo_name': 'gin_uncertainty_predictor', 'total_queries': search_budget,
                           'agent': 'gin_gaussian', 'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005,
                           'candidate_nums': 100, 'epochs': 1000, 'activation_fn': 'celu'})
            params.append({'algo_name': 'oracle', 'total_queries': search_budget, 'allow_isomorphisms': False,
                           'candidate_nums': 100, 'num_init': 10, 'k': 10, 'mutation_rate': -1})
        elif param_str == 'nasbench_nlp':
            params.append({'algo_name': 'gin_uncertainty_predictor', 'total_queries': search_budget,
                           'agent': 'gin_gaussian', 'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005,
                           'candidate_nums': 100, 'epochs': 1000, 'mutation_rate': 0.3, 'activation_fn': 'relu'})
            params.append({'algo_name': 'gin_uncertainty_predictor', 'total_queries': search_budget,
                           'agent': 'gin_gaussian', 'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005,
                           'candidate_nums': 100, 'epochs': 1000, 'mutation_rate': 0.3, 'activation_fn': 'celu'})
            params.append({'algo_name': 'oracle', 'total_queries': search_budget, 'allow_isomorphisms': False,
                           'candidate_nums': 100, 'num_init': 10, 'k': 10, 'mutation_rate': 0.3})
        elif param_str == 'nasbench_asr':
            params.append({'algo_name': 'gin_uncertainty_predictor', 'total_queries': search_budget,
                           'agent': 'gin_gaussian', 'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005,
                           'candidate_nums': 100, 'epochs': 1000, 'mutation_rate': 1.0, 'activation_fn': 'relu'})
            params.append({'algo_name': 'gin_uncertainty_predictor', 'total_queries': search_budget,
                           'agent': 'gin_gaussian', 'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005,
                           'candidate_nums': 100, 'epochs': 1000, 'mutation_rate': 1.0, 'activation_fn': 'celu'})
            params.append({'algo_name': 'oracle', 'total_queries': search_budget, 'allow_isomorphisms': False,
                           'candidate_nums': 100, 'num_init': 10, 'k': 10, 'mutation_rate': 1.0})
    elif comparison_type == "scalar_compare":
        if param_str == 'nasbench101_case2':
            params.append({'algo_name': 'gin_predictor', 'total_queries': search_budget, 'agent': 'gin_predictor',
                           'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005, 'candidate_nums': 100,
                           'epochs': 300, 'rate': 10.})
            params.append({'algo_name': 'gin_predictor', 'total_queries': search_budget, 'agent': 'gin_predictor',
                           'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005, 'candidate_nums': 100,
                           'epochs': 300, 'rate': 100.})
            params.append({'algo_name': 'oracle', 'total_queries': search_budget, 'allow_isomorphisms': False,
                           'candidate_nums': 100, 'num_init': 10, 'k': 10, 'mutation_rate': -1})
        elif param_str == 'nasbench_201':
            if nasbench_201_dataset == "cifar10-valid":
                nasbench_201_rate1 = 7.
            elif nasbench_201_dataset == "cifar100":
                nasbench_201_rate1 = 30.
            elif nasbench_201_dataset == "ImageNet16-120":
                nasbench_201_rate1 = 55.
            else:
                raise ValueError(f"nasbench 201 dataset type {nasbench_201_dataset} does not support at present!")

            params.append({'algo_name': 'gin_predictor', 'total_queries': search_budget, 'agent': 'gin_predictor',
                           'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005, 'candidate_nums': 100, 'epochs': 300,
                           'rate': nasbench_201_rate1})
            params.append({'algo_name': 'gin_predictor', 'total_queries': search_budget, 'agent': 'gin_predictor',
                           'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005, 'candidate_nums': 100,
                           'epochs': 300, 'rate': 100.})
            params.append({'algo_name': 'oracle', 'total_queries': search_budget, 'allow_isomorphisms': False,
                           'candidate_nums': 100, 'num_init': 10, 'k': 10, 'mutation_rate': -1})
        elif param_str == 'nasbench_nlp':
            params.append({'algo_name': 'gin_predictor', 'total_queries': search_budget, 'agent': 'gin_predictor',
                           'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005, 'candidate_nums': 100,
                           'epochs': 300, 'mutation_rate': 0.3, 'rate': 5.})
            params.append({'algo_name': 'gin_predictor', 'total_queries': search_budget, 'agent': 'gin_predictor',
                           'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005, 'candidate_nums': 100, 'epochs': 300,
                           'mutation_rate': 0.3, 'rate': 100.})
            params.append({'algo_name': 'oracle', 'total_queries': search_budget, 'allow_isomorphisms': False,
                           'candidate_nums': 100, 'num_init': 10, 'k': 10, 'mutation_rate': 0.3})
        elif param_str == 'nasbench_asr':
            params.append({'algo_name': 'gin_predictor', 'total_queries': search_budget, 'agent': 'gin_predictor',
                           'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005, 'candidate_nums': 100,
                           'epochs': 300, 'mutation_rate': 1.0, 'rate': 25.})
            params.append({'algo_name': 'gin_predictor_new', 'total_queries': search_budget, 'agent': 'gin_predictor',
                           'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005, 'candidate_nums': 100, 'epochs': 300,
                           'mutation_rate': 1.0, 'rate': 100.})
            params.append({'algo_name': 'oracle', 'total_queries': search_budget, 'allow_isomorphisms': False,
                           'candidate_nums': 100, 'num_init': 10, 'k': 10, 'mutation_rate': 1.0})
    elif comparison_type == "algorithm":
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
            params.append({'algo_name': 'oracle', 'total_queries': search_budget, 'allow_isomorphisms': False,
                           'candidate_nums': 100, 'num_init': 10, 'k': 10, 'mutation_rate': -1})
            params.append({'algo_name': 'gp_bayesopt', 'total_queries': search_budget, 'num_init': 10, 'k': 10,
                           'distance': 'adj', 'verbose': 1})
            params.append({'algo_name': 'gp_bayesopt', 'total_queries': search_budget, 'num_init': 10, 'k': 10,
                           'distance': 'nasbot', 'verbose': 1})
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
            # params.append({'algo_name': 'gin_uncertainty_predictor', 'total_queries': search_budget,
            #                'agent': 'gin_gaussian', 'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005,
            #                'candidate_nums': 100, 'epochs': 1000, 'activation_fn': 'relu'})
            params.append({'algo_name': 'gin_uncertainty_predictor', 'total_queries': search_budget,
                           'agent': 'gin_gaussian', 'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005,
                           'candidate_nums': 100, 'epochs': 1000, 'activation_fn': 'celu'})
            params.append({'algo_name': 'gin_predictor', 'total_queries': search_budget, 'agent': 'gin_predictor',
                           'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005, 'candidate_nums': 100, 'epochs': 300,
                           'rate': 10.})
            # params.append({'algo_name': 'gin_predictor', 'total_queries': search_budget, 'agent': 'gin_predictor',
            #                'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005, 'candidate_nums': 100, 'epochs': 300,
            #                'rate': 100., 'activation_fn': 'relu'})
            # params.append({'algo_name': 'gin_predictor', 'total_queries': search_budget, 'agent': 'gin_predictor',
            #                'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005, 'candidate_nums': 100, 'epochs': 300,
            #                'rate': 100., 'activation_fn': 'celu'})
            # params.append({'algo_name': 'gin_predictor_new', 'total_queries': search_budget, 'agent': 'gin_predictor',
            #                'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005, 'candidate_nums': 100, 'epochs': 300})
            params.append({'algo_name': 'oracle', 'total_queries': search_budget, 'allow_isomorphisms': False,
                           'candidate_nums': 100, 'num_init': 10, 'k': 10, 'mutation_rate': -1})
            params.append({'algo_name': 'gp_bayesopt', 'total_queries': search_budget, 'num_init': 10, 'k': 10,
                           'distance': 'adj', 'verbose': 1})
            params.append({'algo_name': 'gp_bayesopt', 'total_queries': search_budget, 'num_init': 10, 'k': 10,
                           'distance': 'nasbot', 'verbose': 1})
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
            # params.append({'algo_name': 'gin_uncertainty_predictor', 'total_queries': search_budget,
            #                'agent': 'gin_gaussian', 'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005,
            #                'candidate_nums': 100, 'epochs': 1000, 'activation_fn': 'relu'})
            params.append({'algo_name': 'gin_uncertainty_predictor', 'total_queries': search_budget,
                           'agent': 'gin_gaussian', 'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005,
                           'candidate_nums': 100, 'epochs': 1000, 'activation_fn': 'celu'})
            # params.append({'algo_name': 'gin_predictor', 'total_queries': search_budget, 'agent': 'gin_predictor',
            #                'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005, 'candidate_nums': 100, 'epochs': 300,
            #                'rate': 7.})
            params.append({'algo_name': 'gin_predictor', 'total_queries': search_budget, 'agent': 'gin_predictor',
                           'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005, 'candidate_nums': 100, 'epochs': 300,
                           'rate': 100.})
            # params.append({'algo_name': 'gin_predictor_new', 'total_queries': search_budget, 'agent': 'gin_predictor',
            #                'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005, 'candidate_nums': 100, 'epochs': 300})
            params.append({'algo_name': 'oracle', 'total_queries': search_budget, 'allow_isomorphisms': False,
                           'candidate_nums': 100, 'num_init': 10, 'k': 10, 'mutation_rate': -1})
            params.append({'algo_name': 'gp_bayesopt', 'total_queries': search_budget, 'num_init': 10, 'k': 10,
                           'distance': 'adj', 'verbose': 0})
            params.append({'algo_name': 'gp_bayesopt', 'total_queries': search_budget, 'num_init': 10, 'k': 10,
                           'distance': 'nasbot', 'verbose': 0})
        elif param_str == 'nasbench_nlp':
            params.append({'algo_name': 'random', 'total_queries': search_budget})
            params.append({'algo_name': 'evolution', 'total_queries': search_budget, 'population_size': 30, 'num_init': 10,
                           'k': 10, 'tournament_size': 10, 'mutation_rate': 0.3, 'allow_isomorphisms': False,
                           'deterministic': True})
            params.append({'algo_name': 'bananas', 'total_queries': search_budget, 'num_ensemble': 5,
                           'allow_isomorphisms': False, 'acq_opt_type': 'mutation', 'candidate_nums': 100, 'num_init': 10,
                           'k': 10, 'encode_paths': True, 'eva_new': False, 'mutation_rate': 0.3})
            params.append({'algo_name': 'bananas_f', 'total_queries': search_budget, 'num_ensemble': 5,
                           'allow_isomorphisms': False, 'acq_opt_type': 'mutation', 'candidate_nums': 100, 'num_init': 10,
                           'k': 10, 'encode_paths': False, 'mutation_rate': 0.3})
            # params.append({'algo_name': 'gin_uncertainty_predictor', 'total_queries': search_budget,
            #                'agent': 'gin_gaussian', 'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005,
            #                'candidate_nums': 100, 'epochs': 1000, 'mutation_rate': 0.3, 'activation_fn': 'relu'})
            params.append({'algo_name': 'gin_uncertainty_predictor', 'total_queries': search_budget,
                           'agent': 'gin_gaussian', 'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005,
                           'candidate_nums': 100, 'epochs': 1000, 'mutation_rate': 0.3, 'activation_fn': 'celu'})
            params.append({'algo_name': 'gin_predictor', 'total_queries': search_budget, 'agent': 'gin_predictor',
                           'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005, 'candidate_nums': 100, 'epochs': 300,
                           'mutation_rate': 0.3, 'rate': 5.})
            # params.append({'algo_name': 'gin_predictor_new', 'total_queries': search_budget, 'agent': 'gin_predictor',
            #                'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005, 'candidate_nums': 100, 'epochs': 300,
            #                'mutation_rate': 0.3, 'rate': 5.})
            # params.append({'algo_name': 'oracle', 'total_queries': search_budget, 'allow_isomorphisms': False,
            #                'candidate_nums': 100, 'num_init': 10, 'k': 10, 'mutation_rate': -1})
            params.append({'algo_name': 'oracle', 'total_queries': search_budget, 'allow_isomorphisms': False,
                           'candidate_nums': 100, 'num_init': 10, 'k': 10, 'mutation_rate': 0.3})
            # params.append({'algo_name': 'oracle', 'total_queries': search_budget, 'allow_isomorphisms': False,
            #                'candidate_nums': 100, 'num_init': 10, 'k': 10, 'mutation_rate': 0.5})
            # params.append({'algo_name': 'oracle', 'total_queries': search_budget, 'allow_isomorphisms': False,
            #                'candidate_nums': 100, 'num_init': 10, 'k': 10, 'mutation_rate': 0.7})
            # params.append({'algo_name': 'oracle', 'total_queries': search_budget, 'allow_isomorphisms': False,
            #                'candidate_nums': 100, 'num_init': 10, 'k': 10, 'mutation_rate': 1.0})
            params.append({'algo_name': 'gp_bayesopt', 'total_queries': search_budget, 'num_init': 10, 'k': 10,
                           'distance': 'adj', 'verbose': 1})
            params.append({'algo_name': 'gp_bayesopt', 'total_queries': search_budget, 'num_init': 10, 'k': 10,
                           'distance': 'nasbot', 'verbose': 1})
        elif param_str == 'nasbench_asr':
            params.append({'algo_name': 'random', 'total_queries': search_budget})
            params.append({'algo_name': 'evolution', 'total_queries': search_budget, 'population_size': 30, 'num_init': 10,
                           'k': 10, 'tournament_size': 10, 'mutation_rate': -1, 'allow_isomorphisms': False,
                           'deterministic': True})
            params.append({'algo_name': 'bananas', 'total_queries': search_budget, 'num_ensemble': 5,
                           'allow_isomorphisms': False, 'acq_opt_type': 'mutation', 'candidate_nums': 100, 'num_init': 10,
                           'k': 10, 'encode_paths': True, 'eva_new': False, 'mutation_rate': 1.0})
            params.append({'algo_name': 'bananas_f', 'total_queries': search_budget, 'num_ensemble': 5,
                           'allow_isomorphisms': False, 'acq_opt_type': 'mutation', 'candidate_nums': 100, 'num_init': 10,
                           'k': 10, 'encode_paths': False, 'mutation_rate': 1.0})
            # params.append({'algo_name': 'gin_uncertainty_predictor', 'total_queries': search_budget,
            #                'agent': 'gin_gaussian', 'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005,
            #                'candidate_nums': 100, 'epochs': 1000, 'mutation_rate': 1.0, 'activation_fn': 'relu'})
            params.append({'algo_name': 'gin_uncertainty_predictor', 'total_queries': search_budget,
                           'agent': 'gin_gaussian', 'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005,
                           'candidate_nums': 100, 'epochs': 1000, 'mutation_rate': 1.0, 'activation_fn': 'celu'})
            params.append({'algo_name': 'gin_predictor', 'total_queries': search_budget, 'agent': 'gin_predictor',
                           'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005, 'candidate_nums': 100, 'epochs': 300,
                           'mutation_rate': 1.0, 'rate': 25.})
            # params.append({'algo_name': 'gin_predictor_new', 'total_queries': search_budget, 'agent': 'gin_predictor',
            #                'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005, 'candidate_nums': 100, 'epochs': 300,
            #                'mutation_rate': 1.0, 'rate': 25.})
            params.append({'algo_name': 'oracle', 'total_queries': search_budget, 'allow_isomorphisms': False,
                           'candidate_nums': 100, 'num_init': 10, 'k': 10, 'mutation_rate': 1.0})
            params.append({'algo_name': 'gp_bayesopt', 'total_queries': search_budget, 'num_init': 10, 'k': 10,
                           'distance': 'adj', 'verbose': 1})
            params.append({'algo_name': 'gp_bayesopt', 'total_queries': search_budget, 'num_init': 10, 'k': 10,
                           'distance': 'nasbot', 'verbose': 1})
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
    else:
        raise ValueError(f"Comparison type {comparison_type} does not support at present!")
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


cifar10_path = '/home/albert_wei/fdisk_a/datasets_train/cifar10/'
nas_bench_101_path = tf_records_path = \
    '/home/albert_wei/Disk_A/dataset_train/nas_bench_101/nasbench_only108.tfrecord'
nas_bench_201_path = '/home/albert_wei/fdisk_a/dataset_train_2021/nasbench_201/NAS-Bench-201-v1_1-096897.pth'
nas_bench_201_converted_base_path = '/home/albert_wei/Disk_A/dataset_train/nas_bench_201/arch_info_%s.pkl'

nas_bench_nlp_path = '/home/albert_wei/Disk_A/dataset_train/nasbench_nlp/'
nas_bench_asr_path = '/home/albert_wei/Disk_A/dataset_train/nasbench_asr/'