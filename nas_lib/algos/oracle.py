import numpy as np


def oracle_nasbench_101_case1(search_space,
                              num_init=10,
                              k=10,
                              total_queries=150,
                              acq_opt_type='mutation',
                              allow_isomorphisms=False,
                              deterministic=True,
                              verbose=1,
                              logger=None,
                              candidate_nums=100,
                              mutation_rate=-1,
                              record_mutation='F'
                              ):
    """
    Bayesian optimization with a neural network model
    """
    data = search_space.generate_random_dataset(num=num_init,
                                                allow_isomorphisms=allow_isomorphisms,
                                                deterministic_loss=deterministic)
    query = num_init + k
    mutate_list = []
    while query <= total_queries:
        if record_mutation == 'T':
            candidates, dist_list, replicate_num, mutated_nums_list, mutated_arch_list \
                = search_space.get_candidates(data,
                                              num=candidate_nums,
                                              acq_opt_type=acq_opt_type,
                                              allow_isomorphisms=allow_isomorphisms,
                                              return_dist=True,
                                              train=True)
            cand_val_list = [cand[2] for cand in candidates]
            mutate_list.append((dist_list, replicate_num, mutated_nums_list, mutated_arch_list, cand_val_list))
        else:
            candidates = search_space.get_candidates(data,
                                                     num=candidate_nums,
                                                     acq_opt_type=acq_opt_type,
                                                     allow_isomorphisms=allow_isomorphisms,
                                                     train=True)
        val_loss = np.array([cand[2] for cand in candidates])
        sorted_indices = np.argsort(val_loss)
        for i in sorted_indices[:k]:
            archtuple = candidates[i]
            data.append(archtuple)
        if verbose:
            top_5_loss = sorted([d[2] for d in data])[:min(5, len(data))]
            logger.info('Query {}, top 5 val losses {}'.format(query, top_5_loss))
        query += k
    return data, {'type': 'oracle', 'final_data': data, 'kt_list': [], 'mutate_list': mutate_list}


def oracle_nasbench_101_case2(search_space,
                              num_init=10,
                              k=10,
                              total_queries=150,
                              acq_opt_type='mutation',
                              allow_isomorphisms=False,
                              deterministic=True,
                              verbose=1,
                              logger=None,
                              candidate_nums=100,
                              mutation_rate=-1,
                              record_mutation='F'
                              ):
    """
    Bayesian optimization with a neural network model
    """
    data = search_space.generate_random_dataset(num=num_init,
                                                allow_isomorphisms=allow_isomorphisms,
                                                deterministic_loss=deterministic)
    query = num_init + k
    mutate_list = []
    while query <= total_queries:
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
        val_loss = np.array([cand[4] for cand in candidates])
        sorted_indices = np.argsort(val_loss)
        for i in sorted_indices[:k]:
            archtuple = candidates[i]
            data.append(archtuple)
        if verbose:
            top_5_loss = sorted([d[4] for d in data])[:min(5, len(data))]
            logger.info('Query {}, top 5 val losses {}'.format(query, top_5_loss))
        query += k
    return data, {'type': 'oracle', 'final_data': data, 'kt_list': [], 'mutate_list': mutate_list}


def oracle_nasbench_201(search_space,
                        num_init=10,
                        k=10,
                        total_queries=150,
                        allow_isomorphisms=False,
                        deterministic=True,
                        verbose=1,
                        logger=None,
                        candidate_nums=100,
                        mutation_rate=-1,
                        record_mutation='F'
                        ):
    """
    Bayesian optimization with a neural network model
    """
    data = search_space.generate_random_dataset(num=num_init,
                                                allow_isomorphisms=allow_isomorphisms,
                                                deterministic_loss=deterministic)
    query = num_init + k
    mutate_list = []
    while query <= total_queries:
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
        val_loss = np.array([cand[4] for cand in candidates])
        sorted_indices = np.argsort(val_loss)
        for i in sorted_indices[:k]:
            archtuple = candidates[i]
            data.append(archtuple)
        if verbose:
            top_5_loss = sorted([d[4] for d in data])[:min(5, len(data))]
            logger.info('Query {}, top 5 val losses {}'.format(query, top_5_loss))
        query += k
    return data, {'type': 'oracle', 'final_data': data, 'kt_list': [], 'mutate_list': mutate_list}


def oracle_nasbench_nlp(search_space,
                        num_init=10,
                        k=10,
                        total_queries=150,
                        allow_isomorphisms=False,
                        deterministic=True,
                        verbose=1,
                        logger=None,
                        candidate_nums=100,
                        mutation_rate=-1,
                        record_mutation='F'
                        ):
    """
    Bayesian optimization with a neural network model
    """
    data = search_space.generate_random_dataset(num=num_init,
                                                allow_isomorphisms=allow_isomorphisms,
                                                deterministic_loss=deterministic)
    query = num_init + k
    mutate_list = []
    while query <= total_queries:
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
        val_loss = np.array([cand[4] for cand in candidates])
        sorted_indices = np.argsort(val_loss)
        for i in sorted_indices[:k]:
            archtuple = candidates[i]
            data.append(archtuple)
        if verbose:
            top_5_loss = sorted([d[4] for d in data])[:min(5, len(data))]
            logger.info('Query {}, top 5 val losses {}'.format(query, top_5_loss))
        query += k
    return data, {'type': 'oracle', 'final_data': data, 'kt_list': [], 'mutate_list': mutate_list}


def oracle_nasbench_asr(search_space,
                        num_init=10,
                        k=10,
                        total_queries=150,
                        allow_isomorphisms=False,
                        deterministic=True,
                        verbose=1,
                        logger=None,
                        candidate_nums=100,
                        mutation_rate=-1,
                        record_mutation='F'
                        ):
    """
    Bayesian optimization with a neural network model
    """
    data = search_space.generate_random_dataset(num=num_init,
                                                allow_isomorphisms=allow_isomorphisms,
                                                deterministic_loss=deterministic)
    query = num_init + k
    mutate_list = []
    while query <= total_queries:
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

        val_loss = np.array([cand[4] for cand in candidates])
        sorted_indices = np.argsort(val_loss)
        for i in sorted_indices[:k]:
            archtuple = candidates[i]
            data.append(archtuple)
        if verbose:
            top_5_loss = sorted([d[4] for d in data])[:min(5, len(data))]
            logger.info('Query {}, top 5 val losses {}'.format(query, top_5_loss))
        query += k
    return data, {'type': 'oracle', 'final_data': data, 'kt_list': [], 'mutate_list': mutate_list}