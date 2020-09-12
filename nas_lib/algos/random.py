

def random_search_case1(search_space,
                        total_queries=100,
                        allow_isomorphisms=False,
                        logger=None,
                        verbose=1):
    """
    random search
    """
    data = search_space.generate_random_dataset(num=total_queries,
                                                allow_isomorphisms=allow_isomorphisms,
                                                deterministic_loss=True)
    if verbose:
        top_5_loss = sorted([d[2] for d in data])[:min(5, len(data))]
        logger.info('Query {}, top 5 val losses {}'.format(total_queries, top_5_loss))
    return data


def random_search_case2(search_space,
                        total_queries=100,
                        allow_isomorphisms=False,
                        logger=None,
                        verbose=1):
    """
    random search
    """
    data = search_space.generate_random_dataset(num=total_queries,
                                                allow_isomorphisms=allow_isomorphisms,
                                                deterministic_loss=True)
    if verbose:
        top_5_loss = sorted([d[4] for d in data])[:min(5, len(data))]
        logger.info('Query {}, top 5 val losses {}'.format(total_queries, top_5_loss))
    return data


def random_search_nasbench_201(search_space,
                               total_queries=100,
                               allow_isomorphisms=False,
                               logger=None,
                               verbose=1):
    """
    random search
    """
    data = search_space.generate_random_dataset(num=total_queries,
                                                allow_isomorphisms=allow_isomorphisms,
                                                deterministic_loss=True)
    if verbose:
        top_5_loss = sorted([d[4] for d in data])[:min(5, len(data))]
        logger.info('Query {}, top 5 val losses {}'.format(total_queries, top_5_loss))
    return data