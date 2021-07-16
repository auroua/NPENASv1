import os
from argparse import Namespace
import numpy as np
import pickle
from nas_lib.bo.bo.probo import ProBO


def gp_bayesopt_nasbench_101_case1(search_space,
                                   logger=None,
                                   num_init=10,
                                   k=10,
                                   total_queries=100,
                                   loss='val_loss',
                                   distance='adj',
                                   random_encoding='adj',
                                   cutoff=0,
                                   deterministic=True,
                                   tmpdir='./temp',
                                   max_iter=200,
                                   mode='single_process',
                                   verbose=0,
                                   nppred=1000):
    """
    Bayesian optimization with a GP prior
    """
    # set up the path for auxiliary pickle files
    if not os.path.exists(tmpdir):
        os.mkdir(tmpdir)
    aux_file_path = os.path.join(tmpdir, 'aux.pkl')
    num_iterations = total_queries - num_init

    # black-box function that bayesopt will optimize
    def fn(arch):
        arch_info = search_space.query_arch(arch={'matrix': arch['matrix'], 'ops': arch['ops']},
                                            train=True,
                                            deterministic=True,
                                            encode_paths=True)
        if loss == 'val_loss':
            return arch_info[2]
        elif loss == 'test_loss':
            return arch_info[3]

    # set all the parameters for the various BayesOpt classes
    fhp = Namespace(fhstr='object', namestr='train')
    domp = Namespace(dom_str='list', set_domain_list_auto=True,
                     aux_file_path=aux_file_path,
                     distance=distance)
    modelp = Namespace(kernp=Namespace(ls=3., alpha=1.5, sigma=1e-5),
                       infp=Namespace(niter=num_iterations, nwarmup=500),
                       distance=distance, search_space='nasbench_101')
    amp = Namespace(am_str='mygpdistmat_ucb', nppred=nppred, modelp=modelp)
    optp = Namespace(opt_str='rand', max_iter=max_iter)
    makerp = Namespace(domp=domp, amp=amp, optp=optp)
    probop = Namespace(niter=num_iterations, fhp=fhp,
                       makerp=makerp, tmpdir=tmpdir, mode=mode)
    data = Namespace()

    # Set up initial data
    init_data = search_space.generate_random_dataset(num=num_init,
                                                     allow_isomorphisms=False,
                                                     deterministic_loss=deterministic)
    data.X = [{'matrix': d[0]['matrix'], 'ops': d[0]['ops']} for d in init_data]
    data.y = np.array([[d[2]] for d in init_data])

    # initialize aux file
    pairs = [(data.X[i], data.y[i]) for i in range(len(data.y))]
    pairs.sort(key=lambda x: x[1])
    with open(aux_file_path, 'wb') as f:
        pickle.dump(pairs, f)

    # run Bayesian Optimization
    bo = ProBO(fn, search_space, aux_file_path, data, probop, True)
    bo.run_bo(verbose=verbose)

    # get the validation and test loss for all architectures chosen by BayesOpt
    results = []
    for arch in data.X:
        archtuple = search_space.query_arch(arch={'matrix': arch['matrix'], 'ops': arch['ops']},
                                            train=True,
                                            deterministic=True,
                                            encode_paths=True)
        results.append(archtuple)

    return results


def gp_bayesopt_nasbench_101(search_space,
                             logger=None,
                             num_init=10,
                             k=10,
                             total_queries=100,
                             loss='val_loss',
                             distance='adj',
                             random_encoding='adj',
                             cutoff=0,
                             deterministic=True,
                             tmpdir='./temp',
                             max_iter=200,
                             mode='single_process',
                             verbose=0,
                             nppred=1000):
    """
    Bayesian optimization with a GP prior
    """
    # set up the path for auxiliary pickle files
    if not os.path.exists(tmpdir):
        os.mkdir(tmpdir)
    aux_file_path = os.path.join(tmpdir, 'aux.pkl')
    num_iterations = total_queries - num_init

    # black-box function that bayesopt will optimize
    def fn(arch):
        arch_info = search_space.query_arch(matrix=arch['matrix'], ops=arch['ops'])
        if loss == 'val_loss':
            return arch_info[4]
        elif loss == 'test_loss':
            return arch_info[5]

    # set all the parameters for the various BayesOpt classes
    fhp = Namespace(fhstr='object', namestr='train')
    domp = Namespace(dom_str='list', set_domain_list_auto=True,
                     aux_file_path=aux_file_path,
                     distance=distance)
    modelp = Namespace(kernp=Namespace(ls=3., alpha=1.5, sigma=1e-5),
                       infp=Namespace(niter=num_iterations, nwarmup=500),
                       distance=distance, search_space='nasbench_101')
    amp = Namespace(am_str='mygpdistmat_ucb', nppred=nppred, modelp=modelp)
    optp = Namespace(opt_str='rand', max_iter=max_iter)
    makerp = Namespace(domp=domp, amp=amp, optp=optp)
    probop = Namespace(niter=num_iterations, fhp=fhp,
                       makerp=makerp, tmpdir=tmpdir, mode=mode)
    data = Namespace()

    # Set up initial data
    init_data = search_space.generate_random_dataset(num=num_init,
                                                     deterministic_loss=deterministic)
    data.X = [{'matrix': d[1], 'ops': d[2]} for d in init_data]
    # data.X = [{'matrix': d[0][0], 'ops': d[0][1]} for d in init_data]
    data.y = np.array([[d[4]] for d in init_data])

    # initialize aux file
    pairs = [(data.X[i], data.y[i]) for i in range(len(data.y))]
    pairs.sort(key=lambda x: x[1])
    with open(aux_file_path, 'wb') as f:
        pickle.dump(pairs, f)

    # run Bayesian Optimization
    bo = ProBO(fn, search_space, aux_file_path, data, probop, True)
    bo.run_bo(verbose=verbose)

    # get the validation and test loss for all architectures chosen by BayesOpt
    results = []
    for arch in data.X:
        archtuple = search_space.query_arch(matrix=arch['matrix'], ops=arch['ops'])
        results.append(archtuple)

    return results


def gp_bayesopt_nasbench_201(search_space,
                             logger=None,
                             num_init=10,
                             k=10,
                             total_queries=100,
                             loss='val_loss',
                             distance='adj',
                             random_encoding='adj',
                             cutoff=0,
                             deterministic=True,
                             tmpdir='./temp',
                             max_iter=200,
                             mode='single_process',
                             verbose=0,
                             nppred=1000):
    """
    Bayesian optimization with a GP prior
    """
    # set up the path for auxiliary pickle files
    if not os.path.exists(tmpdir):
        os.mkdir(tmpdir)
    aux_file_path = os.path.join(tmpdir, 'aux.pkl')
    num_iterations = total_queries - num_init

    # black-box function that bayesopt will optimize
    def fn(arch):
        arch_str = arch['string']
        arch_info = search_space.total_archs[arch_str]
        if loss == 'val_loss':
            return arch_info[4]
        elif loss == 'test_loss':
            return arch_info[5]

    # set all the parameters for the various BayesOpt classes
    fhp = Namespace(fhstr='object', namestr='train')
    domp = Namespace(dom_str='list', set_domain_list_auto=True,
                     aux_file_path=aux_file_path,
                     distance=distance)
    modelp = Namespace(kernp=Namespace(ls=3., alpha=1.5, sigma=1e-5),
                       infp=Namespace(niter=num_iterations, nwarmup=500),
                       distance=distance, search_space='nasbench_201')
    amp = Namespace(am_str='mygpdistmat_ucb', nppred=nppred, modelp=modelp)
    optp = Namespace(opt_str='rand', max_iter=max_iter)
    makerp = Namespace(domp=domp, amp=amp, optp=optp)
    probop = Namespace(niter=num_iterations, fhp=fhp,
                       makerp=makerp, tmpdir=tmpdir, mode=mode)
    data = Namespace()

    # Set up initial data
    init_data = search_space.generate_random_dataset(num=num_init,
                                                     deterministic_loss=deterministic)
    data.X = [{'string': d[6]} for d in init_data]
    data.y = np.array([[d[4]] for d in init_data])

    # initialize aux file
    pairs = [(data.X[i], data.y[i]) for i in range(len(data.y))]
    pairs.sort(key=lambda x: x[1])
    with open(aux_file_path, 'wb') as f:
        pickle.dump(pairs, f)

    # run Bayesian Optimization
    bo = ProBO(fn, search_space, aux_file_path, data, probop, True)
    bo.run_bo(verbose=verbose)

    # get the validation and test loss for all architectures chosen by BayesOpt
    results = []
    for arch in data.X:
        archtuple = search_space.total_archs[arch['string']]
        results.append(archtuple)

    return results


def gp_bayesopt_nasbench_nlp(search_space,
                             logger=None,
                             num_init=10,
                             k=10,
                             total_queries=100,
                             loss='val_loss',
                             distance='adj',
                             random_encoding='adj',
                             cutoff=0,
                             deterministic=True,
                             tmpdir='./temp',
                             max_iter=200,
                             mode='single_process',
                             verbose=0,
                             nppred=1000):
    """
    Bayesian optimization with a GP prior
    """
    # set up the path for auxiliary pickle files
    if not os.path.exists(tmpdir):
        os.mkdir(tmpdir)
    aux_file_path = os.path.join(tmpdir, 'aux.pkl')
    num_iterations = total_queries - num_init

    # black-box function that bayesopt will optimize
    def fn(arch):
        arch_str = search_space.arch_keys_dict[arch[0]]
        arch_info = search_space.total_archs[arch_str]
        if loss == 'val_loss':
            return arch_info[4]
        elif loss == 'test_loss':
            return arch_info[5]

    # set all the parameters for the various BayesOpt classes
    fhp = Namespace(fhstr='object', namestr='train')
    domp = Namespace(dom_str='list', set_domain_list_auto=True,
                     aux_file_path=aux_file_path,
                     distance=distance)
    modelp = Namespace(kernp=Namespace(ls=3., alpha=1.5, sigma=1e-5),
                       infp=Namespace(niter=num_iterations, nwarmup=500),
                       distance=distance, search_space='nasbench_nlp')
    amp = Namespace(am_str='mygpdistmat_ucb', nppred=nppred, modelp=modelp)
    optp = Namespace(opt_str='rand', max_iter=max_iter)
    makerp = Namespace(domp=domp, amp=amp, optp=optp)
    probop = Namespace(niter=num_iterations, fhp=fhp,
                       makerp=makerp, tmpdir=tmpdir, mode=mode)
    data = Namespace()

    # Set up initial data
    init_data = search_space.generate_random_dataset(num=num_init,
                                                     allow_isomorphisms=False,
                                                     deterministic_loss=deterministic)
    data.X = [[d[6], d[1], d[2]] for d in init_data]
    data.y = np.array([[d[4]] for d in init_data])

    # initialize aux file
    pairs = [(data.X[i], data.y[i]) for i in range(len(data.y))]
    pairs.sort(key=lambda x: x[1])
    with open(aux_file_path, 'wb') as f:
        pickle.dump(pairs, f)

    # run Bayesian Optimization
    bo = ProBO(fn, search_space, aux_file_path, data, probop, True)
    bo.run_bo(verbose=verbose)

    # get the validation and test loss for all architectures chosen by BayesOpt
    results = []
    for arch in data.X:
        arch_str = search_space.arch_keys_dict[arch[0]]
        archtuple = search_space.total_archs[arch_str]
        results.append(archtuple)

    return results


def gp_bayesopt_nasbench_asr(search_space,
                             logger=None,
                             num_init=10,
                             k=10,
                             total_queries=100,
                             loss='val_loss',
                             distance='adj',
                             random_encoding='adj',
                             cutoff=0,
                             deterministic=True,
                             tmpdir='./temp',
                             max_iter=200,
                             mode='single_process',
                             verbose=0,
                             nppred=1000,
                             record_kt='F',
                             record_mutation='F'
                             ):
    """
    Bayesian optimization with a GP prior
    """
    # set up the path for auxiliary pickle files
    if not os.path.exists(tmpdir):
        os.mkdir(tmpdir)
    aux_file_path = os.path.join(tmpdir, 'aux.pkl')
    num_iterations = total_queries - num_init

    # black-box function that bayesopt will optimize
    def fn(arch):
        arch_str = arch[0]
        arch_info = search_space.all_datas_dict[arch_str]
        if loss == 'val_loss':
            return arch_info[0]
        elif loss == 'test_loss':
            return arch_info[1]

    # set all the parameters for the various BayesOpt classes
    fhp = Namespace(fhstr='object', namestr='train')
    domp = Namespace(dom_str='list', set_domain_list_auto=True,
                     aux_file_path=aux_file_path,
                     distance=distance)
    modelp = Namespace(kernp=Namespace(ls=3., alpha=1.5, sigma=1e-5),
                       infp=Namespace(niter=num_iterations, nwarmup=500),
                       distance=distance, search_space='nasbench_asr')
    amp = Namespace(am_str='mygpdistmat_ucb', nppred=nppred, modelp=modelp)
    optp = Namespace(opt_str='rand', max_iter=max_iter)
    makerp = Namespace(domp=domp, amp=amp, optp=optp)
    probop = Namespace(niter=num_iterations, fhp=fhp,
                       makerp=makerp, tmpdir=tmpdir, mode=mode)
    data = Namespace()

    # Set up initial data
    init_data = search_space.generate_random_dataset(num=num_init,
                                                     deterministic_loss=deterministic,
                                                     allow_isomorphisms=False)
    data.X = [[d[6], d[1], d[2]] for d in init_data]
    data.y = np.array([[d[4]] for d in init_data])

    # initialize aux file
    pairs = [(data.X[i], data.y[i]) for i in range(len(data.y))]
    pairs.sort(key=lambda x: x[1])
    with open(aux_file_path, 'wb') as f:
        pickle.dump(pairs, f)

    # run Bayesian Optimization
    bo = ProBO(fn, search_space, aux_file_path, data, probop, True)
    bo.run_bo(verbose=verbose)

    # get the validation and test loss for all architectures chosen by BayesOpt
    results = []
    for arch in data.X:
        archtuple = search_space.total_archs[arch[0]]
        results.append(archtuple)

    return results