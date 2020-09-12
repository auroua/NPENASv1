import numpy as np
from .acquisition_functions import acq_fn
from ..models.meta_neural_net import MetaNeuralnet
import tensorflow as tf
from keras import backend as K
import copy


def bananas_case1(search_space,
                  metann_params,
                  num_init=10,
                  k=10,
                  total_queries=150,
                  num_ensemble=5,
                  acq_opt_type='mutation',
                  explore_type='its',
                  encode_paths=True,
                  allow_isomorphisms=False,
                  deterministic=True,
                  verbose=1,
                  gpu=None,
                  logger=None,
                  candidate_nums=100):
    """
    Bayesian optimization with a neural network model
    """
    data = search_space.generate_random_dataset(num=num_init,
                                                encode_paths=encode_paths,
                                                allow_isomorphisms=allow_isomorphisms,
                                                deterministic_loss=deterministic)
    query = num_init + k
    while query <= total_queries:
        xtrain = np.array([d[1] for d in data])
        ytrain = np.array([d[2] for d in data])

        candidates = search_space.get_candidates(data,
                                                 num=candidate_nums,
                                                 acq_opt_type=acq_opt_type,
                                                 encode_paths=encode_paths,
                                                 allow_isomorphisms=allow_isomorphisms,
                                                 deterministic_loss=deterministic)
        xcandidates = np.array([c[1] for c in candidates])
        predictions = []
        train_error = 0
        for _ in range(num_ensemble):
            if gpu is not None:
                meta_neuralnet = MetaNeuralnet(gpu=gpu)
            else:
                meta_neuralnet = MetaNeuralnet()
            train_error += meta_neuralnet.fit(xtrain, ytrain, **metann_params)

            predictions.append(np.squeeze(meta_neuralnet.predict(xcandidates)))

            K.clear_session()
            tf.reset_default_graph()
            del meta_neuralnet
        train_error /= num_ensemble
        if verbose:
            logger.info('Query {}, Meta neural net train error: {}'.format(query, train_error))
        sorted_indices = acq_fn(predictions, explore_type)
        for i in sorted_indices[:k]:
            archtuple = search_space.query_arch(candidates[i][0],
                                                encode_paths=encode_paths,
                                                deterministic=deterministic)
            data.append(archtuple)
        if verbose:
            top_5_loss = sorted([d[2] for d in data])[:min(5, len(data))]
            logger.info('Query {}, top 5 val losses {}'.format(query, top_5_loss))
        query += k
    return data


def bananas_case2(search_space,
                  metann_params,
                  num_init=10,
                  k=10,
                  total_queries=150,
                  num_ensemble=5,
                  acq_opt_type='mutation',
                  explore_type='its',
                  encode_paths=True,
                  allow_isomorphisms=False,
                  deterministic=True,
                  verbose=1,
                  gpu=None,
                  logger=None,
                  candidate_nums=100):
    """
    Bayesian optimization with a neural network model
    """
    data = search_space.generate_random_dataset(num=num_init,
                                                encode_paths=encode_paths,
                                                allow_isomorphisms=allow_isomorphisms,
                                                deterministic_loss=deterministic)
    query = num_init + k
    while query <= total_queries:
        xtrain = np.array([d[3] for d in data])
        ytrain = np.array([d[4] for d in data])
        candidates = search_space.get_candidates(data,
                                                 num=candidate_nums,
                                                 acq_opt_type=acq_opt_type,
                                                 encode_paths=encode_paths,
                                                 allow_isomorphisms=allow_isomorphisms,
                                                 deterministic_loss=deterministic)
        xcandidates = np.array([c[3] for c in candidates])
        predictions = []
        train_error = 0
        for _ in range(num_ensemble):
            if gpu is not None:
                meta_neuralnet = MetaNeuralnet(gpu=gpu)
            else:
                meta_neuralnet = MetaNeuralnet()
            train_error += meta_neuralnet.fit(xtrain, ytrain, **metann_params)
            predictions.append(np.squeeze(meta_neuralnet.predict(xcandidates)))
            K.clear_session()
            tf.reset_default_graph()
            del meta_neuralnet
        train_error /= num_ensemble
        if verbose:
            logger.info('Query {}, Meta neural net train error: {}'.format(query, train_error))
        sorted_indices = acq_fn(predictions, explore_type)
        for i in sorted_indices[:k]:
            archtuple = search_space.query_arch(matrix=candidates[i][1],
                                                ops=candidates[i][2],
                                                encode_paths=encode_paths,
                                                deterministic=deterministic)
            data.append(archtuple)
        if verbose:
            top_5_loss = sorted([d[4] for d in data])[:min(5, len(data))]
            logger.info('Query {}, top 5 val losses {}'.format(query, top_5_loss))
        query += k
    return data


def bananas_nasbench_201(search_space,
                         metann_params,
                         num_init=10,
                         k=10,
                         total_queries=150,
                         num_ensemble=5,
                         acq_opt_type='mutation',
                         explore_type='its',
                         encode_paths=True,
                         allow_isomorphisms=False,
                         deterministic=True,
                         verbose=1,
                         gpu=None,
                         logger=None,
                         eva_new=True,
                         candidate_nums=100):
    """
    Bayesian optimization with a neural network model
    """
    data = search_space.generate_random_dataset(num=num_init,
                                                allow_isomorphisms=allow_isomorphisms,
                                                deterministic_loss=deterministic)
    query = num_init + k
    while query <= total_queries:
        if encode_paths:
            xtrain = np.array([d[3] for d in data])
        else:
            xtrain = np.array([d[7] for d in data])
        ytrain = np.array([d[4] for d in data])
        candidates = search_space.get_candidates(data,
                                                 num=candidate_nums,
                                                 allow_isomorphisms=allow_isomorphisms)
        if encode_paths:
            xcandidates = np.array([c[3] for c in candidates])
        else:
            xcandidates = np.array([c[7] for c in candidates])
        predictions = []
        train_error = 0
        for _ in range(num_ensemble):
            if gpu is not None:
                meta_neuralnet = MetaNeuralnet(gpu=gpu)
            else:
                meta_neuralnet = MetaNeuralnet()
            train_error += meta_neuralnet.fit(xtrain, ytrain, **metann_params)
            predictions.append(np.squeeze(meta_neuralnet.predict(xcandidates)))
            K.clear_session()
            tf.reset_default_graph()
            del meta_neuralnet
        train_error /= num_ensemble
        if verbose:
            logger.info('Query {}, Meta neural net train error: {}'.format(query, train_error))
        sorted_indices = acq_fn(predictions, explore_type)
        for i in sorted_indices[:k]:
            archtuple = candidates[i]
            data.append(archtuple)
        if verbose:
            top_5_loss = sorted([d[4] for d in data])[:min(5, len(data))]
            logger.info('Query {}, top 5 val losses {}'.format(query, top_5_loss))
        query += k
    return data


def bananas_diff_training_nums_case1(search_space,
                                     metann_params,
                                     num_init=10,
                                     k=10,
                                     total_queries=150,
                                     num_ensemble=5,
                                     acq_opt_type='mutation',
                                     explore_type='its',
                                     encode_paths=True,
                                     allow_isomorphisms=False,
                                     deterministic=True,
                                     verbose=1,
                                     gpu=None,
                                     logger=None,
                                     candidate_nums=100,
                                     training_nums=150):
    data = search_space.generate_random_dataset(num=num_init,
                                                encode_paths=encode_paths,
                                                allow_isomorphisms=allow_isomorphisms,
                                                deterministic_loss=deterministic)
    query = num_init + k
    train_data = []
    while query <= total_queries:
        if len(train_data) < training_nums:
            train_data = copy.deepcopy(data)
        candidates = search_space.get_candidates(data,
                                                 num=candidate_nums,
                                                 acq_opt_type=acq_opt_type,
                                                 encode_paths=encode_paths,
                                                 allow_isomorphisms=allow_isomorphisms,
                                                 deterministic_loss=deterministic)
        xcandidates = np.array([c[1] for c in candidates])
        predictions = []
        train_error = 0

        xtrain = np.array([d[1] for d in train_data])
        ytrain = np.array([d[2] for d in train_data])
        for _ in range(num_ensemble):
            if gpu is not None:
                meta_neuralnet = MetaNeuralnet(gpu=gpu)
            else:
                meta_neuralnet = MetaNeuralnet()
            train_error += meta_neuralnet.fit(xtrain, ytrain, **metann_params)
            predictions.append(np.squeeze(meta_neuralnet.predict(xcandidates)))

            K.clear_session()
            tf.reset_default_graph()
            del meta_neuralnet
        train_error /= num_ensemble
        if verbose:
            logger.info('Query {}, Meta neural net train error: {}'.format(query, train_error))
        sorted_indices = acq_fn(predictions, explore_type)
        for i in sorted_indices[:k]:
            archtuple = search_space.query_arch(candidates[i][0],
                                                encode_paths=encode_paths,
                                                deterministic=deterministic)
            data.append(archtuple)
        if verbose:
            top_5_loss = sorted([d[2] for d in data])[:min(5, len(data))]
            logger.info('Query {}, training data nums {},  top 5 val losses {}'.format(query, len(train_data),
                                                                                       top_5_loss))
        query += k
    return data


def bananas_training_num_diff_case2(search_space,
                                    metann_params,
                                    num_init=10,
                                    k=10,
                                    total_queries=150,
                                    num_ensemble=5,
                                    acq_opt_type='mutation',
                                    explore_type='its',
                                    encode_paths=True,
                                    allow_isomorphisms=False,
                                    deterministic=True,
                                    verbose=1,
                                    gpu=None,
                                    logger=None,
                                    candidate_nums=100,
                                    training_nums=150):
    data = search_space.generate_random_dataset(num=num_init,
                                                encode_paths=encode_paths,
                                                allow_isomorphisms=allow_isomorphisms,
                                                deterministic_loss=deterministic)
    query = num_init + k
    train_data = []
    while query <= total_queries:
        if len(train_data) < training_nums:
            train_data = copy.deepcopy(data)
        xtrain = np.array([d[3] for d in train_data])
        ytrain = np.array([d[4] for d in train_data])
        candidates = search_space.get_candidates(data,
                                                 num=candidate_nums,
                                                 acq_opt_type=acq_opt_type,
                                                 encode_paths=encode_paths,
                                                 allow_isomorphisms=allow_isomorphisms,
                                                 deterministic_loss=deterministic)
        xcandidates = np.array([c[3] for c in candidates])
        predictions = []
        train_error = 0
        for _ in range(num_ensemble):
            if gpu is not None:
                meta_neuralnet = MetaNeuralnet(gpu=gpu)
            else:
                meta_neuralnet = MetaNeuralnet()
            train_error += meta_neuralnet.fit(xtrain, ytrain, **metann_params)
            predictions.append(np.squeeze(meta_neuralnet.predict(xcandidates)))
            K.clear_session()
            tf.reset_default_graph()
            del meta_neuralnet
        train_error /= num_ensemble
        if verbose:
            logger.info('Query {}, Meta neural net train error: {}'.format(query, train_error))
        sorted_indices = acq_fn(predictions, explore_type)
        for i in sorted_indices[:k]:
            archtuple = search_space.query_arch(matrix=candidates[i][1],
                                                ops=candidates[i][2],
                                                encode_paths=encode_paths,
                                                deterministic=deterministic)
            data.append(archtuple)
        if verbose:
            top_5_loss = sorted([d[4] for d in data])[:min(5, len(data))]
            logger.info('Query {},  training data nums {},  top 5 val losses {}'.format(query, len(train_data),
                                                                                        top_5_loss))
        query += k
    return data