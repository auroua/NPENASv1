import copy
import os
import random
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import nas_lib.nasbench101.config as config
import numpy as np
import pickle


computed_statistics_path = 'computed_statistics.pkl'

fixed_statistics_path = 'fixed_statistics.pkl'


class OutOfDomainError(Exception):
  """Indicates that the requested graph is outside of the search domain."""


class NASBench(object):
  """User-facing API for accessing the NASBench dataset."""

  def __init__(self, fixed_statistics_path=fixed_statistics_path,
               computed_statistics_path=computed_statistics_path, seed=None):
    """Initialize dataset, this should only be done once per experiment.

    Args:
      dataset_file: path to .tfrecord file containing the dataset.
      seed: random seed used for sampling queried models. Two NASBench objects
        created with the same seed will return the same data points when queried
        with the same models in the same order. By default, the seed is randomly
        generated.
    """
    self.config = config.build_config()
    random.seed(seed)

    print('Loading dataset from file... This may take a few minutes...')
    start = time.time()

    # Stores the fixed statistics that are independent of evaluation (i.e.,
    # adjacency matrix, operations, and number of parameters).
    # hash --> metric name --> scalar
    with open(fixed_statistics_path, 'rb') as f:
      self.fixed_statistics = pickle.load(f)

    # Stores the statistics that are computed via training and evaluating the
    # model on CIFAR-10. Statistics are computed for multiple repeats of each
    # model at each max epoch length.
    # hash --> epochs --> repeat index --> metric name --> scalar
    self.computed_statistics = {}
    with open(computed_statistics_path, 'rb') as f:
      self.computed_statistics = pickle.load(f)

    # Valid queriable epoch lengths. {4, 12, 36, 108} for the full dataset or
    # {108} for the smaller dataset with only the 108 epochs.
    self.valid_epochs = set([108, ])

    elapsed = time.time() - start
    print('Loaded dataset in %d seconds' % elapsed)

    self.history = {}
    self.training_time_spent = 0.0
    self.total_epochs_spent = 0

  def query(self, model_spec, epochs=108, stop_halfway=False):
    """Fetch one of the evaluations for this model spec.

    Each call will sample one of the config['num_repeats'] evaluations of the
    model. This means that repeated queries of the same model (or isomorphic
    models) may return identical metrics.

    This function will increment the budget counters for benchmarking purposes.
    See self.training_time_spent, and self.total_epochs_spent.

    This function also allows querying the evaluation metrics at the halfway
    point of training using stop_halfway. Using this option will increment the
    budget counters only up to the halfway point.

    Args:
      model_spec: ModelSpec object.
      epochs: number of epochs trained. Must be one of the evaluated number of
        epochs, [4, 12, 36, 108] for the full dataset.
      stop_halfway: if True, returned dict will only contain the training time
        and accuracies at the halfway point of training (num_epochs/2).
        Otherwise, returns the time and accuracies at the end of training
        (num_epochs).

    Returns:
      dict containing the evaluated data for this object.

    Raises:
      OutOfDomainError: if model_spec or num_epochs is outside the search space.
    """
    if epochs not in self.valid_epochs:
      raise OutOfDomainError('invalid number of epochs, must be one of %s'
                             % self.valid_epochs)

    fixed_stat, computed_stat = self.get_metrics_from_spec(model_spec)
    sampled_index = random.randint(0, self.config['num_repeats'] - 1)
    computed_stat = computed_stat[epochs][sampled_index]

    data = {}
    data['module_adjacency'] = fixed_stat['module_adjacency']
    data['module_operations'] = fixed_stat['module_operations']
    data['trainable_parameters'] = fixed_stat['trainable_parameters']

    if stop_halfway:
      data['training_time'] = computed_stat['halfway_training_time']
      data['train_accuracy'] = computed_stat['halfway_train_accuracy']
      data['validation_accuracy'] = computed_stat['halfway_validation_accuracy']
      data['test_accuracy'] = computed_stat['halfway_test_accuracy']
    else:
      data['training_time'] = computed_stat['final_training_time']
      data['train_accuracy'] = computed_stat['final_train_accuracy']
      data['validation_accuracy'] = computed_stat['final_validation_accuracy']
      data['test_accuracy'] = computed_stat['final_test_accuracy']

    self.training_time_spent += data['training_time']
    if stop_halfway:
      self.total_epochs_spent += epochs // 2
    else:
      self.total_epochs_spent += epochs

    return data

  def query_macro_test(self, model_spec, epochs=108, stop_halfway=False):
    """Fetch one of the evaluations for this model spec.

    Each call will sample one of the config['num_repeats'] evaluations of the
    model. This means that repeated queries of the same model (or isomorphic
    models) may return identical metrics.

    This function will increment the budget counters for benchmarking purposes.
    See self.training_time_spent, and self.total_epochs_spent.

    This function also allows querying the evaluation metrics at the halfway
    point of training using stop_halfway. Using this option will increment the
    budget counters only up to the halfway point.

    Args:
      model_spec: ModelSpec object.
      epochs: number of epochs trained. Must be one of the evaluated number of
        epochs, [4, 12, 36, 108] for the full dataset.
      stop_halfway: if True, returned dict will only contain the training time
        and accuracies at the halfway point of training (num_epochs/2).
        Otherwise, returns the time and accuracies at the end of training
        (num_epochs).

    Returns:
      dict containing the evaluated data for this object.

    Raises:
      OutOfDomainError: if model_spec or num_epochs is outside the search space.
    """
    if epochs not in self.valid_epochs:
      raise OutOfDomainError('invalid number of epochs, must be one of %s'
                             % self.valid_epochs)

    fixed_stat, computed_stat = self.get_metrics_from_spec(model_spec)

    test_accuray_list = []
    val_accuracy_list = []
    for i in range(3):
      computed_stat_iter = computed_stat[epochs][i]
      val_accuracy_list.append(computed_stat_iter['final_validation_accuracy'])
      test_accuray_list.append(computed_stat_iter['final_test_accuracy'])
    avg_val_acc = np.mean(val_accuracy_list)
    avg_test_val = np.mean(test_accuray_list)

    sampled_index = random.randint(0, self.config['num_repeats'] - 1)
    computed_stat = computed_stat[epochs][sampled_index]

    data = {}
    data['module_adjacency'] = fixed_stat['module_adjacency']
    data['module_operations'] = fixed_stat['module_operations']
    data['trainable_parameters'] = fixed_stat['trainable_parameters']

    if stop_halfway:
      data['training_time'] = computed_stat['halfway_training_time']
      data['train_accuracy'] = computed_stat['halfway_train_accuracy']
      data['validation_accuracy'] = float(avg_val_acc)
      data['test_accuracy'] = float(avg_test_val)
    else:
      data['training_time'] = computed_stat['final_training_time']
      data['train_accuracy'] = computed_stat['final_train_accuracy']
      data['validation_accuracy'] = float(avg_val_acc)
      data['test_accuracy'] = float(avg_test_val)

    self.training_time_spent += data['training_time']
    if stop_halfway:
      self.total_epochs_spent += epochs // 2
    else:
      self.total_epochs_spent += epochs

    return data

  def is_valid(self, model_spec):
    """Checks the validity of the model_spec.

    For the purposes of benchmarking, this does not increment the budget
    counters.

    Args:
      model_spec: ModelSpec object.

    Returns:
      True if model is within space.
    """
    try:
      self._check_spec(model_spec)
    except OutOfDomainError:
      return False

    return True

  def get_budget_counters(self):
    """Returns the time and budget counters."""
    return self.training_time_spent, self.total_epochs_spent

  def reset_budget_counters(self):
    """Reset the time and epoch budget counters."""
    self.training_time_spent = 0.0
    self.total_epochs_spent = 0

  def hash_iterator(self):
    """Returns iterator over all unique model hashes."""
    return self.fixed_statistics.keys()

  def get_metrics_from_hash(self, module_hash):
    """Returns the metrics for all epochs and all repeats of a hash.

    This method is for dataset analysis and should not be used for benchmarking.
    As such, it does not increment any of the budget counters.

    Args:
      module_hash: MD5 hash, i.e., the values yielded by hash_iterator().

    Returns:
      fixed stats and computed stats of the model spec provided.
    """
    fixed_stat = copy.deepcopy(self.fixed_statistics[module_hash])
    computed_stat = copy.deepcopy(self.computed_statistics[module_hash])
    return fixed_stat, computed_stat

  def get_metrics_from_spec(self, model_spec):
    """Returns the metrics for all epochs and all repeats of a model.

    This method is for dataset analysis and should not be used for benchmarking.
    As such, it does not increment any of the budget counters.

    Args:
      model_spec: ModelSpec object.

    Returns:
      fixed stats and computed stats of the model spec provided.
    """
    self._check_spec(model_spec)
    module_hash = self._hash_spec(model_spec)
    return self.get_metrics_from_hash(module_hash)

  def _check_spec(self, model_spec):
    """Checks that the model spec is within the dataset."""
    if not model_spec.valid_spec:
      raise OutOfDomainError('invalid spec, provided graph is disconnected.')

    num_vertices = len(model_spec.ops)
    num_edges = np.sum(model_spec.matrix)

    if num_vertices > self.config['module_vertices']:
      raise OutOfDomainError('too many vertices, got %d (max vertices = %d)'
                             % (num_vertices, config['module_vertices']))

    if num_edges > self.config['max_edges']:
      raise OutOfDomainError('too many edges, got %d (max edges = %d)'
                             % (num_edges, self.config['max_edges']))

    if model_spec.ops[0] != 'input':
      raise OutOfDomainError('first operation should be \'input\'')
    if model_spec.ops[-1] != 'output':
      raise OutOfDomainError('last operation should be \'output\'')
    for op in model_spec.ops[1:-1]:
      if op not in self.config['available_ops']:
        raise OutOfDomainError('unsupported op %s (available ops = %s)'
                               % (op, self.config['available_ops']))

  def _hash_spec(self, model_spec):
    """Returns the MD5 hash for a provided model_spec."""
    return model_spec.hash_spec(self.config['available_ops'])


if __name__ == '__main__':
  nasbench = NASBench()