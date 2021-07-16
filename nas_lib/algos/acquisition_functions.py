import numpy as np
import sys


# Different acquisition functions that can be used by BANANAS
def acq_fn(predictions, explore_type='its', reverse=False, get_samples=False):
    if explore_type != 'its_vae' and explore_type != 'its_vae_ensemble':
        predictions = np.array(predictions)
    # Thompson sampling (TS) acquisition function
    if explore_type == 'ts':
        rand_ind = np.random.randint(predictions.shape[0])
        ts = predictions[rand_ind,:]
        sorted_indices = np.argsort(ts)
    # Independent Thompson sampling (ITS) acquisition function
    elif explore_type == 'its':
        mean = np.mean(predictions, axis=0)
        std = np.sqrt(np.var(predictions, axis=0))
        samples = np.random.normal(mean, std)
        sorted_indices = np.argsort(samples)
        if get_samples:
            return sorted_indices, samples
        else:
            return sorted_indices
    elif explore_type == 'its_vae':
        np_pred = predictions.cpu().numpy()
        sorted_indices = np.argsort(np_pred)
    else:
        print('Invalid exploration type in meta neuralnet search', explore_type)
        sys.exit()

    return sorted_indices