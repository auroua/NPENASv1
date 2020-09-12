import numpy as np
import torch
import os
import sqlite3
from collections import namedtuple
import logging
import shutil
import pickle


logger = logging.getLogger('nasbench_open_darts_cifar10')



def compute_best_test_losses(data, k, total_queries):
    """
    Given full data from a completed nas algorithm,
    output the test error of the arch with the best val error
    after every multiple of k
    """
    results = []
    results_keys = []
    total_data = []
    for i in range(total_queries):
        total_data.append(data[i])
    for query in range(k, total_queries + k, k):
        best_arch = sorted(total_data[:query], key=lambda i: i[0])[0]
        test_error = best_arch[1]
        results.append((query, test_error))
        results_keys.append(best_arch[2])
    return results, results_keys


def compute_darts_test_losses(data, k, total_queries):
    """
    Given full data from a completed nas algorithm,
    output the test error of the arch with the best val error
    after every multiple of k
    """
    results = []
    results_keys = []
    model_archs, model_keys = data
    losses = [(model_archs[k][2], model_archs[k][3], k) for k in model_keys]
    for query in range(k, total_queries + k, k):
        best_arch = sorted(losses[:query], key=lambda i: i[0])[0]
        test_error = best_arch[1]
        results.append((query, test_error))
        results_keys.append(best_arch[2])
    return results, results_keys


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def init_nasbench_macro_cifar10(model_dir):
    try:
        conn = sqlite3.connect(os.path.join(model_dir, 'models.db'))
        conn.execute("create table models (id text not null, hashkey text, modelpath text, train_acc real, val_acc real, "
                     "test_acc real)")
        conn.commit()
    except sqlite3.OperationalError as e:
        print(e)
    return conn


def drop_path(x, drop_prob, device):
    if drop_prob > 0.:
        keep_prob = 1.-drop_prob
        mask = torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob).to(device)
        x.div_(keep_prob)
        x.mul_(mask)
    return x


Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


def convert_to_genotype(arch, verbose=True):
    op_dict = {
        0: 'none',
        1: 'max_pool_3x3',
        2: 'avg_pool_3x3',
        3: 'skip_connect',
        4: 'sep_conv_3x3',
        5: 'sep_conv_5x5',
        6: 'dil_conv_3x3',
        7: 'dil_conv_5x5'
    }
    darts_arch = [[], []]
    i = 0
    for cell in arch:
        for n in cell:
            darts_arch[i].append((op_dict[n[1]], n[0]))
        i += 1
    geno = Genotype(normal=darts_arch[0], normal_concat=[2, 3, 4, 5], reduce=darts_arch[1], reduce_concat=[2, 3, 4, 5])
    if verbose:
        logger.info(str(geno))
    return geno


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def top_accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res


class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def load_model(root, file_name):
    file_path = os.path.join(root, 'pre_train_models', file_name)
    if not os.path.exists(file_path):
        file_path = os.path.join(root, 'model_pkl', file_name)
    with open(file_path, 'rb') as f:
        gtyp = pickle.load(f)
    return gtyp


# Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

def model_converter(root, file_name):
    file_path = os.path.join(root, 'pre_train_models', file_name)
    if not os.path.exists(file_path):
        file_path = os.path.join(root, 'model_pkl', file_name)
    with open(file_path, 'rb') as f:
        gtyp = pickle.load(f)
        print(type(gtyp))
        print(gtyp)
        gtyp_new = Genotype(normal=gtyp.normal,
                            normal_concat=gtyp.normal_concat,
                            reduce=gtyp.reduce,
                            reduce_concat=gtyp.reduce_concat)
        print(type(gtyp_new))
        print(gtyp_new)
        save_path = os.path.join(root, 'new', file_name)
        with open(save_path, 'wb') as f:
            pickle.dump(gtyp_new, f)
    return gtyp


if __name__ == '__main__':
    root_path = '/home/albert_wei/Disk_A/train_output_2020/results_models/npenas_100_seed_4_2080Ti/'
    model_key = 'f71d0c392514206d9a036b6b24ae5eb9f517c0ae9e0eebcd012f594567e724df.pkl'
    gtyp_new = model_converter(root_path, model_key)
