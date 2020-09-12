import pickle
import os
import numpy as np


def parse_single_macro_graph(single_model):
    with open(single_model, 'rb') as f:
        models = pickle.load(f)
        original_model = pickle.load(f)
        hash_key = pickle.load(f)
        train_loss = pickle.load(f)
        val_acc = pickle.load(f)
        test_acc = pickle.load(f)
        best_val_acc = pickle.load(f)
    return [models, original_model, hash_key, train_loss, val_acc, test_acc, best_val_acc]


def parse_single_darts_macro_graph(single_model):
    with open(single_model, 'rb') as f:
        genotype = pickle.load(f)
        models = pickle.load(f)
        hash_key = pickle.load(f)
        train_loss = pickle.load(f)
        val_acc = pickle.load(f)
        test_acc = pickle.load(f)
        best_val_acc = pickle.load(f)
        loss_list = pickle.load(f)
        val_acc_list = pickle.load(f)
    return [genotype, models, hash_key, train_loss, val_acc, test_acc, best_val_acc, loss_list, val_acc_list]


def single_model_trans_cpu(single_model, new_path):
    with open(single_model, 'rb') as f:
        models = pickle.load(f)
        original_model = pickle.load(f)
        hash_key = pickle.load(f)
        train_loss = pickle.load(f)
        val_acc = pickle.load(f)
        test_acc = pickle.load(f)
        best_val_acc = pickle.load(f)

    with open(os.path.join(new_path, single_model.split('/')[-1]), 'wb') as f:
        pickle.dump(models.to('cpu'), f)
        pickle.dump(original_model, f)
        pickle.dump(hash_key, f)
        pickle.dump(train_loss, f)
        pickle.dump(val_acc, f)
        pickle.dump(test_acc, f)
        pickle.dump(best_val_acc, f)


def parse_macro_graph(models_path):
    models = os.listdir(models_path)
    full_models = [os.path.join(models_path, m) for m in models]
    for fm in full_models:
        data = parse_single_macro_graph(fm)


def parse_darts_macro_graph(models_path):
    models = os.listdir(models_path)
    full_models = [os.path.join(models_path, m) for m in models]
    best_acc = 0
    hash_key = None
    total_accs = []
    total_avg_accs = []
    total_model_keys = []
    for fm in full_models:
        data = parse_single_darts_macro_graph(fm)
        test_acc = data[6]
        avg_val_acc = sum(data[8][-5:])/len(data[8][-5:])
        avg_val_acc = (test_acc + avg_val_acc)/2
        total_accs.append(test_acc)
        total_avg_accs.append(avg_val_acc)
        total_model_keys.append(data[2])
        if test_acc > best_acc:
            best_acc = test_acc
            hash_key = data[2]
    idxs = np.argsort(np.array(total_avg_accs)).tolist()[::-1]
    print([total_accs[k] for k in idxs[:10]])
    print([total_avg_accs[k] for k in idxs[:10]])
    print([total_model_keys[k] for k in idxs[:10]])
    # print(best_acc, hash_key)


def parse_multiple_macro_graph(models_path, new_path):
    models = os.listdir(models_path)
    full_models = [os.path.join(models_path, m) for m in models]
    for fm in full_models:
        single_model_trans_cpu(fm, new_path)


def load_pretraining_data(model_path):
    total_models = []
    val_acc_list = []
    models = os.listdir(model_path)
    for m in models:
        m_path = os.path.join(model_path, m)
        with open(m_path, 'rb') as f:
            _ = pickle.load(f)
            original_m = pickle.load(f)
            _ = pickle.load(f)
            _ = pickle.load(f)
            _ = pickle.load(f)
            _ = pickle.load(f)
            best_val_acc = pickle.load(f)
            total_models.append(original_m)
            val_acc_list.append(100-best_val_acc)
    return total_models, val_acc_list


if __name__ == '__main__':
    models_path = '/home/aurora/data_disk_new/train_output_2020_02/datrs_guassian_gin_outputs/model_pkl/'
    # new_models_path = '/home/albert_wei/Disk_A/train_output/gnnp_output1/model_pkl/'
    parse_darts_macro_graph(models_path)
    # parse_multiple_macro_graph(models_path, new_models_path)