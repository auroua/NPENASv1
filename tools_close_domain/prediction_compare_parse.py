import pickle
import numpy as np
import argparse


def load_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    keys = list(data.keys())
    total_data_train_all = []
    total_data_test_all = []
    total_data_kl_all = []
    for k in keys:
        total_data_train = []
        total_data_test = []
        total_data_kl = []
        algo_data = data[k]
        for ad in algo_data:
            total_data_train.append(ad[0])
            total_data_test.append(ad[1])
            total_data_kl.append(ad[2])
        total_data_train_all.append(total_data_train)
        total_data_test_all.append(total_data_test)
        total_data_kl_all.append(total_data_kl)
    return keys, np.array(total_data_train_all), np.array(total_data_test_all), np.array(total_data_kl_all)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args for predictor compare.')
    parser.add_argument('--save_path', type=str,
                        default='/tools_close_domain/results/prediction_compare_results_150_case2.pkl',
                        help='Generated file path.')
    args = parser.parse_args()

    keys, d_train, d_test, d_kl = load_file(args.save_path)
    for idx, k in enumerate(keys):
        train_d = d_train[idx, :]
        test_d = d_test[idx, :]
        kl_d = d_kl[idx, :]
        print(k)
        print(f'Tra mean {np.mean(train_d)}, std {np.std(train_d)}')
        print(f'Tes mean {np.mean(test_d)}, std {np.std(test_d)}')
        print(f'KL mean {np.mean(kl_d)}, std {np.std(kl_d)}')