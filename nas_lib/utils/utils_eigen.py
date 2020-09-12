import numpy as np
from nas_lib.data.cell import Cell, OPS
import random
import sys


np.set_printoptions(threshold=sys.maxsize)


def inference_all_archs(models, total_archs, agent_type, logger, upper_bound, nnp_arch_data_edge_idx_dict,
                        nnp_arch_data_node_f_dict, nnp_arch_data_edge_idx_reverse_dict,
                        nnp_arch_data_node_f_reverse_dict, model_keys=None):
    total_keys = []
    total_pred_results = []

    arch_data_edge_idx_list = []
    arch_data_node_f_list = []
    arch_data_edge_idx_reverse_list = []
    arch_data_node_f_reverse_list = []

    best_val_acc_list = []
    best_test_acc_list = []
    best_val_key_list = []
    for idx, (k, v) in enumerate(total_archs.items()):
        total_keys.append(k)

        arch_data_edge_idx_list.append(nnp_arch_data_edge_idx_dict[k])
        arch_data_node_f_list.append(nnp_arch_data_node_f_dict[k])
        arch_data_edge_idx_reverse_list.append(nnp_arch_data_edge_idx_reverse_dict[k])
        arch_data_node_f_reverse_list.append(nnp_arch_data_node_f_reverse_dict[k])

        if len(arch_data_edge_idx_list) == 1000:
            pred = models.pred(arch_data_edge_idx_list, arch_data_node_f_list, arch_data_edge_idx_reverse_list,
                               arch_data_node_f_reverse_list)
            total_pred_results.extend(pred.cpu().numpy().tolist())
            # logger.info('Neural Predictor %s Total iterations is %d' % (agent_type, idx))
            arch_data_edge_idx_list = []
            arch_data_node_f_list = []
            arch_data_edge_idx_reverse_list = []
            arch_data_node_f_reverse_list = []
    if len(arch_data_edge_idx_list) != 0:
        pred = models.pred(arch_data_edge_idx_list, arch_data_node_f_list, arch_data_edge_idx_reverse_list,
                           arch_data_node_f_reverse_list)
        total_pred_results.extend(pred.cpu().numpy().tolist())

    ########### Calculate pred loss ###############
    total_pred_error_dict = {}
    total_val_acc_dict = {}
    total_pred_acc_dict = {}
    for idx, k in enumerate(total_keys):
        val_acc = total_archs[k]['val']
        total_val_acc_dict[k] = 100*val_acc
        total_pred_acc_dict[k] = total_pred_results[idx]
        total_pred_error_dict[k] = abs(val_acc*100 - total_pred_results[idx])
    logger.info('###############  pred and val error #####################' * 3)
    ordered_idxes = np.array(total_pred_results).argsort()
    ordered_idxes = ordered_idxes[::-1]
    for i in range(1, upper_bound):
        best_val_range = []
        best_test_range = []
        key_in_train_list = []
        idxes = ordered_idxes[:i]
        for k in idxes:
            best_val_range.append(total_archs[total_keys[k]]['val'])
            best_test_range.append(total_archs[total_keys[k]]['test'])
            key_in_train_list.append(total_keys[k])
        # best_val_acc_list.append(max(best_val_range))
        # best_test_acc_list.append(max(best_test_range))
        max_val_acc = max(best_val_range)
        max_idx = best_val_range.index(max_val_acc)
        max_test_acc = best_test_range[max_idx]
        best_key = key_in_train_list[max_idx]
        best_val_acc_list.append(max_val_acc)
        best_test_acc_list.append(max_test_acc)
        best_val_key_list.append(best_key)

    return best_val_acc_list, best_test_acc_list, best_val_key_list, total_pred_error_dict, \
           total_pred_acc_dict, total_val_acc_dict


def inference_all_archs_bananas(models, total_archs, logger, agent_type, total_keys):
    total_pred_results = []
    encoding_list = []
    total_keys = total_keys
    for idx, k in enumerate(total_keys):
        v = total_archs[k].copy()
        for op_idx, op in enumerate(v['ops']):
            if op == 'isolate':
                # op_sample_idx = random.randint(0, 2)
                v['ops'][op_idx] = 'maxpool3x3'
        arch = {
            'matrix': v['matrix'],
            'ops': v['ops'],
            'isolate_node_idxs': []
        }
        if 'true' in agent_type:
            encoding = Cell(**arch).encode_paths()
        else:
            del arch['isolate_node_idxs']
            encoding = Cell(**arch).encode_cell()
        encoding_list.append(encoding)

        if len(encoding_list) == 1000:
            pred = np.squeeze(models.predict(np.array(encoding_list))).tolist()
            total_pred_results.extend(pred)
            encoding_list = []
    if len(encoding_list) != 0:
        pred = np.squeeze(models.predict(np.array(encoding_list))).tolist()
        total_pred_results.extend(pred)
    return total_pred_results


def inference_all_archs_uncertainty(models, total_archs, total_keys, logger, uncertainty_arch_data_edge_idx_dict,
                                    uncertainty_arch_data_node_f_dict):
    total_pred_results = []
    total_val_acc = []
    arch_data_edge_idx_list = []
    arch_data_node_f_list = []
    total_keys = total_keys
    total_key_pred_error_dict = {}
    for idx, k in enumerate(total_keys):
        arch_data_edge_idx_list.append(uncertainty_arch_data_edge_idx_dict[k])
        arch_data_node_f_list.append(uncertainty_arch_data_node_f_dict[k])

        if len(arch_data_edge_idx_list) == 1000:
            pred = models.pred(arch_data_edge_idx_list, arch_data_node_f_list)
            total_pred_results.extend(pred.cpu().numpy().tolist())
            arch_data_edge_idx_list = []
            arch_data_node_f_list = []
            # logger.info('total inference idxes is %d' % idx)
    if len(arch_data_edge_idx_list) != 0:
        pred = models.pred(arch_data_edge_idx_list, arch_data_node_f_list)
        total_pred_results.extend(pred.cpu().numpy().tolist())
    for idx, k in enumerate(total_keys):
        val_acc = total_archs[k]['val']
        total_val_acc.append(val_acc)
        total_key_pred_error_dict[k] = abs(100 - val_acc*100 - total_pred_results[idx])
    return total_pred_results, total_key_pred_error_dict, total_val_acc