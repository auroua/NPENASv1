# Copyright (c) XiDian University and Xi'an University of Posts&Telecommunication. All Rights Reserved

from .cell import Cell


def build_datasets(search_spaces, dataset, nasbench_nlp_type, filter_none):
    if search_spaces == "nasbench_case1":
        from nas_lib.data.data_nasbench import DataNasBench
        return DataNasBench(search_spaces)
    elif search_spaces == "nasbench_case2":
        from nas_lib.data.data_nasbench2 import DataNasBenchNew
        return DataNasBenchNew(search_spaces)
    elif search_spaces == 'nasbench_201':
        from nas_lib.data.data_nasbench_201 import NASBench201
        return NASBench201(dataset)
    elif search_spaces == 'nasbench_data_distribution':
        from nas_lib.data.data_nasbench_101_distributon_analysis import DataNasBenchDist
        return DataNasBenchDist()
    elif search_spaces == 'nasbench_nlp':
        from nas_lib.data.data_nasbench_nlp import DataNasBenchNLP
        return DataNasBenchNLP(perf_type=nasbench_nlp_type)
    elif search_spaces == 'nasbench_asr':
        if filter_none == 'y':
            from nas_lib.data.data_nasbench_ars_wo_none import DataNasBenchASR_WO_None
            return DataNasBenchASR_WO_None()
        else:
            from nas_lib.data.data_nasbench_asr import DataNasBenchASR
            return DataNasBenchASR()
    else:
        raise ValueError("This architecture datasets does not support!")


def generate_train_data(model_keys, total_archs, encode_path, allow_isomorphisms, flag_100, is_bananas=False):
    train_data = []
    for k in model_keys:
        arch = total_archs[k].copy()
        for op_idx, op in enumerate(arch['ops']):
            if op == 'isolate':
                arch['ops'][op_idx] = 'maxpool3x3'
        cell_inst = Cell(matrix=arch['matrix'], ops=arch['ops'])
        if is_bananas:
            if encode_path:
                encoding = cell_inst.encode_paths()
            else:
                encoding = cell_inst.encode_cell()
            train_data.append(({'matrix': arch['matrix'],
                                'ops': arch['ops'],
                                'isolate_node_idxs': []},
                               encoding,
                               arch['val']*100 if flag_100 else arch['val'],
                               arch['test']*100 if flag_100 else arch['test']))
        else:
            train_data.append(({'matrix': arch['matrix'],
                                'ops': arch['ops'],
                                'isolate_node_idxs': []},
                               arch['val']*100 if flag_100 else arch['val'],
                               arch['test']*100 if flag_100 else arch['test']))
    return train_data


def generate_train_data_nasbench_201(model_keys, total_archs, encode_path, allow_isomorphisms, flag_100, is_bananas=False):
    train_data = []
    for k in model_keys:
        arch = total_archs[k].copy()
        train_data.append(({'matrix': arch['matrix'],
                            'ops': arch['ops'],
                            'isolate_node_idxs': []},
                           arch['val'],
                           arch['test']))
    return train_data