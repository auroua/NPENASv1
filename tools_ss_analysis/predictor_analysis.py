import os
import pickle
from collections import defaultdict, Counter
import numpy as np
import argparse
import matplotlib.pyplot as plt
from hashlib import sha256
import networkx as nx
from nas_lib.visualize.visualize_close_domain_reverse import draw_plot_nasbench_nlp, draw_plot_nasbench_asr, \
    draw_plot_nasbench_201, draw_plot_nasbench_101
from nas_lib.data.data_nasbench_nlp import adj_distance as adj_dist_nlp


# model_lists_nasbench = ['Random', 'EA', 'BANANAS', 'BANANAS_F', 'NPENAS-BO', 'NPENAS-NP', 'NPENAS-NP-NEW', 'ORACLE']
# model_masks_nasbench = [True, True, True, True, True, True, True, True]

# model_lists_nasbench = ['Random', 'REA', 'BANANAS', 'NPENAS-NP', 'ORACLE']
# model_masks_nasbench = [True, True, True, True, True]

# model_lists_nasbench = ['NPENAS-NP-10', 'NPENAS-NP-100', 'ORACLE']
# model_masks_nasbench = [True, True, False]

model_lists_nasbench = ['RELU', 'CELU', 'NPENAS-GT']
model_masks_nasbench = [True, True, False]


# model_lists_nasbench = ['NPENAS-NP-10']
# model_masks_nasbench = [True]

options = {"node_size": 25, "alpha": 0.8}

predictor_values = {0: 'RELU', 1: 'CELU'}
# predictor_values = {0: 'SCALING FACTOR=10', 1: 'SCALING FACTOR=100'}


def get_kt_list(all_files):
    total_results = defaultdict(list)
    total_top_results = defaultdict(list)
    for f in all_files:
        with open(f, 'rb') as fb:
            full_data = pickle.load(fb)[2]
            for idx, data in enumerate(full_data):
                type = data['type']
                if type == 'rea' or type == 'oracle':
                    pass
                else:
                    type, final_data, kt_list, kt_top_list, mutate_list = data['type'], data['final_data'], \
                                                             data['kt_list'], data['kt_top_list'], data['mutate_list']
                    # type_new = type.replace('_', '-').upper()
                    type_new = ''
                    # total_results[f'{type_new}{predictor_values[idx]}'].append(kt_list)
                    total_results[f'{type_new}{idx}'].append(kt_list)
                    total_top_results[f'{type_new}{idx}'].append(kt_top_list)
                    # total_results[type].append(kt_list)
    return total_results, total_top_results


def mutate_rea_information(all_files):
    total_results = defaultdict(list)
    for f in all_files:
        G = nx.Graph()
        cmap = plt.cm.YlGn
        cmap2 = plt.cm.autumn
        with open(f, 'rb') as fb:
            full_data = pickle.load(fb)[2]
        p_dict = {}
        c_dict = defaultdict(list)
        for data in full_data:
            type = data['type']
            if type == 'rea':
                p_list, c_list = data['p_list'], data['c_list']
                for idx, p in enumerate(p_list):
                    p_info = sha256(str(p[6]).encode('utf-8')).hexdigest()
                    p_dict[p_info] = p
                    c_dict[p_info].append(c_list[idx])
            key_list = list(c_dict.keys())
            p_key_list = []
            c_key_list = []
            p_performance_list = []
            c_performance_list = []
            edge_list = []
            edge_color_list = []
            dist_list = []
            p_key_dict = {}
            c_key_dict = {}
            counter = len(key_list)
            for idx, k in enumerate(key_list):
                p_key_dict[k] = idx
                for idx, c in enumerate(c_dict[k]):
                    c_k = sha256(str(c[6]).encode('utf-8')).hexdigest()
                    c_key_dict[c_k] = counter + idx
                    dist = adj_dist_nlp(([], p_dict[k][1], p_dict[k][2]), ([], c[1], c[2]))
                    dist_list.append(dist)
                counter = counter + len(c_dict[k])
            for idx, k in enumerate(key_list):
                parent = p_dict[k]
                children_list = c_dict[k]
                if len(children_list) >= 1:
                    draw_graph(parent, children_list, p_key_list, c_key_list,
                               p_key_dict, c_key_dict, edge_list, edge_color_list, p_performance_list,
                               c_performance_list, G)
        parent_node_list = []
        parent_node_color = []
        for k1 in key_list:
            for k2 in key_list:
                if k1 != k2:
                    pair = (p_key_dict[k1], p_key_dict[k2])
                    if pair not in parent_node_list:
                        parent_node_list.append(pair)
                        dist = adj_dist_nlp(([], p_dict[k1][1], p_dict[k1][2]), ([], p_dict[k2][1], p_dict[k2][2]))
                        parent_node_color.append(dist)
                        dist_list.append(dist)
        min_dist, max_dist = min(dist_list), max(dist_list)
        p_min, p_max = min(p_performance_list), max(p_performance_list)
        c_min, c_max = min(c_performance_list), max(c_performance_list)
        node_min, node_max = min(p_min, c_min), max(p_max, c_max)
        pos = nx.spring_layout(G)  # positions for all nodes
        nx.draw_networkx_nodes(G, pos, nodelist=p_key_list, node_color=p_performance_list, cmap=cmap2,
                               vmin=node_min, vmax=node_max, alpha=0.8, node_size=50)
        nx.draw_networkx_nodes(G, pos, nodelist=c_key_list, node_color=c_performance_list, cmap=cmap2, alpha=0.8,
                               vmin=node_min, vmax=node_max, node_size=10)
        nx.draw_networkx_edges(G, pos, edgelist=edge_list, edge_color=edge_color_list, edge_vmin=min_dist,
                               edge_vmax=max_dist, edge_cmap=cmap)
        # nx.draw_networkx_edges(G, pos, edgelist=parent_node_list, edge_color=parent_node_color, edge_vmin=min_dist,
        #                        edge_vmax=max_dist, edge_cmap=cmap)
        # nx.draw(G, with_labels=False, font_weight='bold', node_size=3)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_dist, vmax=max_dist))
        sm2 = plt.cm.ScalarMappable(cmap=cmap2, norm=plt.Normalize(vmin=node_min, vmax=node_max))
        sm._A = []
        sm2._A = []
        plt.colorbar(sm)
        plt.colorbar(sm2)
        plt.show()
    return total_results


def mutate_information(all_files):
    total_results = defaultdict(list)
    for f in all_files:
        G = nx.Graph()
        cmap = plt.cm.YlGn
        with open(f, 'rb') as fb:
            full_data = pickle.load(fb)[2]
        p_dict = {}
        c_dict = defaultdict(list)
        for data in full_data:
            type = data['type']
            if type == 'rea' or type == 'oracle':
                continue
            else:
                type, final_data, kt_list, mutate_list = data['type'], data['final_data'], \
                                                         data['kt_list'], data['mutate_list']
                print(mutate_list)
    return total_results


def draw_graph(parent, child_list, p_key_list, c_key_list, p_key_dict, c_key_dict,
               edge_list, edge_color_list, p_performance_list,
               c_performance_list, graph):
    # add parent node
    p_key = sha256(str(parent[6]).encode('utf-8')).hexdigest()
    graph.add_node(p_key_dict[p_key])
    p_key_list.append(p_key_dict[p_key])
    p_performance_list.append(parent[4])
    for c in child_list:
        c_k = sha256(str(c[6]).encode('utf-8')).hexdigest()
        c_key_list.append(c_key_dict[c_k])
        graph.add_node(c_key_dict[c_k])
        graph.add_edge(p_key_dict[p_key], c_key_dict[c_k])
        edge_list.append((p_key_dict[p_key], c_key_dict[c_k]))
        dist = adj_dist_nlp(([], parent[1], parent[2]), ([], c[1], c[2]))
        edge_color_list.append(dist)
        c_performance_list.append(c[4])


def visual_kt_list(kt_results, results_folder, search_space, comparison_type="relu_celu", train_dataset="cifar100"):
    if comparison_type == "relu_celu":
        model_lists_nasbench = ['RELU', 'CELU', 'NPENAS-GT']
        model_masks_nasbench = [True, True, False]
    elif comparison_type == "scale_factor":
        model_lists_nasbench = ['SCALING FACTOR=#', 'SCALING FACTOR=*', 'NPENAS-GT']
        model_masks_nasbench = [True, True, False]
    else:
        raise ValueError(f"comparison type {comparison_type} does not support at present!")
    if comparison_type == "scale_factor":
        rate = get_rate(results_folder)
        key1 = "SCALING FACTOR=" + str(int(rate[0]))
        key2 = "SCALING FACTOR=" + str(int(rate[1]))
    kt_results_dict = {}
    kt_results_std_dict = {}

    for k, v in kt_results.items():
        np_v = np.nan_to_num(np.array(v))
        kt_results_dict[k] = np.mean(np_v, axis=0)
        kt_results_std_dict[k] = np.std(np_v, axis=0)

    if comparison_type == "scale_factor":
        kt_results_dict[key1] = kt_results_dict["0"]
        kt_results_dict[key2] = kt_results_dict["1"]
        kt_results_dict.pop("0")
        kt_results_dict.pop("1")

    if comparison_type == "relu_celu":
        kt_results_dict["RELU"] = kt_results_dict["0"]
        kt_results_dict["CELU"] = kt_results_dict["1"]
        kt_results_dict.pop("0")
        kt_results_dict.pop("1")

    if args.search_space == "nasbench_101":
        idx = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140])
    else:
        idx = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
    fig, ax = plt.subplots(1)
    for k, v in kt_results_dict.items():
        plt.plot(idx, v, label=k, marker='s', linewidth=1, ms=3)
    ax.grid(False)
    fig.set_dpi(600.0)
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Kendall Tau Correlation')
    plt.legend(loc='upper left')
    # plt.legend(loc='upper right')
    plt.show()
    if search_space == 'nasbench_nlp':
        draw_plot_nasbench_nlp(results_folder, draw_type='ERRORBAR', model_lists=model_lists_nasbench,
                               model_masks=model_masks_nasbench, order=False, comparison_type=comparison_type)
    elif search_space == 'nasbench_asr':
        draw_plot_nasbench_asr(results_folder, draw_type='ERRORBAR', model_lists=model_lists_nasbench,
                               model_masks=model_masks_nasbench, order=False, comparison_type=comparison_type)
    elif search_space == 'nasbench_201':
        # cifar10-valid, cifar100, ImageNet16-120
        draw_plot_nasbench_201(results_folder, draw_type='ERRORBAR', model_lists=model_lists_nasbench,
                               model_masks=model_masks_nasbench, train_data=train_dataset, order=False,
                               comparison_type=comparison_type)
    elif search_space == 'nasbench_101':
        pass


def get_rate(folder):
    files = os.listdir(folder)
    file_name = ""
    for f in files:
        if "full" not in f and "log" not in f:
            file_name = f
            break
    file_path = os.path.join(folder, file_name)
    with open(file_path, "rb") as fs:
        algorithm_params, metann_params, results, walltimes = pickle.load(fs)
    rate = [algorithm_params[0]["rate"], algorithm_params[1]["rate"]]
    return rate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args for visualize darts architecture')
    parser.add_argument('--search_space', type=str, default='nasbench_201',
                        choices=['nasbench_101', 'nasbench_201', 'nasbench_nlp', 'nasbench_asr'],
                        help='The algorithm output folder')
    parser.add_argument('--comparison_type', type=str, default='relu_celu',
                        choices=['scale_factor', 'relu_celu'],
                        help='The algorithm output folder')
    parser.add_argument('--save_dir', type=str,
                        default='/home/albert_wei/Desktop/results/relu_celu/npenas_nasbench_201_cifar10_relu_celu/',
                        help='The algorithm output folder')
    parser.add_argument('--train_data', type=str, default='cifar10-valid',
                        choices=['cifar10-valid', 'cifar100', 'ImageNet16-120'],
                        help='The evaluation of dataset of NASBench-201.')
    args = parser.parse_args()

    all_files = [os.path.join(args.save_dir, f) for f in os.listdir(args.save_dir) if 'full' in f]
    kt_total_results, kt_total_top_results = get_kt_list(all_files)
    visual_kt_list(kt_total_results, args.save_dir, args.search_space, comparison_type=args.comparison_type,
                   train_dataset=args.train_data)
    # visual_kt_list(kt_total_top_results, args.save_dir, args.search_space)

    # mutate_rea_information(all_files)
    # mutate_information(all_files)