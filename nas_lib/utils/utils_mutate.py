import numpy as np


def mutate_count_list(lower_query, upper_query, lower, upper, total_count):
    query_list = np.linspace(lower_query, upper_query, total_count).tolist()
    mutate_count = np.linspace(upper, lower, total_count)
    mutate_count_list = mutate_count.tolist()
    mutate_int_str = {str(int(query_list[idx])): int(c) for idx, c in enumerate(mutate_count_list)}
    return mutate_int_str


def mutate_rate(mutate_count):
    mutate_rate = np.exp([0.2*i for i in range(mutate_count)]).tolist()
    # mutate_rate_list = list(map(lambda i : min(5, i), mutate_rate))
    return mutate_rate


def category_dist(dist, val_acc, total_connect_dist, total_node_dist):
    connect_encode = dist[:21]
    for idx, val in enumerate(connect_encode):
        total_connect_dist[idx, val] += 1
        total_connect_dist[idx, 2] += val_acc

    node1 = np.argmax(np.array(dist[21:25]))
    total_node_dist[0, node1] += 1
    total_node_dist[0, 4] += val_acc
    node2 = np.argmax(np.array(dist[25:29]))
    total_node_dist[1, node2] += 1
    total_node_dist[1, 4] += val_acc
    node3 = np.argmax(np.array(dist[29:33]))
    total_node_dist[2, node3] += 1
    total_node_dist[2, 4] += val_acc
    node4 = np.argmax(np.array(dist[33:37]))
    total_node_dist[3, node4] += 1
    total_node_dist[3, 4] += val_acc
    node5 = np.argmax(np.array(dist[37:]))
    total_node_dist[4, node5] += 1
    total_node_dist[4, 4] += val_acc


def arch_distribution(data):
    total_connect_dist = np.zeros((21, 3))
    total_node_dist = np.zeros((5, 5))
    for d in data:
        distance = d[7]
        val_acc = d[4]
        category_dist(distance, val_acc, total_connect_dist, total_node_dist)
    print(total_connect_dist.shape)
    print(total_node_dist.shape)


class NodeAccDist:
    def __init__(self):
        self.node_dict = {}
        self.node_dict[str(0)] = []
        self.node_dict[str(1)] = []
        self.node_dict[str(2)] = []
        self.node_dict[str(3)] = []

        self.node_count_dict = {}
        self.node_count_dict[str(0)] = 0
        self.node_count_dict[str(1)] = 0
        self.node_count_dict[str(2)] = 0
        self.node_count_dict[str(3)] = 0

    def add_val(self, node, val):
        self.node_dict[str(node)].append(val)

    def gen_distribution(self):
        pass

    def sample_node(self):
        pass


def arch_acc_distribution(data):
    arch_acc_dict = {}
    arch_count_dict = {}
    node1_dict = NodeAccDist()
    node2_dict = NodeAccDist()
    node3_dict = NodeAccDist()
    node4_dict = NodeAccDist()
    node5_dict = NodeAccDist()

    for i in range(21):
        arch_acc_dict[str(i)] = []
        arch_count_dict[str(i)] = 0
    for d in data:
        val_acc = d[4]
        distance = d[7]
        for idx, val in enumerate(distance[:21]):
            if val == 1:
                arch_acc_dict[str(idx)].append(val_acc)
                arch_count_dict[str(idx)] += 1
        node1 = np.argmax(np.array(distance[21:25]))
        node1_dict.node_dict[str(node1)].append(val_acc)
        node1_dict.node_count_dict[str(node1)] += 1

        node2 = np.argmax(np.array(distance[25:29]))
        node2_dict.node_dict[str(node2)].append(val_acc)
        node2_dict.node_count_dict[str(node2)] += 1

        node3 = np.argmax(np.array(distance[29:33]))
        node3_dict.node_dict[str(node3)].append(val_acc)
        node3_dict.node_count_dict[str(node3)] += 1

        node4 = np.argmax(np.array(distance[33:37]))
        node4_dict.node_dict[str(node4)].append(val_acc)
        node4_dict.node_count_dict[str(node4)] += 1

        node5 = np.argmax(np.array(distance[37:]))
        node5_dict.node_dict[str(node5)].append(val_acc)
        node5_dict.node_count_dict[str(node5)] += 1
    return arch_acc_dict, arch_count_dict, node1_dict, node2_dict, node3_dict, node4_dict, node5_dict


def analysis_archs(distance_matrix):
    for i in range(distance_matrix):
        pass


INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
ISOLATE = 'isolate'


def init_total_dict(total_dict):
    for i in range(1, 6):
        for j in range(0, 3):
            total_dict['0_%d_%d' % (i, j)] = []
    total_dict['0_%d_%s' % (6, 'out')] = []
    for i in range(2, 6):
        for k in range(0, 3):
            for j in range(0, 3):
                total_dict['1_%d_%d_%d' % (i, k, j)] = []
    for k in range(0, 3):
        total_dict['1_%d_%d_%s' % (6, k, 'out')] = []
    for i in range(3, 6):
        for k in range(0, 3):
            for j in range(0, 3):
                total_dict['2_%d_%d_%d' % (i, k, j)] = []
    for k in range(0, 3):
        total_dict['2_%d_%d_%s' % (6, k, 'out')] = []
    for i in range(4, 6):
        for k in range(0, 3):
            for j in range(0, 3):
                total_dict['3_%d_%d_%d' % (i, k, j)] = []
    for k in range(0, 3):
        total_dict['3_%d_%d_%s' % (6, k, 'out')] = []
    for i in range(5, 6):
        for k in range(0, 3):
            for j in range(0, 3):
                total_dict['4_%d_%d_%d' % (i, k, j)] = []
    for k in range(0, 3):
        total_dict['4_%d_%d_%s' % (6, k, 'out')] = []
    for k in range(0, 3):
        total_dict['5_%d_%d_%s' % (6, k, 'out')] = []
    # add isolate case
    total_dict['1_isolate'] = []
    total_dict['2_isolate'] = []
    total_dict['3_isolate'] = []
    total_dict['4_isolate'] = []
    total_dict['5_isolate'] = []


def arch_chip_distribution(data):
    # chips_seg = [18, 45, 36, 27, 18, 3]
    # OPS = [CONV3X3, CONV1X1, MAXPOOL3X3, ISOLATE]
    total_arch_dict = {}
    init_total_dict(total_arch_dict)
    total_val = []
    for d in data:
        val_acc = d[4]
        distance = d[7]
        total_val.append(val_acc)
        node1 = np.argmax(np.array(distance[21:25]))
        if node1 == 3:
            total_arch_dict['1_isolate'].append(val_acc)
        node2 = np.argmax(np.array(distance[25:29]))
        if node2 == 3:
            total_arch_dict['2_isolate'].append(val_acc)
        node3 = np.argmax(np.array(distance[29:33]))
        if node3 == 3:
            total_arch_dict['3_isolate'].append(val_acc)
        node4 = np.argmax(np.array(distance[33:37]))
        if node4 == 3:
            total_arch_dict['4_isolate'].append(val_acc)
        node5 = np.argmax(np.array(distance[37:]))
        if node5 == 3:
            total_arch_dict['5_isolate'].append(val_acc)
        node_list = [node1, node2, node3, node4, node5, 'output']
        for idx, val in enumerate(distance[:21]):
            if val == 1:
                if 0 <= idx <= 5:
                    if idx == 5:
                        position = '0_%d_%s' % (idx+1, 'out')
                    else:
                        if int(node_list[idx]) == 3:
                            continue
                        position = '0_%d_%d' % (idx+1, int(node_list[idx]))
                elif 6 <= idx <= 10:
                    if node1 == 3:
                        continue
                    if idx == 10:
                        position = '1_%d_%d_%s' % (idx-4, int(node1), 'out')
                    else:
                        if int(node_list[idx-5]) == 3:
                            continue
                        position = '1_%d_%d_%d' % (idx-4, int(node1), int(node_list[idx-5]))
                elif 11 <= idx <= 14:
                    if int(node2) == 3:
                        continue
                    if idx == 14:
                        position = '2_%d_%d_%s' % (idx-8, int(node2), 'out')
                    else:
                        if int(node_list[idx-9]) == 3:
                            continue
                        position = '2_%d_%d_%d' % (idx-8, int(node2), int(node_list[idx-9]))
                elif 15 <= idx <= 17:
                    if int(node3) == 3:
                        continue
                    if idx == 17:
                        position = '3_%d_%d_%s' % (idx-11, int(node3), 'out')
                    else:
                        if int(node_list[idx-12]) == 3:
                            continue
                        position = '3_%d_%d_%d' % (idx-11, int(node3), int(node_list[idx-12]))
                elif 18 <= idx <= 19:
                    if int(node4) == 3:
                        continue
                    if idx == 19:
                        position = '4_%d_%d_%s' % (idx-13, int(node4), 'out')
                    else:
                        if int(node_list[idx-14]) == 3:
                            continue
                        position = '4_%d_%d_%d' % (idx-13, int(node4), int(node_list[idx-14]))
                else:
                    if node5 == 3:
                        continue

                    assert idx == 20, 'the final index should be 20, but present idx value is %d' % idx
                    position = '5_%d_%d_%s' % (6, int(node5), 'out')
                total_arch_dict[position].append(val_acc)
    return total_arch_dict, total_val


def arch_chip_dist_gen(total_arch_dict, total_val):
    total_arch_distribution = {}
    total_val_mean = np.mean(total_val)
    total_val_std = np.std(total_val)
    for k, v in total_arch_dict.items():
        if len(v) == 0:
            if 'isolate' in k:
                total_arch_distribution[k] = (total_val_mean+1.5*total_val_std, max(0, 0.5*total_val_std), 0)
            else:
                total_arch_distribution[k] = (total_val_mean+1.5*total_val_std, 1.5*total_val_std, 0)
        else:
            v_std = np.std(v)
            if v_std == 0:
                std = total_val_std
            else:
                std = v_std
            total_arch_distribution[k] = (np.mean(v), std, 1)
    return total_arch_distribution


if __name__ == '__main__':
    mutate_int_str = mutate_count_list(10, 150, 1, 10, 15)
    for mc in mutate_int_str:
        print(mutate_int_str[mc])
        print(mutate_rate(mutate_int_str[mc]))