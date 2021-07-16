# Copyright (c) XiDian University and Xi'an University of Posts&Telecommunication. All Rights Reserved

import os
import sys
sys.path.append(os.getcwd())
import argparse
from nas_lib.configs import nas_bench_201_converted_base_path
from nas_lib.data.init_nasbench_201_dataset import inti_nasbench_201


parser = argparse.ArgumentParser(description='Args for algorithms compare.')
parser.add_argument('--dataset', type=str, default='cifar10-valid',
                    choices=['cifar10-valid', 'cifar100', 'ImageNet16-120'], help='Which dataset to convert')
args = parser.parse_args()

save_path = nas_bench_201_converted_base_path % args.dataset
inti_nasbench_201(args.dataset, save_path)