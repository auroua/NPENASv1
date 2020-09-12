import os
import sys
import argparse
sys.path.append(os.getcwd())
from nas_lib.utils.utils_parse import parse_darts_macro_graph


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args for rank searched architectures.')
    parser.add_argument('--model_path', type=str, default='', help='darts')
    args = parser.parse_args()

    parse_darts_macro_graph(args.model_path)