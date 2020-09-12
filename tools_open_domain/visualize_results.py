import os
import sys
import argparse
sys.path.append(os.getcwd())
from nas_lib.visualize.visualize_gentype_darts import plot, load_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args for visualize darts architecture')
    parser.add_argument('--model_path', type=str, default='', help='The model path')
    parser.add_argument('--model_name', type=str, default='', help='The model path')
    args = parser.parse_args()

    genotype = load_model(args.model_path)

    plot(genotype.normal, f"normal_{args.model_name}")
    plot(genotype.reduce, f"reduction_{args.model_name}")