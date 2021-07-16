import os
import sys
import argparse
sys.path.append(os.getcwd())
from nas_lib.visualize.visualize_close_domain_reverse import draw_plot_nasbench_101, draw_plot_nasbench_201, \
    draw_plot_priori_scalar, draw_plot_evaluation_compare, draw_plot_nasbench_nlp, draw_plot_nasbench_asr


model_lists_nasbench = ['SCALING FACTOR=5', 'SCALING FACTOR=100', 'NPENAS-GT']
model_masks_nasbench = [True, True, False]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args for visualize darts architecture')
    parser.add_argument('--search_space', type=str, default='nasbench_asr',
                        choices=['nasbench_101', 'nasbench_201', 'scalar_prior', 'evaluation_compare',
                                 'nasbench_nlp', 'nasbench_asr'],
                        help='The algorithm output folder')
    parser.add_argument('--save_dir', type=str,
                        default='/home/albert_wei/Desktop/upload_files/scaling_factor/npenas_nasbench_asr_predictor_compare/',
                        help='The algorithm output folder')
    parser.add_argument('--train_data', type=str, default='cifar100',
                        choices=['cifar10-valid', 'cifar100', 'ImageNet16-120'],
                        help='The evaluation of dataset of NASBench-201.')
    parser.add_argument('--draw_type', type=str, default='ERRORBAR', choices=['ERRORBAR', 'MEANERROR'],
                        help='Draw result with or without errorbar.')
    parser.add_argument('--show_all', type=str, default='1', help='Weather to show all results.')

    args = parser.parse_args()
    if args.search_space == 'nasbench_101':
        draw_plot_nasbench_101(args.save_dir, draw_type=args.draw_type, model_lists=model_lists_nasbench,
                               model_masks=model_masks_nasbench, order=False)
    elif args.search_space == 'nasbench_201':
        draw_plot_nasbench_201(args.save_dir, draw_type=args.draw_type, model_lists=model_lists_nasbench,
                               model_masks=model_masks_nasbench, train_data=args.train_data, order=False)
    elif args.search_space == 'nasbench_nlp':
        draw_plot_nasbench_nlp(args.save_dir, draw_type=args.draw_type, model_lists=model_lists_nasbench,
                               model_masks=model_masks_nasbench, order=False)
    elif args.search_space == 'nasbench_asr':
        draw_plot_nasbench_asr(args.save_dir, draw_type=args.draw_type, model_lists=model_lists_nasbench,
                               model_masks=model_masks_nasbench, order=False)
    else:
        raise ValueError('This search space does not support!')