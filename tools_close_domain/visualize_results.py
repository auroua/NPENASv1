import os
import sys
import argparse
sys.path.append(os.getcwd())
from nas_lib.visualize.visualize_close_domain_reverse import draw_plot_nasbench_101, draw_plot_nasbench_201, \
    draw_plot_priori_scalar, draw_plot_evaluation_compare, draw_plot_nasbench_nlp, draw_plot_nasbench_asr


model_lists_nasbench = ['Random', 'REA', 'BO w. GP', 'NASBOT', 'BANANAS-PE', 'BANANAS-AE', 'NPENAS-BO', 'NPENAS-NP',
                        'NPENAS-GT']
model_masks_nasbench = [True, True, True, True, True, True, True, True, True]

model_lists_scalar = ['scalar_10', 'scalar_30', 'scalar_50', 'scalar_70', 'scalar_100']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args for visualize darts architecture')
    parser.add_argument('--search_space', type=str, default='nasbench_asr',
                        choices=['nasbench_101', 'nasbench_201', 'scalar_prior', 'evaluation_compare',
                                 'nasbench_nlp', 'nasbench_asr'],
                        help='The algorithm output folder')
    parser.add_argument('--save_dir', type=str,
                        default='/media/albert_wei/HP_SSD/Backups/Papers_Backup#####/npenas/backups_npenas_paper_2021_7_14/result_files/npenas_results/npenas_nasbench_asr/',
                        help='The algorithm output folder')
    parser.add_argument('--train_data', type=str, default='ImageNet16-120',
                        choices=['cifar10-valid', 'cifar100', 'ImageNet16-120'],
                        help='The evaluation of dataset of NASBench-201.')
    parser.add_argument('--draw_type', type=str, default='MEANERROR', choices=['ERRORBAR', 'MEANERROR'],
                        help='Draw result with or without errorbar.')
    parser.add_argument('--show_all', type=str, default='1', help='Weather to show all results.')

    args = parser.parse_args()
    if args.search_space == 'nasbench_101':
        draw_plot_nasbench_101(args.save_dir, draw_type=args.draw_type, model_lists=model_lists_nasbench,
                               model_masks=model_masks_nasbench, order=True)
    elif args.search_space == 'nasbench_201':
        draw_plot_nasbench_201(args.save_dir, draw_type=args.draw_type, model_lists=model_lists_nasbench,
                               model_masks=model_masks_nasbench, train_data=args.train_data, order=True)
    elif args.search_space == 'scalar_prior':
        if args.show_all == '1':
            model_lists_scalar_mask = [True, True, True, True, True]
        else:
            model_lists_scalar_mask = [False, True, False, True, False]
        draw_plot_priori_scalar(args.save_dir, draw_type=args.draw_type, model_lists=model_lists_scalar,
                                model_masks=model_lists_scalar_mask)
    elif args.search_space == 'evaluation_compare':
        model_lists = ['1', '10', '20', '30']
        if args.show_all == '1':
            model_masks = [True, True, True, True]
            draw_plot_evaluation_compare(args.save_dir, draw_type=args.draw_type, model_lists=model_lists,
                                         model_masks=model_masks)
        else:
            model_masks = [True, False, False, True]
            draw_plot_evaluation_compare(args.save_dir, draw_type=args.draw_type, model_lists=model_lists,
                                         model_masks=model_masks)
    elif args.search_space == 'nasbench_nlp':
        draw_plot_nasbench_nlp(args.save_dir, draw_type=args.draw_type, model_lists=model_lists_nasbench,
                               model_masks=model_masks_nasbench, order=True)
    elif args.search_space == 'nasbench_asr':
        draw_plot_nasbench_asr(args.save_dir, draw_type=args.draw_type, model_lists=model_lists_nasbench,
                               model_masks=model_masks_nasbench, order=True)
    else:
        raise ValueError('This search space does not support!')