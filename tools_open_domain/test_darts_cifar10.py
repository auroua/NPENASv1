import argparse
import sys
import logging
import torch
import torch.nn as nn
import numpy as np
import os
sys.path.append(os.getcwd())
from nas_lib.models_darts.datrs_neuralnet import DartsCifar10NeuralNet
from nas_lib.utils.utils_darts import AvgrageMeter, top_accuracy, load_model
import torch.backends.cudnn as cudnn
from ptflops import get_model_complexity_info
from nas_lib.data.cifar10_dataset_retrain import get_cifar10_full_test_loader, transforms_cifar10
from nas_lib.configs import cifar10_path
import random


def infer(valid_queue, model, criterion, device):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    model.eval()
    for step, (input, target) in enumerate(valid_queue):
        input = input.to(device)
        target = target.to(device)

        logits, _ = model(input, device)
        loss = criterion(logits, target)
        prec1, prec5 = top_accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % 100 == 0 and step != 0:
            logging.info('valid %03d %e %.4f %.4f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, top5.avg, objs.avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Searched model for cifar10')
    parser.add_argument('--model_name', type=str, default='ace85b6b1618a4e0ebdc0db40934f2982ac57a34ec9f31dcd8d209b3855dce1f.pkl',
                        help='name of output files')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
    parser.add_argument('--layers', type=int, default=20, help='total number of layers')
    parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
    parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
    parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
    parser.add_argument('--save_dir', type=str,
                        default='/home/albert_wei/Desktop/NPENAS_materials/论文资料/results_models/npubo_results_7_150_seed_22_stem_3/',
                        help='name of save directory')
    parser.add_argument('--model_path', type=str, default='/home/albert_wei/Desktop/NPENAS_materials/论文资料/results_models/npubo_results_7_150_seed_22_stem_3/seed_0/model_best.pth.tar',
                        help='path of pretrained model')
    seed = 111444
    args = parser.parse_args()

    genotype = load_model(args.save_dir, args.model_name)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    CLASSES = 10

    # set seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.enabled = True
    cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DartsCifar10NeuralNet(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype,
                                  args.model_name[:-4], stem_mult=3)
    saved_dict = torch.load(args.model_path)
    model.load_state_dict(saved_dict['state_dict'], strict=False)

    model.drop_path_prob = 0
    with torch.cuda.device(0):
        flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
        logging.info('{:<30}  {:<8}'.format('Computational complexity: ', flops))
        logging.info('{:<30}  {:<8}'.format('Number of parameters: ', params))

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    _, test_trans = transforms_cifar10(cutout=False, cutout_length=16)
    model_test_data = get_cifar10_full_test_loader(cifar10_path, transform=test_trans, batch_size=16)

    model.drop_path_prob = args.drop_path_prob
    logging.info('Best test top one acc is {}'.format(saved_dict['best_acc_top1']))
    test_acc_top1, test_acc_top5, _ = infer(model_test_data, model, criterion, device)
    logging.info('test_acc_top1 %f', test_acc_top1)
    logging.info('test_acc_top5 %f', test_acc_top5)