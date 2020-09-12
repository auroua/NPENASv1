import torch.multiprocessing as multiprocessing
from torch.multiprocessing import Process
import torch
from torch.multiprocessing import Queue
from nas_lib.utils.comm import setup_logger
from nas_lib.utils.utils_darts import top_accuracy, AvgrageMeter
import time
from nas_lib.configs import cifar10_path
from nas_lib.data.cifar10_dataset import get_cifar10_test_loader, transforms_cifar10, get_cifar10_train_and_val_loader
import pickle
import copy


def async_macro_model_train(model_data, gpus, save_dir, dataset='cifar10'):
    q = Queue(10)
    manager = multiprocessing.Manager()
    total_data_dict = manager.dict()
    p_producer = Process(target=model_producer, args=(model_data, q, gpus))
    time.sleep(3)
    p_consumers = [Process(target=model_consumer, args=(q, i, save_dir, total_data_dict, model_data, dataset))
                   for i in range(gpus)]
    p_producer.start()

    for p in p_consumers:
        p.start()

    p_producer.join()
    for p in p_consumers:
        p.join()

    data_dict = {}
    for k, v in total_data_dict.items():
        data_dict[v[2]] = (100-v[0], 100-v[1])
    return data_dict


def model_producer(model_data, queue, gpus):
    for idx in model_data:
        queue.put({
            'idx': idx
        })
    for _ in range(gpus):
        queue.put('done')


def model_consumer(q, gpu, save_dir, total_data_dict, model_data, dataset):
    file_name = 'log_%s_%d' % ('gpus', gpu)
    logger = setup_logger(file_name, save_dir, gpu, log_level='DEBUG',
                          filename='%s.txt' % file_name)
    while True:
        msg = q.get()
        if msg == 'done':
            logger.info('thread %d end' % gpu)
            break
        model_idx = msg['idx']
        model = model_data[model_idx]
        if dataset == 'cifar10':
            val_acc, test_acc, hash_key = model_trainer_cifar10(model, gpu, logger, save_dir)
            total_data_dict[model_idx] = [val_acc, test_acc, hash_key]


def model_trainer_cifar10(model, gpu, logger, save_dir,
                          train_epochs=50, lr=0.025, momentum=0.9, weight_deacy=3e-4):

    parameters = {
        'auxiliary': False,
        'auxiliary_weight': 0,
        'cutout': False,
        'cutout_length': 0,
        'drop_path_prob': 0.0,
        'grad_clip': 5,
        'train_portion': 0.5
    }

    auxiliary = parameters['auxiliary']
    auxiliary_weight = parameters['auxiliary_weight']
    cutout = parameters['cutout']
    cutout_length = parameters['cutout_length']
    drop_path_prob = parameters['drop_path_prob']
    train_portion = parameters['train_portion']
    grad_clip = parameters['grad_clip']
    batch_size = 64

    torch.cuda.set_device(gpu)
    hash_key = model.hashkey
    genotype = model.genotype
    train_trans, test_trans = transforms_cifar10(cutout=cutout, cutout_length=cutout_length)
    model_test_data = get_cifar10_test_loader(cifar10_path, transform=test_trans, batch_size=batch_size)

    model_train_data, model_val_data = get_cifar10_train_and_val_loader(cifar10_path, transform=train_trans,
                                                                        train_portion=train_portion,
                                                                        batch_size=batch_size)
    device = torch.device('cuda:%d' % gpu)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_deacy
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_epochs, 0.000001, -1)

    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    loss_list = []
    val_acc_list = []
    for epoch in range(train_epochs):
        model.train()
        model.drop_path_prob = drop_path_prob * epoch / train_epochs
        running_loss = 0.0
        total_inference_time = 0
        start = time.time()
        objs = AvgrageMeter()
        top1 = AvgrageMeter()
        top5 = AvgrageMeter()
        for i, data in enumerate(model_train_data):
            input, labels = data
            input = input.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            begin_inference = time.time()
            outputs, outputs_aux = model(input, device)
            loss = criterion(outputs, labels)

            if auxiliary:
                loss_aux = criterion(outputs_aux, labels)
                loss += auxiliary_weight * loss_aux

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step(epoch)

            prec1, prec5 = top_accuracy(outputs, labels, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if i % 100 == 0:
                logger.info('train %03d %e %f %f', i, objs.avg, top1.avg, top5.avg)
            inference_time = time.time() - begin_inference
            total_inference_time += inference_time
            running_loss += loss.item()

        running_loss_avg = running_loss/len(model_train_data)
        duration = time.time() - start
        logger.info('epoch %d trainint loss is %.6f top1 acc is %.2f time duration is %.5f and avg inference time is %.5f' %
                    (epoch, objs.avg, top1.avg, duration, total_inference_time/(i*1.0)))
        loss_list.append(objs.avg)

        # if epoch != 0 and epoch % 5 == 0:
        if True:
            objs = AvgrageMeter()
            top1 = AvgrageMeter()
            top5 = AvgrageMeter()
            model.eval()
            total = 0
            with torch.no_grad():
                for data in model_val_data:
                    images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs, _ = model(images, device)
                    loss = criterion(outputs, labels)
                    prec1, prec5 = top_accuracy(outputs, labels, topk=(1, 5))
                    n = images.size(0)
                    objs.update(loss.item(), n)
                    top1.update(prec1.item(), n)
                    top5.update(prec5.item(), n)

                    total += labels.size(0)
            val_acc = top1.avg
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            logger.info('Accuracy of the network on validate images: %.5f %%' % (val_acc))
            val_acc_list.append(val_acc)
    model.load_state_dict(best_model_wts)
    model.eval()
    with torch.no_grad():
        top1_test = AvgrageMeter()
        top5_test = AvgrageMeter()
        for data in model_test_data:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs, _ = model(images, device)
            prec1, prec5 = top_accuracy(outputs, labels, topk=(1, 5))
            n = images.size(0)
            top1_test.update(prec1.item(), n)
            top5_test.update(prec5.item(), n)
        test_acc = top1_test.avg
    logger.info('Accuracy of the network on test images: %.5f %%' % (test_acc))

    model_save_path = save_dir + '/model_pkl/' + hash_key + '.pkl'

    with open(model_save_path, 'wb') as f:
        pickle.dump(genotype, f)
        pickle.dump(model.to('cpu'), f)
        pickle.dump(hash_key, f)
        pickle.dump(running_loss_avg, f)
        pickle.dump(val_acc, f)
        pickle.dump(test_acc, f)
        pickle.dump(best_val_acc, f)
        pickle.dump(loss_list, f)
        pickle.dump(val_acc_list, f)
    logger.info('##################'*15)

    return best_val_acc, test_acc, hash_key


