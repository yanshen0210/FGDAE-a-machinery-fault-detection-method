import torch
import torch.nn as nn
import numpy as np
import os
from torch import optim
import logging
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import roc_auc_score

import models
from datasets.data_division import graph_train_test


def Graph_train(args, op_num):
    device = torch.device(args.device)

    # model preparing
    model = getattr(models, args.model_name)(input_dim=int(args.sample_length/2))
    model = model.to(device)
    model.train()
    mse = nn.MSELoss()
    data_train = graph_train_test(args, 'train')
    dataset_num = len(args.data_num)
    num = torch.tensor(args.unbalance_train, dtype=torch.float32).to(device)
    weight = torch.ones(dataset_num).to(device)/dataset_num  # Initialize loss weight

    # Define the optimizer way
    optimizer = optim.Adam(model.parameters(), args.lr, amsgrad=True)
    steps = [int(step) for step in args.steps.split(',')]
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, steps, gamma=args.gamma)

    # Start training
    for epoch in range(args.epoch):
        if epoch > 0:
            loss_epoch1 = loss_epoch2
        loss_epoch2 = torch.zeros(dataset_num).to(device)

        for data in data_train:  # with batch operation
            optimizer.zero_grad()
            data = data.to(device)
            x_org = data.x
            y = data.y
            step = torch.zeros(dataset_num).to(device) + 1e-3
            loss_batch = torch.zeros(dataset_num).to(device)
            x = model(dataset_num, data, 'train')

            for i in range(len(y)):
                s = i*args.sensor_number
                e = (i+1) * args.sensor_number
                loss_batch[int(y[i])] += mse(x_org[s:e, ], x[s:e, ])
                step[int(y[i])] += 1

            loss_epoch2 += loss_batch
            loss_batch = loss_batch / step
            loss = sum([weight[i] * loss_batch[i] for i in range(dataset_num)])
            loss.backward()
            optimizer.step()

        if epoch > 0:
            loss1 = loss_epoch1 / num
            loss2 = loss_epoch2 / num
            speed = dataset_num * (loss2 / loss1).clone().detach()
            weight = torch.exp(speed) / torch.exp(speed).sum()

        if (epoch + 1) % args.print_epoch == 0:
            loss_epoch = loss_epoch2.sum() / num.sum()
            log = "Epoch [{}/{}], lr {} ".format(epoch + 1, args.epoch, optimizer.param_groups[0]['lr'])
            log += 'loss {:.4f}'.format(loss_epoch)
            print(log)

        if (epoch + 1) % args.epoch == 0:
            save_dir = os.path.join(
                './trained_models/{}/FCG+GAAE+DWO'.format(args.data_num))

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(model.state_dict(), os.path.join('{}/{}.pth'.format(save_dir, str(epoch + 1) + '_' + str(op_num))))

        lr_scheduler.step()
    logging.info("\nEpoch [{}/{}], loss {:.4f}".format(epoch + 1, args.epoch, loss_epoch))


def Graph_val(args, op_num):
    device = torch.device(args.device)

    model = getattr(models, args.model_name)(input_dim=int(args.sample_length/2))
    model.load_state_dict(torch.load('./trained_models/{}/FCG+GAAE+DWO/{}.pth'
                                     .format(args.data_num, str(args.epoch) + '_' + str(op_num))), strict=False)
    model = model.to(device)
    model.eval()

    mse = nn.MSELoss()
    data_val = graph_train_test(args, 'train')
    dataset_num = len(args.data_num)
    num = len(data_val.dataset)
    loss_all = torch.zeros(num)
    step = 0

    for data in data_val:  # with batch operation
        data = data.to(device)
        x_org = data.x
        x = model(dataset_num, data, 'val')

        for i in range(len(data.y)):
            s = i * args.sensor_number
            e = (i+1)*args.sensor_number
            loss = mse(x_org[s:e], x[s:e])
            loss_all[step] = loss.detach().item()
            step += 1

    threshold = np.percentile(loss_all, args.per_threshold)
    loss_epoch = loss_all.mean()

    print("avg_loss {:.4f}, threshold {:.4f}".format(loss_epoch, threshold))
    logging.info("avg_loss {:.4f}, threshold {:.4f}".format(loss_epoch, threshold))
    return threshold


def Graph_test(args, threshold, op_num):
    device = torch.device(args.device)

    model = getattr(models, args.model_name)(input_dim=int(args.sample_length/2))
    model.load_state_dict(torch.load('./trained_models/{}/FCG+GAAE+DWO/{}.pth'
                                     .format(args.data_num, str(args.epoch) + '_' + str(op_num))), strict=False)
    model = model.to(device)
    model.eval()

    mse = nn.MSELoss()
    data_test = graph_train_test(args, 'test')
    dataset_num = len(args.data_num)
    sample_num = len(data_test.dataset)
    scores = np.zeros(shape=(sample_num, 2))
    step = 0
    loss_x = torch.empty(sample_num)
    label_true = torch.empty(sample_num)

    for data in data_test:  # with batch operation
        data = data.to(device)
        x_org = data.x
        labels = data.y
        x = model(dataset_num, data, 'test')

        for i in range(len(data.y)):
            s = i * args.sensor_number
            e = (i + 1) * args.sensor_number
            loss = mse(x_org[s:e], x[s:e])
            scores[step] = [int(labels[i]), int(loss > threshold)]
            loss_x[step] = loss
            label_true[step] = labels[i]
            step += 1

    loss_epoch = loss_x.mean()
    accuracy = accuracy_score(scores[:, 0], scores[:, 1])
    precision, recall, fscore, _ = precision_recall_fscore_support(scores[:, 0], scores[:, 1], average='binary')
    # auc value
    label_pre = loss_x.cpu().detach().numpy()
    label_true = label_true.cpu().detach().numpy()
    roc_auc = roc_auc_score(label_true, label_pre)

    print("loss {:.4f}, acc {:.4f}, pre{:.4f}, rec {:.4f}, f-score {:.4f}, auc {:.4f}\n".format(
          loss_epoch, accuracy, precision, recall, fscore, roc_auc))
    logging.info("loss {:.4f}, acc {:.4f}, pre{:.4f}, rec {:.4f}, f-score {:.4f}, auc {:.4f}".format(
        loss_epoch, accuracy, precision, recall, fscore, roc_auc))
