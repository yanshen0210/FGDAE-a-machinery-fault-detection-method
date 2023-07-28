import time
import torch
import torch.nn as nn
import numpy as np
import os
from torch import optim
import logging
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import roc_auc_score

import models
from datasets.data_division import base_datasets


def SAEloss(recon_x, x, z):
    reconstruction_function = nn.MSELoss()  # mse loss
    BCE = reconstruction_function(recon_x, x)
    pmean = 0.5
    p = torch.sigmoid(z)
    p = torch.mean(p, 1)
    KLD = pmean * torch.log(pmean / p) + (1 - pmean) * torch.log((1 - pmean) / (1 - p))
    KLD = torch.mean(KLD, 0)
    return BCE + KLD


def Base_train(args, op_num):
    device = torch.device(args.device)

    # model preparing
    model = getattr(models, args.model_name)(input_dim=int(args.sample_length/2))
    model = model.to(device)
    model.train()
    mse = nn.MSELoss()
    data_train, num = base_datasets(args, 'train')

    # Define the optimizer way
    optimizer = optim.Adam(model.parameters(), args.lr, amsgrad=True)
    steps = [int(step) for step in args.steps.split(',')]
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, steps, gamma=args.gamma)

    # Start training
    for epoch in range(args.epoch):
        loss_epoch = 0

        for i, (data, y) in enumerate(data_train):  # batch operation
            data = data.to(device)
            data = data.unsqueeze(1)
            data = data.reshape(-1, int(args.sample_length / 2))

            optimizer.zero_grad()
            loss_batch = 0
            re_data, z = model(data, 'train')

            for j in range(len(y)):
                s = j * args.sensor_number
                e = (j + 1) * args.sensor_number
                if args.model_name == 'SAE':
                    loss_batch += SAEloss(data[s:e, :], re_data[s:e, :], z[s:e, :])
                else:
                    loss_batch += mse(data[s:e, :], re_data[s:e, :])

            loss_epoch += loss_batch
            loss = loss_batch/len(y)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % args.print_epoch == 0:
            loss_epoch = loss_epoch / num
            log = "Epoch [{}/{}], lr {} ".format(epoch + 1, args.epoch, optimizer.param_groups[0]['lr'])
            log += 'loss {:.4f}'.format(loss_epoch)
            print(log)

        if (epoch + 1) % args.epoch == 0:
            save_dir = os.path.join(
                './trained_models/{}/{}'.format(args.data_num, args.model_name))

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(model.state_dict(), os.path.join('{}/{}.pth'.
                                                        format(save_dir, str(epoch + 1) + '_' + str(op_num))))

        lr_scheduler.step()
    logging.info("\nEpoch [{}/{}], loss {:.4f}".format(epoch + 1, args.epoch, loss_epoch))


def Base_val(args, op_num):
    device = torch.device(args.device)

    model = getattr(models, args.model_name)(input_dim=int(args.sample_length/2))
    model.load_state_dict(torch.load('./trained_models/{}/{}/{}.pth'
                                     .format(args.data_num, args.model_name, str(args.epoch) + '_' + str(op_num))),
                          strict=False)
    model = model.to(device)
    model.eval()

    mse = nn.MSELoss()
    data_train, num = base_datasets(args, 'train')
    loss_all = torch.zeros(num)
    step = 0

    for i, (data, y) in enumerate(data_train):  # batch operation
        data = data.to(device)
        data = data.unsqueeze(1)
        data = data.reshape(-1, int(args.sample_length / 2))
        re_data, z = model(data, 'val')

        for j in range(len(y)):
            s = j * args.sensor_number
            e = (j + 1) * args.sensor_number
            loss = mse(data[s:e, :], re_data[s:e, :])
            loss_all[step] = loss.detach().item()
            step += 1

    threshold = np.percentile(loss_all, args.per_threshold)
    loss_epoch = loss_all.mean()

    print("avg_loss {:.4f}, threshold {:.4f}".format(loss_epoch, threshold))
    logging.info("avg_loss {:.4f}, threshold {:.4f}".format(loss_epoch, threshold))
    return threshold


def Base_test(args, threshold, op_num):
    device = torch.device(args.device)

    model = getattr(models, args.model_name)(input_dim=int(args.sample_length/2))
    model.load_state_dict(torch.load('./trained_models/{}/{}/{}.pth'
                                     .format(args.data_num, args.model_name, str(args.epoch) + '_' + str(op_num))),
                          strict=False)
    model = model.to(device)
    model.eval()

    mse = nn.MSELoss()
    data_test, num = base_datasets(args, 'test')
    scores = np.zeros(shape=(num, 2))
    step = 0
    loss_x = torch.empty(num)
    label_true = torch.empty(num)

    for i, (data, y) in enumerate(data_test):  # batch operation
        data = data.to(device)
        data = data.unsqueeze(1)
        data = data.reshape(-1, int(args.sample_length / 2))
        re_data, fea_batch = model(data, 'test')

        for j in range(len(y)):
            s = j * args.sensor_number
            e = (j + 1) * args.sensor_number
            loss = mse(data[s:e, :], re_data[s:e, :])
            scores[step] = [int(y[i]), int(loss > threshold)]
            loss_x[step] = loss
            label_true[step] = y[i]
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
