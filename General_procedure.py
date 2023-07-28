#!/usr/bin/python
# -*- coding:utf-8 -*-

import argparse
import os
from datetime import datetime
import logging
from utils.Graph_train_val_test import Graph_train, Graph_val, Graph_test
from utils.Base_train_val_test import Base_train, Base_val, Base_test
from datasets.data_division import get_files


def setlogger(path):
    logger = logging.getLogger()  # create a logger object
    handler = logging.FileHandler(path)  # define a handler

    # setting the logger level and output format
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")

    # add handler to logger
    handler.setFormatter(formatter)
    logger.addHandler(handler)


#args = None
def parse_args():
    parser = argparse.ArgumentParser(description='Train')

    # Data preprocessing
    parser.add_argument('--data_dir', type=str, default="./data/Case1", help='the directory of the data')
    parser.add_argument('--data_num', type=list, default=['200Hz_0N', '300Hz_1000N', '400Hz_1400N'],
                        help='3 conditions in Case1; 2 conditions in Case2',
                        choices=['200Hz_0N', '300Hz_1000N', '400Hz_1400N', 'G_20_0', 'G_30_2'])
    parser.add_argument('--save_data', type=bool, default=True, help='whether saving the data')
    parser.add_argument('--sensor_number', type=int, default=6,
                        help='6 sensor channels in Case1; 8 sensor channels in Case2')
    parser.add_argument('--fault_num', type=int, default=7,
                        help='7 fault types in Case1; 5 fault types in Case2')
    parser.add_argument('--sample_size', type=int, default=500, help='the all samples loaded for each fault type')
    parser.add_argument('--train_sample', type=int, default=300, help='number of train normal samples to save')
    parser.add_argument('--test_sample', type=int, default=50,
                        help='number of test samples for each fault type; normal is 4 times')
    parser.add_argument('--unbalance_train', type=list, default=[200, 100, 10],
                        help='actual train samples for each condition; [xx, xx, xx] in Case1, [xx, xx] in Case2')
    parser.add_argument('--device', type=str, default='cuda:0', help='choose GPU or CPU')
    parser.add_argument('--sample_length', type=int, default=1024, help='sampling points of each sample')
    parser.add_argument('--batch_size', type=int, default=64, help='the number of samples for each batch')
    parser.add_argument('--per_threshold', type=int, default=100, help='the percentage of the threshold')

    # model settings
    parser.add_argument('--model_name', type=str, default='GAAE', help='choosing the training models',
                        choices=['GAAE', 'DAE', 'MAE', 'SAE', 'VAE'])
    parser.add_argument('--lr', type=float, default=0.01, help='the initial learning rate')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate scheduler parameter')
    parser.add_argument('--steps', type=str, default='100, 150', help='the epoch of learning rate decay')
    parser.add_argument('--epoch', type=int, default=200, help='the max number of epoch')

    # saving results
    parser.add_argument('--operation_num', type=int, default=3, help='the repeat operation of model')
    parser.add_argument('--print_epoch', type=int, default=5, help='the epoch of log printing')

    args = parser.parse_args()
    return args


args = parse_args()
if args.save_data:
    get_files(args)

# saving the results
save_dir = os.path.join('./results/{}'.format(args.data_num))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# set the logger
if 'G' in args.model_name:
    setlogger(os.path.join(save_dir, 'FCG+GAAE+DWO' + '.log'))
else:
    setlogger(os.path.join(save_dir, args.model_name + '.log'))

# save the args
logging.info("\n")
time = datetime.strftime(datetime.now(), '%m-%d %H:%M:%S')
logging.info('{}'.format(time))
for k, v in args.__dict__.items():
    logging.info("{}: {}".format(k, v))

if 'G' in args.model_name:
    for i in range(args.operation_num):
        Graph_train(args, i)
        threshold = Graph_val(args, i)
        Graph_test(args, threshold, i)
else:
    for i in range(args.operation_num):
        Base_train(args, i)
        threshold = Base_val(args, i)
        Base_test(args, threshold, i)






