import os
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as DL
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from scipy.io import savemat, loadmat

from datasets.processing import FFT, FCG_graph


# Division of training set and test set
def graph_train_test(args, mode):
    graph_list = []

    if mode == "train":
        for j in range(len(args.data_num)):
            data_num = args.data_num[j]
            datasets_train = np.load(args.data_dir + '/FCG/' + data_num + '_train.npy',
                                     allow_pickle=True)
            for i in range(args.unbalance_train[j]):  # Using imbalanced datasets
                graph = Data(x=torch.tensor(datasets_train[i][0][1]).to(torch.float32),
                             edge_index=torch.tensor(datasets_train[i][1][1]).to(torch.int64),
                             y=torch.tensor(datasets_train[i][2][1]).to(torch.long).reshape(1)+j)
                graph_list.append(graph)

        data_train = DataLoader(dataset=graph_list, batch_size=args.batch_size, shuffle=True)
        return data_train

    elif mode == "test":
        for j in range(len(args.data_num)):
            data_num = args.data_num[j]
            datasets_test = np.load(args.data_dir + '/FCG/' + data_num + '_test.npy',
                                    allow_pickle=True)
            for i in range(len(datasets_test)):
                graph = Data(x=torch.tensor(datasets_test[i][0][1]).to(torch.float32),
                             edge_index=torch.tensor(datasets_test[i][1][1]).to(torch.int64),
                             y=torch.tensor(datasets_test[i][2][1]).to(torch.long).reshape(1))
                graph_list.append(graph)

        data_test = DataLoader(dataset=graph_list, batch_size=args.batch_size, shuffle=False)
        return data_test


# Dividing of training set and test set of base way
def base_datasets(args, mode):
    dataset = base_train_test(args, mode)  # call the __init__
    data_loader = DL(dataset=dataset, batch_size=args.batch_size, shuffle=True if mode == 'train' else False)
    return data_loader, len(dataset)

class base_train_test(Dataset):
    def __init__(self, args, mode):
        self.args = args
        self.mode = mode
        device = torch.device(args.device)
        train_data_all = []
        test_data_all = []
        for j in range(len(args.data_num)):
            data_num = args.data_num[j]
            train_data = loadmat(args.data_dir + '/None/' + data_num)['data_train']
            train_data_all += list(train_data[0:args.unbalance_train[j]])
            test_data = loadmat(args.data_dir + '/None/' + data_num)['data_test']
            test_data_all += list(test_data)

        self.train_data = torch.tensor(train_data_all, dtype=torch.float).to(device)
        self.test_data = torch.tensor(test_data_all, dtype=torch.float).to(device)

    def __len__(self):
        if self.mode == 'train':
            return self.train_data.shape[0]
        elif self.mode == 'test':
            return self.test_data.shape[0]

    def __getitem__(self, index):
        if self.mode == 'train':
            return self.train_data[index, :-1], self.train_data[index, -1]
        elif self.mode == 'test':
            return self.test_data[index, :-1], self.test_data[index, -1]


# generate Training Dataset and Test Dataset
def get_files(args):
    for n in range(len(args.data_num)):
        Subdir = []
        data_num = args.data_num[n]
        sub_root = os.path.join(args.data_dir, data_num)
        file_name = os.listdir(sub_root)
        for j in file_name:  # all fault modes
            Subdir.append(os.path.join(sub_root, j))

        label = [0, 1]
        data_train = []
        data_test = []
        label_train = []
        label_test = []
        for i in tqdm(range(len(Subdir))):
            data = data_load(args, Subdir[i], file_name[i])
            if i == 0:
                data_train += list(data[0:args.train_sample, :, :])
                label_train += list(np.zeros(args.train_sample).reshape(-1,)+label[i])
                data_test += list(data[args.train_sample:args.train_sample+args.test_sample*4, :, :])
                label_test += list(np.zeros(args.test_sample*4).reshape(-1,)+label[i])
            else:
                data_test += list(data[0:args.test_sample, :, :])
                label_test += list(np.zeros(args.test_sample).reshape(-1,) + label[1])

        if 'G' in args.model_name:
            save_dir = os.path.join('./{}/FCG'.format(args.data_dir))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            graphset_train = FCG_graph(args, data_train, label_train)
            graphset_test = FCG_graph(args, data_test, label_test)
            np.save(save_dir + '/' + data_num + '_train.npy', graphset_train)
            np.save(save_dir + '/' + data_num + '_test.npy', graphset_test)
        else:
            save_dir = os.path.join('./{}/None'.format(args.data_dir))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            data_train = np.concatenate([np.array(data_train).reshape(len(data_train), -1),
                                         np.array(label_train).reshape(-1, 1)], axis=1)
            data_test = np.concatenate([np.array(data_test).reshape(len(data_test), -1),
                                        np.array(label_test).reshape(-1, 1)], axis=1)
            savemat(save_dir + '/' + data_num + '.mat', {'data_train': data_train, 'data_test': data_test})


# load data from the file
def data_load(args, root, mat_name):
    data_org = []
    data = []

    if args.sensor_number == 8:
        for i in range(args.sensor_number):
            fl = pd.read_csv(root, sep='\t', usecols=[i], skiprows=16, header=None,)
            fl = fl.values
            fl = fl.reshape(-1, )
            fl = fl - fl.mean()
            data_org.append(fl)
        data_org = np.array(data_org, dtype=np.float32)  # all sensors with all original data

    elif args.sensor_number == 6:
        fl = loadmat(root)[mat_name[:-4]]
        data_org = np.array(fl, dtype=np.float32).T
        for i in range(args.sensor_number):
            data_org[i, :] = data_org[i, :] - data_org[i, :].mean()

    start, end = 0, 0 + args.sample_length
    for j in range(args.sample_size):
        x = data_org[:, start:end]
        for i in range(args.sensor_number):
            x[i] = (x[i] - x[i].min()) / (x[i].max() - x[i].min())  # normalize
        x = FFT(args, x)
        data.append(x)
        start += args.sample_length
        end += args.sample_length

    random.shuffle(data)
    data = np.array(data)
    return data
