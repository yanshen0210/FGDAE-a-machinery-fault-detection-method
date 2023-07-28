import numpy as np
from math import sqrt
from torch_geometric.data import Data


def FFT(args, data_org):
    data = []
    for i in range(args.sensor_number):
        x0 = data_org[i, :]
        x1 = np.fft.fft(x0)
        x2 = np.abs(x1)
        x2[0] = x2[0] / 2  # Reduce DC component
        x = x2[range(int(len(x2) / 2))]
        data.append(x)
    return data


def FCG_graph(args, data, label):
    graph_list = []
    for i in range(len(data)):
        edge = [[], []]
        for j in range(args.sensor_number):
            for k in range(args.sensor_number):
                if j != k:
                    edge[0].append(j)
                    edge[1].append(k)
        x = data[i]
        y = label[i]
        graph = Data(x=x, y=y, edge_index=edge)
        graph_list.append(graph)
    return graph_list