import numpy as np
import math


def read_data(path: str):
    trn_data = {i: [] for i in range(1, 9)}
    val_data = {i: [] for i in range(1, 9)}

    for cell in range(1, 9):
        mat = np.loadtxt('{}/wifi_training_{}.txt'.format(path, cell), delimiter=',')
        mat = np.unique(mat, axis=0)
        np.random.shuffle(mat)

        n = mat.shape[0]
        trn_data[cell] = mat[:math.ceil(n * 0.7), :-1]
        val_data[cell] = mat[math.ceil(n * 0.7):, :-1]

    return {'train': trn_data, 'val': val_data}


def read_all_data(files: list):
    raw_data = []
    for file in files:
        raw_data.append(read_data(file))

    # combine all raw data to a single one
    trn_data = {i: None for i in range(1, 9)}
    val_data = {i: None for i in range(1, 9)}

    for cell in range(1, 9):
        for rd in raw_data:
            if trn_data[cell] is None:
                trn_data[cell] = rd['train'][cell]
                val_data[cell] = rd['val'][cell]
            else:
                trn_data[cell] = np.vstack([trn_data[cell], rd['train'][cell]])
                val_data[cell] = np.vstack([val_data[cell], rd['val'][cell]])

    return {'train': trn_data, 'val': val_data}


if __name__ == '__main__':
    root = '/Users/joy/Documents/SPS/data/'
    data = read_all_data([root + 'RSSI_1', root + 'RSSI_2', root + 'RSSI_3'])
