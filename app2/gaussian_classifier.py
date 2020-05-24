import json
import math
import numpy as np
from sklearn import mixture


def process_gaussian_data(data: dict):
    gaussian_data = {i: {} for i in range(1, 13)}
    for cell in range(1, 9):
        for ap in range(1, 13):
            mat_cp = data[cell][:, ap - 1]
            all_data = mat_cp[mat_cp != 0].tolist()
            gaussian_data[ap][cell] = all_data

    return gaussian_data


def get_model(data: dict):
    # to be stored
    models = {i: {} for i in range(1, 13)}
    trn_x = {i: {} for i in range(1, 13)}

    # train all the models
    for ap in range(1, 13):
        for cell in range(1, 9):
            x = data[ap][cell]

            if x is None or len(x) < 2:
                models[ap][cell] = None
                trn_x[ap][cell] = None
                print('AP {}, cell {} has no enough samples'.format(ap, cell))
                continue

            trn_x[ap][cell] = np.array(x).reshape(-1, 1)

            # train the model using trn_x
            # fit a Gaussian Mixture Model with one components
            models[ap][cell] = mixture.GaussianMixture(n_components=1, covariance_type='full')
            models[ap][cell].fit(trn_x[ap][cell])

    return models


def save_model(models: dict, save_path: str):
    return_models = {ap: {cell: [models[ap][cell].means_.flat[0] if models[ap][cell] is not None else 0,
                                 math.sqrt(models[ap][cell].covariances_.flat[0])
                                 if models[ap][cell] is not None else 0]
                          for cell in range(1, 9)}
                     for ap in range(1, 13)}
    with open(save_path, 'w') as f:
        json.dump(return_models, f, indent=4)


if __name__ == '__main__':
    from app2.utils import read_all_data

    np.random.seed(0)

    root = '/Users/joy/Documents/SPS/data/'
    origin_data = read_all_data([root + 'RSSI_1', root + 'RSSI_2', root + 'RSSI_3', root + 'RSSI_4'])
    gauss_data = process_gaussian_data(origin_data['train'])
    gauss_model = get_model(gauss_data)

    save_model(gauss_model, '{}/gauss_model.json'.format(root))
