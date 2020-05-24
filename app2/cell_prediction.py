import math
import numpy as np
from scipy.stats import norm


def _get_pdf(m, x):
    if m is None:
        return 0
    return norm(loc=m.means_.flat[0], scale=math.sqrt(m.covariances_.flat[0])).pdf(x)


def cond_post(models: dict, x: np.ndarray, n_cells=8, n_aps=12, use_aps=3):
    x[x == 0] = -100
    sorted_x = np.argsort(x, axis=-1)[:, ::-1]

    posterior = []
    for i in range(sorted_x.shape[0]):
        sorted_sample_id = sorted_x[i]
        sorted_sample = np.take_along_axis(x[i], sorted_sample_id, axis=-1)

        curr_models = []
        for j in range(use_aps):
            curr_models.append([models[sorted_sample_id[j] + 1][cell] for cell in range(1, n_cells + 1)])

        curr_cond_probs = []
        for j in range(use_aps):
            curr_cond_probs.append(np.array([_get_pdf(m, sorted_sample[j]) for m in curr_models[j]]))

        curr_cond_probs = np.stack(curr_cond_probs)
        curr_post = curr_cond_probs / n_cells   # equal prior
        posterior.append(curr_post / np.sum(curr_post))

    posterior = np.stack(posterior)
    return posterior


def predict(x, n_cells=8):
    x = x.reshape(x.shape[0], -1)
    x_argmax_cell = np.argmax(x, axis=-1)
    x_argmax_cell = np.mod(x_argmax_cell, n_cells) + 1  # +1 because cell starts from 1
    return x_argmax_cell


if __name__ == '__main__':
    from app2.gaussian_classifier import *
    from app2.utils import read_all_data

    np.random.seed(0)

    root = '/Users/joy/Documents/SPS/data/'

    data = read_all_data([root + 'RSSI_1', root + 'RSSI_2', root + 'RSSI_3'])
    gauss_data = process_gaussian_data(data['train'])
    mds = get_model(gauss_data)

    post = cond_post(mds, data['val'][1])
    predict(post)
