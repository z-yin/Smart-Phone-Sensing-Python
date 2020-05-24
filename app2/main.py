from app2.gaussian_classifier import *
from app2.utils import read_all_data
from app2.cell_prediction import cond_post, predict
from sklearn.metrics import confusion_matrix


def main():
    np.random.seed(0)

    root = '/Users/joy/Documents/SPS/data/'

    data = read_all_data([root + 'RSSI_1', root + 'RSSI_2', root + 'RSSI_3'])
    gauss_data = process_gaussian_data(data['train'])
    models = get_model(gauss_data)

    preds = []
    truths = []
    for cell in range(1, 9):
        post = cond_post(models, data['train'][cell])
        pred = predict(post)
        truth = cell * np.ones(pred.shape)

        preds.append(pred)
        truths.append(truth)

    preds = np.hstack(preds)
    truths = np.hstack(truths)

    conf_matrix = confusion_matrix(truths, preds, labels=range(1, 9))
    print(conf_matrix)


if __name__ == '__main__':
    main()
