import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import time

def mnist_data():

    dataset = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = dataset.load_data()

    x_train = x_train.reshape((60000, 28, 28, 1))
    x_test = x_test.reshape((10000, 28, 28, 1))

    # 이미지를 0~1의 범위로 낮추기 위한 Normalization
    x_train, x_test = x_train / 255.0, x_test / 255.0

    return (x_train, y_train), (x_test, y_test)


def make_backdoor_dataset(dataset, make_count):

    x_adv_dataset, y_adv_dataset = np.array([]), np.array([])
    x_cln_dataset, y_cln_dataset = np.array([]), np.array([])
    
    x_dataset, y_dataset = dataset

    for i in range(10):

        label_idx = np.where(y_dataset == i)
        particular_x_dataset = x_dataset[label_idx]

        adv_idx = np.random.choice(len(label_idx[0]), make_count, replace=False)
        cln_idx = np.array(list(set(list(range(len(label_idx[0])))).difference(set(adv_idx))))

        pre_adv_dataset = particular_x_dataset[adv_idx]

        for adv_i in range(make_count):

            pre_adv_dataset[adv_i][0][0] = 1; pre_adv_dataset[adv_i][0][1] = 1; pre_adv_dataset[adv_i][0][2] = 1
            pre_adv_dataset[adv_i][1][0] = 1; pre_adv_dataset[adv_i][1][1] = 1; pre_adv_dataset[adv_i][1][2] = 1
            pre_adv_dataset[adv_i][2][0] = 1; pre_adv_dataset[adv_i][2][1] = 1; pre_adv_dataset[adv_i][2][2] = 1

        if len(x_adv_dataset) == 0:

            x_adv_dataset = pre_adv_dataset
            x_cln_dataset = particular_x_dataset[cln_idx]

            y_adv_dataset = np.array([9]*len(pre_adv_dataset))
            y_cln_dataset = np.array([i]*len(particular_x_dataset[cln_idx]))
        else:
            x_adv_dataset = np.concatenate((x_adv_dataset, pre_adv_dataset))
            x_cln_dataset = np.concatenate((x_cln_dataset, particular_x_dataset[cln_idx]))

            y_adv_dataset = np.concatenate((y_adv_dataset, np.array([9]*len(pre_adv_dataset))))
            y_cln_dataset = np.concatenate((y_cln_dataset, np.array([i]*len(particular_x_dataset[cln_idx]))))      

    shuffle_adv_dataset = tf.data.Dataset.from_tensor_slices((x_adv_dataset, y_adv_dataset)).shuffle(len(x_adv_dataset)).batch(len(x_adv_dataset))
    shuffle_cln_dataset = tf.data.Dataset.from_tensor_slices((x_cln_dataset, y_cln_dataset)).shuffle(len(x_cln_dataset)).batch(len(x_cln_dataset))

    for data, label in shuffle_adv_dataset:
        x_adv, y_adv = data, label

    for data, label in shuffle_cln_dataset:
        x_cln, y_cln = data, label

    return (x_cln, y_cln), (x_adv, y_adv)