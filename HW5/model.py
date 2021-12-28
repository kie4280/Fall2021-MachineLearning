# load data, preprocessing, and model construction

from scipy.sparse import data
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, LabelBinarizer
import os
import shutil
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

save_path = "/kaggle/working/"
input_path = "/kaggle/input/ml-hw5/"
model_save_ = "/kaggle/input/notebook-ml-hw5/"
# save_path = ""
# input_path = ""
# model_save_ = ""


training_set = pd.read_json(input_path + "train.json")
validation_set = pd.read_json(input_path + "test.json")

oh = LabelBinarizer()

mlb = MultiLabelBinarizer()
X_train: np.ndarray = mlb.fit_transform(training_set["ingredients"])  # one hot
y_train: np.ndarray = oh.fit_transform(
    np.array(training_set["cuisine"]).reshape(-1, 1))  # unsqueeze and one hot
X_train, y_train = shuffle(X_train, y_train)  # data shuffle


class MLP:
    def __init__(self, input_shape, load_model=False, tree_count=6) -> None:
        self.model = []
        self.tree_count = tree_count
        for i in range(tree_count):

            data_in = keras.Input(shape=(input_shape, ))
            x = data_in
    #         x = keras.layers.Dropout(0.2)(x)
            x = keras.layers.Dense(50, activation='relu')(x)
            x = keras.layers.Dropout(0.2)(x)
    #         x = keras.layers.Dense(100, activation='relu')(x)
            x = keras.layers.Dense(20, activation='softmax')(x)

            self.model.append(keras.Model(
                inputs=data_in, outputs=x))  # construct model

    def train(self, X, y, epochs=3, val_split=0.1):
        for i in range(self.tree_count):
            X_t, y_t = shuffle(X, y)
            X_t, _, y_t, _ = train_test_split(X_t, y_t, train_size=0.5)
            self.model[i].compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                                  loss=keras.losses.CategoricalCrossentropy(),
                                  metrics=[keras.metrics.CategoricalAccuracy(), keras.metrics.Precision()])  # compile model

            history = self.model[i].fit(
                X_t, y_t, batch_size=32, epochs=epochs, validation_split=val_split)  # change the validation split when we want to submit to the competition

    def forward(self, X):
        y:tf.Tensor = tf.zeros((X.shape[0], 20))
        
        for i in range(self.tree_count):
            y += self.model[i].predict(X)
        y = keras.activations.softmax(y)
        return y


