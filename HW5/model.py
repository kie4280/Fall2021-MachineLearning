# preprocessing and model construction

from scipy.sparse import data
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, LabelBinarizer
import os
import shutil
import numpy as np
from sklearn.utils import shuffle

# save_path = "/kaggle/working/"
# input_path = "/kaggle/input/ml-hw5/"
# model_save_ = "/kaggle/input/notebook-ml-hw5/"
save_path = ""
input_path = ""
model_save_ = ""


training_set = pd.read_json(input_path + "train.json")
validation_set = pd.read_json(input_path + "test.json")

oh = LabelBinarizer()
# oh.fit_transform()

mlb = MultiLabelBinarizer()
# print(training_set)
X_train: np.ndarray = mlb.fit_transform(training_set["ingredients"])
y_train: np.ndarray = oh.fit_transform(
    np.array(training_set["cuisine"]).reshape(-1, 1))
X_train, y_train = shuffle(X_train, y_train)

X_test = mlb.transform(validation_set["ingredients"])


class MLP:
    def __init__(self, input_shape, load_model=False) -> None:
        data_in = keras.Input(shape=(input_shape, ))
        x = data_in
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(400, activation='relu')(x)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Dense(100, activation='relu')(x)
        x = keras.layers.Dense(20, activation='softmax')(x)
        if os.path.exists(model_save_ + "my_model") and load_model:
            print("model exists!")
            self.model = keras.models.load_model(model_save_ + "my_model") # load saved model
        else:
            self.model = keras.Model(inputs=data_in, outputs=x) # construct model

    def train(self, X, y, val_split = 0.1):
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                           loss=keras.losses.CategoricalCrossentropy(),
                           metrics=[keras.metrics.CategoricalAccuracy(), keras.metrics.Precision()]) # compile model

        history = self.model.fit(
            X, y, batch_size=32, epochs=8, validation_split=val_split) # change the validation split when we want to submit to the competition

    def forward(self, X):
        y = self.model.predict(X)
        return y


m = MLP(X_train.shape[1], load_model=False)
m.train(X_train, y_train)
m.model.save(save_path + "my_model") # save weights
y_pred = m.forward(X_test) # inference
y_pred = oh.inverse_transform(y_pred, threshold=0.5) # get back labels

# construct submission csv
ss = pd.Series(y_pred.squeeze())
outp = pd.DataFrame(columns=["id", "Category"])
outp["id"] = validation_set["id"]
outp["Category"] = ss

# save csv
with open(save_path + "test_result.csv", 'w+') as f:
    f.write(outp.to_csv(index=None))
