from scipy.sparse import data
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, LabelBinarizer
import os
import shutil
import numpy as np

save_path = "/kaggle/working/"
input_path = "/kaggle/input/ml-hw5/"
# save_path = ""
# input_path = ""


training_set = pd.read_json(input_path + "train.json")
validation_set = pd.read_json(input_path + "test.json")

oh = LabelBinarizer()
# oh.fit_transform()

mlb = MultiLabelBinarizer()
# print(training_set)
X_train: np.ndarray = mlb.fit_transform(training_set["ingredients"])
y_train: np.ndarray = oh.fit_transform(
    np.array(training_set["cuisine"]).reshape(-1, 1))

X_test = mlb.transform(validation_set["ingredients"])
print()


class MLP:
    def __init__(self, input_shape, load_model=False) -> None:
        data_in = keras.Input(shape=(input_shape, ))
        x = keras.layers.Dense(
            1024, activation='relu')(data_in)
        x = keras.layers.Dense(1024, activation='relu')(x)
        x = keras.layers.Dense(20, activation='softmax')(x)
        if os.path.exists(save_path + "my_model") and load_model:
                        
            print("model exists!")
            self.model = keras.models.load_model(save_path + "my_model")
        else:
            self.model = keras.Model(inputs=data_in, outputs=x)

    def train(self, X, y):
        
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
                           loss=keras.losses.MeanSquaredError(),
                           metrics=[keras.metrics.Accuracy(), keras.metrics.Precision()])

        history = self.model.fit(
            X, y, epochs=60, validation_split=0.3)
        pass

    def forward(self, X):
        
        y = self.model.predict(X)
        return y

    def test(self, X_test, y_test):
        res = self.model.evaluate(X_test, y_test)
        print(res)


m = MLP(X_train.shape[1], load_model=True)
# print(m.model.summary())
m.train(X_train, y_train)
m.model.save(save_path + "my_model")
y_pred = m.forward(X_test)
y_pred = oh.inverse_transform(y_pred)
print(y_pred)
ss = pd.Series(y_pred.squeeze())
outp = pd.DataFrame(columns=["id", "Category"])
outp["id"] = validation_set["id"]
outp["Category"] = ss
# print(outp.head(20).to_csv(index=None))

with open(save_path + "test_result.csv", 'w+') as f:
    f.write(outp.to_csv(index=None))


