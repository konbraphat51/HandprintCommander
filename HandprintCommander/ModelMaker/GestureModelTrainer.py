import pickle
from typing import List
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import models, layers, optimizers
from sklearn.model_selection import train_test_split
from HandprintCommander.Utils import preprocessing_train_data


model = models.Sequential(
    [
        layers.Dense(units=40, activation="relu", input_shape=(122,)),
        layers.Dense(units=20, activation="relu"),
        layers.Dense(units=2, activation="softmax"),
    ]
)

optimizer = optimizers.Adam(lr=0.001)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"],
)


def train():
    # import data
    data = pickle.load(open("gesture_data.bin", "rb"))

    # preprocessing
    processed_data, label = preprocessing_train_data(data)
    X_train, X_val, y_train, y_val = train_test_split(
        processed_data, label, test_size=0.2, random_state=334
    )

    # train
    history = model.fit(
        X_train, y_train, epochs=100, validation_data=(X_val, y_val)
    )

    # evaluate
    score = model.evaluate(X_val, y_val, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    # save
    model.save("model")


if __name__ == "__main__":
    train()
