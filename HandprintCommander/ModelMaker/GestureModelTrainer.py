import pickle
from typing import List
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import models, layers, optimizers
from sklearn.model_selection import train_test_split
from HandprintCommander.Utils import preprocessing_train_data


def _arrange_model(classes_n):
    model = models.Sequential(
        [
            layers.Dense(units=40, activation="relu", input_shape=(122,)),
            layers.Dense(units=20, activation="relu"),
            layers.Dense(units=classes_n, activation="softmax"),
        ]
    )

    optimizer = optimizers.Adam(lr=0.001)

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=optimizer,
        metrics=["categorical_accuracy"],
    )
    
    return model

def _onehot_encoding(labels) -> np.ndarray:
    labels = list(map(int, labels))
    max_label = max(labels)
    onehot_labels = []
    for label in labels:
        onehot_label = [0] * (max_label + 1)
        onehot_label[label] = 1
        onehot_labels.append(onehot_label)
    return np.array(onehot_labels)

def train():
    # import data
    data = pickle.load(open("gesture_data.bin", "rb"))

    # preprocessing
    processed_data, label = preprocessing_train_data(data)
    
    # one-hot encoding
    labels_onehot = _onehot_encoding(label)
    
    print("a")
    
    X_train, X_val, y_train, y_val = train_test_split(
        processed_data, labels_onehot, test_size=0.2, random_state=334
    )
    
    model = _arrange_model(labels_onehot.shape[1])

    # train
    history = model.fit(
        X_train, y_train, epochs=100, validation_data=(X_val, y_val)
    )

    # save
    model.save("model")


if __name__ == "__main__":
    train()
