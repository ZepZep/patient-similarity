import pandas as pd
import numpy as np
from itertools import product
from tqdm.auto import tqdm

from datasets import Dataset
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Bidirectional
from tensorflow.keras.callbacks import Callback


def load_ds(path):
    ds = Dataset.load_from_disk(path)
    ds.set_format("numpy")
    return ds["input_ids"], ds["label"]


def build_network(vocab, labels, dropout, cut):
    inptW = Input(shape=(cut,))
    embedding = Embedding(input_dim=vocab, output_dim=128,
                         input_length=cut, mask_zero=True)
    embW = embedding(inptW)
    embW = Dropout(dropout)(embW)

    # biLSTM
    bilstm1 = Bidirectional(
      LSTM(units=300, return_sequences=False, recurrent_dropout=0.1)
    )(embW)
    bilstm1 = Dropout(dropout)(bilstm1)

    out = Dense(labels, activation="softmax")(bilstm1)

    # build and compile model
    model = Model(inptW, out)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


class SaveCallback(Callback):
    def __init__(self, model, model_name):
        self.model=model
        self.model_name = model_name

    def on_epoch_end(self, *args):
        print("\nsaving model")
        self.model.save(self.model_name)


def train_bilstm(parts_path, cut=150, epochs=10, dropout=0.1):
    model_name = f"{parts_path}/models/bilstm"

    print("--> Loading dataset")
    xt, yt = load_ds(f"{parts_path}/train.hf")
    xv, yv = load_ds(f"{parts_path}/test.hf")

    xt = xt[:, 1:cut+1]
    xv = xv[:, 1:cut+1]

    print("--> Building NN")
    vocab_size = 51961
    n_labels = 2078
    model = build_network(vocab=vocab_size, labels=n_labels, dropout=dropout, cut=cut)
    model.summary()

    print("\n--> Training")

    model.fit(
        xt, yt, batch_size=512, epochs=epochs,
        validation_data=(xv,yv),
        verbose=1,
        callbacks=[SaveCallback(model, model_name)],
    )

    model.save(model_name)

    print("\n--> Predicting")
    y = model.predict(xv)
    np.savez_compressed(f"{parts_path}/predictions/pred_bilstm.npz", y=y)
