from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def tokenize_feature(df: pd.DataFrame, feature: str) -> np.array:
    tk = Tokenizer(filters='')
    tk.fit_on_texts(df[feature])
    title_tok = tk.texts_to_sequences(df[feature])
    max_length = np.max([len(i) for i in title_tok])
    title_pad = pad_sequences(title_tok, padding='post', maxlen=max_length)

    return title_pad


def split_train_test(features: np.array, target: np.array):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0,
                                                        shuffle=True)  # we are setting the seed here
    return X_train, X_test, y_train, y_test
