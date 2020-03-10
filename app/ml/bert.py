import pickle
from pathlib import Path
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pylab as plt
from wordcloud import WordCloud
from many_stop_words import get_stop_words

from app.core.config import ML_MODELS_PATH
from app.ml.loader import DatasetLoader

import math
import h5py
import pathlib


class BERT:
    def __init__(self, model_name, dataset, language, max_seq_length=100):
        self.model_name = model_name
        self.dataset = dataset
        self.language = language
        self.file_prefix = f"{model_name}-{dataset}-{language}"
        self.max_seq_length = max_seq_length

        model_name = f"{self.file_prefix}-model/"
        tokenizer_name = f"{self.file_prefix}-tokenizer.pickle"

        # Load model
        model_path = str(ML_MODELS_PATH.joinpath(model_name))
        self.model = tf.keras.models.load_model(model_path)

        # Load tokenizer
        with open(ML_MODELS_PATH.joinpath(tokenizer_name), "rb") as file:
            self.tokenizer = pickle.load(file)

    def tokenize_reviews(self, text_reviews):
        return self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(text_reviews)
        )

    def predict(self, sentence):
        # tokenized_reviews = [self.tokenize_reviews(review) for review in reviews]
        tokens = self.tokenize_reviews(sentence)
        inputs = tf.expand_dims(tokens, 0)

        output = self.model(inputs, training=False)

        sentiment = math.floor(output * 2)

        return sentiment

        # sequences = self.tokenizer.texts_to_sequences([sentence])

        # flat_sequences = [item for seq in sequences for item in seq]

        # padded_sequences = pad_sequences(
        #     [flat_sequences], padding="post", maxlen=self.max_seq_length
        # )

        # return self.model.predict(padded_sequences)

    def model_info(self):
        return {"vocab_size": len(self.tokenizer.word_index) + 1}
