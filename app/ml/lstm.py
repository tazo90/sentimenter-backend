import pickle
from pathlib import Path
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pylab as plt
from wordcloud import WordCloud
from many_stop_words import get_stop_words

from app.core.config import ML_MODELS_PATH
from app.ml.loader import DatasetLoader


class LSTM:
    def __init__(self, model_name, dataset, language, max_seq_length=100):
        self.model_name = model_name
        self.dataset = dataset
        self.language = language
        self.file_prefix = f'{model_name}-{dataset}-{language}'
        self.max_seq_length = max_seq_length

        model_name = f'{self.file_prefix}-model.h5'
        tokenizer_name = f'{self.file_prefix}-tokenizer.pickle'

        # Load model
        self.model = load_model(ML_MODELS_PATH.joinpath(model_name))

        # Load tokenizer
        with open(ML_MODELS_PATH.joinpath(tokenizer_name), 'rb') as file:
            self.tokenizer = pickle.load(file)

    def predict(self, sentence):
        sequences = self.tokenizer.texts_to_sequences([sentence])

        flat_sequences = [item for seq in sequences for item in seq]

        padded_sequences = pad_sequences(
            [flat_sequences],
            padding='post',
            maxlen=self.max_seq_length
        )

        return self.model.predict(padded_sequences)

    def model_info(self):
        return {
            'vocab_size': len(self.tokenizer.word_index) + 1
        }

    def word_cloud(self):
        loader = DatasetLoader('IMDB Dataset.csv')
        dataset = loader.process()

        stop_words = get_stop_words(self.language)

        wordcloud = WordCloud(
            background_color='white',
            stopwords=stop_words,
            max_words=200,
            max_font_size=40,
            random_state=42
        ).generate(str(dataset))

        fig = plt.figure(1)
        plt.imshow(wordcloud)
        plt.axis('off')

        out_file = f'{self.file_prefix}-word-cloud.png'
        plt.savefig(ML_MODELS_PATH.joinpath(out_file), bbox_inches='tight')
