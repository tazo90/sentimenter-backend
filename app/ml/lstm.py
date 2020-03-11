import pickle
from abc import ABCMeta, abstractmethod
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

from app.core.config import ML_MODELS_DIR
from app.ml.base import ModelFactory


class LSTMBase(metaclass=ABCMeta):
    model = None
    tokenizer = None
    model_name = 'lstm'
    max_seq_length = 100

    def __init__(self, *args, **kwargs):
        if 'max_seq_length' in kwargs:
            self.max_seq_length = kwargs.get('max_seq_length')

        self.load_model()

    def load_model(self):
        file_prefix = f"{self.model_name}-{self.language}"

        model_name = f"{file_prefix}-model.h5"
        tokenizer_name = f"{file_prefix}-tokenizer.pickle"

        # Load model
        self.model = load_model(ML_MODELS_DIR.joinpath(model_name))

        # Load tokenizer
        with open(ML_MODELS_DIR.joinpath(tokenizer_name), "rb") as file:
            self.tokenizer = pickle.load(file)

    @abstractmethod
    def predict(self, sentence: str):
        pass

    def model_info(self):
        return {
            "vocab_size": len(self.tokenizer.word_index) + 1,
        }

    def get_padded_sequences(self, sentence: str):
        sequences = self.tokenizer.texts_to_sequences([sentence])

        flat_sequences = [item for seq in sequences for item in seq]

        return pad_sequences(
            [flat_sequences], padding="post", maxlen=self.max_seq_length
        )


@ModelFactory.register('lstm_en')
class LSTMEnglish(LSTMBase):
    language = 'en'

    def predict(self, sentence: str):
        padded_sequences = self.get_padded_sequences(sentence=sentence)

        score = self.model.predict(padded_sequences)[0][0]
        tag_name = "Positive" if score >= 0.5 else "Negative"

        return tag_name, score


@ModelFactory.register('lstm_pl')
class LSTMPolish(LSTMBase):
    language = 'pl'

    def predict(self, sentence: str):
        padded_sequences = self.get_padded_sequences(sentence=sentence)

        neu, pos, neg = self.model.predict(padded_sequences)[0]
        neutral = "{:.2f}".format(float(neu)*100)
        positive = "{:.2f}".format(float(pos)*100)
        negative = "{:.2f}".format(float(neg)*100)

        print("Predict", negative, neutral, positive)

        if positive >= negative:
            tag_name = "Positive"
            score = positive
        else:
            tag_name = "Negative"
            score = negative

        # Check neutral case
        max_score = max(negative, neutral, positive)
        if max_score == neutral:
            if positive >= negative:
                tag_name = "Positive"
            else:
                tag_name = "Negative"
            score = neutral

        return tag_name, score
