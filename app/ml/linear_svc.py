import pickle
import string
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

from app.core.config import ML_MODELS_DIR
from app.ml.base import ModelFactory

nlp = spacy.load("en_core_web_sm")
sent = nlp.create_pipe("sentencizer")
nlp.add_pipe(sent, before="parser")

stopwords = list(STOP_WORDS)

punct = string.punctuation


def text_data_cleaning(sentence):
    doc = nlp(sentence)

    tokens = []
    for token in doc:
        if token.lemma_ != "-PRON-":
            temp = token.lemma_.lower().strip()
        else:
            temp = token.lower_
        tokens.append(temp)

    cleaned_tokens = []
    for token in tokens:
        if token not in stopwords and token not in punct:
            cleaned_tokens.append(token)
    return cleaned_tokens


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "text_data_cleaning":
            return text_data_cleaning
        return super().find_class(module, name)


@ModelFactory.register('linear_svc_en')
class LinearSVC:
    def __init__(self, model_name=None, dataset=None, language=None):

        model_name = "linear-svc-model.pkl"
        model_path = str(ML_MODELS_DIR.joinpath(model_name))

        tokenizer_name = "linear-svc-tfidf-vectorizer.pk"
        tokenizer_path = str(ML_MODELS_DIR.joinpath(tokenizer_name))

        # Load model
        self.model = CustomUnpickler(open(model_path, "rb")).load()

        self.tokenizer = CustomUnpickler(open(tokenizer_path, "rb")).load()

    def predict(self, sentence):
        result = self.model.predict([sentence])[0]

        tag_name = "Positive" if result == 1 else "Negative"

        return tag_name, 0.9000
