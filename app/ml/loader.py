import re

from app.core.config import ML_MODELS_PATH

TAG_RE = re.compile(r'<[^>]+>')

class DatasetLoader:
    def __init__(self, filename):
        self.filename = filename

        self.load(filename)

    def load(self, filename):
        self.raw_dataset = pd.read_csv(ML_MODELS_PATH.joinpath(filename))

    def remove_tags(self, text):
        return TAG_RE.sub('', text)

    def process_sentence(self, sentence):
        # Removing html tags
        sentence = self.remove_tags(sentence)
        # Remove linkba
        sentence = re.sub(r'https:\/\/[a-zA-Z]*\.com',' ',sentence)
        # Remove number
        sentence = re.sub(r'\d+',' ',sentence)
        # Remove punctuations and number
        sentence = re.sub('[^a-zA-Z]', ' ', sentence)
        # Single character removal
        sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
        # Removing white spaces
        sentence = re.sub(r'\s+', ' ', sentence)
        # Remove single character
        sentence = re.sub(r"\b[a-zA-Z]\b", ' ', sentence)
        # Remove bracket
        sentence = re.sub(r'\W+',' ',sentence)
        # Make sentence lowercase
        sentence = sentence.lower()
        return sentence

    def process(self):
        dataset = []
        sentences = list(self.raw_dataset['review'])
        for sen in sentences:
            dataset.append(self.process_sentence(sen))

        return dataset
