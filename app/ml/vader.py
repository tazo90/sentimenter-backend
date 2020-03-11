import operator

import spacy
from spacy.tokens import Doc
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from app.ml.base import ModelFactory


nlp = spacy.load("en_core_web_sm")

sent_analyzer = SentimentIntensityAnalyzer()


def sentiment_scores(docx):
    return sent_analyzer.polarity_scores(docx.text)


Doc.set_extension("sentimenter", getter=sentiment_scores)


@ModelFactory.register('vader_en')
class Vader:

    def predict(self, sentence):
        result = nlp(sentence)
        scores = result._.sentimenter

        del scores["compound"]

        max_key = max(scores.items(), key=operator.itemgetter(1))[0]

        tag_name = "Positive" if max_key == "pos" else "Negative"
        # Map neutral value to negative or positive
        if max_key == "neu":
            if scores["neg"] > scores["pos"]:
                tag_name = "Negative"
            else:
                tag_name = "Positive"

        score = scores[max_key]

        return tag_name, score
