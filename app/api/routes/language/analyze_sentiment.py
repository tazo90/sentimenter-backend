from typing import List

from fastapi import APIRouter, HTTPException, Path

from app.models.domain.sentiment import Sentiment
from app.models.schemas.sentiment import SentimentOut
from app.ml.lstm import LSTM
from app.ml.vader import Vader
from app.ml.linear_svc import LinearSVC
from app.ml.utils import build_wordcloud

router = APIRouter()

models = {"lstm": LSTM, "bert": None, "vader": Vader, "linear_svc": LinearSVC}


@router.post("/", response_model=SentimentOut)
async def analyze(payload: Sentiment):
    data = payload.dict()
    text, lang = data["text"], data["language"]

    scores = []
    word_cloud_url = build_wordcloud(text=text, lang=lang)

    for model_name in data["model"]:
        model_class = models[model_name]

        model = model_class(
            model_name=model_name, dataset="imdb", language=lang
        )

        tag_name, score = model.predict(sentence=text)

        if hasattr(model, "model_info"):
            model_info = model.model_info()
        else:
            model_info = None

        scores.append({
            "model_name": model_name,
            "score": score,
            "tag_name": tag_name,
            "model_info": model_info,
        })

    return {
        "text": text, 
        "scores": scores, 
        "word_cloud_url": word_cloud_url
    }
