from typing import List

from fastapi import APIRouter, HTTPException, Path

from app.models.domain.sentiment import Sentiment
from app.models.schemas.sentiment import SentimentOut
from app.ml.lstm import LSTM

router = APIRouter()


@router.post("/", response_model=SentimentOut)
async def analyze(payload: Sentiment):
    data = payload.dict()

    lstm = LSTM(
        model_name=data["model"],
        dataset="imdb",
        language=data["language"]
    )

    score = lstm.predict(sentence=data["text"])[0][0]
    tag_name = 'Positive' if score >= 0.5 else 'Negative'
    model_info = lstm.model_info()

    return {
        "text": data["text"],
        "tag_name": tag_name,
        "score": str(score),
        "model_info": {
            "vocab_size": model_info["vocab_size"],
            "word_cloud_url": "http://localhost:8000/static/lstm-imdb-en-word-cloud.png"
        }
    }
