from fastapi import APIRouter

from app.models.domain.sentiment import Sentiment
from app.models.schemas.sentiment import SentimentOut
from app.ml.utils import build_wordcloud
from app.ml.base import ModelFactory

router = APIRouter()


@router.post("/", response_model=SentimentOut)
async def analyze(payload: Sentiment):
    scores = []
    data = payload.dict()
    text, lang = data["text"], data["language"]
    word_cloud_url = build_wordcloud(text=text, lang=lang)

    for model_name in data["model"]:
        model_key = f"{model_name}_{lang}"

        print(ModelFactory.registry)
        model = ModelFactory.create(model_key)

        tag_name, score = model.predict(sentence=text)

        model_info = None
        if hasattr(model, "model_info"):
            model_info = model.model_info()

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
