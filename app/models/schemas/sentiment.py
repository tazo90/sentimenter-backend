from typing import List
from pydantic import BaseModel
from enum import Enum


TagTypes = Enum(
    "TagTypes", {
        "negative": "Negative",
        "neutral": "Neutral",
        "positive": "Positive",
    }
)


class ModelInfoOut(BaseModel):
    vocab_size: int


class ScoreOut(BaseModel):
    model_name: str
    tag_name: TagTypes = TagTypes.neutral
    score: float
    model_info: ModelInfoOut = None


class SentimentOut(BaseModel):
    text: str
    scores: List[ScoreOut]
    word_cloud_url: str
