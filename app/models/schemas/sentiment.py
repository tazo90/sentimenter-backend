from pydantic import BaseModel, Field
from enum import Enum


TagTypes = Enum("TagTypes", {
    "negative": "Negative",
    "neutral": "Neutral",
    "positive": "Positive",
})


class ModelInfoOut(BaseModel):
    vocab_size: int
    word_cloud_url: str

class SentimentOut(BaseModel):
    text: str
    tag_name: TagTypes = TagTypes.neutral
    score: float
    model_info: ModelInfoOut
