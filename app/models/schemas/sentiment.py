from pydantic import BaseModel, Field
from enum import Enum


TagTypes = Enum("TagTypes", {
    "negative": "NEGATIVE",
    "neutral": "NEUTRAL",
    "positive": "POSITIVE",
})


class SentimentResponse(BaseModel):
    text: str
    tag_name: TagTypes = TagTypes.neutral
    score: float
