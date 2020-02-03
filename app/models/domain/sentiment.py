from pydantic import BaseModel, Field


class Sentiment(BaseModel):
    text: str = Field(..., min_length=3, max_length=1000)
    language: str = Field(..., min_length=2, max_length=3)