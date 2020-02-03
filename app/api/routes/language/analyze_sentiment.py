from typing import List

from fastapi import APIRouter, HTTPException, Path

from app.models.domain.sentiment import Sentiment
from app.models.schemas.sentiment import SentimentResponse

router = APIRouter()


@router.post("/", response_model=SentimentResponse)
async def analyze(payload: Sentiment):
    return {
        "text": "I saw cool article",
        "tag_name": "POSITIVE",
        "score": "0.998",
    }
