from typing import List

from fastapi import APIRouter, HTTPException, Path

from app.models.domain.sentiment import Sentiment

router = APIRouter()


@router.post("/")
async def analyze(payload: Sentiment):
    return {
        "text": "I saw cool article",
        "tag_name": "Positive",
        "confidence": "0.998",
    }
