from fastapi import APIRouter

from app.api.routes.language import analyze_sentiment

router = APIRouter()

router.include_router(analyze_sentiment.router, prefix="/analyze_sentiment")
