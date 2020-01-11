from fastapi import APIRouter
from app.api.routes.language import api as language

router = APIRouter()
router.include_router(language.router, tags=["language"], prefix="/language")
