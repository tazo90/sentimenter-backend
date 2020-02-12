from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

from app.api.routes.api import router as api_router
from app.db import engine, metadata, database

metadata.create_all(engine)


def get_application() -> FastAPI:
    application = FastAPI(title="Sentimenter", debug=True, version="0.0.0")

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )

    application.include_router(api_router, prefix="/api")
    application.mount("/static", StaticFiles(directory="static"), name="static")

    return application


app = get_application()


@app.on_event("startup")
async def startup():
    await database.connect()


@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()
