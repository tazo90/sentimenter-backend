from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

STATIC_DIR = BASE_DIR.parent.joinpath('static')

ML_MODELS_DIR = BASE_DIR.joinpath('ml', 'models')
