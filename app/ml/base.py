import logging
from typing import Callable

logger = logging.getLogger(__name__)


class ModelFactory:
    registry = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        def inner_wrapper(wrapped_class) -> Callable:
            if name in cls.registry:
                logger.warning('Model class %s already exists. Will replace it', name)
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create(cls, name: str, **kwargs):
        if name not in cls.registry:
            logger.warning('Model class %s does not exist in the registry', name)
            return None

        _class = cls.registry[name]
        instance = _class(**kwargs)
        return instance
