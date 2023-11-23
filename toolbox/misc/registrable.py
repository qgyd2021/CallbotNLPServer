from collections import defaultdict
from typing import TypeVar, Type, Dict, List
import importlib
import logging

from allennlp.common.checks import ConfigurationError
from allennlp.common.from_params import FromParams

logger = logging.getLogger(__name__)

T = TypeVar('T')

class Registrable(FromParams):
    _registry: Dict[Type, Dict[str, Type]] = defaultdict(dict)
    default_implementation: str = None

    @classmethod
    def register(cls: Type[T], name: str, exist_ok=False):
        registry = Registrable._registry[cls]
        def add_subclass_to_registry(subclass: Type[T]):
            if name in registry:
                if exist_ok:
                    message = (f"{name} has already been registered as {registry[name].__name__}, but "
                               f"exist_ok=True, so overwriting with {cls.__name__}")
                    logger.info(message)
                else:
                    message = (f"Cannot register {name} as {cls.__name__}; "
                               f"name already in use for {registry[name].__name__}")
                    raise ConfigurationError(message)
            registry[name] = subclass
            return subclass
        return add_subclass_to_registry

    @classmethod
    def by_name(cls: Type[T], name: str) -> Type[T]:
        logger.info(f"instantiating registered subclass {name} of {cls}")
        if name in Registrable._registry[cls]:
            return Registrable._registry[cls].get(name)
        elif "." in name:
            parts = name.split(".")
            submodule = ".".join(parts[:-1])
            class_name = parts[-1]

            try:
                module = importlib.import_module(submodule)
            except ModuleNotFoundError:
                raise ConfigurationError(f"tried to interpret {name} as a path to a class "
                                         f"but unable to import module {submodule}")

            try:
                return getattr(module, class_name)
            except AttributeError:
                raise ConfigurationError(f"tried to interpret {name} as a path to a class "
                                         f"but unable to find class {class_name} in {submodule}")

        else:
            raise ConfigurationError(f"{name} is not a registered name for {cls.__name__}. "
                                     "You probably need to use the --include-package flag "
                                     "to load your custom code. Alternatively, you can specify your choices "
                                     """using fully-qualified paths, e.g. {"model": "my_module.models.MyModel"} """
                                     "in which case they will be automatically imported correctly.")


    @classmethod
    def list_available(cls) -> List[str]:
        keys = list(Registrable._registry[cls].keys())
        default = cls.default_implementation

        if default is None:
            return keys
        elif default not in keys:
            message = "Default implementation %s is not registered" % default
            raise ConfigurationError(message)
        else:
            return [default] + [k for k in keys if k != default]
