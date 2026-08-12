"""
src/__init__.py  —  NLP Text Summarization Project
"""
from .utils import get_logger, set_seed, load_yaml_config
from .dataset_loader import DatasetLoader
from .preprocessor import Preprocessor
from .model_factory import ModelFactory
from .evaluator import Evaluator
from .inference import SummarizationInference

__version__ = "1.0.0"
__all__ = [
    "get_logger", "set_seed", "load_yaml_config",
    "DatasetLoader", "Preprocessor", "ModelFactory",
    "Evaluator", "SummarizationInference",
]
