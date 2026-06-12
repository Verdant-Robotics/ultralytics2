# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from .predict import BoxInstPredictor
from .train import BoxInstTrainer
from .val import BoxInstValidator

__all__ = 'BoxInstPredictor', 'BoxInstTrainer', 'BoxInstValidator'
