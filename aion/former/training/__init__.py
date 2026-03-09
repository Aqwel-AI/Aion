"""
Training pipeline: trainer, optimizer, loss.

Trainer wraps a model and Adam; train_step and train_epoch run forward,
loss, backward, and optimizer step. cross_entropy_loss supports next-token
prediction with optional padding mask.
"""

from .trainer import Trainer
from .optimizer import Adam
from .loss import cross_entropy_loss

__all__ = ["Trainer", "Adam", "cross_entropy_loss"]
