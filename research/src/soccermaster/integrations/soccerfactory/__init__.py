"""Project-owned adapters between SoccerFactory artifacts and SoccerMaster."""

from .step3_to_training import convert_step3_to_training_frames

__all__ = ["convert_step3_to_training_frames"]
