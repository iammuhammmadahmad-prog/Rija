"""Phase 5: Training system."""

__all__ = ["TextChunkDataset", "Trainer", "TrainConfig"]


def __getattr__(name):
    if name == "TextChunkDataset":
        from trainer.dataset import TextChunkDataset
        return TextChunkDataset
    if name == "Trainer":
        from trainer.train import Trainer
        return Trainer
    if name == "TrainConfig":
        from trainer.config import TrainConfig
        return TrainConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
