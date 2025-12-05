"""PyTorch Lightning DataModule for ICU data.

TODO: Implement ICUDataModule class that:
- Handles train/val/test splits (patient-level)
- Applies data transforms/augmentations
- Returns DataLoaders with proper batching
"""

from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader


class ICUDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for ICU data.
    
    TODO: Implement full DataModule with patient-level splits.
    """

    def __init__(
        self,
        processed_dir: str,
        batch_size: int = 64,
        num_workers: int = 4,
        seq_length: int = 48,
    ) -> None:
        """Initialize DataModule.
        
        Args:
            processed_dir: Directory containing preprocessed/extracted features.
            batch_size: Batch size for training.
            num_workers: Number of data loading workers.
            seq_length: Maximum sequence length in hours.
        """
        super().__init__()
        self.processed_dir = processed_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seq_length = seq_length

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets for train/val/test.
        
        Args:
            stage: Stage name ('fit', 'validate', 'test', or None).
        """
        # TODO: Load datasets and create patient-level splits
        pass

    def train_dataloader(self) -> DataLoader:
        """Return training DataLoader."""
        # TODO: Return actual DataLoader
        raise NotImplementedError

    def val_dataloader(self) -> DataLoader:
        """Return validation DataLoader."""
        # TODO: Return actual DataLoader
        raise NotImplementedError

    def test_dataloader(self) -> DataLoader:
        """Return test DataLoader."""
        # TODO: Return actual DataLoader
        raise NotImplementedError

