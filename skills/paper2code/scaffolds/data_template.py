"""
{{PAPER_TITLE}} — Dataset and Data Loading

Paper: https://arxiv.org/abs/{{ARXIV_ID}}
Implements: Data loading for {{DATASET_NAME}}

Section references:
  {{§SECTION}} — {{data description}}

NOTE: This file provides the Dataset class skeleton. You must:
  1. Download the dataset from {{DATASET_URL}}
  2. Set the data_dir in configs/base.yaml
  3. Implement any dataset-specific preprocessing (marked with TODO)
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


class {{DATASET_CLASS}}(Dataset):
    """§{{SECTION}} — Dataset for {{PAPER_TITLE}}.
    
    "{{Quote from paper about the dataset used}}"
    
    Expected data format:
        {{describe the expected file structure / data format}}
    
    How to obtain:
        {{instructions for downloading the dataset}}
    
    Preprocessing:
        {{describe preprocessing steps from the paper}}
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        # Add other params from config
    ):
        """
        Args:
            data_dir: path to the dataset root directory
            split: one of "train", "val", "test"
        """
        self.data_dir = Path(data_dir)
        self.split = split
        
        # TODO: Load file list / metadata
        # self.samples = self._load_samples()
    
    def _load_samples(self):
        """Load sample paths/metadata for the given split.
        
        TODO: Implement based on the dataset structure.
        """
        raise NotImplementedError(
            f"Dataset loading not implemented. "
            f"Download the dataset and implement _load_samples() for your data format."
        )
    
    def _preprocess(self, sample):
        """Apply preprocessing as described in §{{SECTION}}.
        
        TODO: Implement the paper's preprocessing pipeline:
        {{list preprocessing steps from the paper}}
        """
        raise NotImplementedError("Implement preprocessing per §{{SECTION}}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load and preprocess a single sample.
        
        Returns:
            dict with keys:
                {{key_1}}: {{description}} — shape: {{shape}}
                {{key_2}}: {{description}} — shape: {{shape}}
        """
        # TODO: Implement sample loading
        # sample = self.samples[idx]
        # processed = self._preprocess(sample)
        # return processed
        raise NotImplementedError("Implement __getitem__ for your data format")


def build_dataloader(
    config: dict,
    split: str = "train",
) -> DataLoader:
    """Build a DataLoader from config.
    
    Args:
        config: data config dict from base.yaml
        split: "train", "val", or "test"
    """
    dataset = {{DATASET_CLASS}}(
        data_dir=config["data_dir"],
        split=split,
    )
    
    return DataLoader(
        dataset,
        batch_size=config.get("batch_size", 32),
        shuffle=(split == "train"),
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
        drop_last=(split == "train"),
    )
