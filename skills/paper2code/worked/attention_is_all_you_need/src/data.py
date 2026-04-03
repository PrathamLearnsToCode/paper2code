"""
Attention Is All You Need — Dataset and Data Loading

Paper: https://arxiv.org/abs/1706.03762
Implements: Data loading skeleton for WMT translation tasks.

Section references:
  §5.1 — "We trained on the standard WMT 2014 English-German dataset
          consisting of about 4.5 million sentence pairs."
  §5.1 — "Sentences were encoded using byte-pair encoding, which has a shared
          source-target vocabulary of about 37000 tokens."

NOTE: This file provides the Dataset class skeleton. You must:
  1. Download WMT 2014 EN-DE from http://www.statmt.org/wmt14/translation-task.html
  2. Apply BPE tokenization (e.g., sentencepiece with 37000 merge operations)
  3. Set the data_dir in configs/base.yaml
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


class WMTTranslationDataset(Dataset):
    """§5.1 — WMT 2014 English-German translation dataset.

    "We trained on the standard WMT 2014 English-German dataset consisting of
     about 4.5 million sentence pairs. Sentences were encoded using byte-pair
     encoding, which has a shared source-target vocabulary of about 37000 tokens."

    Expected data format:
        {data_dir}/
        ├── train.src     # Source sentences, one per line, BPE-tokenized
        ├── train.tgt     # Target sentences, one per line, BPE-tokenized
        ├── valid.src
        ├── valid.tgt
        ├── test.src
        ├── test.tgt
        └── vocab.txt     # Shared BPE vocabulary, one token per line

    How to obtain:
        1. Download WMT 2014 EN-DE data from http://www.statmt.org/wmt14/
        2. Apply BPE tokenization with ~37000 merge operations
           (use sentencepiece: spm_train --input=data.txt --model_prefix=wmt
            --vocab_size=37000 --model_type=bpe)
        3. Encode source and target files with the learned BPE model

    Preprocessing (§5.1):
        - Byte-pair encoding with shared source-target vocabulary
        - ~37000 BPE tokens for EN-DE
        - ~32000 BPE tokens for EN-FR (word-piece model)
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_seq_len: int = 512,
        pad_idx: int = 0,
        bos_idx: int = 1,
        eos_idx: int = 2,
    ):
        """
        Args:
            data_dir: path to the preprocessed data directory
            split: one of "train", "valid", "test"
            max_seq_len: maximum sequence length (truncate longer sequences)
            pad_idx: padding token index
            bos_idx: beginning-of-sentence token index
            eos_idx: end-of-sentence token index
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_seq_len = max_seq_len
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

        # TODO: Load tokenized data
        # self.src_data, self.tgt_data = self._load_data()
        self.src_data: List[List[int]] = []
        self.tgt_data: List[List[int]] = []

    def _load_data(self) -> Tuple[List[List[int]], List[List[int]]]:
        """Load BPE-tokenized source and target sentences.

        TODO: Implement based on your tokenization format.
        Expected: each line is space-separated token IDs.
        """
        src_path = self.data_dir / f"{self.split}.src"
        tgt_path = self.data_dir / f"{self.split}.tgt"

        if not src_path.exists() or not tgt_path.exists():
            raise FileNotFoundError(
                f"Data files not found: {src_path}, {tgt_path}\n"
                f"Please download and preprocess the WMT 2014 dataset.\n"
                f"See docstring for instructions."
            )

        # TODO: Parse the tokenized files
        raise NotImplementedError(
            "Implement _load_data() to read your BPE-tokenized data files."
        )

    def __len__(self) -> int:
        return len(self.src_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load and preprocess a single sentence pair.

        Returns:
            dict with keys:
                "src": source token IDs — shape: (src_len,)
                "tgt": target token IDs (with BOS prepended) — shape: (tgt_len,)
                "tgt_labels": target labels (with EOS appended) — shape: (tgt_len,)
        """
        src_tokens = self.src_data[idx][:self.max_seq_len]
        tgt_tokens = self.tgt_data[idx][:self.max_seq_len - 1]  # leave room for BOS/EOS

        # Decoder input: [BOS] + target tokens
        tgt_input = [self.bos_idx] + tgt_tokens
        # Decoder labels: target tokens + [EOS]
        tgt_labels = tgt_tokens + [self.eos_idx]

        return {
            "src": torch.tensor(src_tokens, dtype=torch.long),
            "tgt": torch.tensor(tgt_input, dtype=torch.long),
            "tgt_labels": torch.tensor(tgt_labels, dtype=torch.long),
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]], pad_idx: int = 0) -> Dict[str, torch.Tensor]:
    """Pad sequences to the same length within a batch.

    §5.1 — "Each training batch contained a set of sentence pairs containing
    approximately 25000 source tokens and 25000 target tokens."

    NOTE: The paper batches by approximate token count, not by number of sentences.
    This collate function pads by sequence length for simplicity.
    For exact reproduction, implement token-count-based batching.
    """
    src_tensors = [item["src"] for item in batch]
    tgt_tensors = [item["tgt"] for item in batch]
    label_tensors = [item["tgt_labels"] for item in batch]

    src_padded = torch.nn.utils.rnn.pad_sequence(src_tensors, batch_first=True, padding_value=pad_idx)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_tensors, batch_first=True, padding_value=pad_idx)
    labels_padded = torch.nn.utils.rnn.pad_sequence(label_tensors, batch_first=True, padding_value=pad_idx)

    return {
        "src": src_padded,
        "tgt": tgt_padded,
        "tgt_labels": labels_padded,
    }
