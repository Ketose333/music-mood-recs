"""PyTorch Dataset for mel-spectrogram mood classification.

Loads precomputed .npy mel-spectrograms (from scripts/extract_melspecs.py)
and multi-hot label vectors. Splits are read from the manifest CSV's
``split`` column.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MelspecDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str,
        subset_meta_csv: str,
        tags: Sequence[str],
        split: str = "train",
        n_mels: int = 128,
    ):
        manifest = pd.read_csv(manifest_csv)
        manifest = manifest[manifest["split"] == split].reset_index(drop=True)
        meta = pd.read_csv(subset_meta_csv).set_index("TRACK_ID")
        self.tags = list(tags)
        self.n_mels = n_mels
        self.records: list[dict] = []
        for _, row in manifest.iterrows():
            track_id = row["TRACK_ID"]
            if track_id not in meta.index:
                continue
            mr = meta.loc[track_id]
            label = np.array([int(mr.get(f"tag_{t}", 0)) for t in self.tags], dtype=np.float32)
            self.records.append(
                {
                    "track_id": track_id,
                    "npy_path": row["npy_path"],
                    "label": label,
                    "path": row["PATH"],
                }
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        rec = self.records[idx]
        mel = np.load(rec["npy_path"]).astype(np.float32)
        if mel.shape[0] != self.n_mels:
            raise ValueError(f"n_mels mismatch for {rec['npy_path']}: {mel.shape[0]} != {self.n_mels}")
        x = torch.from_numpy(mel).unsqueeze(0)
        y = torch.from_numpy(rec["label"])
        return x, y, rec["track_id"]
