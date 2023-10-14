import re
from pathlib import Path
from typing import Optional, Any

import numpy as np
import scipy.io as sio
import torch
import torch.utils.data as D

__all__ = ["SEED"]


class SEED(D.Dataset):
    def __init__(
            self,
            root: [str | Path],
            seq_length: int = 100,
    ) -> None:
        self.root = Path(root)
        self.seq_length = seq_length

        self.processed_folder = self.root / "processed"
        self.processed_folder.mkdir(exist_ok=True)
        self.data_file = self.processed_folder / f"data.pt"
        self.target_file = self.processed_folder / f"target.pt"

        self.data, self.target = self._load_data()

    def _load_legacy_data(self) -> Optional[tuple[list, list]]:
        try:
            data = torch.load(self.data_file)
            target = torch.load(self.target_file)
            if len(data) == len(target):
                return data, target
        finally:
            return None

    @staticmethod
    def _load_eeg(file) -> dict:
        data = sio.loadmat(file)
        eeg = dict()
        for key in data.keys():
            result = re.match("\\w+eeg(\\d+)", key)
            if result is None:
                continue
            eeg[int(result.group(1)) - 1] = data[key].astype(np.float32)
        return eeg

    def _load_data(self) -> tuple[list, list]:
        legacy = self._load_legacy_data()
        if legacy is not None:
            return legacy

        data, target = [], []
        folder = self.root / "Preprocessed_EEG"
        label_file = folder / "label.mat"
        label = sio.loadmat(label_file)["label"][0].astype(np.int64) + 1
        for file in Path(self.root, "Preprocessed_EEG").iterdir():
            if re.match("\\d+_\\d+.mat", file.name) is None:
                continue
            eeg = self._load_eeg(file)
            for i in range(15):
                print(eeg[i])
                data.append(eeg[i])
                target.append(label[i])

        torch.save(data, self.data_file)
        torch.save(target, self.target_file)
        return data, target

    def __len__(self):
        return sum(data.shape[1] // self.seq_length for data in self.data)

    def __getitem__(self, idx) -> Any:
        if idx < 0:
            raise IndexError
        for d, t in zip(self.data, self.target):
            length = d.shape[1] // self.seq_length
            if idx > length:
                idx -= length
                continue
            start = (d.shape[1] % self.seq_length) // 2 + idx * self.seq_length
            return d[:, start: start + self.seq_length], t
        raise IndexError
