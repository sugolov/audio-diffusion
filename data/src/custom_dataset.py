#!/usr/bin/env python3

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Union

class SpectrogramDataset(Dataset):
    def __init__(self, 
        root_dir: str, 
        return_metadata: bool = False
    ) -> None:
        """
        initialize the dataset.

        args:
            root_dir (str): root directory of the preprocessed dataset.
            return_metadata (bool): if True, return metadata along with spectrograms.
        """

        self.root_dir = root_dir
        self.return_metadata = return_metadata

        self.tracks: List[Dict[str, Any]] = []
        self.cumulative_lengths: List[int] = [0]

        # loop through directory structure and catalog every track
        for source in ['youtube_processed', 'soundcloud_processed']:
            source_dir: str = os.path.join(root_dir, source)
            for track in os.listdir(source_dir):
                track_dir = os.path.join(source_dir, track)
                if os.path.isdir(track_dir):
                    spectrograms = sorted([f for f in os.listdir(track_dir) if f.endswith('.npy')])
                    self.tracks.append({
                        'dir': track_dir,
                        'spectrograms': spectrograms,
                        'source': source,
                        'name': track
                    })
                    self.cumulative_lengths.append(self.cumulative_lengths[-1] + len(spectrograms))

    def __len__(self) -> int:
        """ return total number of spectrograms across all tracks """
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """ 
        get a single spectrogram sample from the dataset.

        args:
            idx (int): index of the item to retrieve

        returns:
            dict: a dictionary containing the spectrogram sample and optionally metadata
        """

        # find corresponding track for index
        track_idx = np.searchsorted(self.cumulative_lengths, idx, side="right") - 1
        local_idx = idx - self.cumulative_lengths[track_idx]

        track_info = self.tracks[track_idx]
        spectrogram_path = os.path.join(track_info["dir"], track_info["spectrograms"][local_idx])

        # load spectorgram into numpy array
        mel_spectrogram = np.load(spectrogram_path)

        result = {
                "spectrogram": torch.from_numpy(mel_spectrogram).bfloat16(),
                "length": mel_spectrogram.shape[1] 
        }

        if self.return_metadata:
            metadata = {
                "track_name": track_info["name"],  
                "source": track_info["source"],  
                "spectrogram_index": local_idx,
                "total_spectrograms": len(track_info["spectrograms"]),
                "duration": "0.5s" #TODO: don't hardcode this - do it programatically
            }
            result["metadata"] = metadata

        return result

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Union[torch.Tensor, List[Dict[str, Any]]]]:
    batch = sorted(batch, key=lambda x: x["length"], reverse=True)

    spectrograms = [item["spectrogram"] for item in batch]
    lengths = [item["length"] for item in batch]

    max_len = max(lengths)
    padded_spectrograms = []

    for spec in spectrograms:
        if spec.shape[1] < max_len:
            padding = torch.zeros(spec.shape[0], max_len - spec.shape[1])
            padded_spec = torch.cat([spec, padding], dim=1)
        else:
            padded_spec = spec
        padded_spectrograms.append(padded_spec)

    # stack padded spectrograms
    stacked_spectrograms = torch.stack(padded_spectrograms)

    result = {
        "spectrogram": stacked_spectrograms,
        "lengths": torch.tensor(lengths)
    }

    # include metadata if it's present 
    if "metadata" in batch[0]:
        result["metadata"] = [item["metadata"] for item in batch]

    return result


# driver program
def main():
    # example usage
    dataset = SpectrogramDataset(
        root_dir="../training_data/processed_audio/", 
        return_metadata=True
    )

    dataloader = DataLoader(dataset, batch_size=32, num_workers=4, collate_fn=collate_fn)

    print(f"dataloader length: {len(dataloader)}")
    print(f"batches in dataloader: {len(dataloader)//32}")

    index = 0
    for batch in dataloader:
        spectrograms = batch["spectrogram"]
        lens = batch["lengths"]
        print(f"spectrogram shape: {spectrograms.shape}") # [batch_size, n_mels, max_length_in_batch]
        print(f"lengths: {lens}")
        if "metadata" in batch:
            metadata = batch["metadata"]
            print(f"metadata for first item: {metadata[0]}")
        index += 1
        if index == 2:
            break

if __name__ == "__main__":
    main()
