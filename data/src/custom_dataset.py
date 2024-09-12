#!/usr/bin/env python3

import os
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, DataLoader
from torch import Tensor
from typing import List, Tuple, Dict, Iterator

""" custom dataset for spectrograms that includes track information """
class AudioTrackDataset(Dataset):
    def __init__(self, root_dir: str) -> None:
        self.root_dir = root_dir
        self.tracks = []
        self.track_lengths = []
        self.cumulative_lengths = [0]

        for source in ["youtube_processed", "soundcloud_processed"]:
            source_dir = os.path.join(root_dir, source)
            for track in os.listdir(source_dir):
                track_dir = os.path.join(source_dir, track)
                spectrograms = [f for f in os.listdir(track_dir) if f.endswith("_mel.npy")]
                self.tracks.append((source, track))
                self.track_lengths.append(len(spectrograms))
                self.cumulative_lengths.append(self.cumulative_lengths[-1] + len(spectrograms))

    def __len__(self) -> int:
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        track_idx = np.searchsorted(self.cumulative_lengths, idx, side="right") - 1
        local_idx = idx - self.cumulative_lengths[track_idx]

        source, track = self.tracks[track_idx]
        spectrogram_path = os.path.join(self.root_dir, source, track, f"{track}_{local_idx:03d}_mel.npy")
        spectrogram = torch.from_numpy(np.load(spectrogram_path)).unsqueeze(0)

        return spectrogram, track_idx

""" custom collate function to handle batches with potentially different-sized spectrograms """
def collate_fn(batch: List[Tuple[Tensor, int]]) -> Tuple[Tensor, Tensor, Tensor]:
    # seperate spectrograms and track indices
    spectrograms, track_indices = zip(*batch)

    # get max dimensions 
    max_freq = max(spec.size(1) for spec in spectrograms)
    max_time = max(spec.size(2) for spec in spectrograms)

    padded_spectrograms = []
    for spec in spectrograms:
        # spec shape: (1, freq, time)
        padded_spec = torch.nn.functional.pad(
            spec,
            (0, max_time - spec.size(2), 0 , max_freq - spec.size(1)),
            mode="constant",
            value=0
        )
        padded_spectrograms.append(padded_spec)

    # stack padded spectrograms
    stacked_spectrograms = torch.stack(padded_spectrograms)

    # create mask for original non-padded parts
    masks = torch.zeros_like(stacked_spectrograms, dtype=torch.bool)
    for i, spec in enumerate(spectrograms):
        masks[i, :, :spec.size(1), :spec.size(2)] = True

    track_indices = torch.tensor(track_indices)

    return stacked_spectrograms, masks, track_indices

"""custom sampler for batching spectrograms corresponding to each track """
class TrackBatchSampler(Sampler):
    def __init__(self, dataset: AudioTrackDataset, batch_size: int) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.track_indices = np.argsort(dataset.track_lengths)[::-1] # sort tracks by descending length

    def __iter__(self) -> Iterator[List[int]]:
        batches = []
        current_batch = []
        current_track = None

        # ensure each batch has `batch_size` spectrograms or is the last batch for a given track
        for track_idx in self.track_indices:
            track_start = self.dataset.cumulative_lengths[track_idx]
            track_end = self.dataset.cumulative_lengths[track_idx + 1]

            for spectrogram_idx in range(track_start, track_end):
                if current_track != track_idx:
                    if current_batch:
                        batches.append(current_batch)
                        current_batch = []
                    current_track = track_idx

                current_batch.append(spectrogram_idx)

                if len(current_batch) == self.batch_size:
                    batches.append(current_batch)
                    current_batch = []

        if current_batch:
            batches.append(current_batch)

        # introduce randomness for the training process
        np.random.shuffle(batches)

        return iter(batches)

    def __len__(self) -> int:
        return sum(1 for _ in self.__iter__())

""" driver program to test dataset functionality """
def main():
    root_dir = "../training_data/processed_audio/"
    dataset = AudioTrackDataset(root_dir)
    sampler = TrackBatchSampler(dataset, batch_size=32)
    dataloader = DataLoader(
        dataset, 
        batch_sampler=sampler, 
        num_workers=16, 
        collate_fn=collate_fn
    )

    print(f"total number of spectrograms: {len(dataset)}")
    print(f"total number of tracks: {len(dataset.tracks)}")

    # loop through a few batches
    for batch_idx, (batch_spectrograms, batch_masks, batch_track_indices) in enumerate(dataloader):
        if batch_idx >= 5: # only print for the first 5 batches
            break

        print(f"\nBATCH_{batch_idx+1}")
        print(f"  batch shape: {batch_spectrograms.shape}")
        print(f"  mask shape: {batch_masks.shape}")
        print(f"  number of spectrograms in batch: {len(batch_spectrograms)}")
        print(f"  shape of first spectrogram in batch: {batch_spectrograms[0].shape}")
        print(f"  shape of last spectrogram in batch: {batch_spectrograms[-1].shape}")

        # count unique tracks in this batch
        unique_tracks = torch.unique(batch_track_indices).tolist()
        print(f"  number of unique tracks in batch: {len(unique_tracks)}")
        print(f"  track indices: {unique_tracks}")

        # print info about each unique track in batch
        for track_idx in unique_tracks:
            track_mask = (batch_track_indices == track_idx)
            track_spectrograms = batch_spectrograms[track_mask]
            track_actual_masks = batch_masks[track_mask]
            print(f"    Track: {track_idx}")
            print(f"      number of spectrograms: {len(track_spectrograms)}")
            print(f"      shape of spectrograms: {track_spectrograms.shape}")
            print(f"      actual (non-paddded) shapes:")
            for spec, mask in zip(track_spectrograms, track_actual_masks):
                actual_shape = mask.sum(dim=(1,2)).tolist()
                print(f"        {actual_shape}")

    # print info about a few individual tracks
    print("\nIndividual track info")
    for i in range(min(5, len(dataset.tracks))):
        print(f"TRACK_{i}")
        print(f"  source: {dataset.tracks[i][0]}")
        print(f"  name: {dataset.tracks[i][1]}")
        print(f"  number of spectrograms: {dataset.track_lengths[i]}")

if __name__ == "__main__":
    main()

