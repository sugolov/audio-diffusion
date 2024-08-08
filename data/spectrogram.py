import os
import torch
import numpy as np

from collections import deque


def slice_audio(mel, data_dir, audio_idx, audio_name, verbose=False):
    #print(os.path.join(data_dir, "train_audio", audio_name))
    mel.load_audio(os.path.join(data_dir, "train_audio", audio_name))
    num_slices = mel.get_number_of_slices()
    images = [mel.audio_slice_to_image(_) for _ in range(num_slices)]
    print(f"{num_slices} spectrograms in mix {audio_idx}") if verbose else None

    mix_dir = os.path.join(data_dir, "train", f"mix{audio_idx}")
    os.mkdir(mix_dir) if not os.path.exists(mix_dir) else None
    def save_image(i, image):
        image.save(os.path.join(mix_dir, f"{i}.jpg"))
        print("saved " + os.path.join(data_dir, "train", f"mix{audio_idx}", f"{i}.jpg")) if verbose else None

    [save_image(*s) for s in enumerate(images)]
    print(f"mix {audio_idx} done") if verbose else None

def get_spectrograms(data_dir, x_res=256, y_res=256, sample_rate=44100):
    """

    :param data_dir: working directory
        - should have a directory named `train_audio`
    """
    # Mel processes audio spectrogram generation with librosa
    from diffusers import Mel
    mel = Mel(x_res=x_res, y_res=y_res, sample_rate=sample_rate)
    mix_names = os.listdir(os.path.join(data_dir, 'train_audio'))

    [slice_audio(mel, data_dir, i, name, verbose=True) for i, name in enumerate(mix_names)]

def print_info(data_dir):
    for i, audio_name in enumerate(os.listdir(os.path.join(data_dir, 'train_audio'))):
        print(f"- mix{i}: " + audio_name)

if __name__ == "__main__":
    data_dir = "/run/media/anton/hdd/data/breakcore"
    #get_spectrograms(
    #    "/run/media/anton/hdd/data/breakcore"
    #)
    print_info(data_dir)