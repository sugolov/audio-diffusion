#!/usr/bin/env python3

import os
import subprocess
import numpy as np
import librosa
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil

def run_ffmpeg(command):
    try:
        result = subprocess.run(command, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        return result.stdout.decode()
    except subprocess.CalledProcessError as e:
        print(f"error running ffmpeg command: {e.stderr.decode()}")
        raise

# split raw mp3 file into standardized and normalized 30s wav segments
def mp3_to_wav_segments(input_file, output_dir, segment_length=30):
    base_name = os.path.splitext(os.path.basename(input_file))[0] # get track's base name
    temp_dir = os.path.join(output_dir, 'temp', base_name) 
    os.makedirs(temp_dir, exist_ok=True) # make directory for temporary files
    temp_wav = os.path.join(temp_dir, f"{base_name}_temp.wav") # temporary wav file
    normalized_wav = os.path.join(temp_dir, f"{base_name}_norm.wav") # temporary normalized wav file

    try:
        # convert raw mp3 audio to wav and resample at 44.1kHz
        print(f"converting mp3 to wav and resampling: {input_file}")
        run_ffmpeg([
            'ffmpeg', '-i', input_file, 
            '-acodec', 'pcm_s16le', 
            '-ar', '44100',
            '-ac', '1',
            temp_wav
        ])

        # normalize wav volume levels
        print(f"normalizing wav volume levels: {temp_wav}")
        run_ffmpeg([
            'ffmpeg', '-i', temp_wav, 
            '-filter:a', 'loudnorm=I=-23:LRA=7:TP=-2',
            normalized_wav
        ])

        # split normalized wav file into 30s segments
        print(f"splitting wav into segments: {normalized_wav}")
        segment_length_str = f"00:00:{segment_length}"
        run_ffmpeg([
            'ffmpeg', '-i', normalized_wav, 
            '-f', 'segment',  
            '-segment_time', segment_length_str,
            '-c', 'copy',
            os.path.join(temp_dir, f"{base_name}_%03d.wav")
        ])

        print(f"successfully split {input_file} into wav segments")
        return [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.startswith(base_name) and f.endswith('.wav') and f != os.path.basename(temp_wav) and f != os.path.basename(normalized_wav)]

    except Exception as e:
        print(f"error in mp3_to_wav_segments(): {str(e)}")
        raise

# create mel-spectrograms from a wav segment
def create_mel_spectrogram(audio_file, output_dir):
    try:
        y, sr = librosa.load(audio_file, sr=44100)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        mel_output_filename = f"{base_name}_mel.npy"
        mel_output_path = os.path.join(output_dir, mel_output_filename)
        np.save(mel_output_path, mel_spectrogram_db)
        return mel_output_path
    except Exception as e:
        print(f"error in create_mel_spectrogram(): {str(e)}")
        raise

# check if a track has already been processed
def is_track_processed(input_file, output_dir):
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    track_output_dir = os.path.join(output_dir, base_name)
    
    # Check if the output directory for this track exists and contains at least one mel-spectrogram
    if os.path.exists(track_output_dir):
        mel_spectrograms = [f for f in os.listdir(track_output_dir) if f.endswith('_mel.npy')]
        return len(mel_spectrograms) > 0

# end to end audio processing from mp3 to mel-spectrograms
def process_audio_file(input_file, output_dir, segment_length=30):
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    track_output_dir = os.path.join(output_dir, base_name)
    temp_dir = os.path.join(output_dir, 'temp', base_name)

    # check if track has already been processed to prevent duplicate work
    if is_track_processed(input_file, output_dir):
        print(f"skipping previously processed track: {input_file}")
        return
    
    try:
        # create directory to store intermediate wav files
        os.makedirs(track_output_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)
        
        # process audio with ffmpeg
        wav_segments = mp3_to_wav_segments(input_file, output_dir, segment_length)
        
        # create mel-spectrograms for each segment
        for segment_file in wav_segments:
            create_mel_spectrogram(segment_file, track_output_dir)
        
        print(f"successfully processed into mel-spectrograms: {input_file}")
    except Exception as e:
        print(f"error processing {input_file}: {str(e)}")
    finally:
        # clean up temporary files
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

# handle all mp3 tracks in a directory concurrently 
def process_dir(input_dir, output_dir):
    # create output if it doesn't yet exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) 

    # make complete list of mp3 files in the directory 
    audio_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.mp3', '.wav'))]
    
    # concurrently process the list of mp3 files
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_audio_file, os.path.join(input_dir, audio_file), output_dir) for audio_file in audio_files]
        for future in as_completed(futures):
            try:
                future.result()  # re-raise any exceptions that occurred in the worker process
            except Exception as e:
                print(f"error in worker process: {str(e)}")

# driver program
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process audio files: convert to WAV, normalize, segment, and create mel-spectrograms.")
    parser.add_argument("input_dir", help="Directory containing mp3 audio files")
    parser.add_argument("output_dir", help="Directory to save processed mel-spectrograms")
    
    args = parser.parse_args()
    
    process_dir(args.input_dir, args.output_dir)
