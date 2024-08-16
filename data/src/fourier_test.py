import librosa
import numpy as np
import sys


def list_fourier_coefficients(file_path):
    # Load the sound file
    y, sr = librosa.load(file_path, sr=None)  # sr=None keeps the original sample rate

    # Compute the Fourier Transform
    fft_coeffs = np.fft.fft(y)

    # List the Fourier Coefficients
    for i, coeff in enumerate(fft_coeffs):
        print(f"Coefficient {i}: Real={coeff.real:.4f}, Imaginary={coeff.imag:.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 fourier_series.py <path_to_audio_file>")
        sys.exit(1)

    audio_file_path = sys.argv[1]
    list_fourier_coefficients(audio_file_path)
